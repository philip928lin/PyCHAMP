r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Jun 10, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""
import random
import numpy as np
from dotmap import DotMap
import mesa
from ..opt_model import OptModel
from .field import Field
from .well import Well
from .finance import Finance

class Farmer(mesa.Agent):
    """
    A farmer agent class.

    Attributes
    ----------
    agt_id : str or int
        Agent id.
    config : dict or DotMap
        General info of the model.
    agt_attrs : dict
        This dictionary contains all attributes and settings to the agent's
        simulation.
    prec_aw_dict : dict
        A dictionary with field_id as its key and annual precipitation during
        growing season as its value.
    prec_dict : dict
        A dictionary with field_id as its key and annual precipitation as its
        value.
    aquifers : dict
        A dictionary contain aquifer objects. The key is aquifer id.
    model : object
        MESA model object for a datacollector.
    """
    # def __init__(self, agt_id, config, agt_attrs,
    #              prec_aw_dict, prec_dict, temp_dict, aquifers):
    def __init__(self, agt_id, config, agt_attrs,
                 prec_aw_dict, prec_dict, aquifers, model=None):
        """
        Setup an agent (farmer).

        """
        super().__init__(agt_id, model)
        # MESA required attributes
        self.unique_id = agt_id

        #========
        self.agt_id = agt_id
        config = DotMap(config)
        self.config = config
        agt_attrs = DotMap(agt_attrs)
        self.agt_attrs = agt_attrs
        self.aquifers = DotMap(aquifers)

        self.perceived_prec_aw = agt_attrs.perceived_prec_aw
        if agt_attrs.decision_making.alphas is None:
            self.agt_attrs.decision_making.alphas = config.consumat.alpha
        self.agt_attrs.decision_making.scale = config.consumat.scale
        dm_args = self.agt_attrs.decision_making
        crop_options = dm_args.crop_options
        tech_options = dm_args.tech_options

        # Initiate simulation objects
        fields = DotMap()
        for fk, v in agt_attrs.fields.items():
            fields[fk] = Field(
                field_id=fk, config=config, te=v.init.tech, crop=v.init.crop,
                lat=v.lat, dz=v.dz, crop_options=crop_options,
                tech_options=tech_options, aquifer_id=v.aquifer_id
                )
            fields[fk].rain_fed = v.rain_fed # for later convenience

        wells = DotMap()
        for wk, v in agt_attrs.wells.items():
            wells[wk] = Well(
                well_id=wk, config=config, r=v.r, k=v.k,
                st=aquifers[v.aquifer_id].st, sy=v.sy, l_wt=v.l_wt,
                eff_pump=v.eff_pump, eff_well=v.eff_well,
                aquifer_id=v.aquifer_id
                )
            wells[wk].pumping_capacity = v.pumping_capacity
        self.fields = fields
        self.wells = wells
        self.finance = Finance(config=config)
        # Note that do not store OptModel(name=agt_id) as agent's attribute as
        # OptModel is not pickable for parallel computing.

        # Initialize CONSUMAT
        self.sa_thre = config.consumat.satisfaction_threshold
        self.un_thre = config.consumat.uncertainty_threshold
        self.state = None
        self.satisfaction = None
        self.expected_sa = None
        self.uncertainty = None
        self.irrigation = None
        self.yield_pct = None
        self.needs = DotMap()
        self.agt_ids_in_network = agt_attrs.agt_ids_in_network
        self.agts_in_network = {}     # This will be dynamically updated in a simulation
        self.selected_agt_id_in_network = None # This will be populated after social comparison

        # Initialize dm_sol (mimicing opt_model's output)
        n_s = config["field"]["area_split"]
        n_c = len(crop_options)
        n_te = len(tech_options)

        dm_sols = DotMap()
        for fk, v in agt_attrs.fields.items():
            i_crop = np.zeros((n_s, n_c, 1))
            crop = v.init.crop
            if isinstance(crop, str):
                i_c = crop_options.index(crop)
                i_crop[:, i_c, 0] = 1
            else:
                for i, c in enumerate(crop):
                    i_c = crop_options.index(c)
                    i_crop[i, i_c, 0] = 1
            dm_sols[fk]["i_crop"] = i_crop
            dm_sols[fk]["pre_i_crop"] = crop

            i_te = np.zeros(n_te)
            i_te[tech_options.index(v.init.tech)] = 1
            dm_sols[fk]["i_te"] = i_te
            dm_sols[fk]["pre_i_te"] = v.init.tech
        # Run the optimization to solve irr depth with every other variables
        # fixed.
        self.dm_sols = self.make_dm(None, dm_sols=dm_sols, init=True)
        # Run the simulation to calculate satisfication and uncertainty
        self.update_climate_input(prec_aw_dict, prec_dict)
        self.run_simulation() # aquifers

        # Some other attributes
        self.current_step = 0

    def update_climate_input(self, prec_aw_dict, prec_dict):
        """
        Update the climate input before the step simulation.

        Parameters
        ----------
        prec_aw_dict : dict
            A dictionary with field_id as its key and annual precipitation during
            growing season as its value.
        prec_dict : dict
            A dictionary with field_id as its key and annual precipitation as its
            value.

        Returns
        -------
        None.

        """
        self.prec_aw_dict = prec_aw_dict
        self.prec_dict = prec_dict

    def step(self):
        """
        Simulate a single timestep.

        Returns
        -------
        self (for parallel computing purpose)

        """
        self.current_step += 1

        ### Optimization
        # Make decisions based on CONSUMAT theory
        state = self.state
        if state == "Imitation":
            self.make_dm_imitation()
        elif state == "Social comparison":
            self.make_dm_social_comparison()
        elif state == "Repetition":
            self.make_dm_repetition()
        elif state == "Deliberation":
            self.make_dm_deliberation()

        ### Simulation
        # Note prec_aw_dict, prec_dict, temp_dict have to be updated first.
        self.run_simulation()
        return self

    def run_simulation(self):
        prec_aw_dict, prec_dict = self.prec_aw_dict, self.prec_dict

        aquifers = self.aquifers
        fields = self.fields
        wells = self.wells

        # Optimization's output
        dm_sols = self.dm_sols

        # Simulate fields
        for fk, field in fields.items():
            irr = dm_sols[fk].irr
            i_crop = dm_sols[fk].i_crop
            i_te = dm_sols[fk].i_te
            field.step(
                irr=irr, i_crop=i_crop, i_te=i_te, prec_aw=prec_aw_dict[fk],
                prec=prec_dict[fk]
                )

        # Simulate wells
        allo_r = dm_sols.allo_r         # Well allocation ratio from opt
        allo_r_w = dm_sols.allo_r_w     # Well allocation ratio from opt
        field_ids = dm_sols.field_ids
        well_ids = dm_sols.well_ids
        total_v = sum([field.v for _, field in fields.items()])
        self.irrigation = total_v
        for k, wid in enumerate(well_ids):
            well = wells[wid]

            # Select the first year over the planning horizon from opt
            v = total_v * allo_r_w[k, 0]
            q = sum([fields[fid].q * allo_r[f,k,0] for f, fid in enumerate(field_ids)])
            l_pr = sum([fields[fid].l_pr * allo_r[f,k,0] for f, fid in enumerate(field_ids)])
            dwl = aquifers[well.aquifer_id].dwl

            well.step(v=v, dwl=dwl, q=q, l_pr=l_pr)

        # Calulate profit and pumping cost
        self.finance.step(fields=fields, wells=wells)

        # Form evaluation metric
        profit = self.finance.profit
        y_y = sum([field.y_y for _, field in fields.items()])/len(fields)
        self.yield_pct = y_y

        # Calculate satisfaction and uncertainty
        needs = self.needs
        dm_args = self.agt_attrs.decision_making
        eval_metric_vars = {
            "profit": profit/dm_args.scale.profit,
            "yield_pct": y_y/dm_args.scale.yield_pct}
        self.eval_metric_vars = DotMap(eval_metric_vars)

        def func(x, alpha=1):
            return 1-np.exp(-alpha * x)
        for a, alpha in dm_args.alphas.items():
            if alpha is None:
                continue
            needs[a] = func(eval_metric_vars[a], alpha=alpha)
        self.needs = DotMap(needs)

        eval_metric = dm_args.eval_metric
        satisfaction = needs[eval_metric]
        expected_sa = dm_sols.Sa[eval_metric]
        uncertainty = abs(expected_sa - satisfaction)

        # Update CONSUMAT state
        #self.expected_sa = expected_sa
        self.satisfaction = satisfaction
        self.expected_sa = expected_sa
        self.uncertainty = uncertainty
        sa_thre = self.sa_thre
        un_thre = self.un_thre
        if satisfaction >= sa_thre and uncertainty >= un_thre:
            self.state = "Imitation"
        elif satisfaction < sa_thre and uncertainty >= un_thre:
            self.state = "Social comparison"
        elif satisfaction >= sa_thre and uncertainty < un_thre:
            self.state = "Repetition"
        elif satisfaction < sa_thre and uncertainty < un_thre:
            self.state = "Deliberation"

    def make_dm(self, state, dm_sols, init=False):
        """
        Make decisions.

        Parameters
        ----------
        state : str
            State of CONSUMAT. Imitation, Social comparison, Repetition, and
            Deliberation.
        dm_sols : DotMap
            Solution dictionary from dm_model.
        init : bool, optional
            Is it for the initial run. The default is False.

        Returns
        -------
        DotMap
            Solution dictionary from dm_model.

        """
        config = self.config
        aquifers = self.aquifers
        dm_args = self.agt_attrs.decision_making

        fields = self.fields
        wells = self.wells

        # Locally create OptModel to make the Farmer object pickable for
        # parallel computing.
        console_output = config.gurobi.get("LogToConsole")
        dm = OptModel(name=self.agt_id, LogToConsole=console_output)
        dm.setup_ini_model(
            config=config, horizon=dm_args.horizon,
            eval_metric=dm_args.eval_metric, crop_options=dm_args.crop_options,
            tech_options=dm_args.tech_options,
            approx_horizon=dm_args.approx_horizon
            )

        for f, field in fields.items():
            if init:
                dm.setup_constr_field(
                    field_id=f,
                    prec_aw=dm_args.perceived_prec_aw,
                    pre_i_crop=dm_sols[f].pre_i_crop,
                    pre_i_te=dm_sols[f].pre_i_te,
                    rain_fed=field.rain_fed,
                    i_crop=dm_sols[f].i_crop,
                    i_rain_fed=None,
                    i_te=dm_sols[f].i_te)
                continue

            if state == "Deliberation":
                dm.setup_constr_field(
                    field_id=f,
                    prec_aw=dm_args.perceived_prec_aw,
                    pre_i_crop=dm_sols[f].pre_i_crop,
                    pre_i_te=dm_sols[f].pre_i_te,
                    rain_fed=field.rain_fed,
                    i_crop=None,
                    i_rain_fed=None,
                    i_te=None
                    )
            else:
                dm.setup_constr_field(
                    field_id=f,
                    prec_aw=dm_args.perceived_prec_aw,
                    pre_i_crop=dm_sols[f].pre_i_crop,
                    pre_i_te=dm_sols[f].pre_i_te,
                    rain_fed=field.rain_fed,
                    i_crop=dm_sols[f].i_crop,
                    i_rain_fed=dm_sols[f].i_rain_fed,
                    i_te=dm_sols[f].i_te
                    )

        for w, well in wells.items():
            aquifer_id = well.aquifer_id
            proj_dwl = np.mean(aquifers[aquifer_id].dwl_list[-dm_args.n_dwl:])
            dm.setup_constr_well(
                well_id=w, dwl=proj_dwl, st=well.st,
                l_wt=well.l_wt, r=well.r, k=well.k,
                sy=well.sy, eff_pump=well.eff_pump,
                eff_well=well.eff_well,
                pumping_capacity=well.pumping_capacity
                )

        water_rights = dm_sols.water_rights
        for wr_id, v in self.agt_attrs.water_rights.items():
            if v.status: # Check whether the wr is activated
                # Extract the water right setting from the previous opt run,
                # which we record the remaining water right fromt the previous
                # year. If the wr is newly activate in a simulation, then we
                # use the input to setup the wr.
                wr_args = water_rights.get(wr_id)
                if wr_args is None:
                    dm.setup_constr_wr(
                        water_right_id=wr_id, wr=v.wr,
                        field_id_list=v.field_id_list,
                        time_window=v.time_window, i_tw=v.i_tw,
                        remaining_wr=v.remaining_wr,
                        tail_method=v.tail_method
                        )
                else:
                    dm.setup_constr_wr(
                        water_right_id=wr_id, wr=wr_args.wr,
                        field_id_list=wr_args.field_id_list,
                        time_window=wr_args.time_window,
                        i_tw=wr_args.i_tw,
                        remaining_wr=wr_args.remaining_wr,
                        tail_method=wr_args.tail_method
                        )

        dm.setup_constr_finance()
        dm.setup_obj(alpha_dict=dm_args.alphas)
        dm.finish_setup(display_summary=dm_args.display_summary)
        dm.solve(
            keep_gp_model=dm_args.keep_gp_model,
            keep_gp_output=dm_args.keep_gp_output,
            display_report=dm_args.display_report
            )
        dm_sols = dm.sols
        dm.depose_gp_env()  # Release memory
        return dm_sols

    def make_dm_deliberation(self):
        """
        Make decision under deliberation status.

        Returns
        -------
        None.

        """
        self.dm_sols = self.make_dm(self.state, dm_sols=self.dm_sols)

    def make_dm_repetition(self):
        """
        Make decision under repetition status.

        Returns
        -------
        None.

        """
        self.dm_sols = self.make_dm(self.state, self.dm_sols)

    def make_dm_social_comparison(self):
        """
        Make decision under social comparison status.

        Returns
        -------
        None.

        """
        agt_ids_in_network = self.agt_ids_in_network
        agts_in_network = self.agts_in_network
        # Evaluate comparable
        dm_sols_list = []
        for agt_id in agt_ids_in_network:
            # !!! Here we assume no. fields, n_c and split are the same across agents
            dm_sols = self.make_dm(agts_in_network[agt_id].dm_sols)
            dm_sols_list.append(dm_sols)
        objs = [s.obj for s in dm_sols_list]
        selected_agt_obj = max(objs)
        select_agt_index = objs.index(selected_agt_obj)
        self.selected_agt_id_in_network = agt_ids_in_network[select_agt_index]

        # Agent's original choice
        dm_sols = self.make_dm_repetition()
        if dm_sols.obj >= selected_agt_obj:
            self.dm_sols = dm_sols
        else:
            self.dm_sols = dm_sols_list[select_agt_index]

    def make_dm_imitation(self):
        """
        Make decision under imitation status.

        Returns
        -------
        None.

        """
        selected_agt_id_in_network = self.selected_agt_id_in_network
        if selected_agt_id_in_network is None:
            selected_agt_id_in_network = random.choice(self.agt_ids_in_network)

        agts_in_network = self.agts_in_network

        dm_sols = self.make_dm(self.state, agts_in_network[selected_agt_id_in_network].dm_sols)
        self.dm_sols = dm_sols

# Archive
# class Farmer(mesa.Agent):
#     """
#     A farmer agent class.

#     Attributes
#     ----------
#     agt_id : str or int
#         Agent id.
#     config : dict or DotMap
#         General info of the model.
#     agt_attrs : dict
#         This dictionary contains all attributes and settings to the agent's
#         simulation.
#     prec_aw_dict : dict
#         A dictionary with field_id as its key and annual precipitation during
#         growing season as its value.
#     prec_dict : dict
#         A dictionary with field_id as its key and annual precipitation as its
#         value.
#     temp_dict : dict
#         A dictionary with field_id as its key and daily temporature list as its
#         value.
#     aquifers : dict
#         A dictionary contain aquifer objects. The key is aquifer id.
#     model : object
#         MESA model object for a datacollector.
#     """
#     # def __init__(self, agt_id, config, agt_attrs,
#     #              prec_aw_dict, prec_dict, temp_dict, aquifers):
#     def __init__(self, agt_id, config, agt_attrs,
#                  prec_aw_dict, prec_dict, temp_dict, aquifers, model=None):
#         """
#         Setup an agent (farmer).

#         """
#         super().__init__(agt_id, model)
#         # MESA required attributes
#         self.unique_id = agt_id

#         #========
#         self.agt_id = agt_id
#         config = DotMap(config)
#         self.config = config
#         agt_attrs = DotMap(agt_attrs)
#         self.agt_attrs = agt_attrs
#         self.aquifers = DotMap(aquifers)

#         self.perceived_prec_aw = agt_attrs.perceived_prec_aw
#         if agt_attrs.decision_making.alphas is None:
#             self.agt_attrs.decision_making.alphas = config.consumat.alpha
#         self.agt_attrs.decision_making.scale = config.consumat.scale
#         dm_args = self.agt_attrs.decision_making
#         crop_options = dm_args.crop_options
#         tech_options = dm_args.tech_options

#         # Initiate simulation objects
#         fields = DotMap()
#         for fk, v in agt_attrs.fields.items():
#             fields[fk] = Field(
#                 field_id=fk, config=config, te=v.init.tech, crop=v.init.crop,
#                 lat=v.lat, dz=v.dz, crop_options=crop_options,
#                 tech_options=tech_options, aquifer_id=v.aquifer_id
#                 )
#             fields[fk].rain_fed = v.rain_fed # for later convenience

#         wells = DotMap()
#         for wk, v in agt_attrs.wells.items():
#             wells[wk] = Well(
#                 well_id=wk, config=config, r=v.r, k=v.k,
#                 st=aquifers[v.aquifer_id].st, sy=v.sy, l_wt=v.l_wt,
#                 eff_pump=v.eff_pump, eff_well=v.eff_well,
#                 aquifer_id=v.aquifer_id
#                 )
#             wells[wk].pumping_capacity = v.pumping_capacity
#         self.fields = fields
#         self.wells = wells
#         self.finance = Finance(config=config)
#         # Note that do not store OptModel(name=agt_id) as agent's attribute as
#         # OptModel is not pickable for parallel computing.

#         # Initialize CONSUMAT
#         self.sa_thre = config.consumat.satisfaction_threshold
#         self.un_thre = config.consumat.uncertainty_threshold
#         self.state = None
#         self.satisfaction = None
#         self.uncertainty = None
#         self.irrigation = None
#         self.yield_pct = None
#         self.needs = DotMap()
#         self.agt_ids_in_network = agt_attrs.agt_ids_in_network
#         self.agts_in_network = {}     # This will be dynamically updated in a simulation
#         self.selected_agt_id_in_network = None # This will be populated after social comparison

#         # Initialize dm_sol (mimicing opt_model's output)
#         n_s = config["field"]["area_split"]
#         n_c = len(crop_options)
#         n_te = len(tech_options)

#         dm_sols = DotMap()
#         for fk, v in agt_attrs.fields.items():
#             i_crop = np.zeros((n_s, n_c, 1))
#             crop = v.init.crop
#             if isinstance(crop, str):
#                 i_c = crop_options.index(crop)
#                 i_crop[:, i_c, 0] = 1
#             else:
#                 for i, c in enumerate(crop):
#                     i_c = crop_options.index(c)
#                     i_crop[i, i_c, 0] = 1
#             dm_sols[fk]["i_crop"] = i_crop
#             dm_sols[fk]["pre_i_crop"] = crop

#             i_te = np.zeros(n_te)
#             i_te[tech_options.index(v.init.tech)] = 1
#             dm_sols[fk]["i_te"] = i_te
#             dm_sols[fk]["pre_i_te"] = v.init.tech
#         # Run the optimization to solve irr depth with every other variables
#         # fixed.
#         self.dm_sols = self.make_dm(None, dm_sols=dm_sols, init=True)
#         # Run the simulation to calculate satisfication and uncertainty
#         self.update_climate_input(prec_aw_dict, prec_dict, temp_dict)
#         self.run_simulation() # aquifers

#         # Some other attributes
#         self.current_step = 0

#     def update_climate_input(self, prec_aw_dict, prec_dict, temp_dict):
#         """
#         Update the climate input before the step simulation.

#         Parameters
#         ----------
#         prec_aw_dict : dict
#             A dictionary with field_id as its key and annual precipitation during
#             growing season as its value.
#         prec_dict : dict
#             A dictionary with field_id as its key and annual precipitation as its
#             value.
#         temp_dict : dict
#             A dictionary with field_id as its key and daily temporature list as its
#             value.

#         Returns
#         -------
#         None.

#         """
#         self.prec_aw_dict = prec_aw_dict
#         self.prec_dict = prec_dict
#         self.temp_dict = temp_dict

#     def step(self):
#         """
#         Simulate a single timestep.

#         Returns
#         -------
#         self (for parallel computing purpose)

#         """
#         self.current_step += 1

#         ### Optimization
#         # Make decisions based on CONSUMAT theory
#         state = self.state
#         if state == "Imitation":
#             self.make_dm_imitation()
#         elif state == "Social comparison":
#             self.make_dm_social_comparison()
#         elif state == "Repetition":
#             self.make_dm_repetition()
#         elif state == "Deliberation":
#             self.make_dm_deliberation()

#         ### Simulation
#         # Note prec_aw_dict, prec_dict, temp_dict have to be updated first.
#         self.run_simulation()
#         return self

#     def run_simulation(self):
#         prec_aw_dict, prec_dict, temp_dict = \
#             self.prec_aw_dict, self.prec_dict, self.temp_dict

#         aquifers = self.aquifers
#         fields = self.fields
#         wells = self.wells

#         # Optimization's output
#         dm_sols = self.dm_sols

#         # Simulate fields
#         for fk, field in fields.items():
#             irr = dm_sols[fk].irr
#             i_crop = dm_sols[fk].i_crop
#             i_te = dm_sols[fk].i_te
#             field.step(
#                 irr=irr, i_crop=i_crop, i_te=i_te, prec_aw=prec_aw_dict[fk],
#                 prec=prec_dict[fk], temp=temp_dict[fk]
#                 )

#         # Simulate wells
#         allo_r = dm_sols.allo_r         # Well allocation ratio from opt
#         allo_r_w = dm_sols.allo_r_w     # Well allocation ratio from opt
#         field_ids = dm_sols.field_ids
#         well_ids = dm_sols.well_ids
#         total_v = sum([field.v for _, field in fields.items()])
#         self.irrigation = total_v
#         for k, wid in enumerate(well_ids):
#             well = wells[wid]

#             # Select the first year over the planning horizon from opt
#             v = total_v * allo_r_w[k, 0]
#             q = sum([fields[fid].q * allo_r[f,k,0] for f, fid in enumerate(field_ids)])
#             l_pr = sum([fields[fid].l_pr * allo_r[f,k,0] for f, fid in enumerate(field_ids)])
#             dwl = aquifers[well.aquifer_id].dwl

#             well.step(v=v, dwl=dwl, q=q, l_pr=l_pr)

#         # Calulate profit and pumping cost
#         self.finance.step(fields=fields, wells=wells)

#         # Form evaluation metric
#         profit = self.finance.profit
#         y_y = sum([field.y_y for _, field in fields.items()])/len(fields)
#         self.yield_pct = y_y

#         # Calculate satisfaction and uncertainty
#         needs = self.needs
#         dm_args = self.agt_attrs.decision_making
#         eval_metric_vars = {
#             "profit": profit/dm_args.scale.profit,
#             "yield_pct": y_y/dm_args.scale.yield_pct}
#         self.eval_metric_vars = DotMap(eval_metric_vars)

#         def func(x, alpha=1):
#             return 1-np.exp(-alpha * x)
#         for a, alpha in dm_args.alphas.items():
#             if alpha is None:
#                 continue
#             needs[a] = func(eval_metric_vars[a], alpha=alpha)
#         self.needs = DotMap(needs)

#         eval_metric = dm_args.eval_metric
#         satisfaction = needs[eval_metric]
#         expected_sa = dm_sols.Sa[eval_metric]
#         uncertainty = abs(expected_sa - satisfaction)

#         # Update CONSUMAT state
#         #self.expected_sa = expected_sa
#         self.satisfaction = satisfaction
#         self.uncertainty = uncertainty
#         sa_thre = self.sa_thre
#         un_thre = self.un_thre
#         if satisfaction >= sa_thre and uncertainty >= un_thre:
#             self.state = "Imitation"
#         elif satisfaction < sa_thre and uncertainty >= un_thre:
#             self.state = "Social comparison"
#         elif satisfaction >= sa_thre and uncertainty < un_thre:
#             self.state = "Repetition"
#         elif satisfaction < sa_thre and uncertainty < un_thre:
#             self.state = "Deliberation"

#     def make_dm(self, state, dm_sols, init=False):
#         """
#         Make decisions.

#         Parameters
#         ----------
#         state : str
#             State of CONSUMAT. Imitation, Social comparison, Repetition, and
#             Deliberation.
#         dm_sols : DotMap
#             Solution dictionary from dm_model.
#         init : bool, optional
#             Is it for the initial run. The default is False.

#         Returns
#         -------
#         DotMap
#             Solution dictionary from dm_model.

#         """
#         config = self.config
#         aquifers = self.aquifers
#         dm_args = self.agt_attrs.decision_making

#         fields = self.fields
#         wells = self.wells

#         # Locally create OptModel to make the Farmer object pickable for
#         # parallel computing.
#         console_output = config.gurobi.get("LogToConsole")
#         dm = OptModel(name=self.agt_id, LogToConsole=console_output)
#         dm.setup_ini_model(
#             config=config, horizon=dm_args.horizon,
#             eval_metric=dm_args.eval_metric, crop_options=dm_args.crop_options,
#             tech_options=dm_args.tech_options,
#             approx_horizon=dm_args.approx_horizon
#             )

#         for f, field in fields.items():
#             if init:
#                 dm.setup_constr_field(
#                     field_id=f,
#                     prec_aw=dm_args.perceived_prec_aw,
#                     pre_i_crop=dm_sols[f].pre_i_crop,
#                     pre_i_te=dm_sols[f].pre_i_te,
#                     rain_fed=field.rain_fed,
#                     i_crop=dm_sols[f].i_crop,
#                     i_rain_fed=None,
#                     i_te=dm_sols[f].i_te)
#                 continue

#             if state == "Deliberation":
#                 dm.setup_constr_field(
#                     field_id=f,
#                     prec_aw=dm_args.perceived_prec_aw,
#                     pre_i_crop=dm_sols[f].pre_i_crop,
#                     pre_i_te=dm_sols[f].pre_i_te,
#                     rain_fed=field.rain_fed,
#                     i_crop=None,
#                     i_rain_fed=None,
#                     i_te=None
#                     )
#             else:
#                 dm.setup_constr_field(
#                     field_id=f,
#                     prec_aw=dm_args.perceived_prec_aw,
#                     pre_i_crop=dm_sols[f].pre_i_crop,
#                     pre_i_te=dm_sols[f].pre_i_te,
#                     rain_fed=field.rain_fed,
#                     i_crop=dm_sols[f].i_crop,
#                     i_rain_fed=dm_sols[f].i_rain_fed,
#                     i_te=dm_sols[f].i_te
#                     )

#         for w, well in wells.items():
#             aquifer_id = well.aquifer_id
#             proj_dwl = np.mean(aquifers[aquifer_id].dwl_list[-dm_args.n_dwl:])
#             dm.setup_constr_well(
#                 well_id=w, dwl=proj_dwl, st=well.st,
#                 l_wt=well.l_wt, r=well.r, k=well.k,
#                 sy=well.sy, eff_pump=well.eff_pump,
#                 eff_well=well.eff_well,
#                 pumping_capacity=well.pumping_capacity
#                 )

#         water_rights = dm_sols.water_rights
#         for wr_id, v in self.agt_attrs.water_rights.items():
#             if v.status: # Check whether the wr is activated
#                 # Extract the water right setting from the previous opt run,
#                 # which we record the remaining water right fromt the previous
#                 # year. If the wr is newly activate in a simulation, then we
#                 # use the input to setup the wr.
#                 wr_args = water_rights.get(wr_id)
#                 if wr_args is None:
#                     dm.setup_constr_wr(
#                         water_right_id=wr_id, wr=v.wr,
#                         field_id_list=v.field_id_list,
#                         time_window=v.time_window, i_tw=v.i_tw,
#                         remaining_wr=v.remaining_wr,
#                         tail_method=v.tail_method
#                         )
#                 else:
#                     dm.setup_constr_wr(
#                         water_right_id=wr_id, wr=wr_args.wr,
#                         field_id_list=wr_args.field_id_list,
#                         time_window=wr_args.time_window,
#                         i_tw=wr_args.i_tw,
#                         remaining_wr=wr_args.remaining_wr,
#                         tail_method=wr_args.tail_method
#                         )

#         dm.setup_constr_finance()
#         dm.setup_obj(alpha_dict=dm_args.alphas)
#         dm.finish_setup(display_summary=dm_args.display_summary)
#         dm.solve(
#             keep_gp_model=dm_args.keep_gp_model,
#             keep_gp_output=dm_args.keep_gp_output,
#             display_report=dm_args.display_report
#             )
#         dm_sols = dm.sols
#         dm.depose_gp_env()  # Release memory
#         return dm_sols

#     def make_dm_deliberation(self):
#         """
#         Make decision under deliberation status.

#         Returns
#         -------
#         None.

#         """
#         self.dm_sols = self.make_dm(self.state, dm_sols=self.dm_sols)

#     def make_dm_repetition(self):
#         """
#         Make decision under repetition status.

#         Returns
#         -------
#         None.

#         """
#         self.dm_sols = self.make_dm(self.state, self.dm_sols)

#     def make_dm_social_comparison(self):
#         """
#         Make decision under social comparison status.

#         Returns
#         -------
#         None.

#         """
#         agt_ids_in_network = self.agt_ids_in_network
#         agts_in_network = self.agts_in_network
#         # Evaluate comparable
#         dm_sols_list = []
#         for agt_id in agt_ids_in_network:
#             # !!! Here we assume no. fields, n_c and split are the same across agents
#             dm_sols = self.make_dm(agts_in_network[agt_id].dm_sols)
#             dm_sols_list.append(dm_sols)
#         objs = [s.obj for s in dm_sols_list]
#         selected_agt_obj = max(objs)
#         select_agt_index = objs.index(selected_agt_obj)
#         self.selected_agt_id_in_network = agt_ids_in_network[select_agt_index]

#         # Agent's original choice
#         dm_sols = self.make_dm_repetition()
#         if dm_sols.obj >= selected_agt_obj:
#             self.dm_sols = dm_sols
#         else:
#             self.dm_sols = dm_sols_list[select_agt_index]

#     def make_dm_imitation(self):
#         """
#         Make decision under imitation status.

#         Returns
#         -------
#         None.

#         """
#         selected_agt_id_in_network = self.selected_agt_id_in_network
#         if selected_agt_id_in_network is None:
#             selected_agt_id_in_network = random.choice(self.agt_ids_in_network)

#         agts_in_network = self.agts_in_network

#         dm_sols = self.make_dm(self.state, agts_in_network[selected_agt_id_in_network].dm_sols)
#         self.dm_sols = dm_sols