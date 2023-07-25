r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Jun 10, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""
import numpy as np
import mesa
from ..opt_model import OptModel
from ..util import Box

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
    aquifers : dict
        A dictionary contain aquifer objects. The key is aquifer id.
    model : object
        MESA model object for a datacollector.
    """
    def __init__(self, agt_id, config, agt_attrs, fields, wells, finance,
                 aquifers, prec_aw_dict, model,
                 crop_options, tech_options, **kwargs):
        """
        Setup an agent (farmer).

        """
        super().__init__(agt_id, model)
        # MESA required attributes
        self.unique_id = agt_id

        # Load other kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        #========
        self.agt_id = agt_id
        self.crop_options = crop_options
        self.tech_options = tech_options

        # Load agt_attrs
        self.dm_args = agt_attrs["decision_making"]
        self.agt_ids_in_network = agt_attrs["agt_ids_in_network"]
        self.water_rights = agt_attrs["water_rights"]

        # Load config
        self.load_config(config)

        # Assign agt's assets
        # This container is built for dot access to fields and wells

        self.aquifers = aquifers
        self.fields = fields
        self.Fields = Box(fields) # same as self.fields but with dotted access
        self.wells = wells
        self.Wells = Box(wells) # same as self.fields but with dotted access
        self.finance = finance

        # Initialize CONSUMAT
        self.state = None
        self.satisfaction = None
        self.expected_sa = None     # From optimization
        self.uncertainty = None
        self.irr_vol = None         # m-ha
        self.profit = None
        self.scaled_profit = None
        self.scaled_yield_pct = None
        self.yield_pct = None
        self.needs = {}
        self.agts_in_network = {}   # This will be dynamically updated in a simulation
        self.selected_agt_id_in_network = None # This will be populated after social comparison

        # Initialize dm_sol (mimicing opt_model's output)
        dm_sols = {}
        for fi, field in self.fields.items():
            dm_sols[fi] = {}
            dm_sols[fi]["i_crop"] = field.i_crop
            dm_sols[fi]["pre_i_crop"] = field.pre_i_crop
            dm_sols[fi]["i_te"] = field.te
            dm_sols[fi]["pre_i_te"] = field.pre_te
        # Run the optimization to solve irr depth with every other variables
        # fixed.
        self.dm_sols = self.make_dm(None, dm_sols=dm_sols, init=True)
        # Run the simulation to calculate satisfication and uncertainty
        self.update_climate_input(prec_aw_dict)
        self.run_simulation() # aquifers

        # Some other attributes
        self.t = 0

    def load_config(self, config):
        """
        Load config.

        Parameters
        ----------
        config : dict
            General configuration of the model.

        Returns
        -------
        None.

        """
        config_consumat = config["consumat"]
        if self.dm_args["alphas"] is None:
            self.dm_args["alphas"] = config_consumat["alpha"]
        self.dm_args["scale"] = config_consumat["scale"]

        self.sa_thre = config_consumat["satisfaction_threshold"]
        self.un_thre = config_consumat["uncertainty_threshold"]
        self.n_s = config["field"]["area_split"]

        self.config_gurobi = config["gurobi"]
        self.config = config  # for opt only

    def update_climate_input(self, prec_aw_dict):
        """
        Update the climate input before the step simulation.

        Parameters
        ----------
        prec_aw_dict : dict
            A dictionary with field_id as its key and annual precipitation during
            growing season as its value.

        Returns
        -------
        None.

        """
        self.prec_aw_dict = prec_aw_dict

    def step(self):
        """
        Simulate a single timestep.

        Returns
        -------
        self (for parallel computing purpose)

        """
        self.t += 1

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

        # Retrieve opt info
        dm_sols = self.dm_sols
        self.gp_status = dm_sols['gp_status']
        self.gp_MIPGap = dm_sols['gp_MIPGap']
        self.gp_report = dm_sols['gp_report']

        ### Simulation
        # Note prec_aw_dict have to be updated externally first.
        self.run_simulation()

        return self

    def run_simulation(self):
        prec_aw_dict = self.prec_aw_dict

        aquifers = self.aquifers
        fields = self.fields
        wells = self.wells

        # Optimization's output
        dm_sols = self.dm_sols

        # agt dc settings
        dm_args = self.dm_args

        # Simulate fields
        for fi, field in fields.items():
            irr = dm_sols[fi]["irr"][:,:,[0]]

            i_crop = dm_sols[fi]["i_crop"]
            i_te = dm_sols[fi]["i_te"]
            fid = field.field_id
            field.step(
                irr=irr, i_crop=i_crop, i_te=i_te, prec_aw=prec_aw_dict[fid]
                )

        # Simulate wells
        allo_r = dm_sols['allo_r']         # Well allocation ratio from opt
        allo_r_w = dm_sols["allo_r_w"]     # Well allocation ratio from opt
        field_ids = dm_sols["field_ids"]
        well_ids = dm_sols["well_ids"]
        self.irr_vol = sum([field.irr_vol for _, field in fields.items()])

        for k, wid in enumerate(well_ids):
            well = wells[wid]

            # Select the first year over the planning horizon from opt
            v = self.irr_vol * allo_r_w[k, 0]
            q = sum([fields[fid].q * allo_r[f,k,0] for f, fid in enumerate(field_ids)])
            l_pr = sum([fields[fid].l_pr * allo_r[f,k,0] for f, fid in enumerate(field_ids)])
            dwl = aquifers[well.aquifer_id].dwl * dm_args["weight_dwl"]

            well.step(v=v, dwl=dwl, q=q, l_pr=l_pr)

        # Calulate profit and pumping cost
        self.finance.step(fields=fields, wells=wells)

        # Collect variables for evaluation metrices
        self.profit = self.finance.profit
        y_y = sum([field.y_y for _, field in fields.items()])/len(fields)
        self.yield_pct = y_y

        # Calculate satisfaction and uncertainty
        needs = self.needs
        scales = dm_args["scale"]
        self.scaled_profit = self.profit/scales["profit"]
        self.scaled_yield_pct = self.yield_pct/scales["yield_pct"]

        def func(x, alpha=1):
            return 1-np.exp(-alpha * x)
        alphas = dm_args["alphas"]
        for var, alpha in alphas.items():
            if alpha is None:
                continue
            needs[var] = func(eval(f"self.scaled_{var}"), alpha=alpha)

        eval_metric = dm_args["eval_metric"]
        satisfaction = needs[eval_metric]
        expected_sa = dm_sols["Sa"][eval_metric]
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

        aquifers = self.aquifers
        dm_args = self.dm_args

        fields = self.fields
        wells = self.wells

        # Locally create OptModel to make the Farmer object pickable for
        # parallel computing.
        # Note that do not store OptModel(name=agt_id) as agent's attribute as
        # OptModel is not pickable for parallel computing.
        dm = OptModel(name=self.agt_id,
                      LogToConsole=self.config_gurobi.get("LogToConsole"))
        dm.setup_ini_model(
            config=self.config,
            horizon=dm_args["horizon"],
            eval_metric=dm_args["eval_metric"],
            crop_options=self.crop_options,
            tech_options=self.tech_options,
            approx_horizon=dm_args["approx_horizon"]
            )
        for fi, field in fields.items():
            dm_sols_fi = dm_sols[fi]
            if init:
                dm.setup_constr_field(
                    field_id=fi,
                    prec_aw=field.perceived_prec_aw,
                    pre_i_crop=dm_sols_fi['pre_i_crop'],
                    pre_i_te=dm_sols_fi['pre_i_te'],
                    rain_fed=field.rain_fed,
                    i_crop=dm_sols_fi['i_crop'],
                    i_rain_fed=None,
                    i_te=dm_sols_fi['i_te']
                    )
                continue

            if state == "Deliberation":
                dm.setup_constr_field(
                    field_id=fi,
                    prec_aw=field.perceived_prec_aw,
                    pre_i_crop=dm_sols_fi['pre_i_crop'],
                    pre_i_te=dm_sols_fi['pre_i_te'],
                    rain_fed=field.rain_fed,
                    i_crop=None,
                    i_rain_fed=None,
                    i_te=None
                    )
            else:
                dm.setup_constr_field(
                    field_id=fi,
                    prec_aw=field.perceived_prec_aw,
                    pre_i_crop=dm_sols_fi['pre_i_crop'],
                    pre_i_te=dm_sols_fi['pre_i_te'],
                    rain_fed=field.rain_fed,
                    i_crop=dm_sols_fi['i_crop'],
                    i_rain_fed=dm_sols_fi['i_rain_fed'],
                    i_te=dm_sols_fi['i_te']
                    )

        for wi, well in wells.items():
            aquifer_id = well.aquifer_id
            proj_dwl = np.mean(aquifers[aquifer_id].dwl_list[-dm_args['n_dwl']:])
            dm.setup_constr_well(
                well_id=wi, dwl=proj_dwl, st=well.st,
                l_wt=well.l_wt, r=well.r, k=well.k,
                sy=well.sy, eff_pump=well.eff_pump,
                eff_well=well.eff_well,
                pumping_capacity=well.pumping_capacity
                )


        if init: # Inputted
            water_rights = self.water_rights
        else: # Use agent's own water rights (for social comparison and imitation)
            water_rights = self.dm_sols["water_rights"]

        for wr_id, v in self.water_rights.items():
            if v["status"]: # Check whether the wr is activated
                # Extract the water right setting from the previous opt run,
                # which we record the remaining water right fromt the previous
                # year. If the wr is newly activate in a simulation, then we
                # use the input to setup the wr.
                wr_args = water_rights.get(wr_id)
                if wr_args is None: # when first time introduce the water rights
                    dm.setup_constr_wr(
                        water_right_id=wr_id, wr=v["wr"],
                        field_id_list=v['field_id_list'],
                        time_window=v['time_window'],
                        i_tw=v['i_tw'],
                        remaining_wr=v['remaining_wr'],
                        tail_method=v['tail_method']
                        )
                else:
                    dm.setup_constr_wr(
                        water_right_id=wr_id, wr=wr_args['wr'],
                        field_id_list=wr_args['field_id_list'],
                        time_window=wr_args['time_window'],
                        i_tw=wr_args['i_tw'],
                        remaining_wr=wr_args['remaining_wr'],
                        tail_method=wr_args['tail_method']
                        )

        dm.setup_constr_finance()
        dm.setup_obj(alpha_dict=dm_args['alphas'])
        dm.finish_setup(display_summary=dm_args['display_summary'])
        dm.solve(
            keep_gp_model=dm_args['keep_gp_model'],
            keep_gp_output=dm_args['keep_gp_output'],
            display_report=dm_args['display_report']
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
        self.dm_sols = self.make_dm(state=self.state, dm_sols=self.dm_sols)

    def make_dm_repetition(self):
        """
        Make decision under repetition status.

        Returns
        -------
        None.

        """
        self.dm_sols = self.make_dm(state=self.state, dm_sols=self.dm_sols)

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
            dm_sols = self.make_dm(
                state=self.state,
                dm_sols=agts_in_network[agt_id].dm_sols
                )
            dm_sols_list.append(dm_sols)
        objs = [s['obj'] for s in dm_sols_list]
        selected_agt_obj = max(objs)
        select_agt_index = objs.index(selected_agt_obj)
        self.selected_agt_id_in_network = agt_ids_in_network[select_agt_index]

        # Agent's original choice
        self.make_dm_repetition()
        dm_sols = self.dm_sols
        if dm_sols['obj'] >= selected_agt_obj:
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
            selected_agt_id_in_network = self.rngen.choice(self.agt_ids_in_network)

        agts_in_network = self.agts_in_network

        dm_sols = self.make_dm(
            state=self.state,
            dm_sols=agts_in_network[selected_agt_id_in_network].dm_sols
            )
        self.dm_sols = dm_sols

