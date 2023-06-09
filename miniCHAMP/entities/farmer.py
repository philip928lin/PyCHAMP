r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

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

class Farmer(mesa.Agent):#ap.Agent):
    # def setup(self, agt_id, config, agt_attrs, prec_aw_dict, prec_dict, temp_dict, aquifers,
    #           crop_options=["corn", "sorghum", "soybeans", "fallow"],
    #           tech_options=["center pivot", "center pivot LEPA"]):
    def __init__(self, agt_id, config, agt_attrs,
                 prec_aw_dict, prec_dict, temp_dict, aquifers):
        """
        Setup an agent (farmer).

        Parameters
        ----------
        agt_id : str or int
            Agent id.
        config : dict or DotMap
            A dictionary store in DotMap format. DotMap is a python package.
        agent_dict : dict
            A dictionary containing an agent's settings.
            horizon : int
                The planing horizon [yr]. The default is 5.
            eval_metric : str
                evaluation metric.
            perceived_prec_aw : float
                Annual precipitation of a specific quantile of historical records.
        fields_dict : dict
            A dictionary containing an agent's field settings.
        wells_dict : dict
            A dictionary containing an agent's well settings.
        water_rights_dict : dict
            A dictionary containing water rights settings.
        prec_aw_dict : dict
            The annual precipitation during grow season for each field. The
            format is e.g.,
            {"field1": 0.8}.
        prec_dict : dict
            The annual precipitation for each field. The format is e.g.,
            {"field1": 0.8}.
        temp_dict : dict
            The daily temperature for each field. The format is e.g.,
            {"field1": DataFrame}. The DataFrame should have a datetime index.
        aquifers : dict
            A dictionary containing aquifer object(s). The format is e.g.,
            {"aquifer1": aquifer object}.
        crop_options : list, optional
            A list of crop type options. They must exist in the config. The
            default is ["corn", "sorghum", "soybeans", "fallow"].
        tech_options : list, optional
            A list of irrigation technologies. They must exist in the config.
            The default is ["center pivot", "center pivot LEPA"].

        Returns
        -------
        None.

        """
        # MESA required attributes
        self.unique_id = agt_id

        self.agt_id = agt_id
        agt_attrs = DotMap(agt_attrs)
        # self.fdict & self.wdict & agtdict can be deleted to save memory
        self.agt_attrs = agt_attrs

        self.aquifers = DotMap(aquifers)

        self.field_list = list(agt_attrs.fields.keys())
        self.well_list = list(agt_attrs.wells.keys())

        config = DotMap(config)
        self.config = config
        self.perceived_prec_aw = agt_attrs.perceived_prec_aw
        if agt_attrs.decision_making.alphas is None:
            self.agt_attrs.decision_making.alphas = config.consumat.alpha
        dm_args = self.agt_attrs.decision_making
        crop_options = dm_args.crop_options
        tech_options = dm_args.tech_options

        # Create containers for simulation objects
        fields = DotMap()
        wells = DotMap()
        for f, v in agt_attrs.fields.items():
            fields[f] = Field(field_id=f, config=config, te=v.init.tech,
                              crop=v.init.crop,
                              lat=v.lat,
                              dz=v.dz,
                              crop_options=crop_options,
                              tech_options=tech_options)
            fields[f].rain_fed_option = v.rain_fed_option
        for w, v in agt_attrs.wells.items():
            wells[w] = Well(well_id=w, config=config, r=v.r, k=v.k,
                            st=aquifers[v.aquifer_id].st, sy=v.sy,
                            l_wt=v.l_wt, eff_pump=v.eff_pump,
                            eff_well=v.eff_well, aquifer_id=v.aquifer_id)
            wells[w].pumping_capacity = v.pumping_capacity
        self.fields = fields
        self.wells = wells
        self.finance = Finance(config=config)
        # self.dm_model = OptModel(name=agt_id) # Not pickable for parallel computing

        # Initialize CONSUMAT
        self.sa_thre = config.consumat.satisfaction_threshold
        self.un_thre = config.consumat.uncertainty_threshold
        self.state = None
        self.satisfaction = None
        self.uncertainty = None
        self.needs = DotMap()
        self.comparable_agt_ids = agt_attrs.comparable_agt_ids
        self.comparable_agts = {}   # This should be dynamically updated in the simulation
        self.comparable_agt_id = None # This will be populated after social comparison

        # Initialize dm_sol
        n_te = len(tech_options)
        n_s = config["field"]["area_split"]
        n_c = len(crop_options)

        dm_sols = DotMap()
        for f, v in agt_attrs.fields.items():
            i_crop = np.zeros((n_s, n_c, 1))
            crop = v.init.crop
            if isinstance(crop, str):
                i_c = crop_options.index(crop)
                i_crop[:, i_c, 0] = 1
            else:
                for i, c in enumerate(crop):
                    i_c = crop_options.index(c)
                    i_crop[i, i_c, 0] = 1
            dm_sols[f]["i_crop"] = i_crop
            dm_sols[f]["pre_i_crop"] = crop

            i_te = np.zeros(n_te)
            i_te[tech_options.index(v.init.tech)] = 1
            dm_sols[f]["i_te"] = i_te

            dm_sols[f]["pre_i_te"] = v.init.tech
        self.dm_sols = self.make_dm(None, dm_sols=dm_sols, init=True)

        self.prec_aw_dict, self.prec_dict, self.temp_dict = prec_aw_dict, prec_dict, temp_dict
        self.run_simulation(prec_aw_dict, prec_dict, temp_dict) # aquifers

    def update_climate_input(self, prec_aw_dict, prec_dict, temp_dict):
        self.prec_aw_dict, self.prec_dict, self.temp_dict = prec_aw_dict, prec_dict, temp_dict

    def step(self): #, prec_aw_dict, prec_dict, temp_dict):
        """
        Simulate a single timestep.

        Parameters
        ----------
        prec_aw_dict : dict
            The annual precipitation during grow season for each field. The
            format is e.g.,
            {"field1": 0.8}.
        prec_dict : dict, optional
            The annual precipitation for each field. The format is e.g.,
            {"field1": 0.8} The default is {}.
        temp_dict : dict, optional
            The daily temperature for each field. The format is e.g.,
            {"field1": DataFrame}. The DataFrame should have a datetime index.
            The default is {}.
        aquifers : dict, optional
            A dictionary containing aquifer object(s). The format is e.g.,
            {"aquifer1": aquifer object}. The default is {}.

        Returns
        -------
        None

        """
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

        ### Simulation (note prec_aw_dict, prec_dict, temp_dict have to be updated)
        self.run_simulation(self.prec_aw_dict, self.prec_dict, self.temp_dict)

    def run_simulation(self, prec_aw_dict, prec_dict, temp_dict):  # aquifers
        aquifers = self.aquifers
        fields = self.fields
        wells = self.wells

        dm_sols = self.dm_sols
        # Simulate over fields
        for f, field in fields.items():
            irr = dm_sols[f].irr
            i_crop = dm_sols[f].i_crop
            i_te = dm_sols[f].i_te
            field.step(irr=irr, i_crop=i_crop, i_te=i_te,
                           prec_aw=prec_aw_dict[f],
                           prec=prec_dict[f], temp=temp_dict[f])

        # Simulate over wells
        allo_r = dm_sols.allo_r
        allo_r_w = dm_sols.allo_r_w     # Well allocation ratio from opt
        field_ids = dm_sols.field_ids
        well_ids = dm_sols.well_ids
        total_v = sum([field.v for f, field in fields.items()])
        for k, wid in enumerate(well_ids):
            well = wells[wid]
            # select the first year
            v = total_v * allo_r_w[k, 0]
            q = sum([fields[fid].q * allo_r[f,k,0] for f, fid in enumerate(field_ids)])
            l_pr = sum([fields[fid].l_pr * allo_r[f,k,0] for f, fid in enumerate(field_ids)])
            dwl = aquifers[well.aquifer_id].dwl

            # Here we simply adopt opt solutions for demonstration
            # v = dm_sols[w].v
            # q = dm_sols[w].q
            # l_pr = dm_sols[w].l_pr
            well.step(v=v, dwl=dwl, q=q, l_pr=l_pr)

        # Calulate profit and pumping cost
        self.finance.step(fields=self.fields, wells=self.wells)
        profit = self.finance.profit

        y_y = sum([field.y_y for f, field in fields.items()])/len(fields)
        eval_metric_vars = {
            "profit": profit,
            "yield_pct": y_y}

        # Calculate satisfaction and uncertainty
        needs = self.needs
        dm_args = self.agt_attrs.decision_making
        eval_metric = dm_args.eval_metric
        def func(x, alpha=1):
            return 1-np.exp(-alpha * x)
        for a, alpha in dm_args.alphas.items():
            if alpha is None:
                continue
            needs[a] = func(eval_metric_vars[a], alpha=alpha)

        satisfaction = needs[eval_metric]
        expected_sa = dm_sols.Sa[eval_metric]
        uncertainty = abs(expected_sa - satisfaction)

        self.satisfaction = satisfaction
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
        dm : object
            dm_model.
        dm_sols : DotMap, optional
            Solutions of dm_model. The default is None.
        init : bool, optional
            Is it for the initial run. The default is False.

        Returns
        -------
        DotMap
            Solutions of dm_model.

        """
        aquifers = self.aquifers
        config = self.config
        dm_args = self.agt_attrs.decision_making

        fields = self.fields
        wells = self.wells


        # Locally create OptModel to make the Farmer object pickable for parallel computing
        dm = OptModel(name=self.agt_id)
        dm.setup_ini_model(
            config=config, horizon=dm_args.horizon,
            eval_metric=dm_args.eval_metric, crop_options=dm_args.crop_options,
            tech_options=dm_args.tech_options,
            approx_horizon=dm_args.approx_horizon
            )

        for f, field in fields.items():
            for f, field in fields.items():
                if init:
                    dm.setup_constr_field(
                        field_id=f,
                        prec_aw=dm_args.perceived_prec_aw,
                        pre_i_crop=dm_sols[f].pre_i_crop,
                        pre_i_te=dm_sols[f].pre_i_te,
                        rain_fed_option=field.rain_fed_option,
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
                        rain_fed_option=field.rain_fed_option,
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
                        rain_fed_option=field.rain_fed_option,
                        i_crop=dm_sols[f].i_crop,
                        i_rain_fed=dm_sols[f].i_rain_fed,
                        i_te=dm_sols[f].i_te
                        )

        for w, well in wells.items():
            #proj_dwl = 0
            aquifer_id = aquifer_id = well.aquifer_id
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
        dm.depose_gp_env() # Release memory
        return dm_sols

    def make_dm_deliberation(self):
        """
        Make decision with deliberation.

        Returns
        -------
        None.

        """
        self.dm_sols = self.make_dm(self.state, dm_sols=self.dm_sols)

    def make_dm_repetition(self):
        """
        Make decision with repetition.

        Returns
        -------
        None.

        """
        self.dm_sols = self.make_dm(self.state, self.dm_sols)

    def make_dm_social_comparison(self):
        """
        Make decision with social comparison.

        Returns
        -------
        None.

        """
        comparable_agt_ids = self.comparable_agt_ids
        comparable_agts = self.comparable_agts
        # Evaluate comparable
        dm_sols_list = []
        for agt_id in comparable_agt_ids:
            # !!! Here we assume no. fields, n_c and split are the same across agents
            dm_sols = self.make_dm(comparable_agts[agt_id].dm_sols)
            dm_sols_list.append(dm_sols)
        objs = [s.obj for s in dm_sols_list]
        selected_agt_obj = max(objs)
        select_agt_index = objs.index(selected_agt_obj)
        self.comparable_agt_id = comparable_agt_ids[select_agt_index]

        # Agent's original choice
        dm_sols = self.make_dm_repetition()
        if dm_sols.obj >= selected_agt_obj:
            self.dm_sols = dm_sols
        else:
            self.dm_sols = dm_sols_list[select_agt_index]

    def make_dm_imitation(self):
        """
        Make decision with imitation.

        Returns
        -------
        None.

        """
        comparable_agt_id = self.comparable_agt_id
        if comparable_agt_id is None:
            comparable_agt_id = random.choice(self.comparable_agt_ids)

        comparable_agts = self.comparable_agts

        dm_sols = self.make_dm(self.state, comparable_agts[comparable_agt_id].dm_sols)
        self.dm_sols = dm_sols