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
from ..opt_model import OptModel
from .field import Field
from .well import Well
from .finance import Finance

class Farmer():#ap.Agent):
    def setup(self, agt_id, config, agent_dict, fields_dict, wells_dict,
              prec_dict, temp_dict, aquifers,
              crop_options=["corn", "sorghum", "soybean", "fallow"],
              tech_options=["center pivot", "center pivot LEPA"]):
        """
        Setup an agent (farmer).

        Parameters
        ----------
        agt_id : str or int
            Agent id.
        config : dict or DotMap
            A dictionary store in DotMap format. DotMap is a python package.
        agent_dict : dict
            A dictionary containing an agent's settings. E.g.,\n
            agent_dict = {"horizon": 5,
                          "eval_metric": "profit",
                          "risk_attitude_prec": 30,
                          "n_dwl": 5,
                          "comparable_agt_ids": [],
                          "alphas": None,
                          "init": {
                              "te": "center pivot",
                              "crop_type": "corn"}}
            horizon : int
                The planing horizon [yr]. The default is 5.
            eval_metric : str
                evaluation metric.
            risk_attitude_prec : float
                Annual precipitation of a specific quantile of historical records.
        fields_dict : dict
            A dictionary containing an agent's field settings. E.g.,\n
            fields_dict = {"field1": {"te": "center pivot",
                                      "lat": 39.4,
                                      "dz": None,
                                      "rain_fed_option": False}}
        wells_dict : dict
            A dictionary containing an agent's well settings. E.g.,\n
            wells_dict = {"well1": {"r": 0.05,
                                    "tr": 1,
                                    "sy": 1,
                                    "l_wt": 10,
                                    "eff_pump": 0.77,
                                    "eff_well": 0.5,
                                    "aquifer_id": "ac1",
                                    "pumping_capacity": None}}
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
            default is ["corn", "sorghum", "soybean", "fallow"].
        tech_options : list, optional
            A list of irrigation technologies. They must exist in the config.
            The default is ["center pivot", "center pivot LEPA"].

        Returns
        -------
        None.

        """
        self.agt_id = agt_id
        fdict = DotMap(fields_dict)
        wdict = DotMap(wells_dict)
        agtdict = DotMap(agent_dict)
        # self.fdict & self.wdict & agtdict can be deleted to save memory
        self.fdict = fdict
        self.wdict = wdict
        self.agtdict = agtdict

        self.aquifers = aquifers  #!!! check this!

        self.field_list = list(fields_dict.keys())
        self.well_list = list(wells_dict.keys())
        self.n_h = agtdict.horizon
        self.n_dwl = agtdict.n_dwl
        self.eval_metric = agtdict.eval_metric
        self.crop_options = crop_options
        self.tech_options = tech_options
        config = DotMap(config)
        self.config = config
        self.risk_attitude_prec = agtdict.risk_attitude_prec
        if agtdict.alphas is None:
            self.alphas = config.consumat.alpha
        else:
            self.alphas = agtdict.alphas

        # Create containers for simulation objects
        fields = DotMap()
        wells = DotMap()
        for f, v in fdict.items():
            fields[f] = Field(field_id=f, config=config, te=v.te, lat=v.lat, dz=v.dz,
                              crop_options=crop_options,
                              tech_options=tech_options)
            fields[f].rain_fed_option = v.rain_fed_option
        for w, v in wdict.items():
            wells[w] = Well(well_id=w, config=config, r=v.r, tr=v.tr, sy=v.sy,
                            l_wt=v.l_wt, eff_pump=v.eff_pump,
                            eff_well=v.eff_well, aquifer_id=v.aquifer_id)
            wells[w].pumping_capacity = v.pumping_capacity
        self.fields = fields
        self.wells = wells
        self.finance = Finance(config=config, crop_options=crop_options)
        # self.dm_model = OptModel(name=agt_id) # Not pickable for parallel computing

        # Initialize CONSUMAT
        self.sa_thre = config.consumat.satisfaction_threshold
        self.un_thre = config.consumat.uncertainty_threshold
        self.state = None
        self.satisfaction = None
        self.uncertainty = None
        self.needs = DotMap()
        self.comparable_agt_ids = agtdict.comparable_agt_ids
        self.comparable_agts = {}   # This should be dynamically updated in the simulation
        self.comparable_agt_id = None # This will be populated after social comparison

        # Initialize dm_sol
        n_te = len(tech_options)
        i_te = np.zeros(n_te)
        i_te[tech_options.index(agtdict.init.te)] = 1

        n_s = config["field"]["area_split"]
        n_c = len(crop_options)
        i_area = np.zeros((n_s, n_c, 1))
        crop_type = agtdict.init.crop_type
        if isinstance(crop_type, str):
            i_c = crop_options.index(crop_type)
            i_area[:, i_c, 0] = 1
        else:
            for i, c in enumerate(crop_type):
                i_c = crop_options.index(c)
                i_area[i, i_c, 0] = 1
        dm_sols = DotMap()
        for f, v in fdict.items():
            dm_sols[f]["i_area"] = i_area
            dm_sols[f]["i_te"] = i_te
        self.dm_sols = self.make_dm(dm_sols=dm_sols, init=True)
        self.make_simulation(prec_dict, temp_dict) # aquifers

    def sim_step(self, prec_dict, temp_dict):
        """
        Simulate a single timestep.

        Parameters
        ----------
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

        ### Simulation
        self.make_simulation(prec_dict, temp_dict)

    def make_simulation(self, prec_dict, temp_dict):  # aquifers
        aquifers = self.aquifers
        eval_metric = self.eval_metric
        alphas = self.alphas
        fields = self.fields
        wells = self.wells

        dm_sols = self.dm_sols
        # Simulate over fields
        for f, field in fields.items():
            irr = dm_sols[f].irr
            i_area = dm_sols[f].i_area
            i_te = dm_sols[f].i_te
            field.sim_step(irr=irr, i_area=i_area, i_te=i_te,
                           prec=prec_dict[f], temp=temp_dict[f])

        # Simulate over wells
        for w, well in wells.items():
            # Here we simply adopt opt solutions for demonstration
            v = dm_sols[w].v
            dwl = aquifers[well.aquifer_id].dwl
            q = dm_sols[w].q
            l_pr = dm_sols[w].l_pr
            well.sim_step(v=v, dwl=dwl, q=q, l_pr=l_pr)

        # Calulate profit and pumping cost
        y = sum([field.y for f, field in fields.items()])
        y_y = sum([field.y_y for f, field in fields.items()])/len(fields)
        e = sum([well.e for w, well in wells.items()])
        self.finance.sim_step(e=e, y=y)
        profit = self.finance.profit

        eval_metric_vars = {
            "profit": profit,
            "yield_pct": y_y}

        # Calculate satisfaction and uncertainty
        needs = self.needs
        eval_metric = self.eval_metric
        def func(x, alpha=1):
            return 1-np.exp(-alpha * x)
        for a, alpha in alphas.items():
            if alpha is None:
                continue
            needs[a] = func(eval_metric_vars[a], alpha=alpha)

        satisfaction = needs[eval_metric]
        obj_val = dm_sols.obj
        uncertainty = abs(obj_val - satisfaction)

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

    def make_dm(self, dm_sols=None, init=False):
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
        n_h = self.n_h
        n_dwl = self.n_dwl
        crop_options = self.crop_options
        tech_options = self.tech_options
        risk_attitude_prec = self.risk_attitude_prec
        eval_metric = self.eval_metric
        fields = self.fields
        wells = self.wells
        alphas = self.alphas

        # Locally create OptModel to make the Farmer object pickable for parallel computing
        dm = OptModel(name=self.agt_id)
        dm.setup_ini_model(config=config, horizon=n_h, eval_metric=eval_metric,
                           crop_options=crop_options, tech_options=tech_options)

        for f, field in fields.items():
            for f, field in fields.items():
                if init:
                    dm.setup_constr_field(field_id=f, prec=risk_attitude_prec,
                                          i_area=dm_sols[f].i_area,
                                          i_rain_fed=None,
                                          rain_fed_option=field.rain_fed_option,
                                          i_te=dm_sols[f].i_te)
                    continue

                if dm_sols is None:
                    dm.setup_constr_field(field_id=f, prec=risk_attitude_prec,
                                          i_area=None,
                                          i_rain_fed=None,
                                          rain_fed_option=field.rain_fed_option,
                                          i_te=None)
                else:
                    dm.setup_constr_field(field_id=f, prec=risk_attitude_prec,
                                          i_area=dm_sols[f].i_area,
                                          i_rain_fed=dm_sols[f].i_rain_fed,
                                          rain_fed_option=field.rain_fed_option,
                                          i_te=dm_sols[f].i_te)

        for w, well in wells.items():
            #proj_dwl = 0
            aquifer_id = self.wdict[w]["aquifer_id"]
            proj_dwl = np.mean(aquifers[aquifer_id].dwl_list[-n_dwl:])
            dm.setup_constr_well(well_id=w, dwl=proj_dwl, l_wt=well.l_wt,
                                 r=well.r, tr=well.tr, sy=well.sy,
                                 eff_pump=well.eff_pump,
                                 eff_well=well.eff_well,
                                 pumping_capacity=well.pumping_capacity)
        #!!! missing water rights here (do it later)
        dm.setup_constr_finance()
        dm.setup_obj(alpha_dict=alphas)
        dm.finish_setup()
        dm.solve(keep_gp_model=False, keep_gp_output=False)
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
        self.dm_sols = self.make_dm(dm_sols=None)

    def make_dm_repetition(self):
        """
        Make decision with repetition.

        Returns
        -------
        None.

        """
        self.dm_sols = self.make_dm(self.dm_sols)

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

        dm_sols = self.make_dm(comparable_agts[comparable_agt_id].dm_sols)
        self.dm_sols = dm_sols