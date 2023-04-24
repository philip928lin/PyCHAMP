# -*- coding: utf-8 -*-
"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on April 24, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.

To do:
    Complete et calculation
"""
import random
import numpy as np
from pandas import to_numeric
from dotmap import DotMap
from .miniCHAMP_opt import OptModel

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
        self.dm_model = OptModel(name=agt_id)

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
        self.dm_sols = self.make_dm(dm=self.dm_model, dm_sols=dm_sols, init=True)
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

    def make_dm(self, dm, dm_sols=None, init=False):
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
        return dm.sols

    def make_dm_deliberation(self):
        """
        Make decision with deliberation.

        Returns
        -------
        None.

        """
        self.dm_sols = self.make_dm(self.dm_model, dm_sols=None)

    def make_dm_repetition(self):
        """
        Make decision with repetition.

        Returns
        -------
        None.

        """
        self.dm_sols = self.make_dm(self.dm_model, self.dm_sols)

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
            dm_sols = self.make_dm(self.dm_model, comparable_agts[agt_id].dm_sols)
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

        dm_sols = self.make_dm(self.dm_model, comparable_agts[comparable_agt_id].dm_sols)
        self.dm_sols = dm_sols

def cal_pet_Hamon(temp, lat, dz=None):
    """Calculate potential evapotranspiration (pet) with Hamon (1961) equation.

    Parameters
    ----------
    temp : numpy.ndarray
        Daily mean temperature [degC].
    lat : float
        Latitude [deg].
    dz : float, optional
        Altitude temperature adjustment [m], by default None.

    Returns
    -------
    numpy.ndarray
        Potential evapotranspiration [cm/day]

    Note
    ----
    The code is adopted from HydroCNHS (Lin et al., 2022).
    Lin, C. Y., Yang, Y. C. E., & Wi, S. (2022). HydroCNHS: A Python Package of
    Hydrological Model for Coupled Natural–Human Systems. Journal of Water
    Resources Planning and Management, 148(12), 06022005.
    """
    pdDatedateIndex = temp.index
    temp = temp.values.flatten()
    # Altitude temperature adjustment
    if dz is not None:
        # Assume temperature decrease 0.6 degC for every 100 m elevation.
        tlaps = 0.6
        temp = temp - tlaps*dz/100
    # Calculate Julian days
    # data_length = len(temp)
    # start_date = to_datetime(start_date, format="%Y/%m/%d")
    # pdDatedateIndex = date_range(start=start_date, periods=data_length,
    #                              freq="D")
    JDay = to_numeric(pdDatedateIndex.strftime('%j')) # convert to Julian days
    # Calculate solar declination [rad] from day of year (JDay) based on
    # equations 24 in ALLen et al (1998).
    sol_dec = 0.4093 * np.sin(2. * 3.141592654 / 365. * JDay - 1.39)
    lat_rad = lat*np.pi/180
    # Calculate sunset hour angle from latitude and solar declination [rad]
    # based on equations 25 in ALLen et al (1998).
    omega = np.arccos(-np.tan(sol_dec) * np.tan(lat_rad))
    # Calculate maximum possible daylight length [hr]
    dl = 24 / np.pi * omega
    # From Prudhomme(hess, 2013)
    # https://hess.copernicus.org/articles/17/1365/2013/hess-17-1365-2013-supplement.pdf
    # Slightly different from what we used to.
    pet = (dl / 12) ** 2 * np.exp(temp / 16)
    pet = np.array(pet/10)         # Convert from mm to cm
    pet[np.where(temp <= 0)] = 0   # Force pet = 0 when temperature is below 0.
    return pet      # [cm/day]


class Field():
    """
    A field simulator.

    Attributes
    ----------
    field_id : str or int
        Field id.
    config : dict or DotMap
        General info of the model.
    te : str, optional
        Irrigation technology. The default is None. The default is "center
        pivot".
    lat : float, optional
        Latitude of the field [deg]. This will be used in calculating
        evapotranspiration. The default is 39.4.
    dz : float, optional
        Gauge height adjustment. This will be used in calculating
        evapotranspiration. The default is None.
    crop_options : list, optional
        A list of crop type options. They must exist in the config. The
        default is ["corn", "sorghum", "soybean", "fallow"].
    tech_options : list, optional
        A list of irrigation technologies. They must exist in the config. The
        default is ["center pivot", "center pivot LEPA"].
    """

    def __init__(self, field_id, config, te="center pivot", lat=39.4, dz=None,
                 crop_options=["corn", "sorghum", "soybean", "fallow"],
                 tech_options=["center pivot", "center pivot LEPA"]):
        # for name_, value_ in vars().items():
        #     if name_ != 'self' and name_ != 'config':
        #         setattr(self, name_, value_)
        self.field_id = field_id
        self.lat = lat
        self.dz = dz
        config = DotMap(config)
        crop = np.array([config.field.crop[c] for c in crop_options])
        self.ymax = crop[:, 0].reshape((-1, 1))     # (n_c, 1)
        self.wmax = crop[:, 1].reshape((-1, 1))     # (n_c, 1)
        self.a = crop[:, 2].reshape((-1, 1))        # (n_c, 1)
        self.b = crop[:, 3].reshape((-1, 1))        # (n_c, 1)
        self.c = crop[:, 4].reshape((-1, 1))        # (n_c, 1)
        self.n_s = config.field.area_split
        self.unit_area = config.field.field_area/self.n_s

        self.tech_options = tech_options
        self.techs = config.field.tech
        self.update_irr_tech(te)

        self.cal_pet_Hamon = cal_pet_Hamon

    def update_irr_tech(self, i_te):
        """
        Update the irri

        Parameters
        ----------
        i_te : 1darray
            An array outputted from OptModel() that represent the indicator
            matrix for the irrigation technology choices in following year. The
            dimension of the array should be (n_te).

        Returns
        -------
        None.

        """
        if isinstance(i_te, str):
            new_te = i_te
        else:
            new_te = self.tech_options[np.where(i_te==1)[0][0]]
        self.a_te, self.b_te, self.l_pr = self.techs[new_te]
        self.te = new_te

    def sim_step(self, irr, i_area, i_te, prec, temp):
        """
        Simulate a single timestep.

        Parameters
        ----------
        irr : 3darray
            An array outputted from OptModel() that represent the irrigation
            depth for the following year. The dimension of the array should be
            (n_s, n_c, 1).
        i_area : 3darray
            An array outputted from OptModel() that represent the indicator
            matrix for the crop choices in following year. The dimension of the
            array should be (n_s, n_c, 1).
        i_te : 1darray
            An array outputted from OptModel() that represent the indicator
            matrix for the irrigation technology choices in following year. The
            dimension of the array should be (n_te).
        prec : float
            The annual precipitation amount.
        temp : DataFrame
            The daily mean temperature storing in DataFrame format with
            datetime index.

        Returns
        -------
        y : 3darray
            Crop yield with the dimension (n_s, n_c, 1).
        y_y : 3darray
            Crop yield/maximum yield with the dimension (n_s, n_c, 1).
        v : float
            Irrigation amount.
        inflow : float
            Inflow to the aquifer.

        """
        # Crop yield
        a = self.a
        b = self.b
        c = self.c
        ymax = self.ymax
        wmax = self.wmax
        n_s = self.n_s
        unit_area = self.unit_area

        w = irr + prec
        w = w * i_area
        w_ = w/wmax
        y_ = (a * w_**2 + b * w_ + c)
        # y_ = yw_ * i_area
        y = y_ * ymax
        v_c = irr * unit_area
        v = np.sum(v_c)
        y_y = np.sum(y_) / n_s

        self.y, self.y_y, self.v = y, y_y, v
        # Can be deleted in the future
        #assert w_ >= 0 and w_ <= 1, f"w_ in [0, 1] expected, got: {w_}"
        #assert y_ >= 0 and y_ <= 1, f"y_ in [0, 1]  expected, got: {y_}"

        # Tech
        self.update_irr_tech(i_te)  # update tech
        a_te = self.a_te
        b_te = self.b_te
        q = a_te * v + b_te
        self.q = q

        # Annual ET for aquifer
        # !!!! Calculate et + remember to average over the corner
        wv = np.sum(w * unit_area)
        pet = self.cal_pet_Hamon(temp, self.lat, self.dz)
        Kc = 1 # not yet decide how

        # Adopt the stress coeficient from GWLF
        Ks = np.ones(w_.shape)
        Ks[w_ <= 0.5] = 2 * w_[w_ <= 0.5]

        et = Ks * Kc * sum(pet) * unit_area
        inflow = wv - et    # volumn
        self.inflow = inflow
        return y, y_y, v, inflow

class Well():
    """
    A well simulator.

    Attributes
    ----------
    well_id : str or int
        Well id.
    config : dict or DotMap
        General info of the model.
    r : float
        Well radius.
    tr : float
        Transmissivity.
    sy : float
        Specific yield.
    l_wt : float
        Initial head for the lift from the water table to the ground
        surface at the start of the pumping season.
    eff_pump : float
        Pump efficiency. The default is 0.77.
    eff_well : float
        Well efficiency. The default is 0.5.
    aquifer_id : str or int, optional
        Aquifer id. The default is None.

    """
    def __init__(self, well_id, config, r, tr, sy, l_wt,
                 eff_pump=0.77, eff_well=0.5, aquifer_id=None):
        # for name_, value_ in vars().items():
        #     if name_ != 'self' and name_ != 'config':
        #         setattr(self, name_, value_)
        self.well_id, self.r, self.tr, self.sy, self.l_wt = \
            well_id, r, tr, sy, l_wt
        self.eff_pump, self.eff_well = eff_pump, eff_well
        self.aquifer_id = aquifer_id

        config = DotMap(config)
        self.rho = config.well.rho
        self.g = config.well.g

        self.t = 0

    def sim_step(self, v, dwl, q, l_pr):
        """
        Simulate a single timestep.

        Parameters
        ----------
        v : float
            Irrigation amount that will be withdraw from this well.
        dwl : float
            Groudwater level change.
        q : float
            Average daily pumping rate.
        l_pr : float
            The effective lift due to pressurization and of water and pipe
            losses necessary for the allocated irrigation system.

        Returns
        -------
        e : float
            Energy consumption.

        """
        # update groundwater level change from the last year
        self.l_wt += dwl
        l_wt = self.l_wt

        r, tr, sy = self.r, self.tr, self.sy
        eff_well, eff_pump = self.eff_well, self.eff_pump
        rho, g = self.rho, self.g

        cm_ha_2_m3 = 1000
        fpitr = 4 * np.pi * tr
        l_cd_l_wd = (1+eff_well) * q/fpitr \
                    * (-0.5772 - np.log(r**2*sy/fpitr)) * cm_ha_2_m3
        l_t = l_wt + l_cd_l_wd + l_pr
        e = rho * g * v * l_t / eff_pump * cm_ha_2_m3

        # record
        self.t += 1
        e = e[0]
        self.e = e
        return e

class Finance():
    """
    An finance simulator.

    Attributes
    ----------
    config : dict or DotMap
        General info of the model.
    crop_options : list, optional
        A list of crop type options. They must exist in the config. The
        default is ["corn", "sorghum", "soybean", "fallow"].

    """
    def __init__(self, config,
                 crop_options=["corn", "sorghum", "soybean", "fallow"]):
        config = DotMap(config)
        self.energy_price = config.finance.energy_price
        self.crop_profit = config.finance.crop_profit
        self.crop_options = crop_options

    def sim_step(self, e, y):
        """
        Simulate a single timestep.

        Parameters
        ----------
        e : float
            Total energy consumption.
        y : 3darray
            Crop yield with the dimension (n_s, n_c, 1).

        Returns
        -------
        profit : float
            Annual profit.

        """
        ep = self.energy_price
        cp = self.crop_profit
        crop_options = self.crop_options

        cost_e = e * ep
        rev = sum([y[i,j,:] * cp[c] for i in range(y.shape[0]) \
                   for j, c in enumerate(crop_options)])
        profit = rev - cost_e
        profit = profit[0]
        self.profit = profit
        return profit

class Aquifer():
    """
    An aquifer simulator.

    Attributes
    ----------

    """
    def __init__(self, aquifer_id, sy, area, lag, ini_inflow=0, ini_dwl=0):
        """


        Parameters
        ----------
        aquifer_id : str or int
            Aquifer id.
        sy : float
            Specific yield.
        area : float
            Area.
        lag : int
            Vertical percolation time lag for infiltrated water to contribute
            to the groundwater level change.
        ini_inflow : float, optional
            Initial inflow. This will be assigned for the first number of the
            "lag" years. The default is 0.
        ini_dwl : float, optional
            Initial groundwater level change. The default is 0.

        Returns
        -------
        None.

        """
        # for name_, value_ in vars().items():
        #     if name_ != 'self':
        #         setattr(self, name_, value_)
        self.aquifer_id, self.sy, self.area, self.lag = aquifer_id, sy, area, lag
        self.in_list = [ini_inflow]*lag
        self.t = 0
        self.dwl_list = [ini_dwl]
        self.dwl = ini_dwl

    def sim_step(self, inflow, v):
        """
        Simulate a single timestep.

        Parameters
        ----------
        inflow : float
            Inflow of the aquifer.
        v : float
            Total water withdraw from the aquifer.

        Returns
        -------
        dwl : float
            Groundwater level change.

        """
        in_list = self.in_list
        sy, area = self.sy, self.area

        in_list.append(inflow)
        inflow_lag = in_list.pop(0)
        dwl = 1/(area * sy) * (inflow_lag - v)

        self.dwl_list.append(dwl)
        self.t += 1
        self.dwl = dwl
        return dwl