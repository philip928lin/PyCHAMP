# -*- coding: utf-8 -*-
"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.

To do:
    Add example documentation (wait for the completion of the miniCHAMP)
"""
#import os
import json
import numpy as np
import gurobipy as gp
from dotmap import DotMap
#from .util import dict_to_string
def dict_to_string(dictionary, prefix="", indentor="  ", level=2, roun=None):
    """Ture a dictionary into a printable string.
    Parameters
    ----------
    dictionary : dict
        A dictionary.
    indentor : str, optional
        Indentor, by default "  ".

    Returns
    -------
    str
        A printable string.
    """
    def dict_to_string_list(dictionary, indentor="  ", count=1, string=[]):
        for key, value in dictionary.items():
            string.append(prefix + indentor * count + str(key))
            if isinstance(value, dict) and count < level:
                string = dict_to_string_list(value, indentor, count+1, string)
            elif isinstance(value, dict) is False and count == level:
                string[-1] += ":\t" + str(value)
            else:
                if roun is not None and isinstance(value, float):
                    string.append(prefix + indentor * (count+1) + str(round(value, roun)))
                else:
                    string.append(prefix + indentor * (count+1) + str(value))
        return string
    return "\n".join(dict_to_string_list(dictionary, indentor))
#################

class OptModel():
    """
    A class to represent a farmer making decisions on irrigation depth, crop
    types, rain-fed option, and irrigation technologies by solving an
    optimization model to maximize a farmer's satisfication (profit or yield_pct).
    The optimization problem is formulated on an annual scale. Crop types,
    rain-fed option, and irrigation technologies are optional decision
    variables. They can be given.

    The model can be built to address a farmer with multiple crop fields and
    groundwater wells. Water rights can be added as contraints to all fields or
    a subset of fields. The water rights on the point of diversion can be
    implemented by assigning pumping capacity to a well.

    Multiple crop types can be planted in a single field if the attribute
    'area_split' is larger then 1. For example, if area_split = 4, a field is
    uniformly split into 4 subfields. The farmer can make individual crop and
    irrigation decision for each of the subfields. area_split is extract from
    the 'config.'

    If the 'horizon' is larger than 1, only the irrigation depth are varied in
    each year. crop types, rain-fed option, and irrigation technologies are
    fixed over the planning horizon. User can run the optimization model again
    in the next year to update the decision if needed.

    Notes
    -----
    This class solves the optimization problem through Gurobi solver. It is a
    commercial solver. However, Gurobi provides full feature solver for
    academic use with no cost. Users will need to register a academic license
    and download the solver as well as install gurobipy python package to be
    able to run the code.

    More information can be found here:
    https://www.gurobi.com/academia/academic-program-and-licenses/

    Attributes
    ----------
    name : str, optional
        Name of the model. The default is "".

    Methods
    -------
    Left blank

    Examples
    --------
    Left blank

    """
    def __init__(self, name=""):
        """
        Create an optimization environment and object for a farmer.
        """
        self.name = name
        # Create a gurobi environment to ensure thread safety for parallel
        # computing.
        self.gpenv = gp.Env()
        self.model = gp.Model(name=name, env=self.gpenv)

        # Note from gurobi
        # In general, you should aim to create a single Gurobi environment in
        # your program, even if you plan to work with multiple models. Reusing
        # one environment is much more efficient than creating and destroying
        # multiple environments. The one exception is if you are writing a
        # multi-threaded program, since environments are not thread safe. In
        # this case, you will need a separate environment for each of your
        # threads.
    def depose_gp_env(self):
        """
        Clean gurobi environment. Run only when no longer need the instance.
        New optimization model (i.e., setup_ini_model) CANNOT be created
        after calling this method.

        Returns
        -------
        None.

        """
        self.gpenv.dispose()

    def setup_ini_model(self, config, horizon=5, eval_metric="profit",
                        crop_options=["corn", "sorghum", "soybean", "fallow"],
                        tech_options=["center pivot", "center pivot LEPA"],
                        approx_horizon=True):
        """
        Setup initial setting for an optimization model. This will
        automatically dispose the model created in the previous run. However,
        the model will be created in the same gurobi environment initialized
        with the creation of the class instance.


        Parameters
        ----------
        config : dict or DotMap
            General info of the model.
        horizon : str, optional
            The planing horizon [yr]. The default is 5.
        eval_metric : str, optional
            "profit" or "yield_pct". The default is "profit".
        crop_options : list, optional
            A list of crop type options. They must exist in the config. The
            default is ["corn", "sorghum", "soybean", "fallow"].
        tech_options : list, optional
            A list of irrigation technologies. They must exist in the config. The
            default is ["center pivot", "center pivot LEPA"].
        approx_horizon : bool, optional
            If True, the model will calculate two (start & end) points over the
            given horizon to calcalate the objective, which is the average over
            the horizon. This will significantly improve solving speed with
            long horizon. Most of the time this is equivalent to the original
            problem. This is built upon the assumption that groundwater level
            linearly decrease in the projected future.

        Returns
        -------
        None.

        """
        config = DotMap(config)
        self.crop_options = crop_options
        self.tech_options = tech_options
        self.eval_metric = eval_metric

        ## gurobi pars
        self.gurobi_pars = config.get("gurobi")
        if self.gurobi_pars is None:
            self.gurobi_pars = {}

        ## Dimension coefficients
        self.n_c = len(crop_options)    # NO. crop options (not distinguished
                                        # by rain-fed or irrigated)
        self.n_te = len(tech_options)   # NO. irr tech options
        self.approx_horizon = approx_horizon
        self.horizon = horizon          # Original planning horizon

        if approx_horizon:
            self.n_h = 2                # calculate obj with start and end points
        else:
            self.n_h = horizon

        ## Records fields and wells
        self.field_ids = []
        self.well_ids = []
        self.water_right_ids = []
        self.n_fields = 0
        self.n_wells = 0
        self.n_water_rights = 0

        ## Extract parameters from "config"
        crop = np.array([config.field.crop[c] for c in crop_options])
        self.ymax = crop[:, 0].reshape((-1, 1))     # (n_c, 1)
        self.wmax = crop[:, 1].reshape((-1, 1))     # (n_c, 1)
        self.a = crop[:, 2].reshape((-1, 1))        # (n_c, 1)
        self.b = crop[:, 3].reshape((-1, 1))        # (n_c, 1)
        self.c = crop[:, 4].reshape((-1, 1))        # (n_c, 1)
        self.growth_period_ratio = {c: config.field.growth_period_ratio[c] for c in crop_options}
        self.area_split = config.field.area_split
        self.unit_area = config.field.field_area/self.area_split

        self.rho = config.well.rho
        self.g = config.well.g
        self.techs = config.field.tech

        self.energy_price = config.finance.energy_price
        self.crop_profit = config.finance.crop_profit

        self.alphas = config.consumat.alpha
        self.eval_metrics = [metric for metric, v in self.alphas.items() if v is not None]

        ## Model
        #self.model.dispose()    # release the memory of the previous model
        self.model = gp.Model(name=self.name, env=self.gpenv)
        self.vars = DotMap()
        self.bounds = DotMap()
        self.bounds.ub_w = np.max(self.wmax)
        self.inf = float('inf')

        ## Add shared variables
        m = self.model
        inf = self.inf
        n_c = self.n_c
        n_h = self.n_h
        n_s = self.area_split
        # total irrigation depth per crop per yr
        irr = m.addMVar((n_s, n_c, n_h), vtype="C", name="irr(cm)", lb=0, ub=inf)
        # total irrigation volumn per yr
        v = m.addMVar((n_h), vtype="C", name="v(m-ha)", lb=0, ub=inf)
        # total yield per crop type per yr
        y = m.addMVar((n_s, n_c, n_h), vtype="C", name="y", lb=0, ub=inf)
        # average y_ (i.e., y/ymax) per yr
        y_y = m.addMVar((n_h), vtype="C", name="y_y", lb=0, ub=1)
        # total used electricity (pumping) per yr
        e = m.addMVar((n_h), vtype="C", name="e(PJ)", lb=0, ub=inf)
        # total profit
        profit = m.addMVar((n_h), vtype="C", name="profit(1e4$)", lb=0, ub=inf)

        self.vars.irr = irr
        self.vars.v = v
        self.vars.y = y
        self.vars.y_y = y_y
        self.vars.e = e
        self.vars.profit = profit

        ## Add input msg
        self.msg = {}

    def setup_constr_field(self, field_id, prec, rain_fed_option=False,
                           i_crop=None, i_rain_fed=None, i_te=None):
        """
        Add crop field constriants. Multiple fields can be assigned by calling
        this function multiple times with different field id. If i_crop (and
        i_rain_fed) is (are) given, the model will not optimize over different
        crop type options (and rain-fed options). If te is given, the model
        will not optimize over different irrigation technologies.

        Parameters
        ----------
        field_id : str or int
            Field id distinguishing equation sets for different fields.
        prec : float
            Percieved precipitation amount in the growing season [cm].
        i_crop : 3darray, optional
            Indicator matrix with the dimension of (area_split, number of crop
            type options, 1). 1 means the corresponding crop type is selected.
            The default is None.
        i_rain_fed : 3darray, optional
            Indicator matrix with the dimension of (area_split, number of crop
            type options, 1). 1 means the unit area in a field is rainfed.
            i_rain_fed is only used when rain_fed_option is True. Also, if it
            is given, make sure 1 only exists at where i_crop is also 1. If
            given, rain_fed_option will be forced to be True. The
            default is None.
        rain_fed_option : bool, optional
            True if allow rain_fed crop field options. The default is False.
        i_te : 1darray or str, optional
            Irrigation technology. If given, the program will not optimize over
            different irrigation technologies. The default is None.

        Returns
        -------
        None.

        """
        assert prec <= np.max(self.wmax), f"""prec {prec} is larger than wmax
        {np.max(self.wmax)}. This will lead to infeasible solution."""

        self.field_ids.append(field_id)
        fid = field_id
        if i_rain_fed is not None:
            rain_fed_option = True
        self.msg[fid] = {"Crop types": "optimize",
                         "Irr tech": "optimize",
                         "Rain-fed option": rain_fed_option,
                         "Rain-fed areas": (lambda o: "optimize" if o else None)(rain_fed_option)}

        i_crop_input = i_crop
        i_rain_fed_input = i_rain_fed
        i_te_input = i_te
        m = self.model

        n_c = self.n_c
        n_h = self.n_h
        n_s = self.area_split

        inf = self.inf
        a = self.a
        b = self.b
        c = self.c
        ymax = self.ymax
        wmax = self.wmax
        ub_w = self.bounds.ub_w
        ub_irr = ub_w - prec
        self.bounds[fid].ub_irr = ub_irr


        unit_area = self.unit_area

        ### Add general variables
        irr = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.irr(cm)", lb=0, ub=ub_irr)
        w   = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.w(cm)", lb=0, ub=ub_w)
        w_temp = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.w_temp", lb=0, ub=inf)
        w_  = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.w_", lb=0, ub=1)
        y   = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.y(1e4bu)", lb=0, ub=inf)
        y_  = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.y_", lb=0, ub=1)
        yw_temp  = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.yw_temp", lb=-inf, ub=1)
        yw_  = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.yw_", lb=0, ub=1)
        v_c = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.v_c(m-ha)", lb=0, ub=inf)
        y_y = m.addMVar((n_h), vtype="C", name=f"{fid}.y_y", lb=0, ub=1)    # avg y_ per yr
        v   = m.addMVar((n_h), vtype="C", name=f"{fid}.v(m-ha)", lb=0, ub=inf)
        i_crop = m.addMVar((n_s, n_c, 1), vtype="B", name=f"{fid}.i_crop")
        i_rain_fed = m.addMVar((n_s, n_c, 1), vtype="B", name=f"{fid}.i_rain_fed")

        # Given i_crop as an input (i.e., don't optimize over crop type
        # choices)
        # Otherwise, optimize i_crop (i.e., optimize over crop type choices)
        # Currently, crop type choice has to be the same accross planning
        # horizon.
        if i_crop_input is not None:
            #i_crop_input = np.repeat(i_crop_input[:, :, np.newaxis], 1, axis=2)
            m.addConstr(i_crop == i_crop_input, name=f"c.{fid}.i_crop_input")
            self.msg[fid]["Crop types"] = "user input"
        # One unit area can only be planted one type of crops.
        m.addConstr(gp.quicksum(i_crop[:,j,:] for j in range(n_c)) == 1,
                    name=f"c.{fid}.i_crop")

        ### Include rain-fed option
        if rain_fed_option:
            # Given i_rain_fed,
            if i_rain_fed_input is not None:
                # i_rain_fed_input = np.repeat(
                #     i_rain_fed_input[:, :, np.newaxis], 1, axis=2)
                m.addConstr(i_rain_fed == i_rain_fed_input,
                            name=f"c.{fid}.i_rain_fed_input")
                self.msg[fid]["Rain-fed areas"] = "user input"

            # i_rain_fed[i, j, h] can be 1 only when i_crop[i, j, h] is 1.
            # Otherwise, it has to be zero.
            m.addConstr(i_crop - i_rain_fed >= 0,
                        name=f"c.{fid}.i_rain_fed")

            m.addConstr(irr * i_rain_fed == 0, name=f"c.{fid}.irr_rain_fed")
        else:
            m.addConstr(i_rain_fed == 0,
                        name=f"c.{fid}.no_i_rain_fed")

        # See the numpy broadcast rules:
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        m.addConstr((w == irr + prec), name=f"c.{fid}.w(cm)")
        m.addConstr((w_temp == w/wmax), name=f"c.{fid}.w_temp")
        m.addConstrs((w_[s,c,h] == gp.min_(w_temp[s,c,h], constant=1) \
                    for s in range(n_s) for c in range(n_c) for h in range(n_h)),
                    name=f"c.{fid}.w_")
        # Does not really help to improve the speed.
        #m.addConstr((w_ == w/wmax), name=f"c.{fid}.w_")

        # We force irr to be zero but prec will add to w & w_, which will
        # output positive y_ leading to violation for y_y (< 1)
        # Also, we need to seperate yw_ and y_ into two constraints. Otherwise,
        # gurobi will crush. No idea why.
        m.addConstr((yw_temp == (a * w_**2 + b * w_ + c)), name=f"c.{fid}.yw_temp")
        m.addConstrs((yw_[s,c,h] == gp.max_(yw_temp[s,c,h], constant=0) \
                    for s in range(n_s) for c in range(n_c) for h in range(n_h)),
                    name=f"c.{fid}.yw_")
        # Does not really help to improve the speed.
        #m.addConstr((yw_ == (a * w_**2 + b * w_ + c)), name=f"c.{fid}.yw_")

        m.addConstr((y_ == yw_ * i_crop), name=f"c.{fid}.y_")
        m.addConstr((y == y_ * ymax * unit_area * 1e-4), name=f"c.{fid}.y") # 1e4 bu
        m.addConstr((irr * (1-i_crop) == 0), name=f"c.{fid}.irr(cm)")
        cm2m = 0.1
        m.addConstr((v_c == irr * unit_area * cm2m), name=f"c.{fid}.v_c(m-ha)")
        m.addConstr(v == gp.quicksum(v_c[i,j,:] \
                    for i in range(n_s) for j in range(n_c)),
                    name=f"c.{fid}.v(m-ha)")
        m.addConstr(y_y == gp.quicksum( y_[i,j,:] \
                    for i in range(n_s) for j in range(n_c) ) / n_s,
                    name=f"c.{fid}.y_y")

        # Tech decisions
        techs = self.techs
        n_te = self.n_te
        tech_options = self.tech_options

        q = m.addMVar((n_h), vtype="C", name=f"{fid}.q(m-ha/d)", lb=0, ub=inf)
        l_pr = m.addVar(vtype="C", name=f"{fid}.l_pr(m)", lb=0, ub=inf)

        i_te  = m.addMVar((n_te), vtype="B", name=f"{fid}.i_te")
        m.addConstr(q == gp.quicksum((techs[te][0] * v + techs[te][1]) * i_te[i] \
                    for i, te in enumerate(tech_options)), name=f"c.{fid}.q(m-ha/d)")
        m.addConstr(gp.quicksum(i_te[i] for i in range(n_te)) == 1,
                    name=f"c.{fid}.i_te")
        m.addConstr(l_pr == gp.quicksum( techs[te][2] * i_te[i] \
                    for i, te in enumerate(tech_options) ),
                    name=f"c.{fid}.l_pr(m)")
        # Given tech as an input
        if i_te_input is not None:
            self.msg[fid]["Irr tech"] = "user input"
            if isinstance(i_te_input, str):
                te = i_te_input
                i_te_input = np.zeros(n_te)
                i_te_input[tech_options.index(i_te)] = 1
            else:
                te = tech_options[list(i_te_input).index(1)]
            m.addConstr(i_te == i_te_input, name=f"c.{fid}.i_te_input")
            qa_input, qb_input, l_pr_input = self.techs[te]
            m.addConstr(l_pr == l_pr_input, name=f"c.{fid}.l_pr(m)_input")
            m.addConstr(i_te == i_te_input, name=f"c.{fid}.i_te(m)_input")

        self.vars[fid].v = v
        self.vars[fid].y = y
        self.vars[fid].y_y = y_y
        self.vars[fid].irr = irr
        self.vars[fid].i_crop = i_crop
        self.vars[fid].i_rain_fed = i_rain_fed
        self.vars[fid].i_te = i_te
        self.vars[fid].l_pr = l_pr
        self.vars[fid].q = q

        self.n_fields += 1

    def setup_constr_well(self, well_id, dwl, st, l_wt, r, k, sy, eff_pump,
                          eff_well, pumping_capacity=None):
        """
        Add well constraints. Multiple wells can be assigned by calling
        this function multiple times with different well id.

        Parameters
        ----------
        well_id : str or int
            Well id distinguishing equation sets for different wells.
        dwl : float
            Percieved annual water level change rate [m/yr].
        l_wt : float
            Initial head for the lift from the water table to the ground
            surface at the start of the pumping season [m].
        st: float
            Aquifer saturated thickness [m].
        r : float
            Well radius [m].
        k : float
            Hydraulic conductivity [m/d]. This will be used to calculate
            transmissivity [m2/d] by multiply saturated thickness [m].
        sy : float
            Specific yield.
        eff_pump : float
            Pump efficiency.
        eff_well : float
            Well efficiency.
        pumping_capacity : float
            Pumping capacity [m-ha/yr]

        Returns
        -------
        None.

        """
        self.well_ids.append(well_id)
        wid = well_id

        m = self.model
        n_h = self.n_h

        inf = self.inf
        rho = self.rho
        g = self.g

        approx_horizon = self.approx_horizon
        if approx_horizon:
            dwls = np.array([0, dwl*(self.horizon-1)])
        else:
            dwls = np.array([dwl*(i) for i in range(n_h)])
        # Assume a linear projection to the future
        l_wt = l_wt + dwls

        # Calculate propotion of the irrigation water (v), daily pumping rate
        # (q), and head for irr tech (l_pr) of this well.
        v = m.addMVar((n_h), vtype="C", name=f"{wid}.v(m-ha)", lb=0, ub=inf)
        q = m.addMVar((n_h), vtype="C", name=f"{wid}.q(m-ha/d)", lb=0, ub=inf)
        l_pr = m.addVar(vtype="C", name=f"{wid}.l_pr(m)", lb=0, ub=inf)
        # The allocation constraints are added when run finish setup.
        # E.g., m.addConstr((v == v * a_r[w_c, :]), name=f"c.{wid}.v")
        if pumping_capacity is not None:
            m.addConstr((v <= pumping_capacity),
                        name=f"c.{wid}.pumping_capacity")

        tr = st * k
        fpitr = 4 * np.pi * tr
        e = m.addMVar((n_h), vtype="C", name=f"{wid}.e(PJ)", lb=0, ub=inf)
        l_t = m.addMVar((n_h), vtype="C", name=f"{wid}.l_t(m)", lb=0, ub=inf)
        q_lnx = m.addMVar((n_h),vtype="C", name=f"{wid}.q_lnx", lb=0, ub=inf)
        # The upper bound of q_lny is set to -0.5772 to avoid l_cd_l_wd to be
        # negative.
        q_lny = m.addMVar((n_h),vtype="C", name=f"{wid}.q_lny", lb=-inf, ub=-0.5772)
        l_cd_l_wd = m.addMVar((n_h), vtype="C", name=f"{wid}.l_cd_l_wd(m)", lb=0, ub=inf)

        # 10000 is to convert m*ha to m3
        m_ha_2_m3 = 10000.0
        m.addConstr((q_lnx == r**2*sy/fpitr), name=f"c.{wid}.q_lnx")
        # y = ln(x)  addGenConstrLog(x, y)
        # Due to TypeError: unsupported operand type(s) for *: 'MLinExpr' and
        # 'gurobipy.LinExpr'
        for h in range(n_h):
            m.addGenConstrLog(q_lnx[h], q_lny[h])

        #m.addConstr((q_lny == np.log(r**2*sy/fpitr)), name=f"c.{wid}.q_lny")
        m.addConstr(l_cd_l_wd == (1+eff_well) * q/fpitr * (-0.5772 - q_lny) * m_ha_2_m3,
                    name=f"c.{wid}.l_cd_l_wd(m)")

        m.addConstr((l_t == l_wt + l_cd_l_wd + l_pr), name=f"c.{wid}.l_t(m)")
        #!!!  e could be large. Make sure no numerical issue here.
        # J to PJ (1e-15)
        r_g_m_ha_2_m3_eff = rho * g * m_ha_2_m3 / eff_pump / 1e15
        m.addConstr((e ==  r_g_m_ha_2_m3_eff * v * l_t), name=f"c.{wid}.e(PJ)")
        #m.addConstr((e == rho * g * v * cm_ha_2_m3 * l_t / eff_pump), name=f"c.{wid}.e")
        #m.addConstr((e == 10), name=f"c.{wid}.e_fixed")

        self.vars[wid].e = e
        self.vars[wid].v = v
        self.vars[wid].q = q
        self.vars[wid].l_pr = l_pr
        self.n_wells += 1

    def setup_constr_finance(self):
        """
        Add financial constraints that output profits.
        Output in 1e4$.

        Returns
        -------
        None.

        """
        m = self.model
        energy_price = self.energy_price
        crop_profit = self.crop_profit
        crop_options = self.crop_options
        n_h = self.n_h
        area_split = self.area_split
        inf = self.inf

        e = self.vars.e     # (n_h) [PJ]
        y = self.vars.y     # (n_s, n_c, n_h)

        cost_e = m.addMVar((n_h), vtype="C", name="cost_e(1e6$)", lb=0, ub=inf)
        rev = m.addMVar((n_h), vtype="C", name="rev(1e6$)", lb=0, ub=inf)
        # The profit variable is created in the initial to allow users to add
        # contraints without a specific order.
        profit = self.vars.profit

        m.addConstr((cost_e == e * energy_price), name="c.cost_e")
        m.addConstr(rev == gp.quicksum(y[i,j,:] * crop_profit[c]\
                    for i in range(area_split) for j, c in enumerate(crop_options)),
                    name="c.rev")
        m.addConstr((profit == rev - cost_e), name="c.profit")
        self.vars.cost_e = cost_e
        self.vars.rev = rev

    def setup_constr_wr(self, water_right_id, wr, field_id_list="all",
                        time_window=1, start_index=None, remaining_wr=None,
                        tail_wr="propotion"):
        """
        Add water rights constraints.

        1. 1 field 1 year: field_id=id, time_window=1
        2. 1 field m years: field_id=id, time_window=m
        3. All fields m years: field_id=None, time_window=m

        We do not support adding water rights for a subset of fields.

        Parameters
        ----------
        water_right_id : str or int
            Water right id distinguishing equation sets for different water
            rights.
        wr : float
            Depth of the water right [cm].
        field_id_list : "all" or list, optional
            If given, the water right constraints apply only to the subset
            of the fields. Otherwise, apply to all fields The default is "all".
        time_window : int, optional
            If given, the water right constrains the total irrigation depth
            over the time window. The default is 5.
        start_index : int, optional
            The start index of the time window. This is useful when the
            previous time window has not yet used up. The default is None.
        remaining_wr : float, optional
            The remaining water rights in the previous time window [cm]. The
            default is None.
        tail_wr : "propotion" or "all" or float, optional
            Method to allocate incomplete time window if happened. If a float
            is given, the value will be used. The default is "propotion".

        Returns
        -------
        None.

        """
        m = self.model
        fids = field_id_list
        n_h = self.n_h
        n_c = self.n_c
        n_s = self.area_split
        vars = self.vars
        if fids == "all":
            irr_sub = self.vars.irr         # (n_s, n_c, n_h)
        else:
            for i, fid in enumerate(fids):
                if i == 0:
                    irr_sub = vars[fid].irr
                else:
                    irr_sub += vars[fid].irr
            irr_sub = irr_sub/len(fids)
        self.wr = wr
        self.time_window = time_window

        # Initial period
        # The structure is to fit within a larger simulation framework, which
        # we allow the remaining water rights that are not used in the previous
        # year.
        c_i = 0
        if start_index is not None and remaining_wr is not None:
            if start_index > n_h: # Ensure within the planning horizon
                start_index = max(n_h, start_index+1) - 1
                print("Warning: start_index is larger than (horizon - 1).")
            m.addConstrs((gp.quicksum(irr_sub[i,j,h] \
                        for i in range(n_s) for j in range(n_c))/n_s <= remaining_wr \
                        for h in range(0, start_index + 1)),
                        name=f"c.{water_right_id}.wr_{c_i}(cm)")
            c_i += 1
        else:
            start_index = 0

        remaining_length = n_h - start_index

        # Middle period
        while remaining_length > time_window:
            m.addConstrs((gp.quicksum(irr_sub[i,j,h] \
                        for i in range(n_s) for j in range(n_c))/n_s <= wr \
                        for h in range(start_index, start_index+time_window)),
                        name=f"c.{water_right_id}.wr_{c_i}(cm)")
            c_i += 1
            start_index += time_window
            remaining_length -= time_window

        # Last period
        if remaining_length > 0:
            if tail_wr == "propotion":
                wr_tail = wr * remaining_length/time_window
            elif tail_wr == "wr":
                wr_tail = wr
            # Otherwise, we expect a value given by users.

            m.addConstrs((gp.quicksum(irr_sub[i,j,h] \
                        for i in range(n_s) for j in range(n_c))/n_s <= wr_tail \
                        for h in range(start_index, n_h)),
                        name=f"c.{water_right_id}.wr_{c_i}(cm)")

        self.water_right_ids.append(water_right_id)
        self.n_water_rights += 1

    def setup_obj_Sa(self, alpha_dict=None):
        """
        Add the objective to maximize the agent's expected satisfication.
        The calculation of satisfication involves exp, which require using the
        general expression in gurobi. The piecewise process will introduct
        additional binary integers that can significantly slow down the solving
        process and may encounter unexpected numerical issue. We suggest to use
        setup_obj instead and calculate satisfication afterward. Same solution
        should be expected.

        Parameters
        ----------
        alpha_dict : dict, optional
            Overwrite alpha values retrieved from the config. The default is None.

        Returns
        -------
        None.

        """
        eval_metric = self.eval_metric
        alphas = self.alphas
        vars = self.vars

        # Update alpha list
        if alpha_dict is not None:
            alphas.update(alpha_dict)
            self.eval_metrics = [metric for metric, v in self.alphas.items() if v is not None]

        # Check the selected eval_metric exist
        eval_metrics = self.eval_metrics
        eval_metric = self.eval_metric
        if eval_metric not in eval_metrics:
            raise ValueError(f"Alpha value of metric '{eval_metric}' is not given.")

        # Currently supported metrices
        eval_metric_vars = {
            "profit": vars.profit,
            "yield_pct": vars.y_y
            }

        inf = self.inf
        m = self.model
        n_h = self.n_h

        # Caluculate all active metrics
        def add_metric(metric, alpha):
            N_yr_x =  m.addMVar((n_h), vtype="C", name=f"N_yr_x.{metric}", lb=-inf, ub=0)
            N_yr_y = m.addMVar((n_h), vtype="C", name=f"N_yr_y.{metric}", lb=0, ub=inf)
            N_yr = m.addMVar((n_h), vtype="C", name=f"N_yr.{metric}", lb=0, ub=1)
            Sa = m.addVar(vtype="C", name=f"Sa.{metric}", lb=0, ub=1)

            metric_var = eval_metric_vars.get(metric)
            if metric_var is None:
                raise ValueError(f"""'{eval_metric}' is not supported.
                                 Available metrics includes {list(eval_metric_vars.keys())}""")
            m.addConstr((N_yr_x == -alpha * metric_var), name=f"c.N_yr_x.{metric}")
            for h in range(n_h):   # y = exp(x)  addGenConstrLog(x, y)
                m.addGenConstrExp(N_yr_x[h], N_yr_y[h])
            m.addConstr(N_yr == 1 - N_yr_y, name=f"c.N_yr.{metric}")
            m.addConstr((Sa == gp.quicksum(N_yr[h] for h in range(n_h))/n_h),
                        name=f"c.Sa.{metric}")
            vars.Sa[metric] = Sa

        for metric in eval_metrics:
            # It will be super slow if calculate all needs within opt model.
            #add_metric(metric, alphas[metric])

            # Add objective
            if metric == eval_metric:
                add_metric(metric, alphas[metric])
                m.setObjective(vars.Sa[metric], gp.GRB.MAXIMIZE)
        self.obj_post_calculation = False

    def setup_obj(self, alpha_dict=None):
        """
        Add the objective to maximize the agent's expected satisfication. Note
        that the satisfication is calculate after the optimization which is
        much faster. The solution is equivalent to using satisfication directly
        as objective.

        Parameters
        ----------
        alpha_dict : dict, optional
            Overwrite alpha values retrieved from the config. The default is None.

        Returns
        -------
        None.

        """
        eval_metric = self.eval_metric
        alphas = self.alphas
        vars = self.vars

        # Update alpha list
        if alpha_dict is not None:
            alphas.update(alpha_dict)
            self.eval_metrics = [metric for metric, v in self.alphas.items() if v is not None]

        # Check the selected eval_metric exist
        eval_metrics = self.eval_metrics
        eval_metric = self.eval_metric
        if eval_metric not in eval_metrics:
            raise ValueError(f"Alpha value of metric '{eval_metric}' is not given.")

        # Currently supported metrices
        eval_metric_vars = {
            "profit": vars.profit,
            "yield_pct": vars.y_y
            }

        inf = self.inf
        m = self.model
        n_h = self.n_h

        def add_metric(metric):
            Sa = m.addVar(vtype="C", name=f"Sa.{metric}", lb=0, ub=inf)
            metric_var = eval_metric_vars.get(metric)
            m.addConstr((Sa == gp.quicksum(metric_var[h] for h in range(n_h))/n_h),
                        name=f"c.Sa.{metric}")
            vars.Sa[metric] = Sa

        for metric in eval_metrics:
            # Add objective
            if metric == eval_metric:
                add_metric(metric)
                m.setObjective(vars.Sa[metric], gp.GRB.MAXIMIZE)
        self.obj_post_calculation = True

    def finish_setup(self, display_summary=True):
        """
        This will add the summary sets of constraints and update the pending
        assigments to the gurobi model.

        Returns
        -------
        A high-level summary of the model.

        """
        m = self.model
        vars = self.vars
        fids = self.field_ids
        wids = self.well_ids
        n_h = self.n_h
        n_f = self.n_fields
        n_w = self.n_wells

        ### Add some final constraints
        # Allocation ratios for the amount of water withdraw from each well to
        # satisfy v. Sum of the ratios is equal to 1.
        allo_r = m.addMVar((n_f, n_w, n_h), vtype="C", name="allo_r", lb=0, ub=1)
        allo_r_w = m.addMVar((n_w, n_h), vtype="C", name="allo_r_w", lb=0, ub=1)
        self.vars.allo_r = allo_r
        self.vars.allo_r_w = allo_r_w
        m.addConstr(allo_r_w == gp.quicksum(allo_r[f,:,:] for f in range(n_f))/n_f,
                    name="c.allo_r_w")
        m.addConstrs((gp.quicksum(allo_r[f,k,h] for k in range(n_w)) == 1 \
                      for f in range(n_f) for h in range(n_h)), name="c.allo_r")
        v = vars.v
        for k, wid in enumerate(wids):
            m.addConstr((vars[wid].v == v * allo_r_w[k,:]), name=f"c.{wid}.v(m-ha)")
            m.addConstr((vars[wid].q == gp.quicksum(vars[fid].q * allo_r[f,k,:] \
                        for f, fid in enumerate(fids))), name=f"c.{wid}.q(m-ha/d)")
            m.addConstr((vars[wid].l_pr == gp.quicksum(vars[fid].l_pr * allo_r[f,k,:] \
                        for f, fid in enumerate(fids))), name=f"c.{wid}.l_pr(m)")

        irr = vars.irr
        y = vars.y
        y_y = vars.y_y
        e = vars.e
        # Sum to the total
        def get_sum(ids, vars, var):
            """Sum over ids"""
            for i, v in enumerate(ids):
                if i == 0:
                    acc = vars[v][var]
                else:
                    acc += vars[v][var]
            return acc
        m.addConstr(irr == get_sum(fids, vars, "irr"), name="c.irr(cm)")
        m.addConstr(v == get_sum(fids, vars, "v"), name="c.v(m-ha)")
        m.addConstr(y == get_sum(fids, vars, "y"), name="c.y")
        m.addConstr(y_y == get_sum(fids, vars, "y_y")/n_f, name="c.y_y")
        m.addConstr(e == get_sum(wids, vars, "e"), name="c.e(PJ)")

        m.update()

        if self.approx_horizon:
            h_msg = str(self.n_h) + " (approximate with 2)"
        else:
            h_msg = str(self.n_h)
        msg = dict_to_string(self.msg, prefix="\t\t", level=2)
        summary = f"""
        ########## Model Summary ##########\n
        Name:   {self.name}\n
        Planning horizon:   {h_msg}
        NO. Crop fields:    {self.n_fields}
        NO. splits          {self.area_split}
        NO. Wells:          {self.n_wells}
        NO. Water rights:   {self.n_water_rights}\n
        Decision settings:\n{msg}\n
        ###################################
        """
        self.summary = summary
        if display_summary:
            print(summary)

    def solve(self, keep_gp_model=False, keep_gp_output=False, display_report=False, **kwargs):
        """
        Solve the optimization problem. Default NonConvex = 2 (solve for a
        nonconvex model).

        Parameters
        ----------
        keep_gp_model : bool
            Keep the gurobi model instance for further used. Use with caution.
            The default is False.

        keep_gp_output : bool
            If True, the gurobi model output will be stored at "gp_output" in a
            dictionary format.

        **kwargs : **kwargs
            Pass specific parameters to the gurobi solver.

        Returns
        -------
        None.

        Notes
        -----
        More info:
            https://www.gurobi.com/documentation/9.5/refman/mip_models.html

        """
        def extract_sol(vars):
            sols = DotMap()
            def get_inner_dict(d, new_dotmap):
                for k, v in d.items():
                    if isinstance(v, dict):
                        get_inner_dict(v, new_dotmap[k])
                    else:
                        new_dotmap[k] = v.X
            get_inner_dict(vars, sols)
            return sols

        m = self.model
        gurobi_pars = self.gurobi_pars
        gurobi_pars.update(kwargs)
        if "NonConvex" not in gurobi_pars.keys():
            m.setParam("NonConvex", 2)  # Set to solve non-convex problem
        for k, v in gurobi_pars.items():
            m.setParam(k, v)
        m.optimize()

        if m.Status == 2:   # Optimal solution found
            self.optimal_obj_value = m.objVal
            self.sols = extract_sol(self.vars)
            self.sols.obj = m.objVal

            # Calculate satisfication
            if self.obj_post_calculation:
                eval_metrics = self.eval_metrics
                alphas = self.alphas

                # Currently supported metrices
                if self.approx_horizon:
                    horizon = self.horizon
                    profits = self.sols.profit
                    y_ys = self.sols.y_y
                    eval_metric_vars = {
                        "profit": np.linspace(profits[0], profits[1], num=horizon),
                        "yield_pct": np.linspace(y_ys[0], y_ys[1], num=horizon)
                        }
                else:
                    eval_metric_vars = {
                        "profit": self.sols.profit,
                        "yield_pct": self.sols.y_y
                        }
                for metric in eval_metrics:
                    alpha = alphas[metric]
                    metric_var = eval_metric_vars.get(metric)
                    N_yr = 1 - np.exp(-alpha * metric_var)
                    Sa = np.mean(N_yr)
                    self.sols.Sa[metric] = Sa

            # Display report
            crop_options = self.crop_options
            tech_options = self.tech_options
            n_s = self.area_split
            fids = self.field_ids
            sols = self.sols
            irrs = list(sols.irr.mean(axis=0).sum(axis=0).round(2))
            decisions = {"Irrigation depths": irrs}
            for fid in fids:
                i_crop = sols[fid].i_crop[:, :, 0]
                crop_type = [crop_options[np.where(i_crop[s,:].round(0) == 1)[0][0]] for s in range(n_s)]
                tech = tech_options[np.where(sols[fid].i_te[:].round(0) == 1)[0][0]]
                Irrigated = list((sols[fid].i_rain_fed[:, :, 0].sum(axis=1).round(0) == 0))
                decisions[fid] = {"Crop types": crop_type,
                                  "Irr tech": tech,
                                  "Irrigated": Irrigated}
            decisions = dict_to_string(decisions, prefix="\t\t", level=2)
            msg = dict_to_string(self.msg, prefix="\t\t", level=2)
            sas = dict_to_string(self.sols.Sa, prefix="\t\t", level=2, roun=4)
            if self.approx_horizon:
                h_msg = str(self.n_h) + " (approximate with 2)"
            else:
                h_msg = str(self.n_h)
            report = f"""
        ########## Model Report ##########\n
        Name:   {self.name}\n
        Planning horizon:   {h_msg}
        NO. Crop fields:    {self.n_fields}
        NO. splits          {self.area_split}
        NO. Wells:          {self.n_wells}
        NO. Water rights:   {self.n_water_rights}\n
        Decision settings:\n{msg}\n
        Solutions:\n{decisions}\n
        Satisfication:\n{sas}\n
        ###################################
            """
            self.report = report
            if display_report:
                print(report)

        else:
            print("Optimal solution is not found.")
            self.optimal_obj_value = None

        if keep_gp_output:
            self.gp_output = json.loads(m.getJSONSolution())

        if keep_gp_model is False:
            # release the memory of the previous model
            m.dispose()

    def do_IIS_gp(self, filename=None):
        """
        Compute an Irreducible Inconsistent Subsystem (IIS). This function can
        only be exercuted if the model is infeasible.

        An IIS is a subset of the constraints and variable bounds with the
        following properties:

        - It is still infeasible, and
        - If a single constraint or bound is removed, the subsystem becomes feasible.

        Note that an infeasible model may have multiple IISs. The one returned
        by Gurobi is not necessarily the smallest one; there may exist others
        with fewer constraints or bounds.

        More info: https://www.gurobi.com/documentation/10.0/refman/py_model_computeiis.html

        Parameters
        ----------
        filename : str
            Output filename. The default is None.

        Returns
        -------
        None.

        """
        m = self.model
        # do IIS
        m.computeIIS()
        if m.IISMinimal:
            print('IIS is minimal\n')
        else:
            print('IIS is not minimal\n')
        print('\nThe following constraint(s) cannot be satisfied:')
        for c in m.getConstrs():
            if c.IISConstr:
                print('%s' % c.ConstrName)

        if filename is not None:
            if filename[-4:] != ".ilp":
                filename += ".ilp"
            m.write(filename)

    def write_ilp(self, filename):
        """
        Output the information about the results of the IIS computation to
        .ilp. This function can only be exercuted after do_IIS_gp().

        Parameters
        ----------
        filename : str
            Output filename.

        Returns
        -------
        None.

        """
        if filename[-4:] != ".ilp":
            filename += ".ilp"
        m = self.model
        m.write(filename)

    def write_sol(self, filename):
        """
        Output the solution of the model to .sol.

        Parameters
        ----------
        filename : str
            Output filename.

        Returns
        -------
        None.

        """
        if filename[-4:] != ".sol":
            filename += ".sol"
        m = self.model
        m.write(filename)

    def write_lp(self, filename):
        """
        Output the model to .lp.

        Parameters
        ----------
        filename : str
            Output filename.

        Returns
        -------
        None.

        """
        if filename[-3:] != ".lp":
            filename += ".lp"
        m = self.model
        m.write(filename)

    def write_mps(self, filename):
        """
        Output the solution of the model to .lp.

        Parameters
        ----------
        filename : str
            Output filename.

        Returns
        -------
        None.

        """
        if filename[-3:] != ".mps":
            filename += ".mps"
        m = self.model
        m.write(filename)


#%% ===========================================================================
# Archive code
# =============================================================================
    # def add_starts(self, var_starts=[]):
    #     """
    #     Add initial starts for optimization

    #     Parameters
    #     ----------
    #     var_starts : list, optional
    #         A list of dictionaries. Each dictionary is a start contains varable
    #         names as keys and start values as values. If a varable name is
    #         nested like field1.irr, a tuple should be given like
    #         ("field1", "irr"). The default is [].

    #     Returns
    #     -------
    #     None.

    #     """
    #     m = self.model
    #     vars = self.vars
    #     m.NumStart = len(var_starts)
    #     m.update()
    #     for i, d in enumerate(var_starts):
    #         m.params.StartNumber = i
    #         for k, v in d.items():
    #             if isinstance(k, tuple):
    #                 vars[k[0]][k[1]].Start = v
    #             else:
    #                 vars[k].Start = v



