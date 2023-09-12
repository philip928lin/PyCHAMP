# -*- coding: utf-8 -*-
"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Jun 9, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""
import json
import numpy as np
import gurobipy as gp


#################

class OptModel():
    """
    This class represents a farmer making decisions on irrigation depth,
    crop types, rain-fed or irrigated, and irrigation technologies. The
    farmer's objective is to solve a nonlinear mixed interger optimization
    model that maximizes their satisfaction, which can be measured by metrics
    like profit or yield percentage. The optimization problem is formulated on
    an annual scale. Optional decision variables include crop types, rain-fed
    option, and irrigation technologies, which can be provided as input to the
    class.

    Specifically, this class is designed to accommodate a farmer with multiple
    crop fields and groundwater wells. Each field can have multiple area
    splits, where each split has its own set of decision variables. Water
    rights can be incorporated as constraints for all fields or a selected
    subset of fields with an optional time_window argument allowing farmer to
    allocate their water rights across multiple years. To enforce water rights
    at the point of diversion, pumping capacity can be assigned to individual
    wells.

    If the 'horizon' parameter is set to a value greater than 1, only the
    irrigation depth is varied each year. Other decision variables such as the
    crop types, rain-fed option, and irrigation technologies remain fixed over
    the planning horizon. If necessary, the user has the flexibility to rerun
    the optimization model in the subsequent years to update the decision based
    on any changes or new information.

    This class provides an option to approximate the solving process by setting
    the "approx_horizon" parameter to True. This approximation assumes a linear
    decrement in water levels and helps speed up the solving process. However,
    it's important to note that if there is a water right constraint with a
    time_window argument larger than 1, the approximation option is restricted
    and cannot be used. This ensures accurate handling of water right
    constraints over multiple time periods.

    Notations
    ---------
    n_s: Number of the splits in a field.
    n_c: Number of the crop choices.
    n_h: Planning horizon or approximated horizon.
    n_te: Number of the irrigation technology choices.

    Notes
    -----
    This class utilizes the Gurobi solver to solve the optimization problem.
    Gurobi is a commercial solver, but it also offers a full-featured solver
    for academic use at no cost. Users interested in running the code will need
    to register for an academic license and download the solver. Additionally,
    they will need to install the gurobipy Python package to interface with
    Gurobi and execute the code successfully.

    More information can be found here:
    https://www.gurobi.com/academia/academic-program-and-licenses/

    Attributes
    ----------
    name : str, optional
        Model name. The default is "".

    Methods
    -------
    Left blank

    Examples
    --------
    Left blank

    """
    def __init__(self, name="", LogToConsole=1):
        """
        Instantiate an optimization environment and object for a farmer.

        To suppress all Gurobi logging including the license connection
        parameters, you can set the parameter OutputFlag or LogToConsole to 0
        on an empty environment before it is started.  Please note that this
        only works if you set these parameters before starting the environment.
        """
        # Model name
        self.name = name
        # Create a gurobi environment to ensure thread safety for parallel
        # computing.
        # self.gpenv = gp.Env()

        # This will remove all output from gurobi to the console.
        self.gpenv = gp.Env(empty=True)
        if LogToConsole is not None:
            self.gpenv.setParam("LogToConsole", LogToConsole)
        self.gpenv.start()

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
        Depose the Gurobi environment, ensuring that it is executed only when
        the instance is no longer needed. It's important to note that once this
        method is invoked, a new optimization model (setup_ini_model) cannot be
        created anymore.

        Returns
        -------
        None.

        """
        self.gpenv.dispose()

    def setup_ini_model(self, config, horizon=1, eval_metric="profit",
                        crop_options=["corn", "sorghum", "soybeans", "fallow"],
                        tech_options=["center pivot", "center pivot LEPA"],
                        approx_horizon=False):
        """
        Set up the initial settings for an optimization model. The new
        optimization model will be created within the same Gurobi environment
        that was initialized when the class instance was created.

        Parameters
        ----------
        config : dict or DotMap
            General info of the model.
        horizon : str, optional
            The planing horizon [yr]. The default is 1.
        eval_metric : str, optional
            "profit" or "yield_pct". The default is "profit".
        crop_options : list, optional
            A list of crop type options. They must exist in the config. The
            default is ["corn", "sorghum", "soybeans", "fallow"].
        tech_options : list, optional
            A list of irrigation technologies. They must exist in the config.
            The default is ["center pivot", "center pivot LEPA"].
        approx_horizon : bool, optional
            When set to True, the model will calculate two points (start and
            end) over the given horizon to determine the objective, which is
            the average over the horizon. This approach can significantly
            enhance the solving speed, particularly for long horizons. In most
            cases, this approximation is equivalent to solving the original
            problem. It relies on the assumption that the groundwater level
            will linearly decrease in the projected future. However, it's
            important to note that if there is a water right constraint with a
            time_window argument larger than 1, the approximation option is
            restricted and cannot be used. This ensures accurate handling of
            water right constraints over multiple time periods. The default is
            False.

        Returns
        -------
        None.

        """
        self.config = config
        self.crop_options = crop_options
        self.tech_options = tech_options
        self.eval_metric = eval_metric
        self.approx_horizon = approx_horizon
        self.horizon = horizon # Original planning horizon

        ## The gurobi parameters. These will be fed to the solver in solve().
        self.gurobi_pars = config.get("gurobi")
        if self.gurobi_pars is None:
            self.gurobi_pars = {}

        config_field = config['field']
        ## Dimension coefficients
        self.n_s = config_field['area_split']
        self.n_c = len(crop_options)    # No. of crop choice options
        self.n_te = len(tech_options)   # No. of irr_depth tech options
        if approx_horizon and horizon > 2:
            # calculate obj with start and end points
            self.n_h = 2
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
        crop_par = np.array([config_field['crop'][c] for c in crop_options])
        self.ymax = crop_par[:, 0].reshape((-1, 1))     # (n_c, 1)
        self.wmax = crop_par[:, 1].reshape((-1, 1))     # (n_c, 1)
        self.a = crop_par[:, 2].reshape((-1, 1))        # (n_c, 1)
        self.b = crop_par[:, 3].reshape((-1, 1))        # (n_c, 1)
        self.c = crop_par[:, 4].reshape((-1, 1))        # (n_c, 1)
        try:
            self.min_y_pct = crop_par[:, 5].reshape((-1, 1))        # (n_c, 1)
        except:
            self.min_y_pct = np.zeros((self.n_c, 1))

        self.unit_area = config_field['field_area']/self.n_s
        # For consumat
        config_consumat = config['consumat']
        self.alphas = config_consumat['alpha']
        self.scales = config_consumat['scale']
        self.eval_metrics = [metric for metric, v in self.alphas.items() if v is not None]

        ## Form tech and crop change cost matrix from the config
        n_te = self.n_te
        config_finance = config['finance']
        irr_tech_change_cost = config_finance['irr_tech_change_cost']
        tech_change_cost_matrix = np.zeros((n_te, n_te))
        for k, v in irr_tech_change_cost.items():
            try:
                i = tech_options.index(k[0])
                j = tech_options.index(k[1])
                tech_change_cost_matrix[i, j] = v
            except:
                pass
        self.tech_change_cost_matrix = tech_change_cost_matrix

        n_c = self.n_c
        crop_change_cost = config_finance['crop_change_cost']
        crop_change_cost_matrix = np.zeros((n_c, n_c))
        for k, v in crop_change_cost.items():
            try:
                i = crop_options.index(k[0])
                j = crop_options.index(k[1])
                crop_change_cost_matrix[i, j] = v
            except:
                pass
        self.crop_change_cost_matrix = crop_change_cost_matrix

        ## Optimization Model
        # self.model.dispose()    # release the memory of the previous model
        self.model = gp.Model(name=self.name, env=self.gpenv)
        self.vars = {}    # A container to store variables.
        self.bounds = {}
        self.bounds['ub_w'] = np.max(self.wmax)
        self.inf = float('inf')

        ## Add shared variables
        m = self.model
        inf = self.inf
        n_s = self.n_s
        n_c = self.n_c
        n_h = self.n_h
        # Total irrigation depth per crop per yr
        irr_depth = m.addMVar((n_s, n_c, n_h), vtype="C", name="irr_depth(cm)", lb=0, ub=inf)
        # Total irrigation volumn per yr
        v = m.addMVar((n_h), vtype="C", name="v(m-ha)", lb=0, ub=inf)
        # Total yield per crop type per yr
        y = m.addMVar((n_s, n_c, n_h), vtype="C", name="y(1e4bu)", lb=0, ub=inf)
        # Average y_ (i.e., y/ymax) per yr
        y_y = m.addMVar((n_h), vtype="C", name="y_y", lb=0, ub=1)
        # Total used electricity (pumping) per yr
        e = m.addMVar((n_h), vtype="C", name="e(PJ)", lb=0, ub=inf)
        # Total profit
        profit = m.addMVar((n_h), vtype="C", name="profit(1e4$)", lb=-inf, ub=inf)
     
        ## Record variables
        self.vars['irr_depth'] = irr_depth
        self.vars['v'] = v
        self.vars['y'] = y
        self.vars['y_y'] = y_y
        self.vars['e'] = e
        self.vars['profit'] = profit
        self.bigM = 100
        self.penalties = []

        ## Record msg about the user inputs.
        self.msg = {}

        ## Record water rights info.
        self.wrs_info = {}

    def setup_constr_field(self, field_id, prec_aw, pre_i_crop, pre_i_te,
                           field_type="optimize", i_crop=None, i_rainfed=None,
                           i_te=None, **kwargs):
        """
        Add crop field constriants. You can assign multiple fields by calling
        this function repeatedly with different field_id. If
        i_crop/i_rainfed/i_te is provided, the model will not optimize over
        different crop type options/rain-fed or irrigated/irrigation
        technologies.

        Parameters
        ----------
        field_id : str or int
            The field id serves as a means to differentiate the equation sets
            for different fields.
        prec_aw : dict
            Percieved precipitation in each crop type's growing season [cm].
        pre_i_crop: str or 3darray
            Crop name or the i_crop from the previous time step.
        pre_i_te: str or 3darray
            Irrigation technology or i_te from the previous time step.
        field_type : str or list, optional
            The value can be "rainfed", "irrigated", or "optimize". A list can 
            be given to define field type for each area split. The default is
            "optimize".
        i_crop : 3darray, optional
            The indicator matrix has a dimension of (n_s, n_c, 1). In this
            matrix, a value of 1 indicates that the corresponding crop type
            is selected or chosen. The default is None.
        i_rainfed : 3darray, optional
            The indicator matrix has a dimension of (n_s, n_c, 1). In this
            matrix, a value of 1 indicates that the unit area in a field is
            rainfed. Given i_rainfed will force field_type to be "rainfed".
            Also, if it is given, make sure 1 only exists at where i_crop is
            also 1. The default is None.
        i_te : 1darray or str, optional
            The indicator matrix has a dimension of (n_te). In this
            matrix, a value of 1 indicates that the corresponnding irrigation
            technology is selected. The default is None.
        
        Returns
        -------
        None.
        
        Notes
        -----
        The `kwargs` could contain any additional attributes that you want to
        add to the Farmer agent. Available keywords include
        block_w_interval_for_corn : list
            An interval of (perceived) avaiable water [w_low, w_high] that 
            the corn crop type cannot be chose.  
        """

        self.field_ids.append(field_id)
        fid = field_id

        n_c = self.n_c
        n_h = self.n_h
        n_s = self.n_s
        n_te = self.n_te
        
        # Assign field type for each split of the field. 
        if i_rainfed is not None:
            rain_feds = np.sum(i_rainfed, axis=1).flatten()
            field_type_list = ["rainfed" if r > 0.5 else "irrigated" for r in rain_feds]
        if isinstance(field_type, str):  # Apply to all splits
            field_type_list = [field_type] * n_s
        elif isinstance(field_type, list):
            field_type_list = field_type

        # Summary message for the setting.
        self.msg[fid] = {
            "Crop types": "optimize",
            "Irr tech": "optimize",
            "Field type": field_type
            }

        # Record the input
        i_crop_input = i_crop
        i_rainfed_input = i_rainfed
        i_te_input = i_te
        
        # Take out some values
        m = self.model
        inf = self.inf
        ymax = self.ymax
        wmax = self.wmax
        ub_w = self.bounds['ub_w']
        ub_irr = ub_w #ub_w - prec_aw
        self.bounds[fid] = {}
        self.bounds[fid]['ub_irr'] = ub_irr
        unit_area = self.unit_area

        # Create the available precipitiation for each crop. 
        prec_aw_ = np.ones((n_s, n_c, n_h))
        for ci, crop in enumerate(self.crop_options):
            prec_aw_[:, ci, :] = prec_aw[crop]
            
        # Approximate robust optimization by prorating prec_aw with a ratio 
        # This is only for internal testing.
        # uncertainty_ratio = 1
        # for hi in range(n_h):
        #     prec_aw_[:, :, hi] = prec_aw_[:, :, hi] * uncertainty_ratio**hi

        ### Add general variables
        irr_depth     = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.irr_depth(cm)", lb=0, ub=ub_irr)
        w       = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.w(cm)", lb=0, ub=ub_w)
        w_temp  = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.w_temp", lb=0, ub=inf)
        w_      = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.w_", lb=0, ub=1)
        y       = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.y(1e4bu)", lb=0, ub=inf)
        y_      = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.y_", lb=0, ub=1)
        yw_temp = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.yw_temp", lb=-inf, ub=1)
        yw_bi   = m.addMVar((n_s, n_c, n_h), vtype="B", name=f"{fid}.yw_bi")
        yw_     = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.yw_", lb=0, ub=1)
        v_c     = m.addMVar((n_s, n_c, n_h), vtype="C", name=f"{fid}.v_c(m-ha)", lb=0, ub=inf)
        y_y     = m.addMVar((n_h), vtype="C", name=f"{fid}.y_y", lb=0, ub=1)    # avg y_ per yr
        v       = m.addMVar((n_h), vtype="C", name=f"{fid}.v(m-ha)", lb=0, ub=inf)
        i_crop  = m.addMVar((n_s, n_c, 1), vtype="B", name=f"{fid}.i_crop")
        i_rainfed = m.addMVar((n_s, n_c, 1), vtype="B", name=f"{fid}.i_rainfed")

        # Given crop type 
        ## Crop type is set to be the same accross the planning horizon.
        if i_crop_input is not None:
            m.addConstr(i_crop == i_crop_input, name=f"c.{fid}.i_crop_input")
            self.msg[fid]["Crop types"] = "user input"
            
        # One unit area can only be planted one type of crops.
        m.addConstr(gp.quicksum(i_crop[:,ci,:] for ci in range(n_c)) == 1,
                    name=f"c.{fid}.i_crop")

        ### Include rain-fed option
        for si, field_type in enumerate(field_type_list):
            if field_type == "rainfed":
                # Given i_rainfed,
                if i_rainfed_input is not None:
                    m.addConstr(i_rainfed[si,:,:] == i_rainfed_input[si,:,:],
                                name=f"c.{fid}_{si}.i_rainfed_input")
                    self.msg[fid]["Rainfed field"] = "user input"

                # i_rainfed[si, ci, hi] can be 1 only when i_crop[si, ci, hi] is 1.
                # Otherwise, it has to be zero.
                m.addConstr(i_crop[si,:,:] - i_rainfed[si,:,:] >= 0,
                            name=f"c.{fid}_{si}.i_rainfed")
                m.addConstr(irr_depth[si,:,:] == 0, name=f"c.{fid}_{si}.irr_rain_fed")
                
            elif field_type == "irrigated":
                m.addConstr(i_rainfed[si,:,:] == 0,
                            name=f"c.{fid}_{si}.no_i_rainfed")
                
            elif field_type == "irrigated":
                # i_rainfed[si, ci, hi] can be 1 only when i_crop[si, ci, hi] is 1.
                # Otherwise, it has to be zero.
                m.addConstr(i_crop[si,:,:] - i_rainfed[si,:,:] >= 0,
                            name=f"c.{fid}_{si}.i_rainfed")
                m.addConstr(irr_depth[si,:,:] * i_rainfed[si,:,:] == 0, name=f"c.{fid}_{si}.irr_rainfed")

        # See the numpy broadcast rules:
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        m.addConstr((w == irr_depth + prec_aw_), name=f"c.{fid}.w(cm)")
        m.addConstr((w_temp == w/wmax), name=f"c.{fid}.w_temp")
        m.addConstrs((w_[si,ci,hi] == gp.min_(w_temp[si,ci,hi], constant=1) \
                    for si in range(n_s) for ci in range(n_c) for hi in range(n_h)),
                    name=f"c.{fid}.w_")

        # We force irr_depth to be zero but prec_aw_ will add to w & w_, which will
        # output positive y_ leading to violation for y_y (< 1)
        # Also, we need to seperate yw_ and y_ into two constraints. Otherwise,
        # gurobi will crush. No idea why.
        a = self.a
        b = self.b
        c = self.c
        m.addConstr((yw_temp == (a * w_**2 + b * w_ + c)), name=f"c.{fid}.yw_temp")
        
        # Minimum yield_pct cutoff (aim to capture fallow field)
        min_y_pct = self.min_y_pct
        m.addConstr((yw_bi * (yw_temp - min_y_pct) + (1-yw_bi) * (min_y_pct - yw_temp) >= 0),
                    name=f"c.{fid}.yw_bi")
        m.addConstr((yw_ == yw_bi * yw_temp + (1-yw_bi) * min_y_pct),
                    name=f"c.{fid}.yw_")
        
        m.addConstr((y_ == yw_ * i_crop), name=f"c.{fid}.y_")
        m.addConstr((y == y_ * ymax * unit_area * 1e-4), name=f"c.{fid}.y") # 1e4 bu
        m.addConstr((irr_depth * (1-i_crop) == 0), name=f"c.{fid}.irr_depth(cm)")
        cm2m = 0.01
        m.addConstr((v_c == irr_depth * unit_area * cm2m), name=f"c.{fid}.v_c(m-ha)")
        m.addConstr(v == gp.quicksum(v_c[i,j,:] \
                    for i in range(n_s) for j in range(n_c)),
                    name=f"c.{fid}.v(m-ha)")
        m.addConstr(y_y == gp.quicksum( y_[i,j,:] \
                    for i in range(n_s) for j in range(n_c) ) / n_s,
                    name=f"c.{fid}.y_y")
        
        # Add penalty for a given w interval for corn to block the choice
        block_w_interval_for_corn = kwargs.get("block_w_interval_for_corn")
        if block_w_interval_for_corn is not None:
            penalty = m.addVar(vtype="C", name=f"{fid}.penalty", lb=0, ub=inf)
            penalties = self.penalties
            bigM = self.bigM
            corn_idx = self.crop_options.index("corn")
            corn_w = w[:, corn_idx, :]
            w_bi1 = m.addMVar((n_s, 1, n_h), vtype="B", name=f"{fid}.w_bi1")
            w_bi2 = m.addMVar((n_s, 1, n_h), vtype="B", name=f"{fid}.w_bi2")
        
            #if x<b then y=1
            m.addConstr((58-corn_w <= bigM * w_bi1),
                        name=f"c.{fid}.w_bi1")
            #if x>b then y=1
            m.addConstr((corn_w-42 <= bigM * w_bi2),
                        name=f"c.{fid}.w_bi2")
            m.addConstr(penalty == gp.quicksum(bigM*w_bi1[si,0,hi]*w_bi2[si,0,hi] \
                        for si in range(n_s) for hi in range(n_h)),
                        name=f"c.{fid}.penalty")
            penalties.append(penalty)    
            
        # Create i_crop_change to indicate crop type change
        if isinstance(pre_i_crop, str):
            i_c = self.crop_options.index(pre_i_crop)
            pre_i_crop = np.zeros((n_s, n_c, 1))
            pre_i_crop[:, i_c, :] = 1
        i_crop_change_ = m.addMVar((n_s, n_c, 1), vtype="I", name=f"{fid}.i_crop_change_", lb=-1, ub=1)
        i_crop_change = m.addMVar((n_s, n_c, 1), vtype="B", name=f"{fid}.i_crop_change")
        m.addConstr(i_crop_change_ == i_crop - pre_i_crop, name=f"c.{fid}.i_crop_change_")
        m.addConstrs((i_crop_change[s,c,0] == gp.max_(i_crop_change_[s,c,0], constant=0) \
                    for s in range(n_s) for c in range(n_c)),
                    name=f"c.{fid}.i_crop_change")

        # Tech decisions
        techs = self.config['field']['tech']
        tech_options = self.tech_options

        q       = m.addMVar((n_h), vtype="C", name=f"{fid}.q(m-ha/d)", lb=0, ub=inf)
        l_pr    = m.addVar(vtype="C", name=f"{fid}.l_pr(m)", lb=0, ub=inf)
        i_te    = m.addMVar((n_te), vtype="B", name=f"{fid}.i_te")
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
                i_te_input[tech_options.index(te)] = 1
            else:
                te = tech_options[np.argmax(i_te_input)]
            m.addConstr(i_te == i_te_input, name=f"c.{fid}.i_te_input")
            qa_input, qb_input, l_pr_input = techs[te]
            m.addConstr(l_pr == l_pr_input, name=f"c.{fid}.l_pr(m)_input")
            m.addConstr(i_te == i_te_input, name=f"c.{fid}.i_te(m)_input")

        # Create variable for tech change
        if isinstance(pre_i_te, str):
            i_t = self.tech_options.index(pre_i_te)
            pre_i_te = np.zeros((n_te))
            pre_i_te[i_t] = 1
        i_tech_change_ = m.addMVar((n_te), vtype="I", name=f"{fid}.i_tech_change_", lb=-1, ub=1)
        i_tech_change = m.addMVar((n_te), vtype="B", name=f"{fid}.i_tech_change")
        m.addConstr(i_tech_change_ == i_te - pre_i_te, name=f"c.{fid}.i_tech_change_")
        m.addConstrs((i_tech_change[t] == gp.max_(i_tech_change_[t], constant=0) \
                      for t in range(n_te)),
                      name=f"c.{fid}.i_crop_change")

        self.vars[fid] = {}
        self.vars[fid]['v'] = v
        self.vars[fid]['y'] = y
        self.vars[fid]['y_y'] = y_y
        self.vars[fid]['irr_depth'] = irr_depth
        self.vars[fid]['i_crop'] = i_crop
        self.vars[fid]['i_rainfed'] = i_rainfed
        self.vars[fid]['i_te'] = i_te
        self.vars[fid]['l_pr'] = l_pr
        self.vars[fid]['q'] = q

        self.vars[fid]['pre_i_crop'] = pre_i_crop
        self.vars[fid]['pre_i_te'] = pre_i_te
        self.vars[fid]['i_crop_change'] = i_crop_change
        self.vars[fid]['i_tech_change'] = i_tech_change

        self.vars[fid]['field_type'] = field_type

        self.n_fields += 1

    def setup_constr_well(self, well_id, dwl, st, l_wt, r, k, sy, eff_pump,
                          eff_well, pumping_capacity=None):
        """
        Add well constraints. You can assign multiple wells by calling
        this function repeatedly with different well_id.

        Parameters
        ----------
        well_id : str or int
            The well id serves as a means to differentiate the equation sets
            for different wells.
        dwl : float
            Percieved annual water level change rate [m/yr].
        l_wt : float
            The head required to lift water from the water table to the ground
            surface at the start of the pumping season at the initial time step
            [m].
        st: float
            Aquifer saturated thickness at the initial time step [m].
        r : float
            Well radius [m].
        k : float
            Hydraulic conductivity [m/d]. This will be used to calculate
            transmissivity [m2/d] by multiply the saturated thickness [m].
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

        cw = self.config['well']
        rho = cw['rho']
        g = cw['g']

        # Project the future lift head.
        approx_horizon = self.approx_horizon
        if approx_horizon and self.horizon > 2:
            dwls = np.array([0, dwl * (self.horizon-1)])
        else:
            dwls = np.array([dwl * (i) for i in range(n_h)])
        # Assume a linear projection to the future
        l_wt = l_wt - dwls
        self.l_wt = l_wt

        # Calculate proportion of the irrigation water (v), daily pumping rate
        # (q), and head for irr tech (l_pr) of this well.
        v    = m.addMVar((n_h), vtype="C", name=f"{wid}.v(m-ha)", lb=0, ub=inf)
        q    = m.addMVar((n_h), vtype="C", name=f"{wid}.q(m-ha/d)", lb=0, ub=inf)
        l_pr = m.addVar(vtype="C", name=f"{wid}.l_pr(m)", lb=0, ub=inf)
        # The allocation constraints are added when run finish setup.
        # E.g., m.addConstr((v == v * a_r[w_c, :]), name=f"c.{wid}.v")
        if pumping_capacity is not None:
            m.addConstr((v <= pumping_capacity),
                        name=f"c.{wid}.pumping_capacity")

        tr = st * k
        fpitr = 4 * np.pi * tr
        e     = m.addMVar((n_h), vtype="C", name=f"{wid}.e(PJ)", lb=0, ub=inf)
        l_t   = m.addMVar((n_h), vtype="C", name=f"{wid}.l_t(m)", lb=0, ub=inf)
        q_lnx = m.addMVar((n_h), vtype="C", name=f"{wid}.q_lnx", lb=0, ub=inf)
        # The upper bound of q_lny is set to -0.5772 to avoid l_cd_l_wd to be
        # negative.
        q_lny     = m.addMVar((n_h), vtype="C", name=f"{wid}.q_lny", lb=-inf, ub=-0.5772)
        l_cd_l_wd = m.addMVar((n_h), vtype="C", name=f"{wid}.l_cd_l_wd(m)", lb=0, ub=inf)

        # 10000 is to convert m-ha to m3
        m_ha_2_m3 = 10000
        m.addConstr((q_lnx == r**2*sy/fpitr), name=f"c.{wid}.q_lnx")
        # y = ln(x)  addGenConstrLog(x, y)
        # m.addConstr((q_lny == np.log(r**2*sy/fpitr)), name=f"c.{wid}.q_lny")
        # Due to TypeError: unsupported operand type(s) for *: 'MLinExpr' and
        # 'gurobipy.LinExpr'
        for h in range(n_h):
            m.addGenConstrLog(q_lnx[h], q_lny[h])
        m.addConstr(l_cd_l_wd == (1+eff_well) * q/fpitr * (-0.5772 - q_lny) * m_ha_2_m3,
                    name=f"c.{wid}.l_cd_l_wd(m)")
        m.addConstr((l_t == l_wt + l_cd_l_wd + l_pr), name=f"c.{wid}.l_t(m)")
        # e could be large. Make sure no numerical issue here.
        # J to PJ (1e-15)
        r_g_m_ha_2_m3_eff = rho * g * m_ha_2_m3 / eff_pump / 1e15
        m.addConstr((e ==  r_g_m_ha_2_m3_eff * v * l_t), name=f"c.{wid}.e(PJ)")

        self.vars[wid] = {}
        self.vars[wid]['e'] = e
        self.vars[wid]['v'] = v
        self.vars[wid]['q'] = q
        self.vars[wid]['l_pr'] = l_pr
        self.n_wells += 1

    def setup_constr_finance(self):
        """
        Add financial constraints. The output is in 1e4$.

        Returns
        -------
        None.

        """
        m = self.model
        cf = self.config['finance']
        energy_price = cf['energy_price']    #[1e4$/PJ]
        crop_profit = {c: cf['crop_price'][c] - cf['crop_cost'][c] for c in self.crop_options}
        crop_options = self.crop_options
        tech_options = self.tech_options
        n_h = self.n_h
        n_c = self.n_c
        n_s = self.n_s
        n_te = self.n_te
        inf = self.inf
        vars = self.vars
        field_ids = self.field_ids

        e = vars['e']     # (n_h) [PJ]
        y = vars['y']     # (n_s, n_c, n_h) [1e4 bu]

        cost_e = m.addMVar((n_h), vtype="C", name="cost_e(1e4$)", lb=0, ub=inf)
        rev = m.addMVar((n_h), vtype="C", name="rev(1e4$)", lb=-inf, ub=inf)

        cost_tech = np.array([cf['irr_tech_operational_cost'][te] for te in tech_options])

        annual_tech_cost = 0
        annual_tech_change_cost = 0
        annual_crop_change_cost = 0
        for fid in field_ids:
            i_te = vars[fid]['i_te']
            annual_tech_cost += i_te * cost_tech

            pre_i_te = vars[fid]['pre_i_te']
            tech_change_cost_arr = self.tech_change_cost_matrix[np.argmax(pre_i_te), :] # ==1
            i_tech_change = vars[fid]['i_tech_change']
            annual_tech_change_cost += tech_change_cost_arr * i_tech_change/n_h # uniformly allocate into planning horizon

            for s in range(n_s):
                pre_i_crop = vars[fid]['pre_i_crop'][s, :, 0]
                crop_change_cost_arr = self.crop_change_cost_matrix[np.argmax(pre_i_crop), :] # ==1
                i_crop_change = vars[fid]['i_crop_change'][s, :, 0]
                annual_crop_change_cost += crop_change_cost_arr * i_crop_change/n_h # uniformly allocate into planning horizon
        annual_cost = m.addMVar((n_h), vtype="C", name="annual_cost(1e4$)", lb=-inf, ub=inf)
        m.addConstr(annual_cost == \
                    gp.quicksum(annual_tech_cost[t] for t in range(n_te))
                    + gp.quicksum(annual_tech_change_cost[t] for t in range(n_te))
                    + gp.quicksum(annual_crop_change_cost[c] for c in range(n_c)),
                    name="c.annual_cost(1e4$)")

        # The profit variable is created in the initial to allow users to add
        # contraints without a specific order.
        profit = vars['profit']

        m.addConstr((cost_e == e * energy_price), name="c.cost_e")
        m.addConstr(rev == gp.quicksum(y[i,j,:] * crop_profit[c] \
                    for i in range(n_s) for j, c in enumerate(crop_options)),
                    name="c.rev")
        m.addConstr((profit == rev - cost_e - annual_cost), name="c.profit")
        vars['other_cost'] = annual_cost
        vars['cost_e'] = cost_e
        vars['rev'] = rev

    def setup_constr_wr(self, water_right_id, wr, field_id_list="all",
                        time_window=1, remaining_tw=None, remaining_wr=None,
                        tail_method="proportion"):
        """
        Add water rights constraints. You can assign multiple water rights
        constraints by calling this function repeatedly with different
        water_right_id. Water rights can constrain for all fields or a selected
        subset of fields with an optional time_window argument allowing farmer
        to allocate their water rights across multiple years. To enforce water
        rights at the point of diversion, pumping capacity can be assigned to
        individual wells in setup_constr_well().

        Parameters
        ----------
        water_right_id : str or int
            The water right id serves as a means to differentiate the equation
            sets for different water rights.
        wr : float
            Depth of the water right [cm].
        field_id_list : "all" or list, optional
            A list of field ids. If given, the water right constraints apply
            only to the subset of the fields. The default is "all".
        time_window : int, optional
            If given, the water right constrains the total irrigation depth
            over the time window. The default is 1.
        remaining_tw : int, optional
            Remaining year of time window that remaining_wr will apply to. The 
            default is None.
        remaining_wr : float, optional
            The remaining water rights left in the previous unused time window
            [cm]. The default is None.
        tail_method : "proportion" or "all" or float, optional
            Method to allocate incomplete time window at the end of the
            planning period. "proportion" means wr*(tail length/time_window) is
            apply to the tail part of planning period. "all" means wr is apply
            to the tail part of planning period. If a float is given, the value
            will be applied directly to the tail part of planning period. The
            default is "proportion".

        Returns
        -------
        None.

        """
        # As the msg said.
        if time_window != 1 and self.approx_horizon:
            raise ValueError("Approximation is not allow with the water rights "
                             +"constraints that have time_window larger than 1."
                             +" Please set approx_horizon to False.")

        m = self.model
        fids = field_id_list
        n_h = self.n_h
        n_c = self.n_c
        n_s = self.n_s
        vars = self.vars

        # Collect irrigation depth over the constrainted fields.
        if fids == "all":
            irr_sub = vars['irr_depth']         # (n_s, n_c, n_h)
        else:
            for i, fid in enumerate(fids):
                if i == 0:
                    irr_sub = vars[fid]['irr_depth']
                else:
                    irr_sub += vars[fid]['irr_depth']
            irr_sub = irr_sub/len(fids)

        # Initial period
        # The structure is to fit within a larger simulation framework, which
        # we allow the remaining water rights that are not used in the previous
        # year.
        c_i = 0
        
        if remaining_tw is not None and remaining_wr is not None:
            m.addConstr(
                gp.quicksum(irr_sub[i,j,h] \
                for i in range(n_s) for j in range(n_c) \
                for h in range(remaining_tw))/n_s <= remaining_wr,
                name=f"c.{water_right_id}.wr_{c_i}(cm)"
                )
            c_i += 1
            start_index = remaining_tw
            remaining_length = n_h - remaining_tw
        else:
            start_index = 0
            remaining_length = n_h

        # Middle period
        while remaining_length >= time_window:
            m.addConstr(
                gp.quicksum(irr_sub[i,j,h] \
                for i in range(n_s) for j in range(n_c) \
                for h in range(start_index, start_index+time_window))/n_s <= wr,
                name=f"c.{water_right_id}.wr_{c_i}(cm)"
                )
            c_i += 1
            start_index += time_window
            remaining_length -= time_window

        # Last period (if any)
        if remaining_length > 0:
            if tail_method == "proportion":
                wr_tail = wr * remaining_length/time_window
            elif tail_method == "all":
                wr_tail = wr
            # Otherwise, we expect a value given by users.
            else:
                wr_tail = tail_method

            m.addConstr(
                gp.quicksum(irr_sub[i,j,h] \
                for i in range(n_s) for j in range(n_c) \
                for h in range(start_index, n_h))/n_s <= wr_tail,
                name=f"c.{water_right_id}.wr_{c_i}(cm)"
                )

        self.water_right_ids.append(water_right_id)
        self.n_water_rights += 1

        # Record for the next run. Assume the simulation runs annually and will
        # apply the irr_depth solved by the opt model.
        # This record will be updated in solve() and add to the sols.
        if time_window == 1: 
            remaining_wr = None
            remaining_tw = None
        else:
            if remaining_tw is None: # This is the first year of the tw.
                remaining_wr = wr  # wait to be updated
                remaining_tw = time_window - 1
            elif (remaining_tw - 1) == 0:
                # remaining_tw - 1 = 0 means that next year will be a new round.
                remaining_wr = None # will not updated
                remaining_tw = time_window
            else:
                # remaining_wr = remaining_wr
                remaining_tw -= 1
            
        self.wrs_info[water_right_id] = {
            "wr": wr,
            "field_id_list": field_id_list,
            "time_window": time_window,
            "remaining_tw": remaining_tw, # Assume we optimize on a rolling basis
            "remaining_wr": remaining_wr, # If not None, the number will be updated later
            "tail_method": tail_method
            }

    def setup_obj(self, alpha_dict=None):
        """
        Add the objective to maximize the agent's expected satisfication. Note
        that the satisfication is calculated after the optimization which
        significantly speeds up the optimization process. The resulting
        solution is equivalent to directly using satisfaction as the objective
        function.

        Parameters
        ----------
        alpha_dict : dict, optional
            Overwrite alpha values retrieved from the config. The default is
            None.

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
            "profit": vars['profit'],
            "yield_pct": vars['y_y']
            }

        inf = self.inf
        m = self.model
        n_h = self.n_h

        vars['Sa'] = {}
        def add_metric(metric):
            # fakeSa will be forced to be nonnegative later on for Sa calculation
            fakeSa = m.addVar(vtype="C", name=f"fakeSa.{metric}", lb=-inf, ub=inf)
            metric_var = eval_metric_vars.get(metric)
            m.addConstr((fakeSa == gp.quicksum(metric_var[h] for h in range(n_h))/n_h),
                        name=f"c.Sa.{metric}")
            vars['Sa'][metric] = fakeSa

        penalties = self.penalties
        penalty = 0
        for p in penalties:
            penalty += p
        
        for metric in eval_metrics:
            # Add objective
            if metric == eval_metric:
                add_metric(metric)
                m.setObjective(vars['Sa'][metric] - penalty, gp.GRB.MAXIMIZE)
        self.obj_post_calculation = True

    def finish_setup(self, display_summary=True):
        """
        Complete the model setup.

        Parameters
        ----------
        display_summary : bool, optional
            Display the model summary. The default is True.

        Returns
        -------
        None

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
        vars['allo_r'] = allo_r
        vars['allo_r_w'] = allo_r_w
        m.addConstr(allo_r_w == gp.quicksum(allo_r[f,:,:] for f in range(n_f))/n_f,
                    name="c.allo_r_w")
        m.addConstrs((gp.quicksum(allo_r[f,k,h] for k in range(n_w)) == 1 \
                      for f in range(n_f) for h in range(n_h)), name="c.allo_r")
        v = vars['v']
        for k, wid in enumerate(wids):
            m.addConstr((vars[wid]['v'] == v * allo_r_w[k,:]), name=f"c.{wid}.v(m-ha)")
            m.addConstr((vars[wid]['q'] == gp.quicksum(vars[fid]['q'] * allo_r[f,k,:] \
                        for f, fid in enumerate(fids))), name=f"c.{wid}.q(m-ha/d)")
            m.addConstr((vars[wid]['l_pr'] == gp.quicksum(vars[fid]['l_pr'] * allo_r[f,k,:] \
                        for f, fid in enumerate(fids))), name=f"c.{wid}.l_pr(m)")

        irr_depth = vars['irr_depth']
        y = vars['y']
        y_y = vars['y_y']
        e = vars['e']
        # Sum to the total
        def get_sum(ids, vars, var):
            """Sum over ids"""
            for i, v in enumerate(ids):
                if i == 0:
                    acc = vars[v][var]
                else:
                    acc += vars[v][var]
            return acc
        m.addConstr(irr_depth == get_sum(fids, vars, "irr_depth"), name="c.irr_depth(cm)")
        m.addConstr(v == get_sum(fids, vars, "v"), name="c.v(m-ha)")
        m.addConstr(y == get_sum(fids, vars, "y"), name="c.y(1e4bu)")
        m.addConstr(y_y == get_sum(fids, vars, "y_y")/n_f, name="c.y_y")
        m.addConstr(e == get_sum(wids, vars, "e"), name="c.e(PJ)")

        m.update()

        if self.approx_horizon:
            h_msg = str(self.horizon) + " (approximate with 2)"
        else:
            h_msg = str(self.n_h)
        msg = dict_to_string(self.msg, prefix="\t\t", level=2)
        summary = f"""
        ########## Model Summary ##########\n
        Name:   {self.name}\n
        Planning horizon:   {h_msg}
        NO. Crop fields:    {self.n_fields}
        NO. splits          {self.n_s}
        NO. Wells:          {self.n_wells}
        NO. Water rights:   {self.n_water_rights}\n
        Decision settings:\n{msg}\n
        ###################################
        """
        self.summary = summary
        if display_summary:
            print(summary)

    def solve(self, keep_gp_model=False, keep_gp_output=False,
              display_report=True, **kwargs):
        """
        Solve the optimization problem.
        Note that defaultly we set the gurobi parameter NonConvex = 2 for a
        nonconvex model.

        Parameters
        ----------
        keep_gp_model : bool
            Keep the gurobi model instance for further used. Use with caution.
            The default is False.

        keep_gp_output : bool
            If True, the gurobi model output will be stored at "gp_output" in a
            dictionary format.

        display_report : bool
            Display the summary report if True. The default is True.

        **kwargs : **kwargs
            Pass the gurobi parameters to the gurobi solver.

        Returns
        -------
        None.

        Notes
        -----
        More info:
            https://www.gurobi.com/documentation/9.5/refman/mip_models.html

        """
        def extract_sol(vars):
            sols = {}
            def get_inner_dict(d, new_dict):
                for k, v in d.items():
                    if isinstance(v, dict):
                        new_dict[k] = {}
                        get_inner_dict(v, new_dict[k])
                    else:
                        try:
                            new_dict[k] = v.X   # for variables of gurobi
                        except:
                            new_dict[k] = v     # for all others
            get_inner_dict(vars, sols)
            return sols

        ## Solving model
        m = self.model
        gurobi_pars = self.gurobi_pars
        gurobi_pars.update(kwargs)
        if "NonConvex" not in gurobi_pars.keys():
            m.setParam("NonConvex", 2)  # Set to solve non-convex problem
        for k, v in gurobi_pars.items():
            m.setParam(k, v)
        m.optimize()

        ## Collect the results and do some post calculations.
        if m.Status == 2:   # Optimal solution found
            self.optimal_obj_value = m.objVal
            self.sols = extract_sol(self.vars)
            sols = self.sols
            sols['obj'] = m.objVal
            sols['field_ids'] = self.field_ids
            sols['well_ids'] = self.well_ids
            sols['gp_status'] = m.Status
            sols['gp_MIPGap'] = m.MIPGap

            # Calculate satisfication
            if self.obj_post_calculation:
                eval_metrics = self.eval_metrics
                alphas = self.alphas
                scales = self.scales

                # Currently supported metrices
                if self.approx_horizon and self.horizon > 2:
                    horizon = self.horizon
                    profits = sols['profit']
                    y_ys = sols['y_y']
                    eval_metric_vars = {
                        "profit": np.linspace(profits[0], profits[1], num=horizon)/scales['profit'],
                        "yield_pct": np.linspace(y_ys[0], y_ys[1], num=horizon)/scales['yield_pct']
                        }
                else:
                    eval_metric_vars = {
                        "profit": sols['profit']/scales['profit'],
                        "yield_pct": sols['y_y']/scales['yield_pct']
                        }
                for metric in eval_metrics:
                    alpha = alphas[metric]
                    metric_var = eval_metric_vars.get(metric)
                    # force the minimum value to be zero since exp
                    metric_var[metric_var<0] = 0
                    N_yr = 1 - np.exp(-alpha * metric_var)
                    Sa = np.mean(N_yr)
                    sols['Sa'][metric] = Sa
            # Update rainfed info
            for fid in self.field_ids:
                sols_fid = sols[fid]
                if (sols_fid['field_type'] == "rainfed" 
                    or sols_fid['field_type'] == "optimize"):
                    irr_depth = sols_fid['irr_depth'][:,:,0].sum(axis=1)
                    i_rainfed = sols_fid['i_rainfed']
                    i_rainfed[np.where(irr_depth <= 0), :, :] = 1 # avoid using irr_depth == 0
                    sols_fid['i_rainfed'] = i_rainfed * sols_fid['i_crop']

            # Update remaining water rights
            wrs_info = self.wrs_info
            for k, v in wrs_info.items():
                if v['remaining_wr'] is not None:
                    fids = v['field_id_list']
                    if fids == "all":
                        irr_sub = sols['irr_depth']         # (n_s, n_c, n_h)
                    else:
                        for i, fid in enumerate(fids):
                            if i == 0:
                                irr_sub = sols[fid]['irr_depth'].copy()
                            else:
                                irr_sub += sols[fid]['irr_depth']
                        irr_sub = irr_sub/len(fids)
                    v['remaining_wr'] -= np.sum(irr_sub[:, :, 0])
            sols['water_rights'] = wrs_info

            # Display report
            crop_options = self.crop_options
            tech_options = self.tech_options
            n_s = self.n_s
            fids = self.field_ids
            #sols = self.sols
            irrs = list(sols['irr_depth'].mean(axis=0).sum(axis=0).round(2))
            decisions = {"Irrigation depths": irrs}
            for fid in fids:
                sols_fid = sols[fid]
                i_crop = sols_fid['i_crop'][:, :, 0]
                # Avoid using == 0 or 1 => some time have numerical issue
                crop_type = [crop_options[np.argmax(i_crop[s,:])] for s in range(n_s)]
                tech = tech_options[np.argmax(sols_fid['i_te'][:])]
                Irrigated = list((sols_fid['i_rainfed'][:, :, 0].sum(axis=1).round(0) <= 0))
                decisions[fid] = {"Crop types": crop_type,
                                  "Irr tech": tech,
                                  "Irrigated": Irrigated}
            self.decisions = decisions
            decisions = dict_to_string(decisions, prefix="\t\t", level=2)
            msg = dict_to_string(self.msg, prefix="\t\t", level=2)
            sas = dict_to_string(sols['Sa'], prefix="\t\t", level=2, roun=4)
            if self.approx_horizon:
                h_msg = str(self.horizon) + " (approximate with 2)"
            else:
                h_msg = str(self.n_h)
            gp_report = f"""
        ########## Model Report ##########\n
        Name:   {self.name}\n
        Planning horizon:   {h_msg}
        NO. Crop fields:    {self.n_fields}
        NO. splits          {self.n_s}
        NO. Wells:          {self.n_wells}
        NO. Water rights:   {self.n_water_rights}\n
        Decision settings:\n{msg}\n
        Solutions (gap {round(m.MIPGap * 100, 4)}%):\n{decisions}\n
        Satisfication:\n{sas}\n
        ###################################
            """
            self.gp_report = gp_report
            if display_report:
                print(gp_report)
            sols['gp_report'] = gp_report
            #self.sols = sols
        else:
            print("Optimal solution is not found.")
            self.optimal_obj_value = None
            sols = {}
            sols['gp_report'] = "Optimal solution is not found."
            self.sols = sols

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
        Output the model to .mps.

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

# Utility code
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
