# The code is developed by Chung-Yi Lin at Virginia Tech, in June 2024.
# Email: chungyi@vt.edu
import json

import gurobipy as gp
import numpy as np


class Optimization_1f1w_ci:
    """
    This class is specifically designed for 1 field and 1 well with fixed
    irrigation technology to facilitate the solving speed.
    """

    def __init__(self):
        pass

    def setup_ini_model(
        self,
        unique_id,
        gpenv,
        activate_ci,
        horizon=1,
        crop_options=None,
    ):
        ## Basic information
        if crop_options is None:
            crop_options = ["corn", "sorghum", "soybeans", "fallow"]
        self.unique_id = unique_id
        self.horizon = horizon
        self.crop_options = crop_options
        self.activate_ci = activate_ci

        ## Dimension coefficients
        self.n_c = len(crop_options)  # No. of crop choice options
        self.n_h = horizon

        ## Records fields and wells
        self.field_ids = []
        self.well_ids = []
        self.water_right_ids = []
        self.n_fields = 0
        self.n_wells = 0
        self.n_water_rights = 0

        ## Optimization Model
        self.model = gp.Model(name=unique_id, env=gpenv)
        self.vars_ = {}  # A container to store variables.
        self.bounds = {}
        self.inf = float("inf")

        ## Add shared variables
        m = self.model
        inf = self.inf
        n_c = self.n_c
        n_h = self.n_h
        # Total irrigation depth per split per crop per yr
        irr_depth = m.addMVar((n_c, n_h), vtype="C", name="irr_depth(cm)", lb=0, ub=inf)
        # Total irrigation volumn per yr
        v = m.addMVar((n_h), vtype="C", name="v(m-ha)", lb=0, ub=inf)
        # Total yield per split per crop type per yr
        y = m.addMVar((n_c, n_h), vtype="C", name="y(1e4bu)", lb=0, ub=inf)
        # Average y_ (i.e., y/ymax) per yr
        y_y = m.addMVar((n_h), vtype="C", name="y_y", lb=0, ub=1)
        # Total energy (PJ) used for pumping per yr
        e = m.addMVar((n_h), vtype="C", name="e(PJ)", lb=0, ub=inf)
        # Total profit
        profit = m.addMVar((n_h), vtype="C", name="profit(1e4$)", lb=-inf, ub=inf)

        ## Record variables
        self.vars_["irr_depth"] = irr_depth
        self.vars_["v"] = v
        self.vars_["y"] = y
        self.vars_["e"] = e
        ## Average values over fields
        self.vars_["y_y"] = y_y
        self.vars_["profit"] = profit

        ## Record msg about the user inputs.
        self.msg = {}

        ## Record water rights info.
        self.wrs_info = {}

    # We will extract premium_dict from a field and store here.
    # premium_dict = ["rainfed": {"corn": xxx}, "irrigated": {"corn": xxx}]
    def setup_constr_field(
        self,
        field_id,
        field_area,
        prec_aw,
        water_yield_curves,
        field_type="optimize",
        i_crop=None,
        i_rainfed=None,
        premium_dict=None,
        aph_yield_dict=None,
        **kwargs,
    ):
        ## Append field_id
        self.field_ids.append(field_id)
        fid = field_id

        ## Crop options and dimensions
        crop_options = self.crop_options
        n_c = self.n_c
        n_h = self.n_h

        ## Extract parameters from water_yield_curves
        crop_par = np.array([water_yield_curves[c] for c in crop_options])
        ymax = crop_par[:, 0].reshape((-1, 1))  # (n_c, 1)
        wmax = crop_par[:, 1].reshape((-1, 1))  # (n_c, 1)
        a = crop_par[:, 2].reshape((-1, 1))  # (n_c, 1)
        b = crop_par[:, 3].reshape((-1, 1))  # (n_c, 1)
        c = crop_par[:, 4].reshape((-1, 1))  # (n_c, 1)
        try:
            min_y_ratio = crop_par[:, 5].reshape((-1, 1))  # (n_c, 1)
        except:
            min_y_ratio = np.zeros((n_c, 1))

        ## Overwrite field_type if i_rainfed is given.
        if i_rainfed is not None:
            if (
                np.sum(i_rainfed) > 0.5
            ):  # Avoid numerical issue (should be 1 if rainfed)
                field_type = "rainfed"
            else:
                field_type = "irrigated"

        ## Summary message for the setting.
        self.msg[fid] = {
            "Crop types": "optimize",
            "Irr tech": "optimize",
            "Field type": field_type,
        }

        # Record the input
        i_crop_input = i_crop
        i_rainfed_input = i_rainfed

        ## Add constraints
        m = self.model
        inf = self.inf
        self.bounds["ub_w"] = np.max(wmax)
        ub_w = self.bounds["ub_w"]
        ub_irr = (
            ub_w  # ub_w - prec_aw (maximum water required - available precipitation)
        )
        self.bounds[fid] = {}
        self.bounds[fid]["ub_irr"] = ub_irr

        ## Compute the available precipitiation for each crop.
        prec_aw_ = np.ones((n_c, n_h))
        for ci, crop in enumerate(crop_options):
            prec_aw_[ci, :] = prec_aw[crop]

        ## Add general variables
        w = m.addMVar((n_c, n_h), vtype="C", name=f"{fid}.w(cm)", lb=0, ub=ub_w)
        w_temp = m.addMVar((n_c, n_h), vtype="C", name=f"{fid}.w_temp", lb=0, ub=inf)
        w_ = m.addMVar((n_c, n_h), vtype="C", name=f"{fid}.w_", lb=0, ub=1)
        y_ = m.addMVar((n_c, n_h), vtype="C", name=f"{fid}.y_", lb=0, ub=1)
        yw_temp = m.addMVar((n_c, n_h), vtype="C", name=f"{fid}.yw_temp", lb=-inf, ub=1)
        yw_bi = m.addMVar((n_c, n_h), vtype="B", name=f"{fid}.yw_bi")
        yw_ = m.addMVar((n_c, n_h), vtype="C", name=f"{fid}.yw_", lb=0, ub=1)
        v_c = m.addMVar((n_c, n_h), vtype="C", name=f"{fid}.v_c(m-ha)", lb=0, ub=inf)
        i_crop = m.addMVar((n_c, 1), vtype="B", name=f"{fid}.i_crop")
        i_rainfed = m.addMVar((n_c, 1), vtype="B", name=f"{fid}.i_rainfed")

        ## Extract global opt variables
        irr_depth = self.vars_["irr_depth"]
        y = self.vars_["y"]
        y_y = self.vars_["y_y"]
        v = self.vars_["v"]

        ## Given crop type input
        if i_crop_input is not None:
            m.addConstr(i_crop == i_crop_input, name=f"c.{fid}.i_crop_input")
            self.msg[fid]["Crop types"] = "user input"

        ## One unit area can be occupied by only one type of crop.
        m.addConstr(
            gp.quicksum(i_crop[ci, :] for ci in range(n_c)) == 1, name=f"c.{fid}.i_crop"
        )

        ### Include rain-fed option
        if field_type == "rainfed":
            # Given i_rainfed,
            if i_rainfed_input is not None:
                m.addConstr(
                    i_rainfed == i_rainfed_input, name=f"c.{fid}.i_rainfed_input"
                )
                self.msg[fid]["Rainfed field"] = "user input"

            # i_rainfed[si, ci, hi] can be 1 only when i_crop[si, ci, hi] is 1.
            # Otherwise, it has to be zero.
            m.addConstr(i_crop - i_rainfed >= 0, name=f"c.{fid}.i_rainfed")
            m.addConstr(irr_depth == 0, name=f"c.{fid}.irr_rain_fed")

        elif field_type == "irrigated":
            m.addConstr(i_rainfed == 0, name=f"c.{fid}.no_i_rainfed")

        elif field_type == "optimize":
            # i_rainfed[si, ci, hi] can be 1 only when i_crop[si, ci, hi] is 1.
            # Otherwise, it has to be zero.
            m.addConstr(i_crop - i_rainfed >= 0, name=f"c.{fid}.i_rainfed")
            m.addConstr(irr_depth * i_rainfed == 0, name=f"c.{fid}.irr_rainfed")
        else:
            raise ValueError(f"{field_type} is not a valid value for field_type.")

        # See the numpy broadcast rules:
        # https://numpy.org/doc/stable/user/basics.broadcasting.html
        m.addConstr((w == irr_depth + prec_aw_), name=f"c.{fid}.w(cm)")
        m.addConstr((w_temp == w / wmax), name=f"c.{fid}.w_temp")
        m.addConstrs(
            (
                w_[ci, hi] == gp.min_(w_temp[ci, hi], constant=1)
                for ci in range(n_c)
                for hi in range(n_h)
            ),
            name=f"c.{fid}.w_",
        )  # w_ = minimum of 1 or w/w_max

        # We force irr_depth to be zero but prec_aw_ will add to w & w_, which will
        # output positive y_ leading to violation for y_y (< 1)
        # Also, we need to seperate yw_ and y_ into two constraints. Otherwise,
        # gurobi will crash. No idea why.

        m.addConstr((yw_temp == (a * w_**2 + b * w_ + c)), name=f"c.{fid}.yw_temp")

        # Minimum yield_rate cutoff (aim to capture fallow field)
        m.addConstr(
            (
                yw_bi * (yw_temp - min_y_ratio) + (1 - yw_bi) * (min_y_ratio - yw_temp)
                >= 0
            ),
            name=f"c.{fid}.yw_bi",
        )  # yw_bi is 1 or 0 based on yw_temp is greater or less than min_y_ratio
        m.addConstr((yw_ == yw_bi * yw_temp), name=f"c.{fid}.yw_")

        m.addConstr((y_ == yw_ * i_crop), name=f"c.{fid}.y_")
        m.addConstr((y == y_ * ymax * field_area * 1e-4), name=f"c.{fid}.y")  # 1e4 bu
        m.addConstr((irr_depth * (1 - i_crop) == 0), name=f"c.{fid}.irr_depth(cm)")
        cm2m = 0.01
        m.addConstr((v_c == irr_depth * field_area * cm2m), name=f"c.{fid}.v_c(m-ha)")
        m.addConstr(
            v == gp.quicksum(v_c[j, :] for j in range(n_c)), name=f"c.{fid}.v(m-ha)"
        )
        m.addConstr(
            y_y == gp.quicksum(y_[j, :] for j in range(n_c)), name=f"c.{fid}.y_y"
        )

        self.vars_[fid] = {}
        self.vars_[fid]["i_crop"] = i_crop
        self.vars_[fid]["i_rainfed"] = i_rainfed
        self.vars_[fid]["field_type"] = field_type
        self.premium_dict = premium_dict  # assuming single field
        self.aph_yield_dict = aph_yield_dict  # assuming single field

        self.n_fields += 1

    def setup_constr_well(
        self,
        well_id,
        dwl,
        B,
        l_wt,
        eff_pump,
        pumping_capacity=None,
        rho=1000.0,
        g=9.8016,
    ):
        self.well_ids.append(well_id)
        wid = well_id

        m = self.model
        n_h = self.n_h

        v = self.vars_["v"]
        if pumping_capacity is not None:
            m.addConstr((v <= pumping_capacity), name=f"c.{wid}.pumping_capacity")

        # Project the future lift head.
        dwls = np.array([dwl * (i) for i in range(n_h)])
        # Assume a linear projection to the future
        l_wt = l_wt - dwls
        #!!!! From our precalculation for sd6
        B = B - 0.00015 * dwls
        self.l_wt = l_wt
        self.B = B

        #!!!! Center pivot LEPA (fixed)
        tech_a = 0.0058
        tech_b = 0.212206
        l_pr = 12.65

        A = rho * g / eff_pump * 1e-11
        AaB = A * tech_a * B  # (n_h)
        A_L_bB = A * (l_wt + l_pr + tech_b * B)  # (n_h)

        e = self.vars_["e"]
        m.addConstr((e == AaB * v * v + A_L_bB * v), name=f"c.{wid}.e(PJ)")

        self.n_wells += 1

    # Modify this to include crop insurance
    # Finance_dict should contain projected and harvast pricess
    def setup_constr_finance(self, finance_dict, payout_ratio=1, premium_ratio=1):
        m = self.model
        crop_options = self.crop_options
        n_h = self.n_h
        n_c = self.n_c
        inf = self.inf
        vars_ = self.vars_

        fid = self.field_ids[-1]
        i_crop = vars_[fid]["i_crop"]
        i_rainfed = vars_[fid]["i_rainfed"]
        premium_dict = self.premium_dict
        aph_yield_dict = self.aph_yield_dict

        energy_price = finance_dict["energy_price"]  # [1e4$/PJ]
        crop_profit = {
            c: finance_dict["crop_price"][c] - finance_dict["crop_cost"][c]
            for c in crop_options
        }
        harvest_price = {c: finance_dict["harvest_price"][c] for c in crop_options}

        projected_price = {c: finance_dict["projected_price"][c] for c in crop_options}

        cost_tech = 1.876  # center pivot LEPA

        e = vars_["e"]  # (n_h) [PJ]
        y = vars_["y"]  # (n_c, n_h) [1e4 bu]

        cost_e = m.addMVar((n_h), vtype="C", name="cost_e(1e4$)", lb=0, ub=inf)
        rev = m.addMVar((n_h), vtype="C", name="rev(1e4$)", lb=-inf, ub=inf)

        annual_cost = m.addMVar(
            (n_h), vtype="C", name="annual_cost(1e4$)", lb=-inf, ub=inf
        )
        m.addConstr(annual_cost == cost_tech, name="c.annual_cost(1e4$)")

        m.addConstr((cost_e == e * energy_price), name="c.cost_e")
        m.addConstr(
            rev
            == gp.quicksum(
                y[j, :] * crop_profit[c] for j, c in enumerate(crop_options)
            ),
            name="c.rev",
        )

        # Premium constraint
        premium = m.addMVar((n_h), vtype="C", name="premium(1e4$)", lb=0, ub=inf)
        if self.activate_ci:
            for j, c in enumerate(crop_options):
                for r in range(n_c):
                    # Rainfed
                    i_cr = m.addVar(vtype="B", name=f"i_cr_{c}_{r}_rainfed")
                    m.addConstr(i_cr <= i_crop[j])
                    m.addConstr(i_cr <= i_rainfed[r])
                    m.addConstr(i_cr >= i_crop[j] + i_rainfed[r] - 1)
                    m.addConstr(
                        premium == i_cr * premium_dict["rainfed"][c],
                        name=f"c.premium_{c}_{r}_rainfed",
                    )

                    # Irrigated
                    i_cr = m.addVar(vtype="B", name=f"i_cr_{c}_{r}_irrigated")
                    m.addConstr(i_cr <= i_crop[j])
                    m.addConstr(i_cr <= 1 - i_rainfed[r])
                    m.addConstr(i_cr >= i_crop[j] + i_rainfed[r] - 1)
                    m.addConstr(
                        premium == i_cr * premium_dict["irrigated"][c] * premium_ratio,
                        name=f"c.premium_{c}_{r}_irrigated",
                    )
        else:
            premium = m.addConstr(premium == 0, name="c.null_premium")

        # Payout constraint
        harvest_price = {c: finance_dict["harvest_price"][c] for c in crop_options}

        projected_price = {c: finance_dict["projected_price"][c] for c in crop_options}

        i_rainfed = vars_[self.field_ids[-1]]["i_rainfed"]  # Assuming just one field.
        payout = m.addMVar((n_h), vtype="C", name="premium(1e4$)", lb=0, ub=inf)
        if self.activate_ci:
            payout_crops = []
            for j, c in enumerate(crop_options):
                h_or_p = max(harvest_price[c], projected_price[c])

                payout_c = m.addMVar(
                    (n_h), vtype="C", name=f"payout_{c}(1e4$)", lb=0, ub=inf
                )

                payout_temp = m.addMVar(
                    (n_h), vtype="C", name="payout_temp", lb=-inf, ub=inf
                )
                m.addConstrs(
                    (
                        payout_temp[hi]
                        == h_or_p
                        * 0.75
                        * aph_yield_dict["irrigated"][c]
                        * (1 - i_rainfed[j, 0])
                        + h_or_p * 0.75 * aph_yield_dict["rainfed"][c] * i_rainfed[j, 0]
                        - harvest_price[c] * y[j, hi]
                    )
                    for hi in range(n_h)
                )

                m.addConstrs(
                    (
                        payout_c[hi] == gp.max_(payout_temp[hi], constant=0)
                        for hi in range(n_h)
                    ),
                    name=f"c.payout_c_{c}",
                )  # w_ = minimum of 1 or w/w_max

                payout_crops.append(payout_c)

            for hi in range(n_h):
                m.addConstr(
                    payout[hi]
                    == gp.quicksum(
                        payout_c[j] * i_crop[j] for j, c in enumerate(crop_options)
                    )
                    * payout_ratio,
                    name="c.payout",
                )
        else:
            m.addConstr(payout == 0, name="c.payout")

        vars_["rev"] = rev
        vars_["cost_e"] = cost_e
        vars_["other_cost"] = annual_cost
        vars_["premium"] = premium
        vars_["payout"] = payout

        # Note the average profit per field is calculated in finish_setup().
        # That way we can ensure the final field numbers added by users.

    def setup_constr_wr(
        self,
        water_right_id,
        wr_depth,
        time_window=1,
        remaining_tw=None,
        remaining_wr=None,
        tail_method="proportion",
    ):
        """
        Set up water right constraints for the optimization model. You can assign multiple water rights
        constraints by calling this function repeatedly with different
        water_right_id. Water rights can constrain all fields or a selected
        subset of fields with an optional time_window argument, allowing the farmer
        to allocate their water rights across multiple years. To enforce water
        rights at the point of diversion, pumping capacity can be assigned to
        individual wells in setup_constr_well() method.

        Parameters
        ----------
        water_right_id : str or int
            The water right id serves as a means to differentiate the equation
            sets for different water rights.
        wr_depth : float
            Depth of the water right [cm].
        time_window : int, optional
            If given, the water right constrains the total irrigation depth
            over the time window [yr]. The default is 1.
        remaining_tw : int, optional
            Remaining years of time window that the remaining_wr will be applied to [yr]. The
            default is None.
        remaining_wr : float, optional
            The remaining water rights left unused from the previous time window
            [cm]. The default is None.
        tail_method : "proportion" or "all" or float, optional
            Method to allocate water rights to the incomplete part of the time window at the end of the
            planning period.

            "proportion" means water equivalent to wr_depth*(tail length/time_window) is
            applied to the tail part of the planning period.

            "all" means water equivalent to wr_depth is applied to the tail part of planning period.

            If a float is given, the given value
            will be applied directly to the tail part of planning period.

            The default is "proportion".

        Returns
        -------
        None.

        """
        m = self.model
        n_h = self.n_h
        n_c = self.n_c
        vars_ = self.vars_

        irr_sub = vars_["irr_depth"]

        # Initial period
        # The structure is to fit within a larger simulation framework, which
        # we allow the remaining water rights that are not used in the previous
        # year.
        c_i = 0

        if remaining_tw is not None and remaining_wr is not None:
            m.addConstr(
                gp.quicksum(
                    irr_sub[j, h] for j in range(n_c) for h in range(remaining_tw)
                )
                <= remaining_wr,
                name=f"c.{water_right_id}.wr_{c_i}(cm)",
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
                gp.quicksum(
                    irr_sub[j, h]
                    for j in range(n_c)
                    for h in range(start_index, start_index + time_window)
                )
                <= wr_depth,
                name=f"c.{water_right_id}.wr_{c_i}(cm)",
            )
            c_i += 1
            start_index += time_window
            remaining_length -= time_window

        # Last period (if any)
        if remaining_length > 0:
            if tail_method == "proportion":
                wr_tail = wr_depth * remaining_length / time_window
            elif tail_method == "all":
                wr_tail = wr_depth
            # Otherwise, we expect a value given by users.
            else:
                wr_tail = tail_method

            m.addConstr(
                gp.quicksum(
                    irr_sub[j, h] for j in range(n_c) for h in range(start_index, n_h)
                )
                <= wr_tail,
                name=f"c.{water_right_id}.wr_{c_i}(cm)",
            )

        self.water_right_ids.append(water_right_id)
        self.n_water_rights += 1

        # Record for the next run. Assume the simulation runs annually and will
        # apply the irr_depth solved by the opt model.
        # This record will be updated in solve() and added to the sols.
        if time_window == 1:
            remaining_wr = None
            remaining_tw = None
        else:
            if remaining_tw is None:  # This is the first year of the tw.
                remaining_wr = wr_depth  # wait to be updated
                remaining_tw = time_window - 1
            elif (remaining_tw - 1) == 0:
                # remaining_tw - 1 = 0 means that next year will be a new round.
                remaining_wr = None  # will not update
                remaining_tw = time_window
            else:
                # remaining_wr = remaining_wr
                remaining_tw -= 1

        self.wrs_info[water_right_id] = {
            "wr_depth": wr_depth,
            "time_window": time_window,
            "remaining_tw": remaining_tw,  # Assume we optimize on a rolling basis
            "remaining_wr": remaining_wr,  # If not None, the number will be updated later
            "tail_method": tail_method,
        }

    def setup_obj(
        self,
        target="profit",
        consumat_dict=None,
    ):
        """
        This method sets the objective of the optimization model, i.e., to maximize the agent's expected satisfaction. Note
        that the satisfaction value is calculated after the optimization process, which
        significantly speeds up the optimization process. The resulting
        solution is equivalent to directly using satisfaction as the objective
        function.

        Returns
        -------
        None.

        """
        if consumat_dict is None:
            consumat_dict = {"alpha": {"profit": 1}, "scale": {"profit": 0.23 * 50}}
        self.target = target

        # For consumat
        self.alphas = consumat_dict["alpha"]
        self.scales = consumat_dict["scale"]

        vars_ = self.vars_

        # Currently supported metrices
        # We use average value per field (see finish_setup())
        eval_metric_vars = {"profit": vars_["profit"], "yield_rate": vars_["y_y"]}

        if target not in eval_metric_vars:
            print(f"{target} is not a valid metric.")

        inf = self.inf
        m = self.model
        n_h = self.n_h

        vars_["Sa"] = {}

        def add_metric(metric):
            # fakeSa will be forced to be nonnegative later on for Sa calculation
            fakeSa = m.addVar(vtype="C", name=f"fakeSa.{metric}", lb=-inf, ub=inf)
            metric_var = eval_metric_vars.get(metric)
            m.addConstr(
                (fakeSa == gp.quicksum(metric_var[h] for h in range(n_h)) / n_h),
                name=f"c.Sa.{metric}",
            )
            vars_["Sa"][metric] = fakeSa  # fake Sa for each metric (profit and y_Y)

        # Add objective
        add_metric(target)
        m.setObjective(vars_["Sa"][target], gp.GRB.MAXIMIZE)
        self.obj_post_calculation = True

    def finish_setup(self, display_summary=True):
        """
        This method completes the setup for the optimization model.

        Parameters
        ----------
        display_summary : bool, optional
            Display the model summary. The default is True.

        Returns
        -------
        None

        """
        m = self.model
        vars_ = self.vars_
        n_f = self.n_fields

        # Calculate average value per field
        profit = vars_["profit"]
        rev = vars_["rev"]
        cost_e = vars_["cost_e"]
        annual_cost = vars_["other_cost"]

        if self.activate_ci:
            premium = vars_["premium"]
            payout = vars_["payout"]
            m.addConstr(
                (profit == (payout + rev - cost_e - annual_cost - premium) / n_f),
                name="c.profit",
            )
        else:
            m.addConstr(
                (profit == (rev - cost_e - annual_cost) / n_f),
                name="c.profit",
            )

        m.update()

        h_msg = str(self.n_h)

        msg = dict_to_string(self.msg, prefix="\t\t", level=2)
        summary = f"""
        ########## Model Summary ##########\n
        Name:   {self.unique_id}\n
        Planning horizon:   {h_msg}
        No. of Crop fields:    {self.n_fields}
        No. of Wells:          {self.n_wells}
        NO. of Water rights:   {self.n_water_rights}\n
        Decision settings:\n{msg}\n
        ###################################
        """
        self.summary = summary
        if display_summary:
            print(summary)

    def solve(
        self, keep_gp_model=False, keep_gp_output=False, display_report=True, **kwargs
    ):
        def extract_sol(vars_):
            sols = {}

            def get_inner_dict(d, new_dict):
                for k, v in d.items():
                    if isinstance(v, dict):
                        new_dict[k] = {}
                        get_inner_dict(v, new_dict[k])
                    else:
                        try:
                            new_dict[
                                k
                            ] = v.X  # for variables associated with the gurobi solver
                        except:
                            new_dict[k] = v  # for all others

            get_inner_dict(vars_, sols)
            return sols

        ## Solving model
        m = self.model
        m.optimize()

        ## Collect the results and do some post calculations.
        # Optimal solution found or reach time limit
        if m.Status == 2 or m.Status == 9:
            self.optimal_obj_value = m.objVal
            self.sols = extract_sol(self.vars_)
            sols = self.sols
            sols["obj"] = m.objVal
            sols["field_ids"] = self.field_ids
            sols["well_ids"] = self.well_ids
            sols["gp_status"] = m.Status
            sols["gp_MIPGap"] = m.MIPGap

            # Calculate satisfaction
            if self.obj_post_calculation:
                alphas = self.alphas
                scales = self.scales
                metric = self.target

                # Currently supported metrices
                eval_metric_vars = {
                    "profit": sols["profit"] / scales["profit"],
                    # "yield_rate": sols['y_y']/scales['yield_rate']
                }

                alpha = alphas[metric]
                metric_var = eval_metric_vars.get(metric)
                # force the minimum value to be zero since there is an exponential function
                metric_var[metric_var < 0] = 0
                N_yr = 1 - np.exp(-alpha * metric_var)
                Sa = np.mean(N_yr)
                sols["Sa"][metric] = Sa

            # Update rainfed info
            for fid in self.field_ids:
                sols_fid = sols[fid]
                irr_depth = sum(sols["irr_depth"][:, 0])
                i_rainfed = sols_fid["i_rainfed"]
                if irr_depth <= 0:
                    i_rainfed[:, :] = 1  # avoid using irr_depth == 0
                sols_fid["i_rainfed"] = i_rainfed * sols_fid["i_crop"]

            # Update remaining water rights
            wrs_info = self.wrs_info
            for _k, v in wrs_info.items():
                if v["remaining_wr"] is not None:
                    irr_sub = sols["irr_depth"]  # (n_c, n_h)
                    v["remaining_wr"] -= np.sum(irr_sub[:, 0])
            sols["water_rights"] = wrs_info

            # Display report
            crop_options = self.crop_options
            fids = self.field_ids
            irrs = sols["irr_depth"].mean().round(2)
            decisions = {"Irrigation depths": irrs}
            for fid in fids:
                sols_fid = sols[fid]
                i_crop = sols_fid["i_crop"][:, 0]
                # Avoid using == 0 or 1 => it can have numerical issues
                crop_type = crop_options[np.argmax(i_crop)]
                Irrigated = sols_fid["i_rainfed"][:, 0].sum().round(0) <= 0
                decisions[fid] = {
                    "Crop types": crop_type,
                    "Irr tech": "center pivot LEPA",
                    "Irrigated": Irrigated,
                }
            self.decisions = decisions
            decisions = dict_to_string(decisions, prefix="\t\t", level=2)
            msg = dict_to_string(self.msg, prefix="\t\t", level=2)
            sas = dict_to_string(sols["Sa"], prefix="\t\t", level=2, roun=4)
            h_msg = str(self.n_h)
            gp_report = f"""
        ########## Model Report ##########\n
        Name:   {self.unique_id}\n
        Planning horizon:   {h_msg}
        No. of Crop fields:    {self.n_fields}
        No. of Wells:          {self.n_wells}
        No. of Water rights:   {self.n_water_rights}\n
        Decision settings:\n{msg}\n
        Solutions (gap {round(m.MIPGap * 100, 4)}%):\n{decisions}\n
        Satisfaction:\n{sas}\n
        ###################################
            """
            self.gp_report = gp_report
            if display_report:
                print(gp_report)
            sols["gp_report"] = gp_report
        else:
            print("Optimal solution is not found.")
            self.optimal_obj_value = None
            sols = {}
            sols["gp_report"] = "Optimal solution is not found."
            self.sols = sols

        if keep_gp_output:
            self.gp_output = json.loads(m.getJSONSolution())

        if keep_gp_model is False:
            # release the memory of the previous model
            m.dispose()

    def do_IIS_gp(self, filename=None):
        """
        Compute an Irreducible Inconsistent Subsystem (IIS). This function can
        only be executed if the model is infeasible.

        An IIS is a subset of the constraints and variable bounds with the
        following properties:

        - It is still infeasible, and
        - If a single constraint or bound is removed, the subsystem becomes feasible.

        Note that an infeasible model may have multiple IISs. The one returned
        by Gurobi is not necessarily the smallest one; others may exist
        with a fewer constraints or bounds.

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
            print("IIS is minimal\n")
        else:
            print("IIS is not minimal\n")
        print("\nThe following constraint(s) cannot be satisfied:")
        for c in m.getConstrs():
            if c.IISConstr:
                print("%s" % c.ConstrName)

        if filename is not None:
            if filename[-4:] != ".ilp":
                filename += ".ilp"
            m.write(filename)

    def write_ilp(self, filename):
        """
        This function outputs the information about the results of the IIS computation to
        .ilp and can only be executed after do_IIS_gp().

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
        This function outputs the solution of the model to a .sol file.

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
        This function outputs the model to a .lp file.

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
        This funtion outputs the model to an .mps file.

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

    def depose_gp_env(self):
        self.gpenv.dispose()


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

    def dict_to_string_list(dictionary, indentor="  ", count=1, string=None):
        if string is None:
            string = []
        for key, value in dictionary.items():
            string.append(prefix + indentor * count + str(key))
            if isinstance(value, dict) and count < level:
                string = dict_to_string_list(value, indentor, count + 1, string)
            elif isinstance(value, dict) is False and count == level:
                string[-1] += ":\t" + str(value)
            else:
                if roun is not None and isinstance(value, float):
                    string.append(
                        prefix + indentor * (count + 1) + str(round(value, roun))
                    )
                else:
                    string.append(prefix + indentor * (count + 1) + str(value))
        return string

    return "\n".join(dict_to_string_list(dictionary, indentor))
