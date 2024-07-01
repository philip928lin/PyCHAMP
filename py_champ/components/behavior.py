# The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
# Email: chungyi@vt.edu
# Last modified on Dec 30, 2023
import warnings

import mesa
import numpy as np
import pandas as pd
from scipy.stats import truncnorm


class Behavior(mesa.Agent):
    """
    This module is a farmer's behavior simulator.

    Parameters
    ----------
    unique_id : int
        A unique identifier for this agent.
    model
        The model instance to which this agent belongs.
    settings : dict
        A dictionary containing behavior-related settings, which includes assets,
        decision-making parameters, and gurobi settings.

        - 'behavior_ids_in_network': IDs of other behavior agents in the agent's social network.
        - 'field_ids': IDs of fields managed by the agent.
        - 'well_ids': IDs of wells managed by the agent.
        - 'finance_id': ID of the finance agent associated with this behavior agent.
        - 'decision_making': Settings and parameters for the decision-making process.
        - 'consumat': Parameters related to the CONSUMAT model, including sensitivities and scales.
        - 'water_rights': Information about water rights, including depth [cm] and fields to which the constraint is applied.
        - 'gurobi': Settings for the Gurobi optimizer, such as logging and output controls.

        >>> # A sample settings dictionary
        >>> settings = {
        >>>     "field_ids": ["f1", "f2"],
        >>>     "well_ids": ["w1"],
        >>>     "finance_id": "finance",
        >>>     "behavior_ids_in_network": ["behavior2", "behavior3"],
        >>>     "decision_making": {
        >>>         "target": "profit",
        >>>         "horizon": 5, # [yr]
        >>>         "n_dwl": 5,
        >>>         "keep_gp_model": False,
        >>>         "keep_gp_output": False,
        >>>         "display_summary": False,
        >>>         "display_report": False
        >>>         },
        >>>     "water_rights": {
        >>>         "<name>": {
        >>>             "wr_depth": None,
        >>>             "applied_field_ids": ["f1_"], # Will automatically update to "f1_"
        >>>             "time_window": 1,
        >>>             "remaining_tw": None,
        >>>             "remaining_wr": None,
        >>>             "tail_method": "proportion",  # tail_method can be "proportion" or "all" or float
        >>>             "status": True
        >>>             }
        >>>         },
        >>>     "consumat": {
        >>>         "alpha": {  # [0-1] Sensitivity factor for the "satisfaction" calculation.
        >>>             "profit":     1,
        >>>             "yield_rate": 1
        >>>             },
        >>>         "scale": {  # Normalize "need" for "satisfaction" calculation.
        >>>             "profit": 1000,
        >>>             "yield_rate": 1
        >>>             },
        >>>         },
        >>>     "gurobi": {
        >>>         "LogToConsole": 1,  # 0: no console output; 1: with console output.
        >>>         "Presolve": -1      # Options are Auto (-1; default), Aggressive (2), Conservative (1), Automatic (-1), or None (0).
        >>>         }
        >>>     }

    pars : dict
        Parameters defining satisfaction and uncertainty thresholds for
        CONSUMAT and the agent's perception of risk and trust in forecasts.
        All four parameters are in the range 0 to 1.

        >>> # A sample pars dictionary
        >>> settings = {
        >>>     'perceived_risk': 0.5,
        >>>     'forecast_trust': 0.5,
        >>>     'sa_thre': 0.5,
        >>>     'un_thre': 0.5
        >>>     }

    fields : dict
        A dictionary of Field agents with their unique IDs as keys.
    wells : dict
        A dictionary of Well agents with their unique IDs as keys.
    finance : Finance
        A Finance agent instance associated with the behavior agent.
    aquifers : dict
        A dictionary of Aquifer agents with their unique IDs as keys.
    **kwargs
        Additional keyword arguments that can be dynamically set as agent attributes.

    Attributes
    ----------
    agt_type : str
        The type of the agent, set to 'Behavior'.
    num_fields : int
        The number of fields managed by the agent.
    num_wells : int
        The number of wells managed by the agent.
    total_field_area : float
        The total area of all fields managed by the agent.
    t : int
        The current time step, initialized to zero.
    state : str
        The current state of the agent based on CONSUMAT theory.

    Notes
    -----
    This method also initializes various attributes related to the agent's
    perception of risks, precipitation, profit, yield rate, and decision-making
    solutions. It calculates initial perceived risks and precipitation availability as well.
    """

    def __init__(
        self,
        unique_id,
        model,
        settings: dict,
        pars: dict,
        fields: dict,
        wells: dict,
        finance,
        aquifers: dict,
        optimization_class: object,
        **kwargs,
    ):
        """Initialize a Behavior agent in the Mesa model."""
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, model)
        self.agt_type = "Behavior"
        self.optimization_class = optimization_class

        # Load other kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.fix_state = kwargs.get("fix_state")  # internal experiment

        # Load settings
        self.load_settings(settings)

        # Parameters = {'perceived_risk': [0, 1], 'forecast_trust': [0, 1],
        #               'sa_thre': [0, 1], 'un_thre': [0, 1]}
        self.pars = pars

        # Assign agt's assets
        self.aquifers = aquifers
        self.fields = fields
        self.wells = wells
        self.finance = finance
        self.num_fields = len(fields)
        self.num_wells = len(wells)
        self.total_field_area = sum([field.field_area for _, field in self.fields.items()])

        # Initialize CONSUMAT
        self.state = None
        self.satisfaction = None
        self.expected_sa = None  # computed in an optimization
        self.uncertainty = None
        self.irr_vol = None  # m-ha
        self.profit = None
        self.avg_profit_per_field = None
        self.yield_rate = None

        self.scaled_profit = None
        self.scaled_yield_rate = None

        self.needs = {}
        self.selected_behavior_id_in_network = (
            None  # This will be populated after social comparison
        )

        # Some other attributes
        self.t = 0
        self.percieved_risks = None
        self.perceived_prec_aw = None
        self.profit = None
        self.yield_rate = None

        # Initial calculation
        self.process_percieved_risks(par_perceived_risk=self.pars["perceived_risk"])
        # Since perceived_risk and forecast_trust are not dynamically updated,
        # we pre-calculate perceived_prec_aw for all years here (not step-wise).
        self.update_perceived_prec_aw(par_forecast_confidence=self.pars["forecast_trust"])

        # Initialize dm_sols (mimicking opt_model's output)
        dm_sols = {}
        for fi, field in self.fields.items():
            dm_sols[fi] = {}
            dm_sols[fi]["i_crop"] = field.i_crop
            dm_sols[fi]["pre_i_crop"] = field.pre_i_crop
            dm_sols[fi]["i_te"] = field.te
            dm_sols[fi]["pre_i_te"] = field.pre_te

        # Run the optimization to solve irr depth with every other variables
        # fixed.
        self.pre_dm_sols = None
        self.dm_sols = self.make_dm(None, dm_sols=dm_sols, init=True)

        # Run the simulation to calculate satisfaction and uncertainty
        self.run_simulation()  # aquifers

    def load_settings(self, settings: dict):
        """
        Load the behavior settings from a dictionary.

        Parameters
        ----------
        settings : dict
            A dictionary containing settings related to the behavior agent. Expected keys include
            'behavior_ids_in_network', 'field_ids', 'well_ids', 'finance_id', 'decision_making',
            'consumat', 'water_rights', and 'gurobi'.
        """
        self.behavior_ids_in_network = settings["behavior_ids_in_network"]
        self.field_ids = settings["field_ids"]
        self.well_ids = settings["well_ids"]
        self.finance_id = settings["finance_id"]

        self.dm_dict = settings["decision_making"]
        self.consumat_dict = settings["consumat"]
        self.wr_dict = settings["water_rights"]
        self.gb_dict = settings["gurobi"]

    def process_percieved_risks(self, par_perceived_risk):
        """
        Compute perceived risks for each field and crop.

        Parameters
        ----------
        par_perceived_risk : float
            The quantile used in an inverse cumulative distribution function.

        Notes
        -----
        This method calculates the perceived risks based on the truncated normal distribution
        parameters for each crop. The calculated values are stored in the `percieved_risks` attribute.
        """
        # Compute percieved_risks (i.e., ECDF^(-1)(qu)) for each field and crop.
        percieved_risks = {}
        for fi, field in self.fields.items():
            truncated_normal_pars = field.truncated_normal_pars
            percieved_risks[fi] = {
                crop: 0
                if crop == "fallow"
                else round(
                    truncnorm.ppf(
                        q=par_perceived_risk,
                        a=truncated_normal_pars[crop][0],
                        b=truncated_normal_pars[crop][1],
                        loc=truncated_normal_pars[crop][2],
                        scale=truncated_normal_pars[crop][3],
                    ),
                    4,
                )
                for crop in self.model.crop_options
            }
        self.percieved_risks = percieved_risks

    def update_perceived_prec_aw(self, par_forecast_confidence, year=None):
        """
        Update the perceived precipitation available water based on forecast trust.

        Parameters
        ----------
        par_forecast_confidence : float
            The forecast trust parameter used in the calculation.
        year : int, optional
            The specific year for which the calculation is done. If None, the calculation
            is done for all years.

        Notes
        -----
        This method updates the `perceived_prec_aw` attribute based on the agent's trust in
        the weather forecast and the perceived risks. It adjusts the available precipitation
        for each crop in each field accordingly.
        """
        prec_aw_step = self.model.prec_aw_step  # read prec_aw_step from mesa model
        fotr = par_forecast_confidence
        percieved_risks = self.percieved_risks

        if year is None:
            # Compute perceived_prec_aw for all years at once since
            # percieved_risks and forecast_trust are constants during the simulation.
            perceived_prec_aw = {}  # [fi][yr][crop] = perceived_prec_aw
            for fi, field in self.fields.items():
                percieved_risks_f = percieved_risks[fi]
                prec_aw_step_f = pd.DataFrame(prec_aw_step[field.prec_aw_id]).T
                for crop, percieved_risk in percieved_risks_f.items():
                    prec_aw_step_f[crop] = prec_aw_step_f[
                        crop
                    ] * fotr + percieved_risk * (1 - fotr)
                perceived_prec_aw[fi] = prec_aw_step_f.round(4).T.to_dict()
            self.perceived_prec_aw = perceived_prec_aw
        else:
            # Step-wise update
            percieved_risks = self.percieved_risks
            perceived_prec_aw = self.perceived_prec_aw
            if perceived_prec_aw is None:
                perceived_prec_aw = {}
            for fi, field in self.fields.items():
                if fi not in perceived_prec_aw:
                    perceived_prec_aw[fi] = {}
                percieved_risks_f = percieved_risks[fi]
                prec_aw = prec_aw_step[field.prec_aw_id][year]
                perceived_prec_aw_f = {
                    crop: round(
                        percieved_risks_f[crop] * (1 - fotr) + prec_aw[crop] * fotr, 4
                    )
                    for crop in percieved_risks_f
                }
                perceived_prec_aw[fi][year] = perceived_prec_aw_f

    def step(self):
        """
        Perform a single step of the behavior agent's actions.

        Notes
        -----
        This method involves several key processes:

        1. Updating agents in the agent's social network.

        2. Making decisions based on the current CONSUMAT state (Imitation, Social Comparison, Repetition, Deliberation).

        3. Running simulations based on these decisions.

        4. Updating various attributes like profit, yield rate, satisfaction, and uncertainty.
        """
        self.t += 1
        # No need for step-wise update for perceived_prec_aw. We pre-calculated it.
        # current_year = self.model.current_year
        # self.update_perceived_prec_aw(self.pars['forecast_trust'], current_year)

        ### Optimization
        # Make decisions based on CONSUMAT theory
        state = self.state
        # print(self.unique_id, ": ", state)
        if state == "Imitation":
            self.make_dm_imitation()
        elif state == "Social comparison":
            self.make_dm_social_comparison()
        elif state == "Repetition":
            self.make_dm_repetition()
        elif state == "Deliberation":
            self.make_dm_deliberation()

        # Internal experiment
        elif state == "FixCrop":
            self.make_dm_deliberation()

        # Retrieve opt info
        dm_sols = self.dm_sols
        self.gp_status = dm_sols.get("gp_status")
        self.gp_MIPGap = dm_sols.get("gp_MIPGap")
        self.gp_report = dm_sols.get("gp_report")

        ### Simulation
        # Note prec_aw_dict have to be updated externally first.
        self.run_simulation()

        return self

    def run_simulation(self):
        """
        Run the simulation for the Behavior agent for one time step.

        This method performs several key operations:

        1. Simulates the fields based on decision-making solutions.

        2. Simulates the wells for energy consumption.

        3. Updates the financial status based on the field and well simulations.

        4. Calculates satisfaction and uncertainty based on CONSUMAT theory.

        5. Updates the CONSUMAT state of the agent.

        Notes
        -----
        The method uses the `dm_sols` attribute, which contains the decision-making solutions,
        to guide the simulation of fields and wells. It then updates various attributes of the
        agent, including profit, yield rate, satisfaction, uncertainty, and the CONSUMAT state.
        """
        current_year = self.model.current_year  # read current year from mesa model
        prec_aw_step = self.model.prec_aw_step  # read prec_aw_step from mesa model
        aquifers = self.aquifers  # aquifer objects
        fields = self.fields  # field objects
        wells = self.wells  # well objects
        dm_dict = self.dm_dict  # decision-making settings
        consumat_dict = self.consumat_dict  # CONSUMAT settings
        dm_sols = self.dm_sols  # optimization outputs

        ##### Simulate fields
        for fi, field in fields.items():
            irr_depth = dm_sols[fi]["irr_depth"][:, :, [0]]
            i_crop = dm_sols[fi]["i_crop"].copy()
            i_te = dm_sols[fi]["i_te"]
            field.step(
                irr_depth=irr_depth,
                i_crop=i_crop,
                i_te=i_te,
                prec_aw=prec_aw_step[field.prec_aw_id][current_year],
            )

        ##### Simulate wells (energy consumption)
        allo_r = dm_sols["allo_r"]  # Well allocation ratio from optimization
        allo_r_w = dm_sols["allo_r_w"]  # Well allocation ratio from optimization
        field_ids = dm_sols["field_ids"]
        well_ids = dm_sols["well_ids"]
        self.irr_vol = sum([field.irr_vol_per_field for _, field in fields.items()])

        for k, wid in enumerate(well_ids):
            well = wells[wid]
            # We only take first year optimization solution for simulation.
            withdrawal = self.irr_vol * allo_r_w[k, 0]
            pumping_rate = sum(
                [
                    fields[fid].pumping_rate * allo_r[f, k, 0]
                    for f, fid in enumerate(field_ids)
                ]
            )
            l_pr = sum(
                [fields[fid].l_pr * allo_r[f, k, 0] for f, fid in enumerate(field_ids)]
            )
            dwl = aquifers[well.aquifer_id].dwl
            # pumping_days remains fixed unless it is given.
            well.step(
                withdrawal=withdrawal, dwl=dwl, pumping_rate=pumping_rate, l_pr=l_pr
            )

        ##### Calulate profit and pumping cost
        self.finance.step(fields=fields, wells=wells)

        ##### CONSUMAT
        # Collect variables for CONSUMAT calculation
        self.profit = self.finance.profit
        self.avg_profit_per_field = self.profit / len(fields)
        self.yield_rate = sum(
            [field.yield_rate_per_field for _, field in fields.items()]
        ) / len(fields)

        # Calculate satisfaction and uncertainty
        alphas = consumat_dict["alpha"]
        scales = consumat_dict["scale"]

        def func(x, alpha=1):
            return 1 - np.exp(-alpha * x)

        # Use the average values per field
        self.scaled_profit = self.avg_profit_per_field / scales["profit"]
        self.scaled_yield_rate = self.yield_rate / scales["yield_rate"]
        needs = {}
        for var, alpha in alphas.items():
            if alpha is None:
                continue
            needs[var] = func(eval(f"self.scaled_{var}"), alpha=alpha)

        target = dm_dict["target"]
        satisfaction = needs[target]
        expected_sa = dm_sols["Sa"][target]

        # We define uncertainty to be the difference between expected_sa at the
        # previous time and satisfaction this year.
        expected_sa_t_1 = self.expected_sa
        if expected_sa_t_1 is None:  # Initial step
            uncertainty = abs(expected_sa - satisfaction)
        else:
            uncertainty = abs(expected_sa_t_1 - satisfaction)

        self.needs = needs
        self.satisfaction = satisfaction
        self.expected_sa = expected_sa
        self.uncertainty = uncertainty

        # Update CONSUMAT state
        sa_thre = self.pars["sa_thre"]
        un_thre = self.pars["un_thre"]

        if satisfaction >= sa_thre and uncertainty >= un_thre:
            self.state = "Imitation"
        elif satisfaction < sa_thre and uncertainty >= un_thre:
            self.state = "Social comparison"
        elif satisfaction >= sa_thre and uncertainty < un_thre:
            self.state = "Repetition"
        elif satisfaction < sa_thre and uncertainty < un_thre:
            self.state = "Deliberation"

        # Internal experiment => Fix state, where state is not dynamically updated
        if self.fix_state is not None:
            self.state = self.fix_state

    def make_dm(self, state, dm_sols, neighbor=None, init=False):
        """
        Create and solve an optimization model for decision-making based on the
        agent's current state.

        Parameters
        ----------
        state : str or None
            The current CONSUMAT state of the agent, which influences the optimization process.
        dm_sols : dict
            The previous decision-making solutions, used as inputs for the optimization model.
        neighbor : dict, optional
            A neighboring agent object, used in states, 'Imitation' and "Social comparison".
        init : bool, optional
            A flag indicating if it is the initial setup of the optimization model.

        Returns
        -------
        dict
            Updated decision-making solutions after solving the optimization model.

        Notes
        -----
        This method sets up and solves an optimization model based on various inputs, including
        field data, well data, water rights, and financial considerations. The type of
        optimization and constraints applied depend on the agent's current state, as defined
        by the CONSUMAT theory. The method returns updated decision-making solutions that
        guide the agent's actions in subsequent steps.
        """
        aquifers = self.aquifers  # aquifer objects
        fields = self.fields  # field objects
        wells = self.wells  # well objects
        dm_dict = self.dm_dict  # decision-making settings
        consumat_dict = self.consumat_dict

        dm = self.optimization_class(
            unique_id=self.unique_id,
            log_to_console=self.gb_dict.get("LogToConsole"),
            gpenv=self.model.gpenv,
        )

        dm.setup_ini_model(
            target=dm_dict["target"],
            horizon=dm_dict["horizon"],
            area_split=self.model.area_split,
            crop_options=self.model.crop_options,
            tech_options=self.model.tech_options,
            consumat_dict=consumat_dict,
            approx_horizon=False,
            gurobi_kwargs={},
        )

        perceived_prec_aw = self.perceived_prec_aw
        current_year = self.model.current_year
        for i, (fi, field) in enumerate(fields.items()):
            # Note: Since field type is given as an input, we do not constrain
            # i_rainfed under any condition. Behavior agent will only adopt
            # i_crop and i_te in certain states.
            dm_sols_fi = dm_sols[fi]

            if init:
                # Optimize irrigation depth with others variables given.
                # Apply the actual prec_aw (not the perceived one)
                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=self.model.prec_aw_step[field.prec_aw_id][current_year],
                    water_yield_curves=field.water_yield_curves,
                    tech_pumping_rate_coefs=field.tech_pumping_rate_coefs,
                    pre_i_crop=dm_sols_fi["i_crop"],
                    pre_i_te=dm_sols_fi["i_te"],
                    field_type=field.field_type,
                    i_crop=dm_sols_fi["i_crop"],
                    i_rainfed=None,
                    i_te=dm_sols_fi["i_te"],
                )

            elif state == "Deliberation":
                # optimize irrigation depth, crop choice, tech choice
                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=perceived_prec_aw[fi][current_year],
                    water_yield_curves=field.water_yield_curves,
                    tech_pumping_rate_coefs=field.tech_pumping_rate_coefs,
                    pre_i_crop=dm_sols_fi["i_crop"],
                    pre_i_te=dm_sols_fi["i_te"],
                    field_type=field.field_type,
                    i_crop=None,
                    i_rainfed=None,
                    i_te=None,
                )

            elif state == "Repetition":
                # only optimize irrigation depth
                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=perceived_prec_aw[fi][current_year],
                    water_yield_curves=field.water_yield_curves,
                    tech_pumping_rate_coefs=field.tech_pumping_rate_coefs,
                    pre_i_crop=dm_sols_fi["i_crop"],
                    pre_i_te=dm_sols_fi["i_te"],
                    field_type=field.field_type,
                    i_crop=dm_sols_fi["i_crop"],
                    i_rainfed=None,
                    i_te=dm_sols_fi["i_te"],
                )

            else:  # social comparason & imitation
                # We assume this behavioral agent has the same number of fields
                # as its neighbor.
                # A small patch may be needed in the future to generalize this.
                fi_neighbor = neighbor.field_ids[i]
                dm_sols_neighbor_fi = neighbor.pre_dm_sols[fi_neighbor]

                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=perceived_prec_aw[fi][current_year],
                    water_yield_curves=field.water_yield_curves,
                    tech_pumping_rate_coefs=field.tech_pumping_rate_coefs,
                    pre_i_crop=dm_sols_fi["i_crop"],
                    pre_i_te=dm_sols_fi["i_te"],
                    field_type=field.field_type,
                    i_crop=dm_sols_neighbor_fi["i_crop"],
                    i_rainfed=None,
                    i_te=dm_sols_neighbor_fi["i_te"],
                )

        for wi, well in wells.items():
            aquifer_id = well.aquifer_id
            proj_dwl = np.mean(aquifers[aquifer_id].dwl_list[-dm_dict["n_dwl"] :])

            dm.setup_constr_well(
                well_id=wi,
                dwl=proj_dwl,
                st=well.st,
                l_wt=well.l_wt,
                r=well.r,
                k=well.k,
                sy=well.sy,
                eff_pump=well.eff_pump,
                eff_well=well.eff_well,
                pumping_days=well.pumping_days,
                pumping_capacity=well.pumping_capacity,
                rho=well.rho,
                g=well.g,
            )

        if init:  # Inputted
            wr_dict = self.wr_dict
        else:  # Use agent's own water rights (for social comparison and imitation)
            wr_dict = dm_sols["water_rights"]

        for wr_id, v in self.wr_dict.items():
            if v["status"]:  # Check whether the wr is activated
                # Extract the water right setting from the previous opt run,
                # which we record as the remaining water right from the previous
                # year. If the wr is newly activated in a simulation, then we
                # use the input to setup the wr.
                wr_args = wr_dict.get(wr_id)
                if (
                    wr_args is None
                ):  # when we introduce the water rights for the first time (LEMA)
                    dm.setup_constr_wr(
                        water_right_id=wr_id,
                        wr_depth=v["wr_depth"],
                        applied_field_ids=v["applied_field_ids"],
                        time_window=v["time_window"],
                        remaining_tw=v["remaining_tw"],
                        remaining_wr=v["remaining_wr"],
                        tail_method=v["tail_method"],
                    )
                else:
                    dm.setup_constr_wr(
                        water_right_id=wr_id,
                        wr_depth=wr_args["wr_depth"],
                        applied_field_ids=wr_args["applied_field_ids"],
                        time_window=wr_args["time_window"],
                        remaining_tw=wr_args["remaining_tw"],
                        remaining_wr=wr_args["remaining_wr"],
                        tail_method=wr_args["tail_method"],
                    )

        dm.setup_constr_finance(self.finance.finance_dict)
        dm.setup_obj(alpha_dict=None)  # no dynamic update for alpha.
        dm.finish_setup(display_summary=dm_dict["display_summary"])
        dm.solve(
            keep_gp_model=dm_dict["keep_gp_model"],
            keep_gp_output=dm_dict["keep_gp_output"],
            display_report=dm_dict["display_report"],
            **self.gb_dict,
        )
        dm_sols = dm.sols
        if dm_sols is None:
            warnings.warn(
                "Gurobi returns empty solutions (likely due to infeasible problem.",
                stacklevel=2,
            )
        dm.depose_gp_env()  # Delete the entire environment to release memory.

        return dm_sols

    def make_dm_deliberation(self):
        """
        Make decision under the "Deliberation" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method updates the `dm_sols` attribute by calling the `make_dm` method
        with the current state set to "Deliberation".
        """
        self.dm_sols = self.make_dm(state="Deliberation", dm_sols=self.pre_dm_sols)

    def make_dm_repetition(self):
        """
        Make decision under the "Repetition" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method updates the `dm_sols` attribute by calling the `make_dm` method
        with the current state set to "Repetition".
        """
        self.dm_sols = self.make_dm(state="Repetition", dm_sols=self.pre_dm_sols)

    def make_dm_social_comparison(self):
        """
        Make decision under the "Social comparison" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method performs several key steps:

        1. Evaluates comparable decision-making solutions from agents in the network.

        2. Selects the agent with the best objective value.

        3. Compares the agent's original choice with the selected agent's choice.

        4. Updates the `dm_sols` attribute based on the comparison.
        """
        behavior_ids_in_network = self.behavior_ids_in_network
        # Evaluate comparable
        dm_sols_list = []
        for behavior_id in behavior_ids_in_network:
            # !!! Here we assume no. fields, n_c and split are the same across agents
            # Keep this for now.
            dm_sols = self.make_dm(
                state="Social comparison",
                dm_sols=self.pre_dm_sols,
                neighbor=self.model.behaviors[behavior_id],
            )
            dm_sols_list.append(dm_sols)
        objs = [s["obj"] for s in dm_sols_list]
        max_obj = max(objs)
        select_behavior_index = objs.index(max_obj)
        self.selected_behavior_id_in_network = behavior_ids_in_network[
            select_behavior_index
        ]

        # Agent's original choice
        dm_sols = self.make_dm(state="Repetition", dm_sols=self.pre_dm_sols)
        if dm_sols["obj"] >= max_obj:
            self.dm_sols = dm_sols
        else:
            self.dm_sols = dm_sols_list[select_behavior_index]

    def make_dm_imitation(self):
        """
        Make decision under the "Imitation" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method performs the following key steps:

        1. Selects an agent based on memory from previous social comparison from the network for imitation.

        2. Updates the `dm_sols` attribute by calling the `make_dm` method
           with the current state set to "Imitation" and using the selected agent's solutions.
        """
        selected_behavior_id_in_network = self.selected_behavior_id_in_network
        if selected_behavior_id_in_network is None:
            try:  # if rngen is given in the model
                selected_behavior_id_in_network = self.rngen.choice(
                    self.behavior_ids_in_network
                )
            except:
                selected_behavior_id_in_network = np.random.choice(
                    self.behavior_ids_in_network
                )

        neighbor = self.model.behaviors[selected_behavior_id_in_network]

        self.dm_sols = self.make_dm(
            state="Imitation", dm_sols=self.pre_dm_sols, neighbor=neighbor
        )


class Behavior4SingleFieldAndWell(mesa.Agent):
    """ Simulate a farmer's behavior. """

    def __init__(
        self,
        unique_id,
        model,
        settings: dict,
        pars: dict,
        fields: dict,
        wells: dict,
        finance,
        aquifers: dict,
        optimization_class: object,
        **kwargs,
    ):
        """Initialize a Behavior agent in the Mesa model.
        
        Parameters
        ----------
        unique_id : int
            A unique identifier for the agent.
        model : object
            The Mesa model object.
        settings : dict
            A dictionary containing settings related to the behavior agent. Expected 
            keys include 'behavior_ids_in_network', 'field_ids', 'well_ids', 
            'finance_id', 'decision_making', 'consumat', and 'water_rights'.
        pars : dict
            Parameters defining satisfaction and uncertainty thresholds for
            CONSUMAT and the agent's perception of risk and confidence in forecasts.
            All four parameters are in the range 0 to 1.
        fields : dict
            A dictionary of Field agents with their unique IDs as keys.
        wells : dict
            A dictionary of Well agents with their unique IDs as keys.
        finance : Finance
            A Finance agent instance associated with the behavior agent.
        aquifers : dict
            A dictionary of Aquifer agents with their unique IDs as keys.
        optimization_class : object
            The optimization class used for decision-making.
        **kwargs
            Additional keyword arguments that can be dynamically set as agent
            attributes.

        Notes
        -----
        This method initializes various attributes related to the agent's perception
        of risks, precipitation, profit, yield rate, and decision-making solutions. It
        calculates initial perceived risks and precipitation availability as well.
        """
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, model)
        self.agt_type = "Behavior"

        # Load other kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.fix_state = kwargs.get("fix_state")  # internal experiment

        # Load optimization class
        self.optimization_class = optimization_class

        # Load settings
        self.load_settings(settings)

        # Parameters = {'perceived_risk': [0, 1], 'forecast_trust': [0, 1],
        #               'sa_thre': [0, 1], 'un_thre': [0, 1]}
        self.pars = pars

        # Assign agt's assets
        self.aquifers = aquifers
        self.fields = fields
        self.wells = wells
        self.finance = finance
        self.num_fields = len(fields)
        self.num_wells = len(wells)
        self.total_field_area = sum([field.field_area for _, field in self.fields.items()])

        # Initialize CONSUMAT
        self.state = None
        self.satisfaction = None
        self.expected_sa = None  # computed in an optimization
        self.uncertainty = None
        self.irr_vol = None  # m-ha
        self.profit = None
        self.avg_profit_per_field = None
        self.yield_rate = None

        self.scaled_profit = None
        self.scaled_yield_rate = None

        self.needs = {}
        self.selected_behavior_id_in_network = (
            None  # This will be populated after social comparison
        )

        # Some other attributes
        self.t = 0
        self.percieved_risks = None
        self.perceived_prec_aw = None
        self.profit = None
        self.yield_rate = None

        # Initial calculation
        self.process_percieved_risks(par_perceived_risk=self.pars["perceived_risk"])
        # Since perceived_risk and forecast_trust are not dynamically updated,
        # we pre-calculate perceived_prec_aw for all years here (not step-wise).
        self.update_perceived_prec_aw(par_forecast_confidence=self.pars["forecast_trust"])

        # Initialize dm_sols (mimicking opt_model's output)
        dm_sols = {}
        for fi, field in self.fields.items():
            dm_sols[fi] = {}
            dm_sols[fi]["i_crop"] = field.i_crop

        # Run the optimization to solve irr depth with every other variables
        # fixed.
        self.pre_dm_sols = None
        self.dm_sols = self.make_dm(None, dm_sols=dm_sols, init=True)

        # Run the simulation to calculate satisfaction and uncertainty
        self.run_simulation()  # aquifers

    def load_settings(self, settings: dict):
        """
        Load the behavior settings from a dictionary.

        Parameters
        ----------
        settings : dict
            A dictionary containing settings related to the behavior agent. Expected
            keys include 'behavior_ids_in_network', 'field_ids', 'well_ids', 
            'finance_id', 'decision_making', 'consumat', and 'water_rights'.
        """
        self.behavior_ids_in_network = settings["behavior_ids_in_network"]
        self.field_ids = settings["field_ids"]
        self.well_ids = settings["well_ids"]
        self.finance_id = settings["finance_id"]

        self.dm_dict = settings["decision_making"]
        self.consumat_dict = settings["consumat"]
        self.wr_dict = settings["water_rights"]

    def process_percieved_risks(self, par_perceived_risk):
        """
        Compute perceived risks for each field and crop.

        Parameters
        ----------
        par_perceived_risk : float
            The quantile used in an inverse cumulative distribution function.

        Notes
        -----
        This method calculates the perceived risks based on the truncated normal 
        distribution parameters for each crop. The calculated values are stored in the
        `percieved_risks` attribute.
        """
        # Compute percieved_risks (i.e., ECDF^(-1)(qu)) for each field and crop.
        percieved_risks = {}
        for fi, field in self.fields.items():
            truncated_normal_pars = field.truncated_normal_pars
            percieved_risks[fi] = {
                crop: 0
                if crop == "fallow"
                else round(
                    truncnorm.ppf(
                        q=par_perceived_risk,
                        a=truncated_normal_pars[crop][0],
                        b=truncated_normal_pars[crop][1],
                        loc=truncated_normal_pars[crop][2],
                        scale=truncated_normal_pars[crop][3],
                    ),
                    4,
                )
                for crop in self.model.crop_options
            }
        self.percieved_risks = percieved_risks

    def update_perceived_prec_aw(self, par_forecast_confidence, year=None):
        """
        Update the perceived precipitation available water based on forecast trust.

        Parameters
        ----------
        par_forecast_confidence : float
            The forecast trust parameter used in the calculation.
        year : int, optional
            The specific year for which the calculation is done. If None, the calculation
            is done for all years.

        Notes
        -----
        This method updates the `perceived_prec_aw` attribute based on the agent's trust in
        the weather forecast and the perceived risks. It adjusts the available precipitation
        for each crop in each field accordingly.
        """
        prec_aw_step = self.model.prec_aw_step  # read prec_aw_step from mesa model
        fotr = par_forecast_confidence
        percieved_risks = self.percieved_risks

        if year is None:
            # Compute perceived_prec_aw for all years at once since
            # percieved_risks and forecast_trust are constants during the simulation.
            perceived_prec_aw = {}  # [fi][yr][crop] = perceived_prec_aw
            for fi, field in self.fields.items():
                percieved_risks_f = percieved_risks[fi]
                prec_aw_step_f = pd.DataFrame(prec_aw_step[field.prec_aw_id]).T
                for crop, percieved_risk in percieved_risks_f.items():
                    prec_aw_step_f[crop] = prec_aw_step_f[
                        crop
                    ] * fotr + percieved_risk * (1 - fotr)
                perceived_prec_aw[fi] = prec_aw_step_f.round(4).T.to_dict()
            self.perceived_prec_aw = perceived_prec_aw
        else:
            # Step-wise update
            percieved_risks = self.percieved_risks
            perceived_prec_aw = self.perceived_prec_aw
            if perceived_prec_aw is None:
                perceived_prec_aw = {}
            for fi, field in self.fields.items():
                if fi not in perceived_prec_aw:
                    perceived_prec_aw[fi] = {}
                percieved_risks_f = percieved_risks[fi]
                prec_aw = prec_aw_step[field.prec_aw_id][year]
                perceived_prec_aw_f = {
                    crop: round(
                        percieved_risks_f[crop] * (1 - fotr) + prec_aw[crop] * fotr, 4
                    )
                    for crop in percieved_risks_f
                }
                perceived_prec_aw[fi][year] = perceived_prec_aw_f

    def step(self):
        """
        Perform a single step of the behavior agent's actions.

        Notes
        -----
        This method involves several key processes:

        1. Updating agents in the agent's social network.

        2. Making decisions based on the current CONSUMAT state (Imitation,
           Social Comparison, Repetition, Deliberation).

        3. Running simulations based on these decisions.

        4. Updating various attributes like profit, yield rate, satisfaction, and
           uncertainty.
        """
        self.t += 1
        # No need for step-wise update for perceived_prec_aw. We pre-calculated it.
        # current_year = self.model.current_year
        # self.update_perceived_prec_aw(self.pars['forecast_trust'], current_year)

        ### Optimization
        # Make decisions based on CONSUMAT theory
        state = self.state
        # print(self.unique_id, ": ", state)
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
        self.gp_status = dm_sols.get("gp_status")
        self.gp_MIPGap = dm_sols.get("gp_MIPGap")
        self.gp_report = dm_sols.get("gp_report")

        ### Simulation
        # Note prec_aw_dict have to be updated externally first.
        self.run_simulation()

        return self

    def run_simulation(self):
        """
        Run the simulation for the Behavior agent for one time step.

        This method performs several key operations:

        1. Simulates the fields based on decision-making solutions.

        2. Simulates the wells for energy consumption.

        3. Updates the financial status based on the field and well simulations.

        4. Calculates satisfaction and uncertainty based on CONSUMAT theory.

        5. Updates the CONSUMAT state of the agent.

        Notes
        -----
        The method uses the `dm_sols` attribute, which contains the decision-making solutions,
        to guide the simulation of fields and wells. It then updates various attributes of the
        agent, including profit, yield rate, satisfaction, uncertainty, and the CONSUMAT state.
        """
        current_year = self.model.current_year  # read current year from mesa model
        prec_aw_step = self.model.prec_aw_step  # read prec_aw_step from mesa model
        aquifers = self.aquifers  # aquifer objects
        fields = self.fields  # field objects
        wells = self.wells  # well objects
        dm_dict = self.dm_dict  # decision-making settings
        consumat_dict = self.consumat_dict  # CONSUMAT settings
        dm_sols = self.dm_sols  # optimization outputs

        ##### Simulate fields
        for fi, field in fields.items():
            irr_depth = dm_sols["irr_depth"][:, [0]]
            i_crop = dm_sols[fi]["i_crop"].copy()
            field.step(
                irr_depth=irr_depth,
                i_crop=i_crop,
                prec_aw=prec_aw_step[field.prec_aw_id][current_year],
            )

        ##### Simulate wells (energy consumption)
        well_ids = dm_sols["well_ids"]
        self.irr_vol = sum([field.irr_vol_per_field for _, field in fields.items()])

        for _, wid in enumerate(well_ids):
            well = wells[wid]
            # We only take first year optimization solution for simulation.
            withdrawal = self.irr_vol
            dwl = aquifers[well.aquifer_id].dwl
            # pumping_days remains fixed unless it is given.
            well.step(withdrawal=withdrawal, dwl=dwl)

        ##### Calulate profit and pumping cost
        self.finance.step(fields=fields, wells=wells)

        ##### CONSUMAT
        # Collect variables for CONSUMAT calculation
        self.profit = self.finance.profit
        self.avg_profit_per_field = self.profit / len(fields)
        self.yield_rate = sum(
            [field.yield_rate_per_field for _, field in fields.items()]
        ) / len(fields)

        # Calculate satisfaction and uncertainty
        alphas = consumat_dict["alpha"]
        scales = consumat_dict["scale"]

        def func(x, alpha=1):
            return 1 - np.exp(-alpha * x)

        # Use the average values per field
        self.scaled_profit = self.avg_profit_per_field / scales["profit"]
        self.scaled_yield_rate = self.yield_rate  # /scales["yield_rate"]
        needs = {}
        for var, alpha in alphas.items():
            if alpha is None:
                continue
            needs[var] = func(eval(f"self.scaled_{var}"), alpha=alpha)

        target = dm_dict["target"]
        satisfaction = needs[target]
        expected_sa = dm_sols["Sa"][target]

        # We define uncertainty to be the difference between expected_sa at the
        # previous time and satisfaction this year.
        expected_sa_t_1 = self.expected_sa
        if expected_sa_t_1 is None:  # Initial step
            uncertainty = abs(expected_sa - satisfaction)
        else:
            uncertainty = abs(expected_sa_t_1 - satisfaction)

        self.needs = needs
        self.satisfaction = satisfaction
        self.expected_sa = expected_sa
        self.uncertainty = uncertainty

        # Update CONSUMAT state
        sa_thre = self.pars["sa_thre"]
        un_thre = self.pars["un_thre"]

        if satisfaction >= sa_thre and uncertainty >= un_thre:
            self.state = "Imitation"
        elif satisfaction < sa_thre and uncertainty >= un_thre:
            self.state = "Social comparison"
        elif satisfaction >= sa_thre and uncertainty < un_thre:
            self.state = "Repetition"
        elif satisfaction < sa_thre and uncertainty < un_thre:
            self.state = "Deliberation"

        # Internal experiment => Fix state, where state is not dynamically updated
        if self.fix_state is not None:
            self.state = self.fix_state

    def make_dm(self, state, dm_sols, neighbor=None, init=False):
        """
        Create and solve an optimization model for decision-making based on the
        agent's current state.

        Parameters
        ----------
        state : str or None
            The current CONSUMAT state of the agent, which influences the optimization
            process.
        dm_sols : dict
            The previous decision-making solutions, used as inputs for the optimization
            model.
        neighbor : dict, optional
            A neighboring agent object, used in states, 'Imitation' and
            "Social comparison".
        init : bool, optional
            A flag indicating if it is the initial setup of the optimization model.

        Returns
        -------
        dict
            Updated decision-making solutions after solving the optimization model.

        Notes
        -----
        This method sets up and solves an optimization model based on various inputs, 
        including field data, well data, water rights, and financial considerations. The 
        type of optimization and constraints applied depend on the agent's current 
        state, as defined by the CONSUMAT theory. The method returns updated 
        decision-making solutions that guide the agent's actions in subsequent steps.
        """
        aquifers = self.aquifers  # aquifer objects
        fields = self.fields  # field objects
        wells = self.wells  # well objects
        dm_dict = self.dm_dict  # decision-making settings
        consumat_dict = self.consumat_dict

        dm = self.optimization_class()

        dm.setup_ini_model(
            unique_id=self.unique_id,
            gpenv=self.model.gpenv,  # share one environment for the entire simulation.
            horizon=dm_dict["horizon"],
            crop_options=self.model.crop_options,
        )

        perceived_prec_aw = self.perceived_prec_aw
        current_year = self.model.current_year
        for i, (fi, field) in enumerate(fields.items()):
            # Note: Since field type is given as an input, we do not constrain
            # i_rainfed under any condition. Behavior agent will only adopt
            # i_crop and i_te in certain states.
            dm_sols_fi = dm_sols[fi]

            if init:
                # Optimize irrigation depth with others variables given.
                # Apply the actual prec_aw (not the perceived one)
                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=self.model.prec_aw_step[field.prec_aw_id][current_year],
                    water_yield_curves=field.water_yield_curves,
                    field_type=field.field_type,
                    i_crop=dm_sols_fi["i_crop"],
                    i_rainfed=None,
                )

            elif state == "Deliberation":
                # optimize irrigation depth, crop choice, tech choice
                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=perceived_prec_aw[fi][current_year],
                    water_yield_curves=field.water_yield_curves,
                    field_type=field.field_type,
                    i_crop=None,
                    i_rainfed=None,
                )

            elif state == "Repetition":
                # only optimize irrigation depth
                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=perceived_prec_aw[fi][current_year],
                    water_yield_curves=field.water_yield_curves,
                    field_type=field.field_type,
                    i_crop=dm_sols_fi["i_crop"],
                    i_rainfed=None,
                )

            else:  # social comparason & imitation
                # We assume this behavioral agent has the same number of fields
                # as its neighbor.
                # A small patch may be needed in the future to generalize this.
                fi_neighbor = neighbor.field_ids[i]
                dm_sols_neighbor_fi = neighbor.pre_dm_sols[fi_neighbor]

                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=perceived_prec_aw[fi][current_year],
                    water_yield_curves=field.water_yield_curves,
                    field_type=field.field_type,
                    i_crop=dm_sols_neighbor_fi["i_crop"],
                    i_rainfed=None,
                )

        for wi, well in wells.items():
            aquifer_id = well.aquifer_id
            proj_dwl = np.mean(aquifers[aquifer_id].dwl_list[-dm_dict["n_dwl"] :])

            dm.setup_constr_well(
                well_id=wi,
                dwl=proj_dwl,
                B=well.B,
                l_wt=well.l_wt,
                eff_pump=well.eff_pump,
                pumping_capacity=well.pumping_capacity,
                rho=well.rho,
                g=well.g,
            )

        if init:  # Inputted
            wr_dict = self.wr_dict
        else:  # Use agent's own water rights (for social comparison and imitation)
            wr_dict = dm_sols["water_rights"]

        for wr_id, v in self.wr_dict.items():
            if v["status"]:  # Check whether the wr is activated
                # Extract the water right setting from the previous opt run,
                # which we record as the remaining water right from the previous
                # year. If the wr is newly activated in a simulation, then we
                # use the input to setup the wr.
                wr_args = wr_dict.get(wr_id)
                if (
                    wr_args is None
                ):  # when we introduce the water rights for the first time (LEMA)
                    dm.setup_constr_wr(
                        water_right_id=wr_id,
                        wr_depth=v["wr_depth"],
                        time_window=v["time_window"],
                        remaining_tw=v["remaining_tw"],
                        remaining_wr=v["remaining_wr"],
                        tail_method=v["tail_method"],
                    )
                else:
                    dm.setup_constr_wr(
                        water_right_id=wr_id,
                        wr_depth=wr_args["wr_depth"],
                        time_window=wr_args["time_window"],
                        remaining_tw=wr_args["remaining_tw"],
                        remaining_wr=wr_args["remaining_wr"],
                        tail_method=wr_args["tail_method"],
                    )

        dm.setup_constr_finance(self.finance.finance_dict)
        dm.setup_obj(
            target=dm_dict["target"], consumat_dict=consumat_dict
        )  # no dynamic update for alpha.
        dm.finish_setup(display_summary=dm_dict["display_summary"])
        dm.solve(
            keep_gp_model=dm_dict["keep_gp_model"],
            keep_gp_output=dm_dict["keep_gp_output"],
            display_report=dm_dict["display_report"],
        )
        dm_sols = dm.sols
        if dm_sols is None:
            warnings.warn(
                "Gurobi returns empty solutions (likely due to infeasible problem.",
                stacklevel=2,
            )

        return dm_sols

    def make_dm_deliberation(self):
        """
        Make decision under the "Deliberation" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method updates the `dm_sols` attribute by calling the `make_dm` method
        with the current state set to "Deliberation".
        """
        self.dm_sols = self.make_dm(state="Deliberation", dm_sols=self.pre_dm_sols)

    def make_dm_repetition(self):
        """
        Make decision under the "Repetition" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method updates the `dm_sols` attribute by calling the `make_dm` method
        with the current state set to "Repetition".
        """
        self.dm_sols = self.make_dm(state="Repetition", dm_sols=self.pre_dm_sols)

    def make_dm_social_comparison(self):
        """
        Make decision under the "Social comparison" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method performs several key steps:

        1. Evaluates comparable decision-making solutions from agents in the network.

        2. Selects the agent with the best objective value.

        3. Compares the agent's original choice with the selected agent's choice.

        4. Updates the `dm_sols` attribute based on the comparison.
        """
        behavior_ids_in_network = self.behavior_ids_in_network
        # Evaluate comparable
        dm_sols_list = []
        for behavior_id in behavior_ids_in_network:
            dm_sols = self.make_dm(
                state="Social comparison",
                dm_sols=self.pre_dm_sols,
                neighbor=self.model.behaviors[behavior_id],
            )
            dm_sols_list.append(dm_sols)
        objs = [s["obj"] for s in dm_sols_list]
        max_obj = max(objs)
        select_behavior_index = objs.index(max_obj)
        self.selected_behavior_id_in_network = behavior_ids_in_network[
            select_behavior_index
        ]

        # Agent's original choice
        dm_sols = self.make_dm(state="Repetition", dm_sols=self.pre_dm_sols)
        if dm_sols["obj"] >= max_obj:
            self.dm_sols = dm_sols
        else:
            self.dm_sols = dm_sols_list[select_behavior_index]

    def make_dm_imitation(self):
        """
        Make decision under the "Imitation" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method performs the following key steps:

        1. Selects an agent based on memory from previous social comparison from the
           network for imitation.

        2. Updates the `dm_sols` attribute by calling the `make_dm` method
           with the current state set to "Imitation" and using the selected agent's
           solutions.
        """
        selected_behavior_id_in_network = self.selected_behavior_id_in_network
        if selected_behavior_id_in_network is None:
            selected_behavior_id_in_network = self.model.rngen.choice(
                self.behavior_ids_in_network
            )

        neighbor = self.model.behaviors[selected_behavior_id_in_network]

        self.dm_sols = self.make_dm(
            state="Imitation", dm_sols=self.pre_dm_sols, neighbor=neighbor
        )


class Behavior_1f1w_ci(mesa.Agent):
    """
    This module is a farmer's behavior simulator.

    Parameters
    ----------
    unique_id : int
        A unique identifier for this agent.
    model
        The model instance to which this agent belongs.
    settings : dict
        A dictionary containing behavior-related settings, which includes assets,
        decision-making parameters, and gurobi settings.

        - 'behavior_ids_in_network': IDs of other behavior agents in the agent's social network.
        - 'field_ids': IDs of fields managed by the agent.
        - 'well_ids': IDs of wells managed by the agent.
        - 'finance_id': ID of the finance agent associated with this behavior agent.
        - 'decision_making': Settings and parameters for the decision-making process.
        - 'consumat': Parameters related to the CONSUMAT model, including sensitivities and scales.
        - 'water_rights': Information about water rights, including depth [cm] and fields to which the constraint is applied.
        - 'gurobi': Settings for the Gurobi optimizer, such as logging and output controls.

        >>> # A sample settings dictionary
        >>> settings = {
        >>>     "field_ids": ["f1", "f2"],
        >>>     "well_ids": ["w1"],
        >>>     "finance_id": "finance",
        >>>     "behavior_ids_in_network": ["behavior2", "behavior3"],
        >>>     "decision_making": {
        >>>         "target": "profit",
        >>>         "horizon": 5, # [yr]
        >>>         "n_dwl": 5,
        >>>         "keep_gp_model": False,
        >>>         "keep_gp_output": False,
        >>>         "display_summary": False,
        >>>         "display_report": False
        >>>         },
        >>>     "water_rights": {
        >>>         "<name>": {
        >>>             "wr_depth": None,
        >>>             "applied_field_ids": ["f1_"], # Will automatically update to "f1_"
        >>>             "time_window": 1,
        >>>             "remaining_tw": None,
        >>>             "remaining_wr": None,
        >>>             "tail_method": "proportion",  # tail_method can be "proportion" or "all" or float
        >>>             "status": True
        >>>             }
        >>>         },
        >>>     "consumat": {
        >>>         "alpha": {  # [0-1] Sensitivity factor for the "satisfaction" calculation.
        >>>             "profit":     1,
        >>>             "yield_rate": 1
        >>>             },
        >>>         "scale": {  # Normalize "need" for "satisfaction" calculation.
        >>>             "profit": 1000,
        >>>             "yield_rate": 1
        >>>             },
        >>>         },
        >>>     "gurobi": {
        >>>         "LogToConsole": 1,  # 0: no console output; 1: with console output.
        >>>         "Presolve": -1      # Options are Auto (-1; default), Aggressive (2), Conservative (1), Automatic (-1), or None (0).
        >>>         }
        >>>     }

    pars : dict
        Parameters defining satisfaction and uncertainty thresholds for
        CONSUMAT and the agent's perception of risk and trust in forecasts.
        All four parameters are in the range 0 to 1.

        >>> # A sample pars dictionary
        >>> settings = {
        >>>     'perceived_risk': 0.5,
        >>>     'forecast_trust': 0.5,
        >>>     'sa_thre': 0.5,
        >>>     'un_thre': 0.5
        >>>     }

    fields : dict
        A dictionary of Field agents with their unique IDs as keys.
    wells : dict
        A dictionary of Well agents with their unique IDs as keys.
    finance : Finance
        A Finance agent instance associated with the behavior agent.
    aquifers : dict
        A dictionary of Aquifer agents with their unique IDs as keys.
    **kwargs
        Additional keyword arguments that can be dynamically set as agent attributes.

    Attributes
    ----------
    agt_type : str
        The type of the agent, set to 'Behavior'.
    num_fields : int
        The number of fields managed by the agent.
    num_wells : int
        The number of wells managed by the agent.
    total_field_area : float
        The total area of all fields managed by the agent.
    t : int
        The current time step, initialized to zero.
    state : str
        The current state of the agent based on CONSUMAT theory.

    Notes
    -----
    This method also initializes various attributes related to the agent's
    perception of risks, precipitation, profit, yield rate, and decision-making
    solutions. It calculates initial perceived risks and precipitation availability as well.
    """

    def __init__(
        self,
        unique_id,
        model,
        settings: dict,
        pars: dict,
        fields: dict,
        wells: dict,
        finance,
        aquifers: dict,
        Optimization: object,
        **kwargs,
    ):
        """Initialize a Behavior agent in the Mesa model."""
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, model)
        self.agt_type = "Behavior"

        # Load other kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.fix_state = kwargs.get("fix_state")  # internal experiment

        self.optimization_class = Optimization

        # Load settings
        self.load_settings(settings)

        # Parameters = {'perceived_risk': [0, 1], 'forecast_trust': [0, 1],
        #               'sa_thre': [0, 1], 'un_thre': [0, 1]}
        self.pars = pars

        # Assign agt's assets
        self.aquifers = aquifers
        self.fields = fields
        self.wells = wells
        self.finance = finance
        self.num_fields = len(fields)
        self.num_wells = len(wells)
        self.total_field_area = sum([field.field_area for _, field in self.fields.items()])

        # Initialize CONSUMAT
        self.state = None
        self.satisfaction = None
        self.expected_sa = None  # computed in an optimization
        self.uncertainty = None
        self.irr_vol = None  # m-ha
        self.profit = None
        self.avg_profit_per_field = None
        self.yield_rate = None

        self.scaled_profit = None
        self.scaled_yield_rate = None

        self.needs = {}
        self.selected_behavior_id_in_network = (
            None  # This will be populated after social comparison
        )

        # Some other attributes
        self.t = 0
        self.percieved_risks = None
        self.perceived_prec_aw = None
        self.profit = None
        self.yield_rate = None

        # Initial calculation
        self.process_percieved_risks(par_perceived_risk=self.pars["perceived_risk"])
        # Since perceived_risk and forecast_trust are not dynamically updated,
        # we pre-calculate perceived_prec_aw for all years here (not step-wise).
        self.update_perceived_prec_aw(par_forecast_confidence=self.pars["forecast_trust"])

        # Initialize dm_sols (mimicking opt_model's output)
        dm_sols = {}
        for fi, field in self.fields.items():
            dm_sols[fi] = {}
            dm_sols[fi]["i_crop"] = field.i_crop

        # Run the optimization to solve irr depth with every other variables
        # fixed.
        self.pre_dm_sols = None
        self.dm_sols = self.make_dm(None, dm_sols=dm_sols, init=True)

        # Run the simulation to calculate satisfaction and uncertainty
        self.run_simulation()  # aquifers

    def load_settings(self, settings: dict):
        """
        Load the behavior settings from a dictionary.

        Parameters
        ----------
        settings : dict
            A dictionary containing settings related to the behavior agent. Expected keys include
            'behavior_ids_in_network', 'field_ids', 'well_ids', 'finance_id', 'decision_making',
            'consumat', 'water_rights', and 'gurobi'.
        """
        self.behavior_ids_in_network = settings["behavior_ids_in_network"]
        self.field_ids = settings["field_ids"]
        self.well_ids = settings["well_ids"]
        self.finance_id = settings["finance_id"]

        self.dm_dict = settings["decision_making"]
        self.consumat_dict = settings["consumat"]
        self.wr_dict = settings["water_rights"]

    def process_percieved_risks(self, par_perceived_risk):
        """
        Compute perceived risks for each field and crop.

        Parameters
        ----------
        par_perceived_risk : float
            The quantile used in an inverse cumulative distribution function.

        Notes
        -----
        This method calculates the perceived risks based on the truncated normal distribution
        parameters for each crop. The calculated values are stored in the `percieved_risks` attribute.
        """
        # Compute percieved_risks (i.e., ECDF^(-1)(qu)) for each field and crop.
        percieved_risks = {}
        for fi, field in self.fields.items():
            truncated_normal_pars = field.truncated_normal_pars
            percieved_risks[fi] = {
                crop: 0
                if crop == "fallow"
                else round(
                    truncnorm.ppf(
                        q=par_perceived_risk,
                        a=truncated_normal_pars[crop][0],
                        b=truncated_normal_pars[crop][1],
                        loc=truncated_normal_pars[crop][2],
                        scale=truncated_normal_pars[crop][3],
                    ),
                    4,
                )
                for crop in self.model.crop_options
            }
        self.percieved_risks = percieved_risks

    def update_perceived_prec_aw(self, par_forecast_confidence, year=None):
        """
        Update the perceived precipitation available water based on forecast trust.

        Parameters
        ----------
        par_forecast_confidence : float
            The forecast trust parameter used in the calculation.
        year : int, optional
            The specific year for which the calculation is done. If None, the calculation
            is done for all years.

        Notes
        -----
        This method updates the `perceived_prec_aw` attribute based on the agent's trust in
        the weather forecast and the perceived risks. It adjusts the available precipitation
        for each crop in each field accordingly.
        """
        prec_aw_step = self.model.prec_aw_step  # read prec_aw_step from mesa model
        fotr = par_forecast_confidence
        percieved_risks = self.percieved_risks

        if year is None:
            # Compute perceived_prec_aw for all years at once since
            # percieved_risks and forecast_trust are constants during the simulation.
            perceived_prec_aw = {}  # [fi][yr][crop] = perceived_prec_aw
            for fi, field in self.fields.items():
                percieved_risks_f = percieved_risks[fi]
                prec_aw_step_f = pd.DataFrame(prec_aw_step[field.prec_aw_id]).T
                for crop, percieved_risk in percieved_risks_f.items():
                    prec_aw_step_f[crop] = prec_aw_step_f[
                        crop
                    ] * fotr + percieved_risk * (1 - fotr)
                perceived_prec_aw[fi] = prec_aw_step_f.round(4).T.to_dict()
            self.perceived_prec_aw = perceived_prec_aw
        else:
            # Step-wise update
            percieved_risks = self.percieved_risks
            perceived_prec_aw = self.perceived_prec_aw
            if perceived_prec_aw is None:
                perceived_prec_aw = {}
            for fi, field in self.fields.items():
                if fi not in perceived_prec_aw:
                    perceived_prec_aw[fi] = {}
                percieved_risks_f = percieved_risks[fi]
                prec_aw = prec_aw_step[field.prec_aw_id][year]
                perceived_prec_aw_f = {
                    crop: round(
                        percieved_risks_f[crop] * (1 - fotr) + prec_aw[crop] * fotr, 4
                    )
                    for crop in percieved_risks_f
                }
                perceived_prec_aw[fi][year] = perceived_prec_aw_f

    def step(self):
        """
        Perform a single step of the behavior agent's actions.

        Notes
        -----
        This method involves several key processes:

        1. Updating agents in the agent's social network.

        2. Making decisions based on the current CONSUMAT state (Imitation, Social Comparison, Repetition, Deliberation).

        3. Running simulations based on these decisions.

        4. Updating various attributes like profit, yield rate, satisfaction, and uncertainty.
        """
        self.t += 1
        # No need for step-wise update for perceived_prec_aw. We pre-calculated it.
        # current_year = self.model.current_year
        # self.update_perceived_prec_aw(self.pars['forecast_trust'], current_year)

        ### Optimization
        # Make decisions based on CONSUMAT theory
        state = self.state
        # print(self.unique_id, ": ", state)
        if state == "Imitation":
            self.make_dm_imitation()
        elif state == "Social comparison":
            self.make_dm_social_comparison()
        elif state == "Repetition":
            self.make_dm_repetition()
        elif state == "Deliberation":
            self.make_dm_deliberation()

        # Internal experiment
        elif state == "FixCrop":
            self.make_dm_deliberation()

        # Retrieve opt info
        dm_sols = self.dm_sols
        self.gp_status = dm_sols.get("gp_status")
        self.gp_MIPGap = dm_sols.get("gp_MIPGap")
        self.gp_report = dm_sols.get("gp_report")

        ### Simulation
        # Note prec_aw_dict have to be updated externally first.
        self.run_simulation()

        return self

    def run_simulation(self):
        """
        Run the simulation for the Behavior agent for one time step.

        This method performs several key operations:

        1. Simulates the fields based on decision-making solutions.

        2. Simulates the wells for energy consumption.

        3. Updates the financial status based on the field and well simulations.

        4. Calculates satisfaction and uncertainty based on CONSUMAT theory.

        5. Updates the CONSUMAT state of the agent.

        Notes
        -----
        The method uses the `dm_sols` attribute, which contains the decision-making solutions,
        to guide the simulation of fields and wells. It then updates various attributes of the
        agent, including profit, yield rate, satisfaction, uncertainty, and the CONSUMAT state.
        """
        current_year = self.model.current_year  # read current year from mesa model
        prec_aw_step = self.model.prec_aw_step  # read prec_aw_step from mesa model
        aquifers = self.aquifers  # aquifer objects
        fields = self.fields  # field objects
        wells = self.wells  # well objects
        dm_dict = self.dm_dict  # decision-making settings
        consumat_dict = self.consumat_dict  # CONSUMAT settings
        dm_sols = self.dm_sols  # optimization outputs

        finance = self.finance
        ##### Simulate fields
        for fi, field in fields.items():
            irr_depth = dm_sols["irr_depth"][:, [0]]
            i_crop = dm_sols[fi]["i_crop"].copy()
            field.step(
                irr_depth=irr_depth,
                i_crop=i_crop,
                prec_aw=prec_aw_step[field.prec_aw_id][current_year],
            )

        ##### Simulate wells (energy consumption)
        well_ids = dm_sols["well_ids"]
        self.irr_vol = sum([field.irr_vol_per_field for _, field in fields.items()])

        for _k, wid in enumerate(well_ids):
            well = wells[wid]
            # We only take first year optimization solution for simulation.
            withdrawal = self.irr_vol
            dwl = aquifers[well.aquifer_id].dwl
            # pumping_days remains fixed unless it is given.
            well.step(withdrawal=withdrawal, dwl=dwl)

        ##### Calulate profit and pumping cost
        finance.step(fields=fields, wells=wells)

        ##### CONSUMAT
        # Collect variables for CONSUMAT calculation
        self.profit = finance.profit
        self.avg_profit_per_field = self.profit / len(fields)
        self.yield_rate = sum(
            [field.yield_rate_per_field for _, field in fields.items()]
        ) / len(fields)

        # Calculate satisfaction and uncertainty
        alphas = consumat_dict["alpha"]
        scales = consumat_dict["scale"]

        def func(x, alpha=1):
            return 1 - np.exp(-alpha * x)

        # Use the average values per field
        self.scaled_profit = self.avg_profit_per_field / scales["profit"]
        self.scaled_yield_rate = self.yield_rate  # /scales["yield_rate"]
        needs = {}
        for var, alpha in alphas.items():
            if alpha is None:
                continue
            needs[var] = func(eval(f"self.scaled_{var}"), alpha=alpha)

        target = dm_dict["target"]
        satisfaction = needs[target]
        expected_sa = dm_sols["Sa"][target]

        # We define uncertainty to be the difference between expected_sa at the
        # previous time and satisfaction this year.
        expected_sa_t_1 = self.expected_sa
        if expected_sa_t_1 is None:  # Initial step
            uncertainty = abs(expected_sa - satisfaction)
        else:
            uncertainty = abs(expected_sa_t_1 - satisfaction)

        self.needs = needs
        self.satisfaction = satisfaction
        self.expected_sa = expected_sa
        self.uncertainty = uncertainty

        # Update CONSUMAT state
        sa_thre = self.pars["sa_thre"]
        un_thre = self.pars["un_thre"]

        if satisfaction >= sa_thre and uncertainty >= un_thre:
            self.state = "Imitation"
        elif satisfaction < sa_thre and uncertainty >= un_thre:
            self.state = "Social comparison"
        elif satisfaction >= sa_thre and uncertainty < un_thre:
            self.state = "Repetition"
        elif satisfaction < sa_thre and uncertainty < un_thre:
            self.state = "Deliberation"

        # Internal experiment => Fix state, where state is not dynamically updated
        if self.fix_state is not None:
            self.state = self.fix_state

    def make_dm(self, state, dm_sols, neighbor=None, init=False):
        """
        Create and solve an optimization model for decision-making based on the
        agent's current state.

        Parameters
        ----------
        state : str or None
            The current CONSUMAT state of the agent, which influences the optimization process.
        dm_sols : dict
            The previous decision-making solutions, used as inputs for the optimization model.
        neighbor : dict, optional
            A neighboring agent object, used in states, 'Imitation' and "Social comparison".
        init : bool, optional
            A flag indicating if it is the initial setup of the optimization model.

        Returns
        -------
        dict
            Updated decision-making solutions after solving the optimization model.

        Notes
        -----
        This method sets up and solves an optimization model based on various inputs, including
        field data, well data, water rights, and financial considerations. The type of
        optimization and constraints applied depend on the agent's current state, as defined
        by the CONSUMAT theory. The method returns updated decision-making solutions that
        guide the agent's actions in subsequent steps.
        """
        aquifers = self.aquifers  # aquifer objects
        fields = self.fields  # field objects
        wells = self.wells  # well objects
        dm_dict = self.dm_dict  # decision-making settings
        consumat_dict = self.consumat_dict

        dm = self.optimization_class()

        dm.setup_ini_model(
            unique_id=self.unique_id,
            gpenv=self.model.gpenv,  # share one environment for the entire simulation.,
            activate_ci=self.model.activate_ci,
            horizon=dm_dict["horizon"],
            crop_options=self.model.crop_options,
        )

        perceived_prec_aw = self.perceived_prec_aw
        current_year = self.model.current_year
        for i, (fi, field) in enumerate(fields.items()):
            # Note: Since field type is given as an input, we do not constrain
            # i_rainfed under any condition. Behavior agent will only adopt
            # i_crop and i_te in certain states.
            dm_sols_fi = dm_sols[fi]

            # calulate the premium for dm that store in the field
            if self.model.activate_ci:
                for field_type in ["irrigated", "rainfed"]:
                    for crop in self.model.crop_options:
                        field.premium_dict_for_dm[field_type][
                            crop
                        ] = self.finance.cal_APH_revenue_based_premium(
                            df=self.finance.aph_revenue_based_coef,
                            crop=crop,
                            county=field.county,
                            field_type=field_type,
                            aph_yield_dict=field.aph_yield_dict,
                            projected_price=self.finance.projected_price[crop],
                            premium_ratio=self.finance.premium_ratio,
                            coverage_level=0.75,
                        )
                premium_dict = field.premium_dict_for_dm
                aph_yield_dict = field.aph_yield_dict
            else:
                premium_dict = (None,)
                aph_yield_dict = (None,)

            if init:
                # Optimize irrigation depth with others variables given.
                # Apply the actual prec_aw (not the perceived one)
                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=self.model.prec_aw_step[field.prec_aw_id][current_year],
                    water_yield_curves=field.water_yield_curves,
                    field_type=field.field_type,
                    i_crop=dm_sols_fi["i_crop"],
                    i_rainfed=None,
                    premium_dict=premium_dict,
                    aph_yield_dict=aph_yield_dict,
                )

            elif state == "Deliberation":
                # optimize irrigation depth, crop choice, tech choice
                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=perceived_prec_aw[fi][current_year],
                    water_yield_curves=field.water_yield_curves,
                    field_type=field.field_type,
                    i_crop=None,
                    i_rainfed=None,
                    premium_dict=premium_dict,
                    aph_yield_dict=aph_yield_dict,
                )

            elif state == "Repetition":
                # only optimize irrigation depth
                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=perceived_prec_aw[fi][current_year],
                    water_yield_curves=field.water_yield_curves,
                    field_type=field.field_type,
                    i_crop=dm_sols_fi["i_crop"],
                    i_rainfed=None,
                    premium_dict=premium_dict,
                    aph_yield_dict=aph_yield_dict,
                )

            else:  # social comparason & imitation
                # We assume this behavioral agent has the same number of fields
                # as its neighbor.
                # A small patch may be needed in the future to generalize this.
                fi_neighbor = neighbor.field_ids[i]
                dm_sols_neighbor_fi = neighbor.pre_dm_sols[fi_neighbor]

                dm.setup_constr_field(
                    field_id=fi,
                    field_area=field.field_area,
                    prec_aw=perceived_prec_aw[fi][current_year],
                    water_yield_curves=field.water_yield_curves,
                    field_type=field.field_type,
                    i_crop=dm_sols_neighbor_fi["i_crop"],
                    i_rainfed=None,
                    premium_dict=premium_dict,
                    aph_yield_dict=aph_yield_dict,
                )

        for wi, well in wells.items():
            aquifer_id = well.aquifer_id
            proj_dwl = np.mean(aquifers[aquifer_id].dwl_list[-dm_dict["n_dwl"] :])

            dm.setup_constr_well(
                well_id=wi,
                dwl=proj_dwl,
                B=well.B,
                l_wt=well.l_wt,
                eff_pump=well.eff_pump,
                pumping_capacity=well.pumping_capacity,
                rho=well.rho,
                g=well.g,
            )

        if init:  # Inputted
            wr_dict = self.wr_dict
        else:  # Use agent's own water rights (for social comparison and imitation)
            wr_dict = dm_sols["water_rights"]

        for wr_id, v in self.wr_dict.items():
            if v["status"]:  # Check whether the wr is activated
                # Extract the water right setting from the previous opt run,
                # which we record as the remaining water right from the previous
                # year. If the wr is newly activated in a simulation, then we
                # use the input to setup the wr.
                wr_args = wr_dict.get(wr_id)
                if (
                    wr_args is None
                ):  # when we introduce the water rights for the first time (LEMA)
                    dm.setup_constr_wr(
                        water_right_id=wr_id,
                        wr_depth=v["wr_depth"],
                        time_window=v["time_window"],
                        remaining_tw=v["remaining_tw"],
                        remaining_wr=v["remaining_wr"],
                        tail_method=v["tail_method"],
                    )
                else:
                    dm.setup_constr_wr(
                        water_right_id=wr_id,
                        wr_depth=wr_args["wr_depth"],
                        time_window=wr_args["time_window"],
                        remaining_tw=wr_args["remaining_tw"],
                        remaining_wr=wr_args["remaining_wr"],
                        tail_method=wr_args["tail_method"],
                    )

        dm.setup_constr_finance(
            self.finance.finance_dict,
            payout_ratio=self.finance.payout_ratio,
            premium_ratio=self.finance.premium_ratio,
        )
        dm.setup_obj(
            target=dm_dict["target"], consumat_dict=consumat_dict
        )  # no dynamic update for alpha.
        dm.finish_setup(display_summary=dm_dict["display_summary"])
        dm.solve(
            keep_gp_model=dm_dict["keep_gp_model"],
            keep_gp_output=dm_dict["keep_gp_output"],
            display_report=dm_dict["display_report"],
        )
        dm_sols = dm.sols
        if dm_sols is None:
            warnings.warn(
                "Gurobi returns empty solutions (likely due to infeasible problem.",
                stacklevel=2,
            )
        # dm.depose_gp_env()  # Delete the entire environment to release memory.

        return dm_sols

    def make_dm_deliberation(self):
        """
        Make decision under the "Deliberation" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method updates the `dm_sols` attribute by calling the `make_dm` method
        with the current state set to "Deliberation".
        """
        self.dm_sols = self.make_dm(state="Deliberation", dm_sols=self.pre_dm_sols)

    def make_dm_repetition(self):
        """
        Make decision under the "Repetition" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method updates the `dm_sols` attribute by calling the `make_dm` method
        with the current state set to "Repetition".
        """
        self.dm_sols = self.make_dm(state="Repetition", dm_sols=self.pre_dm_sols)

    def make_dm_social_comparison(self):
        """
        Make decision under the "Social comparison" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method performs several key steps:

        1. Evaluates comparable decision-making solutions from agents in the network.

        2. Selects the agent with the best objective value.

        3. Compares the agent's original choice with the selected agent's choice.

        4. Updates the `dm_sols` attribute based on the comparison.
        """
        behavior_ids_in_network = self.behavior_ids_in_network
        # Evaluate comparable
        dm_sols_list = []
        for behavior_id in behavior_ids_in_network:
            # !!! Here we assume no. fields, n_c and split are the same across agents
            # Keep this for now.
            dm_sols = self.make_dm(
                state="Social comparison",
                dm_sols=self.pre_dm_sols,
                neighbor=self.model.behaviors[behavior_id],
            )
            dm_sols_list.append(dm_sols)
        objs = [s["obj"] for s in dm_sols_list]
        max_obj = max(objs)
        select_behavior_index = objs.index(max_obj)
        self.selected_behavior_id_in_network = behavior_ids_in_network[
            select_behavior_index
        ]

        # Agent's original choice
        dm_sols = self.make_dm(state="Repetition", dm_sols=self.pre_dm_sols)
        if dm_sols["obj"] >= max_obj:
            self.dm_sols = dm_sols
        else:
            self.dm_sols = dm_sols_list[select_behavior_index]

    def make_dm_imitation(self):
        """
        Make decision under the "Imitation" CONSUMAT state.

        Returns
        -------
        None

        Notes
        -----
        This method performs the following key steps:

        1. Selects an agent based on memory from previous social comparison from the network for imitation.

        2. Updates the `dm_sols` attribute by calling the `make_dm` method
           with the current state set to "Imitation" and using the selected agent's solutions.
        """
        selected_behavior_id_in_network = self.selected_behavior_id_in_network
        if selected_behavior_id_in_network is None:
            try:  # if rngen is given in the model
                selected_behavior_id_in_network = self.rngen.choice(
                    self.behavior_ids_in_network
                )
            except:
                selected_behavior_id_in_network = np.random.choice(
                    self.behavior_ids_in_network
                )

        neighbor = self.model.behaviors[selected_behavior_id_in_network]

        self.dm_sols = self.make_dm(
            state="Imitation", dm_sols=self.pre_dm_sols, neighbor=neighbor
        )
