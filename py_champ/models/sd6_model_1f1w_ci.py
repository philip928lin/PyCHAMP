from copy import deepcopy

import gurobipy as gp
import mesa
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..components.aquifer import Aquifer
from ..components.behavior import Behavior_1f1w_ci
from ..components.field import Field_1f1w_ci
from ..components.finance import Finance_1f1w_ci
from ..components.optimization_1f1w_ci import Optimization_1f1w_ci
from ..components.well import Well4SingleFieldAndWell
from ..utility.util import (
    BaseSchedulerByTypeFiltered,
    Indicator,
    TimeRecorder,
    get_agt_attr,
)


# % MESA
class SD6Model_1f1w_ci(mesa.Model):
    """
    A Mesa model representing the SD6 simulation for agricultural and environmental systems.

    This model includes various agents like fields, wells, behaviors, and aquifers. It simulates
    interactions and decisions of these agents over a specified time period.

    Parameters
    ----------
    pars : dict
        Parameters used for model calibration and setup.

        >>> settings = {
        >>>     "perceived_risk": 0.52,
        >>>     "forecast_trust": 0.70,
        >>>     "sa_thre": 0.11,
        >>>     "un_thre": 0.11,

    crop_options : list
        List of available crop options for the simulation.
    tech_options : list
        List of available technology options for irrigation and farming.
    area_split : int
        The number of splits or divisions within each field area.
    aquifers_dict : dict
        Settings about the aquifers in the model, mapped by their IDs.
    fields_dict : dict
        Settings about the fields in the model, mapped by their IDs.
    wells_dict : dict
        Settings about the wells in the model, mapped by their IDs.
    finances_dict : dict
        Settings about the finance in the model, mapped by their IDs.
    behaviors_dict : dict
        Settings about the behaviors in the model, mapped by their IDs.
    prec_aw_step : dict
        Time-series data for available precipitation.

        >>> prec_aw_step = {
        >>> "<prec_aw1>": {
        >>>     "<year>": {
        >>>         "<crop1>": "[cm]",
        >>>         "<crop2>": "[cm]",
        >>>         }
        >>>     }
        >>> }

    init_year : int, optional
        The initial year of the simulation (a year before the start year).
    end_year : int, optional
        The final year of the simulation.
    lema_options: tuple, optional
        The option to turn on water rights for Local Enhanced Management Areas
        (LEMA). (bool: flag to indicate whether LEMA regulations are in effect,
        str: water right id, int: year LEMA begin).
        Default is (True, 'wr_LEMA_5yr', 2013).
    fix_state : str, optional
        A fixed state for testing or specific scenarios.
    lema : bool, optional
        Flag to indicate whether LEMA regulations are in effect.
    show_step : bool, optional
        Flag to control the display of step information during simulation.
    seed : int, optional
        Seed for random number generation for reproducibility.
    **kwargs : dict
        Additional keyword arguments including time-step data like crop price, type, irrigation depth, etc.
        - 'crop_price_step': crop_price_step

    Attributes
    ----------
    schedule : BaseSchedulerByTypeFiltered
        The scheduler used to activate agents in the model.
    datacollector : mesa.DataCollector
        Collects and stores data from agents and the model during the simulation.
    (other attributes are initialized within the __init__ method)

    Notes
    -----
    The model is highly configurable with various parameters affecting the behavior of agents
    and their interactions within the simulated environment. It can be used to study the impacts
    of different agricultural practices and environmental policies.
    """

    def __init__(
        self,
        pars,
        crop_options,
        aquifers_dict,
        fields_dict,
        wells_dict,
        finances_dict,
        behaviors_dict,
        prec_aw_step,
        init_year=2007,
        end_year=2022,
        components=None,
        optimization_class=Optimization_1f1w_ci,
        lema_options=(True, "wr_LEMA_5yr", 2013),
        fix_state=None,
        show_step=True,
        seed=None,
        gurobi_dict=None,
        activate_ci=True,
        **kwargs,
    ):
        # Time and Step recorder
        self.time_recorder = TimeRecorder()

        # Prepare the model
        if gurobi_dict is None:
            gurobi_dict = {"LogToConsole": 0, "NonConvex": 2, "Presolve": -1}
        if components is None:
            components = {
                "aquifer": Aquifer,
                "field": Field_1f1w_ci,
                "well": Well4SingleFieldAndWell,
                "finance": Finance_1f1w_ci,
                "behavior": Behavior_1f1w_ci,
            }
        self.optimization_class = optimization_class
        self.crop_options = crop_options  # n_c
        # Create revised mesa scheduler and assign it to the model
        self.schedule = BaseSchedulerByTypeFiltered(self)
        # MESA required attributes
        self.running = True
        # Indicate the activation of the crop insurance feature
        self.activate_ci = activate_ci
        # Simulation period
        self.init_year = init_year
        self.start_year = self.init_year + 1
        self.end_year = end_year
        self.total_steps = self.end_year - self.init_year
        self.current_year = self.init_year
        self.t = 0  # This is initial step
        # LEMA water right info
        self.lema = lema_options[0]
        self.lema_wr_name = lema_options[1]
        self.lema_year = lema_options[2]
        # Show step info during the simulation (default: True)
        self.show_step = show_step

        # mesa has self.random.seed(seed) but it is not usable for truncnorm
        # We create our own random generator.
        self.seed = seed
        self.rngen = np.random.default_rng(seed)

        # Store parameters
        self.pars = pars

        # Input timestep data
        self.prec_aw_step = prec_aw_step  # [prec_aw_id][year][crop]
        self.crop_price_step = kwargs.get("crop_price_step")  # [finance_id][year][crop]

        # Just for internal testing and debugging.
        self.irr_depth_step = kwargs.get("irr_depth_step")
        self.field_type_step = kwargs.get("field_type_step")

        if self.activate_ci:
            self.harvest_price_step = kwargs.get(
                "harvest_price_step"
            )  # [finance_id][year][crop]
            self.projected_price_step = kwargs.get(
                "projected_price_step"
            )  # [finance_id][year][crop]
            self.aph_revenue_based_coef_step = kwargs.get(
                "aph_revenue_based_coef_step"
            )  # [finance_id][year][field_type][crop]
        else:
            self.harvest_price_step = None
            self.projected_price_step = None
            self.aph_revenue_based_coef_step = None

        # Copy the dictionaries before correction from shared_config
        aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict = (
            deepcopy(aquifers_dict),
            deepcopy(fields_dict),
            deepcopy(wells_dict),
            deepcopy(finances_dict),
            deepcopy(behaviors_dict),
        )

        # Setup Gurobi environment, gpenv
        self.gpenv = gp.Env(empty=True)
        for k, v in gurobi_dict.items():
            self.gpenv.setParam(k, v)
        self.gpenv.start()

        # Initialize aquifer environment (this is not associated with farmers)
        aquifers = {}
        for aqid, aquifer_dict in aquifers_dict.items():
            agt_aquifer = components["aquifer"](
                unique_id=aqid, model=self, settings=aquifer_dict
            )
            aquifers[aqid] = agt_aquifer
            self.schedule.add(agt_aquifer)
        self.aquifers = aquifers

        # Initialize fields
        fields = {}
        for fid, field_dict in fields_dict.items():
            # Initialize crop type (considering remove this)
            init_crop = field_dict["init"]["crop"]
            if isinstance(init_crop, list):
                field_dict["init"]["crop"] = self.rngen.choice(init_crop)

            # Initialize fields
            agt_field = components["field"](
                unique_id=fid,
                model=self,
                settings=field_dict,
                # For this model's convenience
                irr_freq=field_dict.get("irr_freq"),
                truncated_normal_pars=field_dict.get("truncated_normal_pars"),
                lat=field_dict.get("lat"),
                lon=field_dict.get("lon"),
                x=field_dict.get("x"),
                y=field_dict.get("y"),
                field_type_rn=None,
            )
            fields[fid] = agt_field
            self.schedule.add(agt_field)
        self.fields = fields

        # Initialize wells
        wells = {}
        for wid, well_dict in wells_dict.items():
            agt_well = components["well"](unique_id=wid, model=self, settings=well_dict)
            wells[wid] = agt_well
            self.schedule.add(agt_well)
        self.wells = wells

        # Initialize behavior and finance agents and append them to the schedule
        ## Don't use parallelization. It is slower!
        self.max_num_fields_per_agt = 0
        self.max_num_wells_per_agt = 0

        behaviors = {}
        finances = {}
        for behavior_id, behavior_dict in tqdm(
            behaviors_dict.items(), desc="Initialize agents"
        ):
            # Initialize finance
            finance_id = behavior_dict["finance_id"]
            finance_dict = finances_dict[finance_id]
            agt_finance = components["finance"](
                unique_id=f"{finance_id}_{behavior_id}",
                model=self,
                settings=finance_dict,
            )
            agt_finance.finance_id = finance_id
            finances[
                behavior_id
            ] = agt_finance  # Assume one behavior agent has one finance object
            self.schedule.add(agt_finance)

            agt_behavior = components["behavior"](
                unique_id=behavior_id,
                model=self,
                settings=behavior_dict,
                pars=self.pars,
                fields={
                    fid: self.fields[fid]
                    for i, fid in enumerate(behavior_dict["field_ids"])
                },
                wells={
                    wid: self.wells[wid]
                    for i, wid in enumerate(behavior_dict["well_ids"])
                },
                finance=agt_finance,
                aquifers=self.aquifers,
                Optimization=self.optimization_class,
                # kwargs
                fix_state=fix_state,
            )
            behaviors[behavior_id] = agt_behavior
            self.schedule.add(agt_behavior)
            self.max_num_fields_per_agt = max(
                self.max_num_fields_per_agt, len(behavior_dict["field_ids"])
            )
            self.max_num_wells_per_agt = max(
                self.max_num_wells_per_agt, len(behavior_dict["well_ids"])
            )
        self.behaviors = behaviors
        self.finances = finances

        if self.crop_price_step is not None:
            for _unique_id, finance in self.finances.items():
                crop_prices = self.crop_price_step.get(finance.finance_id)
                if crop_prices is not None:
                    finance.crop_price = crop_prices[self.current_year]

        agent_reporters = {
            "agt_type": get_agt_attr("agt_type"),
            ### Field
            "field_type": get_agt_attr("field_type"),
            "crop": get_agt_attr("crop"),
            "irr_vol": get_agt_attr("irr_vol_per_field"),  # m-ha
            "yield_rate": get_agt_attr("yield_rate_per_field"),
            "yield": get_agt_attr("y"),  # 1e4 bu/field
            "w": get_agt_attr("w"),
            "field_area": get_agt_attr("field_area"),  # ha
            "aph_yield_dict": get_agt_attr("aph_yield_dict"),  # a dictionary
            ### Behavior
            "Sa": get_agt_attr("satisfaction"),
            "E[Sa]": get_agt_attr("expected_sa"),
            "Un": get_agt_attr("uncertainty"),
            "state": get_agt_attr("state"),
            "total_field_area": get_agt_attr("total_field_area"),  # ha
            "gp_status": get_agt_attr("gp_status"),
            "gp_MIPGap": get_agt_attr("gp_MIPGap"),
            ### Finance
            "profit": get_agt_attr("finance.profit"),  # 1e4 $
            "revenue": get_agt_attr("finance.rev"),  # 1e4 $
            "energy_cost": get_agt_attr("finance.cost_e"),  # 1e4 $
            "tech_cost": get_agt_attr("finance.cost_tech"),  # 1e4 $
            "premium": get_agt_attr("finance.premium"),  # 1e4 $
            "payout": get_agt_attr("finance.payout"),  # 1e4 $
            "premium_dict": get_agt_attr("finance.premium_dict"),  # a dictionary
            ### Well
            "water_depth": get_agt_attr("l_wt"),
            "energy": get_agt_attr("e"),  # PJ
            ### Aquifer
            "withdrawal": get_agt_attr("withdrawal"),  # m-ha
            "GW_st": get_agt_attr("st"),  # m
            "GW_dwl": get_agt_attr("dwl"),  # m
        }

        model_reporters = {}
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
        )

        # Summary info
        estimated_sim_dur = self.time_recorder.sec2str(
            self.time_recorder.get_elapsed_time(strf=False) * self.total_steps
        )
        msg = f"""\n
        Initial year: \t{self.init_year}
        Simulation period:\t{self.start_year} to {self.end_year}
        Number of agents:\t{len(behaviors_dict)}
        Number of aquifers:\t{len(aquifers_dict)}
        Initialiation duration:\t{self.time_recorder.get_elapsed_time()}
        Estimated sim duration:\t{estimated_sim_dur}
        """
        print(msg)

    def step(self):
        """
        Advance the model by one step.

        This method progresses the simulation by one year. It involves updating crop prices,
        deciding field types based on agent behavior, applying LEMA policy, and executing
        the step functions of all agents. Finally, it collects data and updates the model's state.

        The method controls the flow of the simulation, ensuring that each agent and component
        of the model acts according to the current time step and the state of the environment.
        """
        self.current_year += 1
        self.t += 1

        ##### Human (behaviors)
        current_year = self.current_year

        # Update crop price
        if self.crop_price_step is not None:
            for _, finance in self.finances.items():
                crop_prices = self.crop_price_step.get(finance.finance_id)
                if crop_prices is not None:
                    finance.crop_price = crop_prices[self.current_year]
        # Update data, harvest price, projected price, and aph_revenue_based_coef
        if self.activate_ci:
            for _, finance in self.finances.items():
                harvest_price = self.harvest_price_step.get(finance.finance_id)
                if harvest_price is not None:
                    finance.harvest_price = harvest_price[self.current_year]
            for _, finance in self.finances.items():
                projected_price = self.projected_price_step.get(finance.finance_id)
                if projected_price is not None:
                    finance.projected_price = projected_price[self.current_year]
            for _, finance in self.finances.items():
                aph_revenue_based_coef = self.aph_revenue_based_coef_step.get(
                    finance.finance_id
                )
                if aph_revenue_based_coef is not None:
                    finance.aph_revenue_based_coef = aph_revenue_based_coef[
                        self.current_year
                    ]

        # Assign field type based on each behavioral agent.
        for _, behavior in self.behaviors.items():
            # Stocastically select rainfed field
            for _, field in behavior.fields.items():
                rn_irr = True
                if rn_irr:
                    irr_freq = field.irr_freq
                    rn = self.rngen.uniform(0, 1)
                    if rn <= irr_freq:
                        # Let optimization decide the field type
                        field.field_type = "optimize"
                        field.field_type_rn = "optimize"
                    else:
                        # Force the field to be rainfed
                        field.field_type = "rainfed"  # Force the field to be rainfed
                        field.field_type_rn = "rainfed"

            # Activate LEMA (i.e., a water right constraint) starting from 2013
            if self.lema and current_year >= self.lema_year:
                behavior.wr_dict[self.lema_wr_name]["status"] = True
            else:
                behavior.wr_dict[self.lema_wr_name]["status"] = False

            # Save the decisions from the previous step. (very important)
            behavior.pre_dm_sols = behavior.dm_sols

        # Simulation
        # Exercute step() of all behavioral agents in a for loop
        # Note: fields, wells, and finance are simulation within a behavioral
        # agent to better accomondate heterogeneity among behavioral agents
        self.schedule.step(agt_type="Behavior")  # Parallelization makes it slower!

        ##### Nature Environment (aquifers)
        for aq_id, aquifer in self.aquifers.items():
            withdrawal = 0.0
            # Collect all the well withdrawals of a given aquifer
            withdrawal += sum(
                [
                    well.withdrawal if well.aquifer_id == aq_id else 0.0
                    for _, well in self.wells.items()
                ]
            )
            # Update aquifer
            aquifer.step(withdrawal)

        # Collect df_sys and print info
        self.datacollector.collect(self)
        if self.lema and current_year == self.lema_year and self.show_step:
            print("LEMA begin")
        if self.show_step:
            print(
                f"Year {self.current_year} [{self.t}/{self.total_steps}]"
                + f"\t{self.time_recorder.get_elapsed_time()}\n"
            )

        # Termination criteria
        if self.current_year == self.end_year:
            self.running = False
            print("Done!", f"\t{self.time_recorder.get_elapsed_time()}")

    def end(self):
        """Depose the Gurobi environment, ensuring that it is executed only when
        the instance is no longer needed.
        """
        self.gpenv.dispose()

    def get_datacollector_output(self):
        return self.datacollector.get_agent_vars_dataframe()

    @staticmethod
    def get_dfs(model):
        df = model.datacollector.get_agent_vars_dataframe().reset_index()
        df["year"] = df["Step"] + model.init_year
        df.index = df["year"]

        # =============================================================================
        # df_agt
        # =============================================================================
        df_behaviors = df[df["agt_type"] == "Behavior"].dropna(axis=1, how="all")
        df_behaviors["bid"] = df_behaviors["AgentID"]

        df_fields = df[df["agt_type"] == "Field"].dropna(axis=1, how="all")
        df_fields["field_type"] = ""
        df_fields.loc[df_fields["irr_vol"] == 0, "field_type"] = "rainfed"
        df_fields.loc[df_fields["irr_vol"] > 0, "field_type"] = "irrigated"
        df_fields["irr_depth"] = (
            df_fields["irr_vol"] / df_fields["field_area"] * 100
        )  # cm
        df_fields["fid"] = df_fields["AgentID"]

        df_wells = df[df["agt_type"] == "Well"].dropna(axis=1, how="all")
        df_wells["wid"] = df_wells["AgentID"]
        
        if model.activate_ci:
            df_agt = pd.concat(
                [
                    df_behaviors[
                        [
                            "bid",
                            "Sa",
                            "E[Sa]",
                            "Un",
                            "state",
                            "profit",
                            "revenue",
                            "energy_cost",
                            "tech_cost",
                            "premium",
                            "payout",
                            "gp_status",
                            "gp_MIPGap",
                        ]
                    ],
                    df_fields[
                        [
                            "fid",
                            "field_type",
                            "crop",
                            "irr_vol",
                            "yield",
                            "yield_rate",
                            "irr_depth",
                            "w",
                            "field_area",
                        ]
                    ],
                    df_wells[["wid", "water_depth", "withdrawal", "energy"]],
                ],
                axis=1,
            )
            df_agt["yield"] = df_agt["yield"].apply(sum).apply(sum)
            df_agt = df_agt.round(8)

            df_other = pd.concat(
                [
                    df_behaviors[["bid", "premium_dict"]],
                    df_fields[["fid", "aph_yield_dict"]],
                ],
                axis=1,
            )
        else:
            df_agt = pd.concat(
                [
                    df_behaviors[
                        [
                            "bid",
                            "Sa",
                            "E[Sa]",
                            "Un",
                            "state",
                            "profit",
                            "revenue",
                            "energy_cost",
                            "tech_cost",
                            "gp_status",
                            "gp_MIPGap",
                        ]
                    ],
                    df_fields[
                        [
                            "fid",
                            "field_type",
                            "crop",
                            "irr_vol",
                            "yield",
                            "yield_rate",
                            "irr_depth",
                            "w",
                            "field_area",
                        ]
                    ],
                    df_wells[["wid", "water_depth", "withdrawal", "energy"]],
                ],
                axis=1,
            )
            df_agt["yield"] = df_agt["yield"].apply(sum).apply(sum)
            df_agt = df_agt.round(8)

            df_other = None
        # =============================================================================
        # df_sys
        # =============================================================================
        df_sys = pd.DataFrame()

        # Aquifer
        df_aquifers = df[df["agt_type"] == "Aquifer"].dropna(axis=1, how="all")
        df_sys["GW_st"] = df_aquifers["GW_st"]
        df_sys["withdrawal"] = df_aquifers["withdrawal"]

        # Field_Type ratio
        dff = (
            df_agt[["field_type", "field_area"]]
            .groupby([df_agt.index, "field_type"])
            .sum()
        )
        all_years = dff.index.get_level_values("year").unique()
        all_field_types = ["irrigated", "rainfed"]
        new_index = pd.MultiIndex.from_product(
            [all_years, all_field_types], names=["year", "field_type"]
        )
        dff = dff.reindex(new_index).fillna(0)
        df_sys["rainfed"] = (
            dff.xs("rainfed", level="field_type") / dff.groupby("year").sum()
        )

        # Crop type ratio
        dff = df_agt[["crop", "field_area"]].groupby([df_agt.index, "crop"]).sum()
        all_years = dff.index.get_level_values("year").unique()
        all_crop_types = model.crop_options
        new_index = pd.MultiIndex.from_product(
            [all_years, all_crop_types], names=["year", "crop"]
        )
        dff = dff.reindex(new_index).fillna(0)
        total = dff.groupby("year").sum()
        for c in all_crop_types:
            df_sys[f"{c}"] = dff.xs(c, level="crop") / total

        # Behavioral agent state ratio
        dff = df_behaviors[["state"]].groupby([df_behaviors.index, "state"]).size()
        all_years = dff.index.get_level_values("year").unique()
        all_states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]
        new_index = pd.MultiIndex.from_product(
            [all_years, all_states], names=["year", "state"]
        )
        dff = dff.reindex(new_index).fillna(0)
        for s in all_states:
            df_sys[f"{s}"] = dff.xs(s, level="state")
        df_sys = df_sys.round(4)

        return df_sys, df_agt, df_other

    @staticmethod
    def get_metrices(
        df_sys,
        data,
        targets=None,
        indicators_list=None,
    ):
        """
        Calculate various metrics based on system-level data and specified targets.

        Parameters
        ----------
        df_sys : pd.DataFrame
            The system-level dataframe.
        data : dict or pd.DataFrame
            The reference or observed data for comparison.
        targets : list
            List of targets or variables for which metrics are calculated.
        indicators_list : list
            List of indicators to calculate, such as 'r', 'rmse', 'KGE'.

        Returns
        -------
        pd.DataFrame
            A dataframe containing calculated metrics for each target.

        This method is useful for evaluating the performance of the model against real-world data
        or specific objectives, providing insights into the accuracy and reliability of the simulation.
        """
        if targets is None:
            targets = [
                "GW_st",
                "withdrawal",
                "rainfed",
                "corn",
                "sorghum",
                "soybeans",
                "wheat",
                "fallow",
            ]
        if indicators_list is None:
            indicators_list = ["r", "rmse", "kge"]
        indicators = Indicator()
        metrices = []
        for tar in targets:
            metrices.append(
                indicators.cal_indicator_df(
                    x_obv=data[tar],
                    y_sim=df_sys[tar],
                    index_name=tar,
                    indicators_list=indicators_list,
                )
            )
        metrices = pd.concat(metrices)
        return metrices
