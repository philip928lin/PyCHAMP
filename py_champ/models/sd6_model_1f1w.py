# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
import mesa
from ..utility.util import TimeRecorder, Indicator
import gurobipy as gp

class BaseSchedulerByTypeFiltered(mesa.time.BaseScheduler):
    """
    A scheduler that overrides the step method to allow for filtering
    of agents by .agt_type.

    Example:
    >>> scheduler = BaseSchedulerByTypeFiltered(model)
    >>> scheduler.step(agt_type="Behavior")
    """
        
    def step(self, agt_type=None) -> None:
        """Execute the step of all the agents, one at a time."""
        # To be able to remove and/or add agents during stepping
        # it's necessary for the keys view to be a list.
        self.do_each(method="step", agt_type=agt_type)
        self.steps += 1
        self.time += 1

    def do_each(self, method, agent_keys=None, shuffle=False, agt_type=None):
        if agent_keys is None:
            agent_keys = self.get_agent_keys()
        if agt_type is not None:
            agent_keys = [i for i in agent_keys if self._agents[i].agt_type==agt_type]
        if shuffle:
            self.model.random.shuffle(agent_keys)
        for agent_key in agent_keys:
            if agent_key in self._agents:
                getattr(self._agents[agent_key], method)()

#% MESA
class SD6Model_1f1w(mesa.Model):
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
    def __init__(self, pars, crop_options, prec_aw_step,
                 aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict, 
                 aquifer_agtType, field_agtType, well_agtType, finance_agtType, behavior_agtType,
                 optimization_class,
                 init_year=2007, end_year=2022,
                 lema_options=(True, 'wr_LEMA_5yr', 2013),
                 fix_state=None, show_step=True, seed=None, 
                 gurobi_dict={"LogToConsole": 0, "NonConvex": 2, "Presolve": -1},
                 **kwargs):
        # MESA required attributes
        self.running = True     # Required for batch run

        # Time and Step recorder
        self.time_recorder = TimeRecorder()

        self.init_year  = init_year
        self.start_year = self.init_year + 1
        self.end_year   = end_year
        self.total_steps    = self.end_year - self.init_year
        self.current_year   = self.init_year
        self.t              = 0                         # This is initial step
        self.lema           = lema_options[0]
        self.lema_wr_name   = lema_options[1]
        self.lema_year      = lema_options[2]
        self.show_step = show_step

        # mesa has self.random.seed(seed) but it is not usable for truncnorm
        # We create our own random generator.
        self.seed = seed
        self.rngen = np.random.default_rng(seed)
        
        # Store parameters 
        self.pars = pars
        
        # Store agent type
        self.aquifer_agtType = aquifer_agtType
        self.field_agtType = field_agtType
        self.well_agtType = well_agtType
        self.finance_agtType = finance_agtType
        self.behavior_agtType = behavior_agtType
        self.optimization_class = optimization_class

        # Input timestep data
        self.prec_aw_step = prec_aw_step    # [prec_aw_id][year][crop]
        self.crop_price_step = kwargs.get("crop_price_step") # [finance_id][year][crop]
        self.irr_depth_step = kwargs.get("irr_depth_step")
        self.field_type_step = kwargs.get("field_type_step")
        
        # These three variables will be used to define the dimension of the opt
        self.crop_options = crop_options # n_c

        # Create schedule and assign it to the model
        self.schedule = BaseSchedulerByTypeFiltered(self)
        
        # Copy the dictionaries before correction from shared_config
        aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict = \
            (deepcopy(aquifers_dict), deepcopy(fields_dict), deepcopy(wells_dict),
            deepcopy(finances_dict), deepcopy(behaviors_dict))
            
        # Setup Gurobi environment, gpenv
        self.gpenv = gp.Env(empty=True)
        for k, v in gurobi_dict.items():
            self.gpenv.setParam(k, v)
        self.gpenv.start()
        
        # Initialize aquifer environment (this is not associated with farmers)
        aquifers = {}
        for aqid, aquifer_dict in aquifers_dict.items():
            agt_aquifer = self.aquifer_agtType(
                unique_id=aqid, 
                model=self, 
                settings=aquifer_dict)
            aquifers[aqid] = agt_aquifer
            self.schedule.add(agt_aquifer)
        self.aquifers = aquifers

        # Initialize fields
        fields = {}
        for fid, field_dict in fields_dict.items():
            # Initialize crop type   
            init_crop = field_dict['init']['crop']
            if isinstance(init_crop, list):
                field_dict['init']['crop'] = self.rngen.choice(init_crop)
            
            # Initialize fields
            agt_field = self.field_agtType(
                unique_id=fid,
                model=self, 
                settings=field_dict,
                # For this model's convenience
                irr_freq=field_dict.get('irr_freq'),
                truncated_normal_pars=field_dict.get('truncated_normal_pars'),
                lat=field_dict.get("lat"),
                lon=field_dict.get("lon"),
                x=field_dict.get("x"),
                y=field_dict.get("y"),
                )
            fields[fid] = agt_field
            self.schedule.add(agt_field)
        self.fields = fields

        # Initialize wells
        wells = {}
        for wid, well_dict in wells_dict.items():
            agt_well = self.well_agtType(
                unique_id=wid, 
                model=self, 
                settings=well_dict
                )
            wells[wid] = agt_well
            self.schedule.add(agt_well)
        self.wells = wells
        
        # Initialize behavior and finance agents and append them to the schedule
        ## Don't use parallelization. It is slower!
        self.max_num_fields_per_agt = 0
        self.max_num_wells_per_agt = 0
        
        behaviors = {}   
        finances = {}                                    
        for behavior_id, behavior_dict in tqdm(behaviors_dict.items(), desc="Initialize agents"):
            # Initialize finance
            finance_id = behavior_dict['finance_id']
            finance_dict = finances_dict[finance_id]
            agt_finance = self.finance_agtType(
                unique_id=f"{finance_id}_{behavior_id}", 
                model=self, 
                settings=finance_dict
                )
            agt_finance.finance_id = finance_id
            finances[behavior_id] = agt_finance # Assume one behavior agent has one finance object
            self.schedule.add(agt_finance)
            
            agt_behavior = self.behavior_agtType(
                unique_id=behavior_id, 
                model=self, 
                settings=behavior_dict, 
                pars=self.pars,
                fields={fid: self.fields[fid] for i, fid in enumerate(behavior_dict['field_ids'])}, 
                wells={wid: self.wells[wid] for i, wid in enumerate(behavior_dict['well_ids'])}, 
                finance=agt_finance, 
                aquifers=self.aquifers,
                optimization_class=self.optimization_class,
                # kwargs
                rngen=self.rngen,   
                fix_state=fix_state
                )
            behaviors[behavior_id] = agt_behavior
            self.schedule.add(agt_behavior)
            self.max_num_fields_per_agt = max(
                self.max_num_fields_per_agt, len(behavior_dict['field_ids']))
            self.max_num_wells_per_agt = max(
                self.max_num_wells_per_agt, len(behavior_dict['well_ids']))
        self.behaviors = behaviors
        self.finances = finances
        
        if self.crop_price_step is not None:
            for unique_id, finance in self.finances.items():
                crop_prices = self.crop_price_step.get(finance.finance_id)
                if crop_prices is not None:
                    finance.crop_price = crop_prices[self.current_year]

        def get_nested_attr(obj, attr_str):
            """A patch to collect a nested attribute using MESA's datacollector"""
            attrs = attr_str.split('.', 1)
            current_attr = getattr(obj, attrs[0], None)
            if len(attrs) == 1 or current_attr is None:
                return current_attr
            return get_nested_attr(current_attr, attrs[1])
        
        def get_agt_attr(attr_str):
            """
            This replaces, e.g., lambda a: getattr(a, "satisfaction", None)
            We have to do this to return None if the attribute is not exist 
            in the given agent type.
            def func(agent):
                return getattr(agent, attr, None)
            """
            def get_nested_attr(obj):
                def get_nested_attr_(obj, attr_str):
                    attrs = attr_str.split('.', 1)
                    current_attr = getattr(obj, attrs[0], None)
                    if len(attrs) == 1 or current_attr is None:
                        return current_attr
                    return get_nested_attr_(current_attr, attrs[1])
                return get_nested_attr_(obj, attr_str)
            return get_nested_attr
        
        agent_reporters = {
            "agt_type":         get_agt_attr("agt_type"),
            ### Field
            "field_type":       get_agt_attr("field_type"),
            "crop":             get_agt_attr("crop"),
            "irr_vol":    get_agt_attr("irr_vol_per_field"),    # m-ha
            "yield_rate": get_agt_attr("yield_rate_per_field"),
            "w":                get_agt_attr("w"),
            "field_area":       get_agt_attr("field_area"),     # ha
            
            ### Behavior
            "Sa":               get_agt_attr("satisfaction"),
            "E[Sa]":            get_agt_attr("expected_sa"),
            "Un":               get_agt_attr("uncertainty"),
            "state":            get_agt_attr("state"),
            "profit":           get_agt_attr("profit"),                 # 1e4 $
            "revenue":          get_agt_attr("finance.rev"),            # 1e4 $
            "energy_cost":      get_agt_attr("finance.cost_e"),         # 1e4 $
            "tech_cost":        get_agt_attr("finance.tech_cost"),      # 1e4 $              
            "gp_status":        get_agt_attr("gp_status"),
            "gp_MIPGap":        get_agt_attr("gp_MIPGap"),

            ### Well
            "water_depth":      get_agt_attr("l_wt"),
            
            ### Aquifer
            "withdrawal":       get_agt_attr("withdrawal"),             # m-ha
            "GW_st":            get_agt_attr("st"),                     # m
            "GW_dwl":           get_agt_attr("dwl"),                    # m
            }

        model_reporters = {}
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
        )

        # Summary info
        estimated_sim_dur = self.time_recorder.sec2str(
            self.time_recorder.get_elapsed_time(strf=False)*self.total_steps)
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
            for unique_id, finance in self.finances.items():
                crop_prices = self.crop_price_step.get(finance.finance_id)
                if crop_prices is not None:
                    finance.crop_price = crop_prices[self.current_year]
        
        # Assign field type based on each behavioral agent.
        # irr_depth_step = self.irr_depth_step
        # field_type_step = self.field_type_step
        for behavior_id, behavior in self.behaviors.items():
            # Randomly select rainfed field
            for fid_, field in behavior.fields.items():
                rn_irr = True
                if rn_irr:
                    irr_freq = field.irr_freq
                    rn = self.rngen.uniform(0, 1)
                    if rn <= irr_freq:
                        field.field_type = 'optimize' # Optimize it
                        field.field_type_rn = 'optimize' 
                    else:
                        #raise
                        field.field_type = 'rainfed'  # Force the field to be rainfed
                        field.field_type_rn = 'rainfed'
                    
            # Turn on LEMA (i.e., a water right constraint) starting from 2013
            if self.lema and current_year >= self.lema_year:
                behavior.wr_dict[self.lema_wr_name]['status'] = True
            else:
                behavior.wr_dict[self.lema_wr_name]['status'] = False
            
            # Save the decisions from the previous step. (very important)
            behavior.pre_dm_sols = behavior.dm_sols
        
        # Simulation
        # Exercute step() of all behavioral agents in a for loop
        # Note: fields, wells, and finance are simulation within a behavioral 
        # agent to better accomondate heterogeneity among behavioral agents        
        self.schedule.step(agt_type="Behavior")    # Parallelization makes it slower!

        ##### Nature Environment (aquifers)
        for aq_id, aquifer in self.aquifers.items():
            withdrawal = 0.0
            # Collect all the well withdrawals of a given aquifer
            withdrawal += sum([well.withdrawal if well.aquifer_id == aq_id \
                               else 0.0 for _, well in self.wells.items()])
            #withdrawal += sum([well.withdrawal if well.aquifer_id==aq_id else 0 \
               #                for _, well in self.wells.items()])
            # Update aquifer
            aquifer.step(withdrawal)

        # Collect df_sys and print info
        self.datacollector.collect(self)
        if self.lema and current_year == self.lema_year and self.show_step:
            print("LEMA begin")
        if self.show_step:
            print(f"Year {self.current_year} [{self.t}/{self.total_steps}]"
                  +f"\t{self.time_recorder.get_elapsed_time()}\n")

        # Termination criteria
        if self.current_year == self.end_year:
            self.running = False
            print("Done!", f"\t{self.time_recorder.get_elapsed_time()}")
    
    def end(self):
        """Depose the Gurobi environment, ensuring that it is executed only when
        the instance is no longer needed."""
        self.gpenv.dispose()
        
    @staticmethod
    def get_dfs(model):
        df = model.datacollector.get_agent_vars_dataframe().reset_index()
        df["year"] = df["Step"] + model.init_year
        df.index = df["year"]

        # =============================================================================
        # df_agt
        # =============================================================================
        df_agt = pd.DataFrame()

        df_behaviors = df[df["agt_type"]=="Behavior"].dropna(axis=1, how='all')
        df_behaviors['bid'] = df_behaviors['AgentID']
        df_agt = pd.concat([df_agt, df_behaviors[['bid', 'Sa', 'E[Sa]', 'Un', 'state', 'profit',
               'revenue', 'energy_cost', 'gp_status', 'gp_MIPGap']]], axis=1)

        df_fields = df[df["agt_type"]=="Field"].dropna(axis=1, how='all')
        df_fields['field_type'] = np.nan
        df_fields.loc[df_fields['irr_vol'] == 0, 'field_type'] = 'rainfed'
        df_fields.loc[df_fields['irr_vol'] > 0, 'field_type'] = 'irrigated'
        df_fields['irr_depth'] = df_fields['irr_vol']/df_fields['field_area']*100 # cm
        df_fields['fid'] = df_fields['AgentID']
        df_agt = pd.concat([df_agt, df_fields[['fid', 'field_type', 'crop', 'irr_vol',
               'yield_rate', 'irr_depth', 'w', 'field_area']]], axis=1)

        df_wells = df[df["agt_type"]=="Well"].dropna(axis=1, how='all')
        df_wells['wid'] = df_wells['AgentID']
        df_agt = pd.concat([df_agt, df_wells[['wid', 'water_depth', 'withdrawal']]], axis=1)

        df_agt = df_agt.round(4)

        # =============================================================================
        # df_sys
        # =============================================================================
        df_sys = pd.DataFrame()

        # Aquifer
        df_aquifers = df[df["agt_type"]=="Aquifer"].dropna(axis=1, how='all')
        df_sys['GW_st'] = df_aquifers['GW_st']
        df_sys['withdrawal'] = df_aquifers['withdrawal']

        # Field_Type ratio
        dff = df_agt[['field_type', "field_area"]].groupby([df_agt.index, 'field_type']).sum()
        all_years = dff.index.get_level_values('year').unique()
        all_field_types = ['irrigated', 'rainfed']
        new_index = pd.MultiIndex.from_product([all_years, all_field_types], names=['year', 'field_type'])
        dff = dff.reindex(new_index).fillna(0)
        df_sys['rainfed'] = dff.xs('rainfed', level='field_type')/dff.groupby('year').sum()

        # Crop type ratio 
        dff = df_agt[['crop', "field_area"]].groupby([df_agt.index, 'crop']).sum()
        all_years = dff.index.get_level_values('year').unique()
        all_crop_types = model.crop_options
        new_index = pd.MultiIndex.from_product([all_years, all_crop_types], names=['year', 'crop'])
        dff = dff.reindex(new_index).fillna(0)
        total = dff.groupby('year').sum()
        for c in all_crop_types:
            df_sys[f'{c}'] = dff.xs(c, level='crop')/total

        # Behavioral agent state ratio
        dff = df_behaviors[['state']].groupby([df_behaviors.index, 'state']).size()
        all_years = dff.index.get_level_values('year').unique()
        all_states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]
        new_index = pd.MultiIndex.from_product([all_years, all_states], names=['year', 'state'])
        dff = dff.reindex(new_index).fillna(0)
        for s in all_states:
            df_sys[f'{s}'] = dff.xs(s, level='state')
        df_sys = df_sys.round(4)

        return df_sys, df_agt
    
    @staticmethod
    def get_metrices(df_sys, data, 
                     targets=['GW_st', 'withdrawal', 'rainfed']
                         +["corn", "others"],
                     indicators_list=['r', 'rmse', "KGE"]):
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
        indicators = Indicator()
        metrices = []
        for tar in targets:
            metrices.append(indicators.cal_indicator_df(
                x_obv=data[tar], y_sim=df_sys[tar], index_name=tar,
                indicators_list=indicators_list))
        metrices = pd.concat(metrices)
        return metrices

class SD6Model_input_templates():
    def __init__(self):
        self.prec_aw_dict = {
            "<prec_aw1>": {
                "<year>": {
                    "<crop1>": "[cm]",
                    "<crop2>": "[cm]",
                    }
                }
            }
        self.aquifers_dict = {
            "<aquifer1>": {
                "aq_a": "[1e-4 m^-2]",
                "aq_b": "m",
                "area": "1e4 m^2",
                "sy": "[-]",
                "init": {
                    "st": "m",
                    "dwl": "m"
                    }
                }
            }

        self.fields_dict = {
            "<field1>": {
                "field_area": "1e4 m^2",
                "water_yield_curves": {       
                    # If min_yield_ratio is not given, default is zero.
                    "crop1":     ["<ymax [bu]>", "<wmax [cm]>", "<a>", "<b>", "<c>", "optional <min_yield_ratio>"],
                    "crop2":     ["<ymax [bu]>", "<wmax [cm]>", "<a>", "<b>", "<c>", "optional <min_yield_ratio>"],
                    },
                "prec_aw_id": "<prec_aw1>",
                "init":{
                    "crop": "<crop> or a list of <crop> for each area split",
                    "field_type": "<'optimize' or 'irrigated' or 'rainfed'>",  
                    }
                }
            }

        self.wells_dict = {
            "<well1>": {
                "rho": "[kg/m3]",   
                "g": "[m/s2]", 
                "eff_pump": "[-]",
                "aquifer_id": "aquifers_dict's <unique_id>",
                "pumping_capacity": "[1e4 m^3]",
                "init":{
                    "l_wt": "[m]",
                    "B": "[m]",
                    "pumping_days": "[days]" 
                    },
                }
            }

        self.finances_dict = {
            "<finance1>": {
                "energy_price": "[1e4$/PJ]",
                "crop_price": {
                    "<crop1>": "[$/bu]",    
                    "<crop2>": "[$/bu]",    
                    },
                "crop_cost": { 
                    "<crop1>": "[$/bu]",    
                    "<crop2>": "[$/bu]",   
                    },
                "irr_tech_operational_cost": { 
                    "<tech1>": "[1e4$]",
                    "<tech2>": "[1e4$]"
                    }
                }
            }

        self.behaviors_dict = {
            "<behavior1>": {
                "field_ids": ["field1"],
                "well_ids": ["well1"],
                "finance_id": "<finance1>",
                "behavior_ids_in_network": ["<behavior2>", "<behavior3>"],
                "decision_making": {
                    "target": "profit",
                    "horizon": "[year]",
                    "n_dwl": "[year]",
                    "keep_gp_model": False,
                    "keep_gp_output": False,
                    "display_summary": False,
                    "display_report": False
                    },
                "water_rights": {
                    "<water_right1>": {
                        "wr_depth": "[cm]",
                        "time_window": "[year]",
                        "remaining_tw": "[year]",
                        "remaining_wr": "[cm]",
                        "tail_method": "<'proportion' or 'all' or float>",
                        "status": True
                        }
                    },
                "consumat": {
                    # [0-1] Sensitivity factor for the "satisfication" calculation.
                    "alpha": {"profit": 1},
                    # Normalize "need" for "satisfication" calculation.
                    "scale": {"profit": 0.23 * 50} # Use corn 1e4$*bu*ha
                    },
                }
            }
    


    
    
    
    
    
    
    
    