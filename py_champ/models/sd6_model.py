import numpy as np
import pandas as pd
from tqdm import tqdm
import mesa
# import py_champ
from ..util import TimeRecorder, Box, Indicator
from ..entities.aquifer import Aquifer
from ..entities.behavior import Behavior
from ..entities.field import Field
from ..entities.well import Well
from ..entities.finance import Finance

class BaseSchedulerByTypeFiltered(mesa.time.BaseScheduler):
    """
    A scheduler that overrides the step method to allow for filtering
    of agents by .agt_type.

    Example:
    >>> scheduler = BaseSchedulerByTypeFiltered(model)
    >>> scheduler.step(agt_type="Farmer")
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
class SD6Model(mesa.Model):
    def __init__(self, pars, config,
                 fields_dict, wells_dict, agts_dict, aquifers_dict,
                 prec_aw_step, init_year=2007, end_year=2022, lema_year=2013, 
                 fix_state=None, lema=True, show_step=True,
                 crop_options=None, tech_options=None,
                 seed=None, **kwargs):
        """
        The main model class for the SD6 simulation.
    
        Parameters
        ----------
        pars : dict
            Dictionary containing parameters used for calibration.
        config : dict
            Configuration dictionary.
        fields_dict : dict
            Information about the fields in the model.
        wells_dict : dict
            Information about the wells in the model.
        agts_dict : dict
            Information about the agents in the model.
        aquifers_dict : dict
            Information about the aquifers in the model.
        prec_aw_step : dict
            Time-series data on precipitation and available water.
        crop_options : list, optional
            List of available crop options.
        tech_options : list, optional
            List of available technology options.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional keyword arguments.
            'crop_price_step': crop_price_step
            'crop_type_step': crop_type_step
            'field_type_step': field_type_step
            'irr_depth_step': irr_depth_step
            'block_w_interval_for_corn': [43, 57]
            'lema_wr_name': name
        
        """
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
        self.lema_year      = lema_year
            
        self.lema = lema
        self.lema_wr_name = kwargs.get("lema_wr_name", 'wr_LEMA_5yr')
        self.show_step = show_step

        # mesa has self.random.seed(seed) but it is not usable for truncnorm
        # We create our own random generator.
        self.seed = seed
        self.rngen = np.random.default_rng(seed)

        # Input timestep data
        self.prec_aw_step = prec_aw_step       # Available prec for each crop
        self.crop_price_step = kwargs.get("crop_price_step")
        self.crop_type_step = kwargs.get("crop_type_step")
        self.irr_depth_step = kwargs.get("irr_depth_step")
        self.field_type_step = kwargs.get("field_type_step")
        
        # Options
        self.crop_options = crop_options
        self.tech_options = tech_options

        # load model layer pars (for calibration)
        self.pars = pars

        # Update crop profit (Note we consider income only. The cost are ignored)
        self.config = config
        self.config['consumat']['satisfaction_threshold'] = self.pars['sa_thre']
        self.config['consumat']['uncertainty_threshold'] = self.pars['un_thre']
        if self.crop_price_step is not None:
            self.config['finance']['crop_price'] = self.crop_price_step[self.current_year]

        # Create schedule and assign it to the model
        #self.schedule = mesa.time.BaseScheduler(self)
        self.schedule = BaseSchedulerByTypeFiltered(self)
        self.num_agents = len(agts_dict)
        #self.grid = mesa.space.ContinuousSpace(x_max=-100.43, y_max=39.48, x_min=-100.78, y_min=39.30, torus=False)
        self.grid = mesa.space.MultiGrid(36, 24, torus=False)
        
        # Initialize aquifer environment (this is not associated with farmers)
        aquifers = {}
        for aqid, aquifer in aquifers_dict.items():
            agt_aquifer = Aquifer(
                unique_id=aqid,
                mesa_model=self,
                aq_a=aquifer["aq_a"],
                aq_b=aquifer["aq_b"],
                ini_st=aquifer["init"]["st"],
                ini_dwl=aquifer["init"]["dwl"])
            aquifers[aqid] = agt_aquifer
            self.schedule.add(agt_aquifer)
        self.aquifers = aquifers
        self.Aquifers = Box(aquifers)  # to access aquifer in dot way

        # Initialize fields
        fields = {}
        for fid, field in fields_dict.items():
            # Initialize irrigation technology    
            init_tech = field['init']['tech']
            if isinstance(init_tech, float):
                rn = self.rngen.uniform(0, 1)
                if rn < init_tech:
                    init_tech = "center pivot"
                else:
                    init_tech = "center pivot LEPA"
                    
            # Initialize crop type   
            init_crop = field['init']['crop']
            if isinstance(init_crop, list):
                init_crop = self.rngen.choice(init_crop)
            
            # Initialize fields
            agt_field = Field(
                unique_id=fid,
                mesa_model=self,
                config=self.config,
                ini_crop=init_crop,
                ini_te=init_tech,
                ini_field_type=field.get('field_type', 'optimize'),
                crop_options=self.crop_options,
                tech_options=self.tech_options,
                block_w_interval_for_corn=kwargs.get("block_w_interval_for_corn"),
                # For this model's convenience
                prec_aw_step=prec_aw_step[fid],
                irr_freq=field.get('irr_freq'),
                truncated_normal_pars=field.get('truncated_normal_pars'),
                lat=field.get("lat"),
                lon=field.get("lon"),
                x=field.get("x"),
                y=field.get("y"),
                gridmet_id=field.get("gridmet_id"),
                regen=self.rngen
                )
            fields[fid] = agt_field
            self.schedule.add(agt_field)
            self.grid.place_agent(agt_field, (agt_field.x, agt_field.y))
            #coords = (self.random.randrange(0, 10), self.random.randrange(0, 10))
            #self.grid.place_agent(agt_field, coords)
        self.fields = fields

        # Initialize wells
        wells = {}
        for wid, well in wells_dict.items():
            agt_well = Well(
                unique_id=wid,
                mesa_model=self,
                config=self.config,
                r=well['r'],
                k=well['k'],
                sy=well['sy'],
                ini_st=well['init']['st'],
                ini_l_wt=well['init']['l_wt'],
                ini_pumping_days=well['init']['pumping_days'],
                eff_pump=well['eff_pump'],
                eff_well=well['eff_well'],
                aquifer_id=well['aquifer_id'],
                pumping_capacity=well['pumping_capacity']  # as kwargs
                )
            wells[wid] = agt_well
            self.schedule.add(agt_well)
        self.wells = wells

        # Initialize agents (farmers) and append to the schedule
        ## Don't use parallelization. It is slower!
        finances = {}
        self.max_num_fields_per_agt = 0
        self.max_num_wells_per_agt = 0
        for agtid, agt_dict in tqdm(agts_dict.items(), desc="Initialize agents"):
            finances[agtid] = Finance(unique_id=agtid, mesa_model=self, config=self.config)
            # Assign threshold
            agt_farmer = Behavior(
                unique_id=agtid,
                mesa_model=self,
                config=self.config,
                agt_attrs=agt_dict,
                fields={f"f{i+1}_": fields[fid] for i, fid in enumerate(agt_dict['fids'])},
                wells={f"w{i+1}_": wells[wid] for i, wid in enumerate(agt_dict['wids'])},
                finance=finances[agtid],
                aquifers=self.aquifers,
                ini_year=self.init_year,
                crop_options=self.crop_options,
                tech_options=self.tech_options,
                # kwargs
                perceived_risk=self.pars['perceived_risk'], 
                fids=agt_dict['fids'],     
                wids=agt_dict['wids'],     
                rngen=self.rngen,   
                fix_state=fix_state
                )
            agt_farmer.process_percieved_risks(pars['perceived_risk'])
            self.schedule.add(agt_farmer)
            self.max_num_fields_per_agt = max(
                self.max_num_fields_per_agt, len(agt_dict['fids']))
            self.max_num_wells_per_agt = max(
                self.max_num_wells_per_agt, len(agt_dict['wids']))
            
        self.finances = finances

        # Assign agents in a agent's social network to an agent
        for _, agt in self.schedule._agents.items():
            if agt.agt_type != "Farmer":
                continue
            agt.agts_in_network = {
                agtid: self.schedule._agents[agtid]
                    for agtid in agt.agt_ids_in_network
                }

        
        def get_nested_attr(obj, attr_str):
            attrs = attr_str.split('.', 1)
            current_attr = getattr(obj, attrs[0], None)
            if len(attrs) == 1 or current_attr is None:
                return current_attr
            return get_nested_attr(current_attr, attrs[1])
        
        
        def get_agt_attr(attr_str):
            # This replaces lambda a: getattr(a, "satisfaction", None)
            # We have to do this to return None if the attribute is not exist 
            # in the given agent type.
            # def func(agent):
            #     return getattr(agent, attr, None)
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
            # Field
            "field_type":       get_agt_attr("field_type"),
            "crop":             get_agt_attr("crops"),
            "tech":             get_agt_attr("te"),
            "w":                get_agt_attr("w"),
            "irr_vol_per_field":    get_agt_attr("irr_vol_per_field"),
            "yield_rate_per_field": get_agt_attr("yield_rate_per_field"),
            # Farmer
            "Sa":               get_agt_attr("satisfaction"),
            "E[Sa]":            get_agt_attr("expected_sa"),
            "Un":               get_agt_attr("uncertainty"),
            "state":            get_agt_attr("state"),
            "yield_rate":       get_agt_attr("yield_rate"),             # [0, 1]
            "profit":           get_agt_attr("profit"),                 # 1e4 $
            "profit_per_field": get_agt_attr("avg_profit_per_field"),   # 1e4 $
            "revenue":          get_agt_attr("finance.rev"),            # 1e4 $
            "energy_cost":      get_agt_attr("finance.cost_e"),         # 1e4 $
            "tech_cost":        get_agt_attr("finance.tech_cost"),      # 1e4 $
            "irr_vol":          get_agt_attr("irr_vol"),                # m-ha
            "gp_status":        get_agt_attr("gp_status"),
            "gp_MIPGap":        get_agt_attr("gp_MIPGap"),
            "num_fields":       get_agt_attr("num_fields"),
            "num_wells":        get_agt_attr("num_wells"),
            # Well
            "water_depth":      get_agt_attr("l_wt"),
            # Aquifer
            "withdrawal":       get_agt_attr("withdrawal"),             # m-ha
            "GW_st":            get_agt_attr("st"),                     # m
            "GW_dwl":           get_agt_attr("dwl"),                    # m
            #"yield":            get_agt_attr("finance.y"),            # 1e4 bu
            #"scaled_yield_pct": get_agt_attr("scaled_yield_pct"),
            #"scaled_profit":    get_agt_attr("scaled_profit"),
            #"perceived_prec_aw":get_agt_attr("perceived_prec_aw")
            #"lema_remaining_wr": "lema_remaining_wr",
            # "gp_report": "gp_report",
            # "cost_tech": "finance.cost_tech",
            # "tech_change_cost": "finance.tech_change_cost",
            # "crop_change_cost": "finance.crop_change_cost",
            }
        model_reporters = {}
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,
            #exclude_none_values=True
        )

        # Summary info
        estimated_sim_dur = self.time_recorder.sec2str(
            self.time_recorder.get_elapsed_time(strf=False)*self.total_steps)
        msg = f"""\n
        Initial year: \t{self.init_year}
        Simulation period:\t{self.start_year} to {self.end_year}
        Number of agents:\t{len(agts_dict)}
        Number of aquifers:\t{len(aquifers_dict)}
        Initialiation duration:\t{self.time_recorder.get_elapsed_time()}
        Estimated sim duration:\t{estimated_sim_dur}
        """
        print(msg)

        # Note:
        ## We do not record the df_sys in the initial step.

    def step(self):
        """Advance the model by one step."""

        self.current_year += 1
        self.t += 1

        ##### Human (farmers)
        current_year = self.current_year

        # Update crop price input
        if self.crop_price_step is not None:    # Manually update the price
            self.config["finance"]['crop_price'] = self.crop_price_step[current_year]

        for agtid, agt in self.schedule._agents.items():
            if agt.agt_type != "Farmer":
                continue
            
            agt.update_perceived_prec_aw(
                par_forecast_trust=self.pars['forecast_trust'],
                year=current_year
                )
            # Update crop price input
            agt.load_config(self.config)            # opt
            agt.finance.load_config(self.config)    # sim

            # Randomly select rainfed field
            for fid_, field in agt.fields.items():
                irr_depth_step = self.irr_depth_step
                field_type_step = self.field_type_step
                rn_irr = True
                if irr_depth_step is not None:
                    if irr_depth_step.get(current_year) is not None:
                        rn_irr = False
                        if irr_depth_step[current_year][agtid] == 0:
                            field.field_type = 'rainfed'      # Force the field to be rainfed
                        else:
                            field.field_type = 'irrigated'    # Force the field to be irrigated  
                if field_type_step is not None:
                    rn_irr = False
                    if pd.isna(field_type_step[current_year]):
                        irr_freq = field.irr_freq
                    else:
                        irr_freq = 1-field_type_step[current_year]
                    rn = self.rngen.uniform(0, 1)
                    if rn <= irr_freq:
                        field.field_type = 'optimize' # Optimize it
                    else:
                        field.field_type = 'rainfed'  # Force the field to be rainfed
                if rn_irr:
                    irr_freq = field.irr_freq
                    rn = self.rngen.uniform(0, 1)
                    #rn = 0
                    if rn <= irr_freq:
                        field.field_type = 'optimize' # Optimize it
                    else:
                        #raise
                        field.field_type = 'rainfed'  # Force the field to be rainfed
                
                # Assign crop decisions if given
                if self.crop_type_step is not None:
                    crop = self.crop_type_step[current_year][agtid]
                    i_crop = np.zeros((1, len(self.crop_options), 1))
                    i_c = self.crop_options.index(crop)
                    i_crop[:, i_c, 0] = 1
                    agt.dm_sols[fid_]['i_crop'] = i_crop
                    
            # Turn on LEMA (i.e., a water right constraint) starting from 2013
            if self.lema and current_year >= self.lema_year:
                agt.water_rights[self.lema_wr_name]['status'] = True
            else:
                agt.water_rights[self.lema_wr_name]['status'] = False
                
        # Exercute step() of all agents in a for loop
        self.schedule.step(agt_type="Farmer")    # Parallelization makes it slower!

        ##### Nature Environment (aquifers)
        for aq_id, aq in self.aquifers.items():
            withdrawal = 0
            # Collect all the well withdrawals of a given aquifer
            withdrawal += sum([well.withdrawal if well.aquifer_id==aq_id else 0 \
                               for _, well in self.wells.items()])
            # Update aquifer
            aq.step(withdrawal)

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

    @staticmethod
    def get_dfs(model):
        df = model.datacollector.get_agent_vars_dataframe().reset_index()
        df["year"] = df["Step"] + model.init_year
        df.index = df["year"]
        field_area = model.config["field"]["field_area"]    # ha
        df_fields = df[df["agt_type"]=="Field"].dropna(axis=1, how='all')
        df_fields['field_type'] = np.nan
        df_fields.loc[df_fields['irr_vol_per_field'] == 0, 'field_type'] = 'rainfed'
        df_fields.loc[df_fields['irr_vol_per_field'] > 0, 'field_type'] = 'irrigated'
        df_fields['crop'] = [i[0] for i in df_fields['crop']] # assume 1 split
        df_fields['irr_depth_per_field'] = df_fields['irr_vol_per_field']/field_area*100 # cm
        df_wells = df[df["agt_type"]=="Well"].dropna(axis=1, how='all')
        df_aquifers = df[df["agt_type"]=="Aquifer"].dropna(axis=1, how='all')
        df_farmers = df[df["agt_type"]=="Farmer"].dropna(axis=1, how='all')
        df_farmers["irr_depth"] = df_farmers["irr_vol"]/(field_area*df_farmers["num_fields"])*100 # cm
        return df_farmers, df_fields, df_wells, df_aquifers
    
    @staticmethod
    def get_df_sys(model, df_farmers, df_fields, df_wells, df_aquifers):
        df_sys = pd.DataFrame()
        # Aquifer
        df_sys['GW_st'] = df_aquifers['GW_st']
        df_sys['withdrawal'] = df_aquifers['withdrawal']
        # Field_Type ratio
        dff = df_fields[['field_type']].groupby([df_fields.index, 'field_type']).size()
        all_years = dff.index.get_level_values('year').unique()
        all_field_types = ['irrigated', 'rainfed']
        new_index = pd.MultiIndex.from_product([all_years, all_field_types], names=['year', 'field_type'])
        dff = dff.reindex(new_index).fillna(0)
        df_sys['rainfed'] = dff.xs('rainfed', level='field_type')/dff.groupby('year').sum()
        # Crop type ratio
        dff = df_fields[['crop']].groupby([df_fields.index, 'crop']).size()
        all_years = dff.index.get_level_values('year').unique()
        all_crop_types = model.crop_options
        new_index = pd.MultiIndex.from_product([all_years, all_crop_types], names=['year', 'crop'])
        dff = dff.reindex(new_index).fillna(0)
        total = dff.groupby('year').sum()
        for c in all_crop_types:
            df_sys[f'{c}'] = dff.xs(c, level='crop')/total
        # Agent state ratio
        dff = df_farmers[['state']].groupby([df_farmers.index, 'state']).size()
        all_years = dff.index.get_level_values('year').unique()
        all_states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]
        new_index = pd.MultiIndex.from_product([all_years, all_states], names=['year', 'state'])
        dff = dff.reindex(new_index).fillna(0)
        for s in all_states:
            df_sys[f'{s}'] = dff.xs(s, level='state')
        return df_sys
    
    @staticmethod
    def get_metrices(df_sys, data, 
                     targets=['GW_st', 'withdrawal', 'rainfed']
                         +["corn", "sorghum", "soybeans", "wheat", "fallow"],
                         #+["center pivot LEPA"], 
                     indicators_list=['r', 'rmse', "KGE"]):
        indicators = Indicator()
        metrices = []
        for tar in targets:
            metrices.append(indicators.cal_indicator_df(
                x_obv=data[tar], y_sim=df_sys[tar], index_name=tar,
                indicators_list=indicators_list))
        metrices = pd.concat(metrices)
        return metrices
# =============================================================================
#     Add plots
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
cmap = plt.get_cmap("tab10")

class SD6Plots():
        
    @staticmethod
    def plot_crop_ratio(df_sys, data, metrices, crop_options, df_sys_nolema=None, savefig=None):
        ##### Crop ratio
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(5.5, 4), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, crop in enumerate(crop_options):
            ax = axes[i]
            x = np.arange(2008, 2023)
            metrice = metrices[["r", 'rmse', "KGE"]].T.round(2)[crop].to_string()
            ax.plot(x, df_sys[crop], c=cmap(i))
            if df_sys_nolema is not None:
                ax.plot(x, df_sys_nolema[crop], c=cmap(i), ls=":")
            ax.plot(x, data[crop], c="k", ls="--")
            ax.set_title(crop.capitalize())
            ax.legend(title=metrice, title_fontsize=8, edgecolor="white", framealpha=0)
            ax.set_xlim([2008, 2022])
            ax.set_ylim([0, 1])
        fig.delaxes(axes[-1])
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Ratio\n", fontsize=12)
        line_obv = Line2D([0], [0], label='Obv', c='k', ls='--')
        line_sim = Line2D([0], [0], label='Sim', c='k', ls='-')
        line_sim_no = Line2D([0], [0], label='Sim\n(no LEMA)', c='k', ls=':')
        plt.legend(handles=[line_obv, line_sim, line_sim_no], loc="lower right")
        plt.tight_layout()
        
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()
        
    @staticmethod
    def plot_gwrc(df_sys, data, metrices, df_sys_nolema=None, stochastic=[], savefig=None):
        ##### st & withdrawal
        fig, axes = plt.subplots(ncols=1, nrows=4, figsize=(5.5, 6), sharex=True, sharey=False)
        axes = axes.flatten()
        x = np.arange(2008, 2023)
    
        
        for i, v in enumerate(['GW_st', 'withdrawal', "rainfed", "corn"]):
            ax = axes[i]
            ax.plot(x, df_sys[v], c=cmap(i+5))
            if df_sys_nolema is not None:
                ax.plot(x, df_sys_nolema[v], c=cmap(i+5), ls=":")
            ax.plot(x, data[v], c="k", ls="--")
            for df in stochastic:
                ax.plot(x, df[v], c=cmap(i+5), alpha=0.3, lw=0.8)
            metrice = metrices[["r", 'rmse', "KGE"]].T.round(2)[v].to_string()
            ax.legend(title=metrice, title_fontsize=9, edgecolor="white", framealpha=0,
                      loc="lower left")
            ax.axvline(2012.5, c="grey", ls=":")
            ax.axvline(2017.5, c="grey", ls=":")
            ax.set_xlim([2008, 2022])
            ylabels = ["Saturated\nthickness (m)", "Withdrawal\n(m-ha)", "Rainfed\nratio", "Corn\nratio"]
            ax.set_ylabel(ylabels[i], fontsize=12)
        fig.align_ylabels(axes)
    
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Year", fontsize=12)
        line_obv = Line2D([0], [0], label='Obv', c='k', ls='--')
        line_sim = Line2D([0], [0], label='Sim', c='k', ls='-')
        line_sim_no = Line2D([0], [0], label='Sim\n(no LEMA)', c='k', ls=':')
        plt.legend(handles=[line_obv, line_sim, line_sim_no], loc="upper right")
        plt.tight_layout()
        
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()
          
    # @staticmethod
    # def plot_tech_rainfed_ratio(df_sys, data, metrices, savefig=None):
    #     ##### Tech & Rainfed
    #     fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(5.5, 4), sharex=True, sharey=False)
    #     axes = axes.flatten()
    #     x = np.arange(2008, 2023)
    
    #     for i, v in enumerate(['center pivot LEPA', 'rainfed']):
    #         ax = axes[i]
    #         ax.plot(x, df_sys[v], c=cmap(i+7))
    #         ax.plot(x, data[v], c="k", ls="--")
    #         metrice = metrices[["r", 'rmse', "KGE"]].T.round(2)[v].to_string()
    #         if i == 0:
    #             ax.legend(title=metrice, title_fontsize=9, edgecolor="white", framealpha=0,
    #                       loc="lower left")
    #         else:
    #             ax.legend(title=metrice, title_fontsize=9, edgecolor="white", framealpha=0,
    #                       loc="upper left")
    #         ax.axvline(2012.5, c="grey", ls=":")
    #         ax.axvline(2017.5, c="grey", ls=":")
    #         ax.set_xlim([2008, 2022])
    #         ax.set_ylim([0, 1])
    #         ylabels = ["center pivot\nLEPA ratio", "Rainfed\nfield ratio"]
    #         ax.set_ylabel(ylabels[i], fontsize=12)
    #     fig.align_ylabels(axes)
    
    #     fig.add_subplot(111, frameon=False)
    #     plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    #     plt.xlabel("Year", fontsize=12)
    #     line_obv = Line2D([0], [0], label='Obv', c='k', ls='--')
    #     line_sim = Line2D([0], [0], label='Sim', c='k', ls='-')
    #     plt.legend(handles=[line_obv, line_sim], loc="lower right", fontsize=9)
    #     plt.tight_layout()
        
    #     if savefig is not None:
    #         plt.savefig(savefig)
    #     plt.show()

    @staticmethod
    def plot_prec(prec_avg, savefig=None):
        ##### Prec
        fig, axes = plt.subplots(ncols=1, nrows=5, figsize=(5.5, 5), sharex=True, sharey=True)
        axes = axes.flatten()
        x = np.arange(2008, 2023)
    
        for i, v in enumerate(["annual", "corn", "sorghum", "soybeans", "wheat"]):
            ax = axes[i]
            if v == "annual":
                ax.plot(x, prec_avg[v], c="k", ls="-")
            else:
                ax.plot(x, prec_avg[v], c=cmap(i-1))
            ax.axvline(2012.5, c="grey", ls=":")
            ax.axvline(2017.5, c="grey", ls=":")
            ax.set_xlim([2008, 2022])
            ax.set_ylabel(v, fontsize=12)
        fig.align_ylabels(axes)
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Precipitation (cm)\n", fontsize=12)
        plt.tight_layout()
        
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()
        
    @staticmethod
    def plot_agt_status(df_sys, prec=None, savefig=None):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        x = list(df_sys.index)
        states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]
        dff = df_sys[states]
        num_agt = sum(dff.iloc[0, :])
        dff.plot(ax=ax, zorder=20)
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([0, num_agt])
        ax.set_xlabel("Year")
        ax.set_ylabel("Agent count")
        if prec is not None:
            ax2 = ax.twinx()
            ax2.bar(x, prec, alpha=0.5, zorder=10)
            ax2.set_ylim([0, max(prec)*5])
            ax2.set_ylabel("Annual precipitation (cm)")
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()

    @staticmethod
    def plot_agt_violin(df_agts, y="irr_depth", savefig=None):
        fig, ax = plt.subplots()
        sns.violinplot(data=df_agts, x=df_agts.index, y=y, ax=ax, cut=0)
        plt.xticks(rotation=45, ha='right')
        ax.axvline(2012.5-2008, c="red", ls="--")
        ax.axvline(2017.5-2008, c="red", ls="--")
        
        if y=="irr_depth":
            ax.axhline(35.56, c="k", ls="--")
            ax.axhline(27.94, c="k", ls=":")
            ax.axhline(0, c="k", ls="--")
            
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()

    @staticmethod
    def plot_SaUn(df_agts, un='Un', sa="Sa", year="year", thres_un=None, thres_sa=None, savefig=None):
    # Create the figure and axis objects
        if "year" not in df_agts:
            df_agts["year"] = df_agts.index
        fig, ax = plt.subplots()
        
        # Split the data into the specified ranges
        df1 = df_agts[(df_agts[year] >= 2008) & (df_agts[year] <= 2012)]
        df2 = df_agts[(df_agts[year] >= 2013) & (df_agts[year] <= 2017)]
        df3 = df_agts[(df_agts[year] >= 2018) & (df_agts[year] <= 2022)]
        
        # Plot each range with a different colormap
        ms=3; vmin = 2005; vmax = 2025
        sc0 = ax.scatter([], [], c=[], cmap='Greys', vmin=2008, vmax=2022, zorder=0, s=ms)
        sc1 = ax.scatter(df1[un], df1[sa], c=df1[year], cmap='Reds', vmin=vmin, vmax=vmax, zorder=10, s=ms)
        sc2 = ax.scatter(df2[un], df2[sa], c=df2[year], cmap='Blues', vmin=vmin, vmax=vmax, zorder=11, s=ms)
        sc3 = ax.scatter(df3[un], df3[sa], c=df3[year], cmap='Greens', vmin=vmin, vmax=vmax, zorder=12, s=ms)
        
        if thres_un is not None:
            ax.axvline(thres_un, color="k", zorder=0, lw=1)
        
        if thres_sa is not None:
            ax.axhline(thres_sa, color="k", zorder=0, lw=1)
        
        # Add colorbar and labels
        cbar = fig.colorbar(sc0, ax=ax, label='Year')
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Satisfication')
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='2008-2012', markersize=10, markerfacecolor='pink'),
            Line2D([0], [0], marker='o', color='w', label='2013-2017', markersize=10, markerfacecolor='lightblue'),
            Line2D([0], [0], marker='o', color='w', label='2018-2022', markersize=10, markerfacecolor='green')
            ]

        ax.legend(handles=legend_elements, loc='upper right', ncols=3,
                  bbox_to_anchor=(1.02, 1.1), fontsize=9)
        
        # Add text labels at the four corners
        x_min, x_max = ax.get_xlim() 
        y_min, y_max = ax.get_ylim()
        
        ax.text(x_min, y_min, 'Deliberation', verticalalignment='bottom', horizontalalignment='left', fontsize=9)
        ax.text(x_min, y_max, 'Repetition', verticalalignment='top', horizontalalignment='left', fontsize=9)
        ax.text(x_max, y_min, 'Social comparison', verticalalignment='bottom', horizontalalignment='right', fontsize=9)
        ax.text(x_max, y_max, 'Imitation', verticalalignment='top', horizontalalignment='right', fontsize=9)
        
        if savefig is not None:
            plt.savefig(savefig)
            
        plt.show()
    
    
r"""
def get_df_sys(model):
    # Not yet generalized
    crop_options = model.crop_options
    tech_options = model.tech_options

    dc = model.datacollector
    dc.model_vars

    # for a farmer with one field and one well only!!
    df_model = dc.get_model_vars_dataframe()
    df_model = df_model.set_index("year")
    df_agts = dc.get_agent_vars_dataframe().reset_index()
    df_agts["irr_depth"]= df_agts["irr_vol"] / 50 * 100  #!!! cm
    df_agts["yield"]    = [np.sum(y) for y in df_agts["yield"]]
    df_agts["crop_1"]   = [c[0] for c in df_agts["crop_1"]]
    df_agts["rainfed"]  = [1 if irr==0 else 0 for irr in df_agts["irr_vol"]]
    df_agts["ratio"]    = 1
    df_agts["year"] = df_agts["Step"] + model.init_year

    #!!!!
    df_agts = df_agts.drop("perceived_prec_aw", axis=1)

    years = np.arange(model.start_year, model.end_year+1)
    states = ["Imitation", "Social comparison", "Repetition", "Deliberation"]

    df_state = df_agts.groupby(["state", "year"]).count().reindex(
        [(s, y) for s in states for y in years], fill_value=0
        ).reset_index().pivot(index='year', columns='state', values='ratio')
    #df_model = pd.concat([df_model, df_state], axis=1)

    df_sys = pd.DataFrame(index=years)
    df_sys["GW_st"] = df_model["GW_st"]
    df_sys["withdrawal"] = df_model["withdrawal"]

    tech_ratio = df_agts.groupby(["tech_1", "year"]).count()[["ratio"]] / df_agts.groupby(["year"]).sum()[["ratio"]]
    tech_ratio = tech_ratio.reindex([(t, y) for t in tech_options for y in years], fill_value=0).reset_index()
    df_sys = pd.concat([df_sys, tech_ratio.pivot(index='year', columns='tech_1', values='ratio')], axis=1)

    crop_ratio = df_agts.groupby(["crop_1", "year"]).count()[["ratio"]] / df_agts.groupby(["year"]).sum()[["ratio"]]
    crop_ratio = crop_ratio.reindex([(c, y) for c in crop_options for y in years], fill_value=0).reset_index()
    df_sys = pd.concat([df_sys, crop_ratio.pivot(index='year', columns='crop_1', values='ratio')], axis=1)

    rainfed_ratio = df_agts.groupby(["rainfed", "year"]).count()[["ratio"]] / df_agts.groupby(["year"]).sum()[["ratio"]]
    rainfed_ratio = rainfed_ratio.reindex([(r, y) for r in [0, 1] for y in years], fill_value=0)
    df_sys["rainfed"] = rainfed_ratio.xs(1, level="rainfed")
    df_sys = pd.concat([df_sys, df_state], axis=1)

    #df_sys.index = [y for y in range(model.start_year, model.end_year+1)]
    df_agts.index = df_agts["year"]
    return df_sys, df_model, df_agts
"""
    
    
    
    
    
    
    
    
    
    
    
    