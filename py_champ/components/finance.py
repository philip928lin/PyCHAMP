# The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
# Email: chungyi@vt.edu
# Last modified on Dec 30, 2023
import mesa
import numpy as np


class Finance(mesa.Agent):
    """
    This module is a finance simulator.

    Parameters
    ----------
    unique_id : int
        A unique identifier for this agent.
    model
        The model instance to which this agent belongs.
    settings : dict
        A dictionary containing financial settings, which include energy prices,
        crop price, and crop cost, irrigation operational cost,
        irr_tech_change_cost, and crop_change_cost.

        - 'energy_price': The price of energy [1e4 $/PJ].
        - 'crop_price' and 'crop_cost': The price and cost of different crops [$/bu].
        - 'irr_tech_operational_cost': Operational costs for different irrigation technologies [1e4 $].
        - 'irr_tech_change_cost': Costs associated with changing irrigation technologies [1e4 $].
        - 'crop_change_cost': Costs associated with changing crop types [1e4 $].

        >>> # A sample settings dictionary
        >>> settings = {
        >>>     "energy_price": 2777.78,
        >>>     "crop_price": {
        >>>         "corn":     5.39,
        >>>         "sorghum":  6.59,
        >>>         "soybeans": 13.31,
        >>>         "wheat":    8.28,
        >>>         "fallow":   0.
        >>>         },
        >>>     "crop_cost": {
        >>>         "corn":     0,
        >>>         "sorghum":  0,
        >>>         "soybeans": 0,
        >>>         "wheat":    0,
        >>>         "fallow":   0.
        >>>         },
        >>>     "irr_tech_operational_cost": {
        >>>         "center pivot":         1.87,
        >>>         "center pivot LEPA":    1.87
        >>>         },
        >>>     "irr_tech_change_cost": { # If not specified, 0 is the default.
        >>>         ("center pivot", "center pivot LEPA"): 0
        >>>         },
        >>>     "crop_change_cost": { # If not specified, 0 is the default.
        >>>         ("corn", "sorghum"):     0
        >>>         }
        >>>     }

    Attributes
    ----------
    agt_type : str
        The type of the agent, set to 'Finance'.
    profit : float or None
        The profit, initialized to None [1e4 $].
    y : float or None
        The total yield, initialized to None [1e4 bu].
    t : int
        The current time step, initialized to zero.

    """

    def __init__(self, unique_id, model, settings: dict):
        """Initialize a Finance agent in the Mesa model."""
        super().__init__(unique_id, model)
        self.agt_type = "Finance"

        self.load_settings(settings)

        self.cost_e = None
        self.cost_tech = None
        self.tech_change_cost = None
        self.crop_change_cost = None
        self.profit = None
        self.y = None
        self.t = 0

    def load_settings(self, settings: dict):
        """
        Load the financial settings from a dictionary.

        Parameters
        ----------
        settings : dict
            A dictionary containing financial settings. Expected keys include
            'energy_price', 'crop_price', 'crop_cost',
            'irr_tech_operational_cost', 'irr_tech_change_cost',
            and 'crop_change_cost'.

        """
        self.finance_dict = settings

        self.energy_price = settings["energy_price"]
        self.crop_price = settings["crop_price"]
        self.crop_cost = settings["crop_cost"]
        self.irr_tech_operational_cost = settings["irr_tech_operational_cost"]
        self.irr_tech_change_cost = settings["irr_tech_change_cost"]
        self.crop_change_cost = settings["crop_change_cost"]

    def step(self, fields: dict, wells: dict) -> float:
        """
        Perform a single step of financial calculations.

        Parameters
        ----------
        fields : dict
            A dictionary of Field agents with their unique_id as keys.
        wells : dict
            A dictionary of Well agents with their unique_id as keys.

        Returns
        -------
        float
            The profit calculated for this step [1e4 $].

        Notes
        -----
        This method calculates the total yield, energy usage, operational costs,
        technology change costs, crop change costs, energy costs, and the total profit.
        The profit is calculated as revenue minus all associated costs.
        """
        self.t += 1

        # Compute total yield and energy use
        y = sum([field.y for _, field in fields.items()])  # 1e4 bu
        e = sum([well.e for _, well in wells.items()])  # PJ

        # Operational cost only happen when the irrigation amount is not zero.
        cost_tech = sum(
            [
                self.irr_tech_operational_cost[field.te]
                if field.irr_vol_per_field > 0
                else 0
                for _, field in fields.items()
            ]
        )

        # Loop over fields to calculate technology and crop change costs
        tech_change_cost = 0
        crop_change_cost = 0
        for _, field in fields.items():
            # Calculate technology change cost
            key = (field.pre_te, field.te)
            tech_change_cost += self.irr_tech_change_cost.get(key, 0)

            # Calculate crop change cost
            # Assume crop_options are the same accross fields.
            crop_options = self.model.crop_options
            i_crop = field.i_crop
            pre_i_crop = field.pre_i_crop

            cc = (i_crop - pre_i_crop)[:, :, 0]
            for s in range(cc.shape[0]):
                ccc = cc[s, :]
                fr = np.argmin(ccc)  # ccc == -1
                to = np.argmax(ccc)  # ccc == 1
                if fr.size != 0 & to.size != 0:
                    key = (crop_options[fr], crop_options[to])
                    crop_change_cost += self.crop_change_cost.get(key, 0)

        # Calculate energy cost and profit
        cost_e = e * self.energy_price  # 1e4$

        cp = {k: v - self.crop_cost[k] for k, v in self.crop_price.items()}
        # Assume crop_options are the same accross fields.
        rev = sum(
            [
                y[i, j, :] * cp[c]
                for i in range(y.shape[0])
                for j, c in enumerate(crop_options)
            ]
        )[0]
        profit = rev - cost_e - cost_tech - tech_change_cost - crop_change_cost
        self.y = y  # (n_s, n_c, 1) [1e4 bu] of all fields
        self.rev = rev
        self.cost_e = cost_e
        self.cost_tech = cost_tech
        self.tech_change_cost = tech_change_cost
        self.crop_change_cost = crop_change_cost
        self.profit = profit

        return profit

class Finance_1f1w_ci(mesa.Agent):
    """
    This module is a finance simulator.

    Parameters
    ----------
    unique_id : int
        A unique identifier for this agent.
    model
        The model instance to which this agent belongs.
    settings : dict
        A dictionary containing financial settings, which include energy prices, 
        crop price, and crop cost, irrigation operational cost, 
        irr_tech_change_cost, and crop_change_cost.

        - 'energy_price': The price of energy [1e4 $/PJ].
        - 'crop_price' and 'crop_cost': The price and cost of different crops [$/bu].
        - 'irr_tech_operational_cost': Operational costs for different irrigation technologies [1e4 $].
        - 'irr_tech_change_cost': Costs associated with changing irrigation technologies [1e4 $].
        - 'crop_change_cost': Costs associated with changing crop types [1e4 $].
        
        >>> # A sample settings dictionary
        >>> settings = {
        >>>     "energy_price": 2777.78,
        >>>     "crop_price": { 
        >>>         "corn":     5.39,      
        >>>         "sorghum":  6.59,    
        >>>         "soybeans": 13.31,     
        >>>         "wheat":    8.28,
        >>>         "fallow":   0.
        >>>         },
        >>>     "crop_cost": { 
        >>>         "corn":     0,      
        >>>         "sorghum":  0,
        >>>         "soybeans": 0,
        >>>         "wheat":    0,
        >>>         "fallow":   0.
        >>>         },
        >>>     "irr_tech_operational_cost": { 
        >>>         "center pivot":         1.87,
        >>>         "center pivot LEPA":    1.87
        >>>         },
        >>>     "irr_tech_change_cost": { # If not specified, 0 is the default.
        >>>         ("center pivot", "center pivot LEPA"): 0
        >>>         },
        >>>     "crop_change_cost": { # If not specified, 0 is the default.
        >>>         ("corn", "sorghum"):     0
        >>>         }
        >>>     }
        
    Attributes
    ----------
    agt_type : str
        The type of the agent, set to 'Finance'.
    profit : float or None
        The profit, initialized to None [1e4 $].
    y : float or None
        The total yield, initialized to None [1e4 bu].
    t : int
        The current time step, initialized to zero.

    """
    
    def __init__(self, unique_id, model, settings: dict):
        """
        Initialize a Finance agent in the Mesa model.
        """
        super().__init__(unique_id, model)
        self.agt_type = "Finance"

        self.load_settings(settings)
        
        self.cost_e = None
        self.cost_tech = None
        self.profit = None
        self.premium = None
        self.payout = None
        self.y = None
        self.t = 0
    
    def load_settings(self, settings: dict):
        """
        Load the financial settings from a dictionary.
    
        Parameters
        ----------
        settings : dict
            A dictionary containing financial settings. Expected keys include 
            'energy_price', 'crop_price', 'crop_cost',
            'irr_tech_operational_cost', 'irr_tech_change_cost', 
            and 'crop_change_cost'.
        
        """
        self.finance_dict = settings

        self.energy_price = settings["energy_price"]
        self.crop_price = settings["crop_price"]
        self.harvest_price = settings.get("harvest_price")
        self.projected_price = settings.get("projected_price")
        self.aph_revenue_based_coef = settings.get("aph_revenue_based_coef")
        
        self.crop_cost = settings["crop_cost"]
        self.irr_tech_operational_cost = settings["irr_tech_operational_cost"]
        
        self.payout_ratio = settings.get("payout_ratio", 1)
        self.premium_ratio = settings.get("premium_ratio", 1)

    @staticmethod
    def cal_APH_revenue_based_premium(df, crop, county, field_type, aph_yield_dict, projected_price, premium_ratio=1, coverage_level=0.75, field_area=50):
        # Calculate here but store in each individual field.
        
        if crop == "fallow":
            return 0
        
        mask = (df["Crop"]==crop) & (df["County"]==county) & (df["Field Type"]==field_type)
        dff = df.loc[mask, :].squeeze() 
        
        # dynamic (mean of simulated data of a particular field or sd6 average)
        aph_yield = aph_yield_dict[field_type][crop]

        ref_yield = dff.loc["Reference Yield"]     # data 2011 (dynamic) 1e4 bu/field
        exponent = dff.loc["Exponent Value"]      # data 2011 (dynamic) county level => Malena check how to calculate the value
        ref_rate = dff.loc["Reference Rate"]      # data 2011 (dynamic)
        fixed_rate = dff.loc["Fixed Rate"]    # data 2011 (dynamic)

        additional_coverage_rate = 0
        m_factor = 1
        designed_rate = 0
        coverage_level_rate_differential = dff.loc["Coverage Level Rate Diffrential"]


        def calc_continuous_rating_base_rate(aph_yield, ref_yield, exponent, ref_rate, fixed_rate, weight=1.2):
            yield_ratio = round(aph_yield / ref_yield, 2)
            yield_ratio = min(max(yield_ratio, 0.50), 1.50)
            continuous_rating_base_rate = ((yield_ratio**exponent * ref_rate) + fixed_rate) * weight
            return round(continuous_rating_base_rate, 8)
        
        weigthed_current_continuous_rating_base_rate = calc_continuous_rating_base_rate(aph_yield, ref_yield, exponent, ref_rate, fixed_rate, weight=1.2)
        
        current_continuous_rating_base_rate = calc_continuous_rating_base_rate(aph_yield, ref_yield, exponent, ref_rate, fixed_rate, weight=1)
            
        weigthed_pre_continuous_rating_base_rate = calc_continuous_rating_base_rate(aph_yield, ref_yield, exponent, ref_rate, fixed_rate, weight=1.2)
            
        preliminary_base_rate = min(current_continuous_rating_base_rate, weigthed_current_continuous_rating_base_rate, weigthed_pre_continuous_rating_base_rate)
        
        # round 8 (ignore to avoid discrete in opt)
        adjusted_based_rate = max((preliminary_base_rate + additional_coverage_rate) * m_factor, designed_rate)
        
        # round 8 (ignore to avoid discrete in opt)
        base_premium_rate = min(adjusted_based_rate * coverage_level_rate_differential, 0.999)
        
        premium = base_premium_rate * projected_price * coverage_level * aph_yield
        
        return  premium * premium_ratio
    
    def cal_APH_revenue_based_payout(self, harvest_price, projected_price, aph_yield, yield_, coverage_level=0.75):
        payout = max(max(harvest_price, projected_price) * coverage_level * aph_yield - harvest_price * yield_, 0)
        return payout * self.payout_ratio
    
    def step(self, fields: dict, wells: dict) -> float:       
        """
        Perform a single step of financial calculations.
    
        Parameters
        ----------
        fields : dict
            A dictionary of Field agents with their unique_id as keys.
        wells : dict
            A dictionary of Well agents with their unique_id as keys.
    
        Returns
        -------
        float
            The profit calculated for this step [1e4 $].
    
        Notes
        -----
        This method calculates the total yield, energy usage, operational costs, 
        technology change costs, crop change costs, energy costs, and the total profit. 
        The profit is calculated as revenue minus all associated costs.
        """
        self.t +=1

        # Compute total yield and energy use
        y = sum([field.y for _, field in fields.items()])   # 1e4 bu
        e = sum([well.e for _, well in wells.items()])      # PJ

        #!!!! Operational cost only happen when the irrigation amount is not zero.
        cost_tech = sum([self.irr_tech_operational_cost["center pivot LEPA"] \
                         if field.irr_vol_per_field > 0 else 0 \
                             for _, field in fields.items()])
        crop_options = self.model.crop_options

        # Calculate energy cost and profit
        cost_e = e * self.energy_price  # 1e4$
        
        cp = {k: v - self.crop_cost[k] for k, v in self.crop_price.items()}
        rev = sum([y[j,:] * cp[c] for j, c in enumerate(crop_options)])[0]
        
        # Crop insurance
        aph_yield_dicts = [field.aph_yield_dict for _, field in fields.items()]
        premiums = [field.premium_dict["irrigated"][field.crop] if field.irr_vol_per_field > 0 else field.premium_dict["rainfed"][field.crop] for _, field in fields.items()]
        
        field = fields[list(fields.keys())[0]] # assuming on field
        if aph_yield_dicts[0] is not None: # assuming 1 field only
            if field.irr_vol_per_field > 0:
                field_type = "irrigated"
            else:
                field_type = "rainfed"
            payout = sum(
                [self.cal_APH_revenue_based_payout(
                    self.harvest_price[c], self.projected_price[c],
                    aph_yield_dicts[0][field_type][c], y[j,0], coverage_level=0.75) \
                        for j, c in enumerate(crop_options)])
            premium = premiums[0]
        else:
            payout = 0
            premium = 0
        
        profit = payout + rev - cost_e - cost_tech - premium #- tech_change_cost - crop_change_cost
        self.y = y # (n_s, n_c, 1) [1e4 bu] of all fields
        self.rev = rev
        self.cost_e = cost_e
        self.cost_tech = cost_tech
        self.profit = profit
        self.premium = premium
        self.payout = payout

        return profit