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


class Finance4SingleFieldAndWell(mesa.Agent):
    """ Simulate the financial aspect of the model. """
    
    def __init__(self, unique_id, model, settings: dict):
        """
        Initialize a Finance agent in the Mesa model.
        """
        super().__init__(unique_id, model)
        self.agt_type = "Finance"

        self.load_settings(settings)
        
        self.cost_energy = None
        self.cost_tech = None
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
        cost_energy = e * self.energy_price  # 1e4$
        
        cp = {k: v - self.crop_cost[k] for k, v in self.crop_price.items()}
        rev = sum([y[j,:] * cp[c] for j, c in enumerate(crop_options)])[0]
        profit = rev - cost_energy - cost_tech #- tech_change_cost - crop_change_cost
        self.y = y # (n_c, 1) [1e4 bu] of all fields
        self.rev = rev
        self.cost_energy = cost_energy
        self.cost_tech = cost_tech
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
        """Initialize a Finance agent in the Mesa model."""
        super().__init__(unique_id, model)
        self.agt_type = "Finance"

        self.load_settings(settings)

        self.cost_e = None
        self.cost_tech = None
        self.profit = None
        # premium database for all crops and field types assuming a single field.
        self.premium_dict = self.premium_dict_for_dm = {
            "irrigated": {c: None for c in self.model.crop_options},
            "rainfed": {c: None for c in self.model.crop_options},
        }
        # the total premium for all fields of the selected crop and field type.
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
        self.crop_cost = settings["crop_cost"]
        self.irr_tech_operational_cost = settings["irr_tech_operational_cost"]

        if self.model.activate_ci:
            self.harvest_price = settings.get("harvest_price")
            self.projected_price = settings.get("projected_price")
            self.aph_revenue_based_coef = settings.get("aph_revenue_based_coef")
        # If ratios are not provided, default 1 will be used.
        self.payout_ratio = settings.get("payout_ratio", 1)
        self.premium_ratio = settings.get("premium_ratio", 1)

    @staticmethod
    def cal_APH_revenue_based_premium(
        df,
        crop,
        county,
        field_type,
        aph_yield_dict,
        projected_price,
        premium_ratio=1,
        coverage_level=0.75,
    ):
        """Calculate the premium for a given crop, county, and field type.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the crop insurance data.
        crop : str
            The crop type.
        county : str
            The county name.
        field_type : str
            The field type.
        aph_yield_dict : dict
            A dictionary of aph yield for each crop.
        projected_price : float
            The projected price.
        premium_ratio : float, optional
            The premium ratio, by default 1.
        coverage_level : float, optional
            The coverage level, by default 0.75.

        Returns
        -------
        float
            The premium calculated for this step [1e4 $].

        """
        # Calculate here but store in each individual field.

        if crop == "fallow":
            return 0

        mask = (
            (df["Crop"] == crop)
            & (df["County"] == county)
            & (df["Field Type"] == field_type)
        )
        dff = df.loc[mask, :].squeeze()

        # dynamic (mean of simulated data of a particular field or sd6 average)
        aph_yield = aph_yield_dict[field_type][crop]

        ref_yield = dff.loc["Reference Yield"]  # data 2011 (dynamic) 1e4 bu/field
        exponent = dff.loc[
            "Exponent Value"
        ]  # data 2011 (dynamic) county level => Malena check how to calculate the value
        ref_rate = dff.loc["Reference Rate"]  # data 2011 (dynamic)
        fixed_rate = dff.loc["Fixed Rate"]  # data 2011 (dynamic)

        additional_coverage_rate = 0
        m_factor = 1
        designed_rate = 0
        coverage_level_rate_differential = dff.loc["Coverage Level Rate Diffrential"]

        def calc_continuous_rating_base_rate(
            aph_yield, ref_yield, exponent, ref_rate, fixed_rate, weight=1.2
        ):
            """Calculate the continuous rating base rate.

            Parameters
            ----------
            aph_yield : float
                The aph yield.
            ref_yield : float
                The reference yield.
            exponent : float
                The exponent value.
            ref_rate : float
                The reference rate.
            fixed_rate : float
                The fixed rate.
            weight : float, optional
                The weight, by default 1.2.

            Returns
            -------
            float
                The continuous rating base rate.
            """
            yield_ratio = round(aph_yield / ref_yield, 2)
            yield_ratio = min(max(yield_ratio, 0.50), 1.50)
            continuous_rating_base_rate = (
                (yield_ratio**exponent * ref_rate) + fixed_rate
            ) * weight
            return round(continuous_rating_base_rate, 8)

        weigthed_current_continuous_rating_base_rate = calc_continuous_rating_base_rate(
            aph_yield, ref_yield, exponent, ref_rate, fixed_rate, weight=1.2
        )

        current_continuous_rating_base_rate = calc_continuous_rating_base_rate(
            aph_yield, ref_yield, exponent, ref_rate, fixed_rate, weight=1
        )

        weigthed_pre_continuous_rating_base_rate = calc_continuous_rating_base_rate(
            aph_yield, ref_yield, exponent, ref_rate, fixed_rate, weight=1.2
        )

        preliminary_base_rate = min(
            current_continuous_rating_base_rate,
            weigthed_current_continuous_rating_base_rate,
            weigthed_pre_continuous_rating_base_rate,
        )

        # round 8 (ignore to avoid discrete in opt)
        adjusted_based_rate = max(
            (preliminary_base_rate + additional_coverage_rate) * m_factor, designed_rate
        )

        # round 8 (ignore to avoid discrete in opt)
        base_premium_rate = min(
            adjusted_based_rate * coverage_level_rate_differential, 0.999
        )

        # the unit is dependent on the aph_yield unit (1e4 bu/field)
        # Projected price's unit is $/bu
        premium = base_premium_rate * projected_price * coverage_level * aph_yield
        # premium (1e4 $/field)

        return premium * premium_ratio

    def cal_APH_revenue_based_payout(
        self, harvest_price, projected_price, aph_yield, yield_, coverage_level=0.75
    ):
        payout = max(
            max(harvest_price, projected_price) * coverage_level * aph_yield
            - harvest_price * yield_,
            0,
        )
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
        self.t += 1

        # Compute total yield and energy use
        y = sum([field.y for _, field in fields.items()])  # 1e4 bu
        e = sum([well.e for _, well in wells.items()])  # PJ

        #!!!! Operational cost only happen when the irrigation amount is not zero.
        cost_tech = sum(
            [
                self.irr_tech_operational_cost["center pivot LEPA"]
                if field.irr_vol_per_field > 0
                else 0
                for _, field in fields.items()
            ]
        )
        crop_options = self.model.crop_options

        # Calculate energy cost and profit
        cost_e = e * self.energy_price  # 1e4$

        cp = {k: v - self.crop_cost[k] for k, v in self.crop_price.items()}
        rev = sum([y[j, :] * cp[c] for j, c in enumerate(crop_options)])[0]

        # Crop insurance
        field = fields[next(iter(fields.keys()))]  # assuming only one field
        if self.model.activate_ci:
            # Calculate premium for all possible cases for model diagnosis
            for field_type in ["irrigated", "rainfed"]:
                for crop in crop_options:
                    self.premium_dict[field_type][
                        crop
                    ] = self.cal_APH_revenue_based_premium(
                        df=self.aph_revenue_based_coef,
                        crop=crop,
                        county=field.county,
                        field_type=field_type,
                        aph_yield_dict=field.aph_yield_dict,
                        projected_price=self.projected_price[crop],
                        premium_ratio=self.premium_ratio,
                        coverage_level=0.75,
                    )

            # Calculate premium for the selected crop and field type
            if field.irr_vol_per_field > 0:
                field_type = "irrigated"
            else:
                field_type = "rainfed"
            crop = field.crop
            premium = self.premium_dict[field_type][crop]

            # Calculate payout for the selected crop and field type
            payout = self.cal_APH_revenue_based_payout(
                self.harvest_price[crop],
                self.projected_price[crop],
                field.aph_yield_dict[field_type][crop],
                y[crop_options.index(crop), 0],
                coverage_level=0.75,
            )

            # trigger aph_yield_dict update in field
            field.update_aph_yield(field_type, y[crop_options.index(crop), 0])
        else:
            payout = 0
            premium = 0

        profit = payout + rev - cost_e - cost_tech - premium
        self.y = y  # (n_c, 1) [1e4 bu] of all fields
        self.rev = rev
        self.cost_e = cost_e
        self.cost_tech = cost_tech
        self.profit = profit
        self.premium = premium
        self.payout = payout

        return profit
