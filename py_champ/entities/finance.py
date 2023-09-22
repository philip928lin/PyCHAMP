r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Sep 6, 2023
"""
import numpy as np

class Finance():
    """
    A class to simulate the financial aspects of an agricultural system.

    Attributes
    ----------
    config_finance : dict
        Financial parameters read from the config, such as costs and prices.
    cost_e : float
        Energy cost for the current step in units of 1e4$.
    cost_tech : float
        Technology operational cost for the current step in units of 1e4$.
    tech_change_cost : float
        Cost incurred from changing technology for the current step in units of 1e4$.
    crop_change_cost : float
        Cost incurred from changing crop types for the current step in units of 1e4$.
    profit : float
        Calculated profit for the current step in units of 1e4$.
    y : float
        Total crop yield for the current step in units of 1e4 bu (bushels).
    t : int
        Current time step.
    """

    def __init__(self, config):
        """
        Initialize a Finance object.

        Parameters
        ----------
        config : dict
            General configuration information for the model.
        """
        self.load_config(config)
        
        self.cost_e = None
        self.cost_tech = None
        self.tech_change_cost = None
        self.crop_change_cost = None
        self.profit = None
        self.y = None
        self.t = 0

    def load_config(self, config):
        """
        Load a given configuration.

        Parameters
        ----------
        config : dict
            General configuration information for the model.
        """
        self.config_finance = config["finance"]

    def step(self, fields, wells):
        """
        Compute the financial metrics for the current time step.

        Parameters
        ----------
        fields : dict
            A dictionary containing field objects with attributes like crop yield.
        wells : dict
            A dictionary containing well objects with attributes like energy use.

        Returns
        -------
        float
            Calculated profit for the current step in units of 1e4$.

        Notes
        -----
        Assumes that all fields have the same crop choice options and that crop 
        costs and prices are specified in config for all relevant crops and 
        technologies.

        Attributes Modified
        -------------------
        cost_e : float
            Updated energy cost.
        cost_tech : float
            Updated technology operational cost.
        tech_change_cost : float
            Updated technology change cost.
        crop_change_cost : float
            Updated crop change cost.
        profit : float
            Updated profit.
        y : float
            Updated total crop yield.
        t : int
            Updated time step.
        """
        
        self.t +=1

        # Compute total yield and energy use
        y = sum([field.y for _, field in fields.items()])   # 1e4 bu
        e = sum([well.e for _, well in wells.items()])      # PJ

        cf = self.config_finance
        # Operational cost only happen when the irrigation amount is not zero.
        cost_tech = sum([cf["irr_tech_operational_cost"][field.te] \
                         if field.irr_vol_per_field > 0 else 0 \
                             for _, field in fields.items()])
        
        # Loop over fields to calculate technology and crop change costs
        tech_change_cost = 0
        crop_change_cost = 0
        for _, field in fields.items():

            # Calculate technology change cost
            key = (field.pre_te, field.te)
            tech_change_cost += cf["irr_tech_change_cost"].get(key, 0)

            # Calculate crop change cost
            # Assume crop_options are the same accross fields.
            crop_options = field.crop_options
            i_crop = field.i_crop
            pre_i_crop = field.pre_i_crop

            cc = (i_crop - pre_i_crop)[:,:,0]
            for s in range(cc.shape[0]):
                ccc = cc[s, :]
                fr = np.argmin(ccc) # ccc == -1
                to = np.argmax(ccc) # ccc == 1
                if fr.size != 0 & to.size != 0:
                    key = (crop_options[fr], crop_options[to])
                    crop_change_cost += cf["crop_change_cost"].get(key, 0)
        
        # Calculate energy cost and profit
        cost_e = e * cf["energy_price"]  # 1e4$
        
        cp = {k: v - cf["crop_cost"][k] for k, v in cf["crop_price"].items()}
        # Assume crop_options are the same accross fields.
        rev = sum([y[i,j,:] * cp[c] for i in range(y.shape[0]) \
                   for j, c in enumerate(crop_options)])[0]
        profit = rev - cost_e - cost_tech - tech_change_cost - crop_change_cost
        self.y = y # (n_s, n_c, 1) [1e4 bu] of all fields
        self.rev = rev
        self.cost_e = cost_e
        self.cost_tech = cost_tech
        self.tech_change_cost = tech_change_cost
        self.crop_change_cost = crop_change_cost
        self.profit = profit

        return profit