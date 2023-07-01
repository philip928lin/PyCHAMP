r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Jun 9, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""
import numpy as np

class Finance():
    """
    A finance simulator.

    Attributes
    ----------
    config : dict or DotMap
        General info of the model.

    """
    def __init__(self, config):

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
        Load config.

        Parameters
        ----------
        config : dict
            General configuration of the model.

        Returns
        -------
        None.

        """
        self.config_finance = config["finance"]

    def step(self, fields, wells):
        """
        Calculate the profit of the current step.

        Parameters
        ----------
        fields : dict
            A dictionary contains field objects.
        wells : dict
            A dictionary contains well objects.

        Returns
        -------
        profit : float
            [1e4$].

        """
        self.t +=1

        # Calulate profit and pumping cost
        y = sum([field.y for _, field in fields.items()])   # 1e4 bu
        e = sum([well.e for _, well in wells.items()])      # PJ

        cf = self.config_finance
        cost_tech = sum([cf["irr_tech_operational_cost"][field.te] \
                         for _, field in fields.items()])

        tech_change_cost = 0
        crop_change_cost = 0
        for _, field in fields.items():

            # Tech cost
            key = (field.pre_te, field.te)
            ctc = cf["irr_tech_change_cost"].get(key)
            if ctc is None: ctc = 0     # Default value
            tech_change_cost += ctc

            # Crop cost
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
                    ccc = cf["crop_change_cost"].get(key)
                    if ccc is None: ccc = 0     # Default value
                    crop_change_cost += ccc

        cost_e = e * cf["energy_price"]  # 1e4$
        cp = cf["crop_profit"]
        # Assume crop_options are the same accross fields.
        rev = sum([y[i,j,:] * cp[c] for i in range(y.shape[0]) \
                   for j, c in enumerate(crop_options)])[0]
        profit = rev - cost_e - cost_tech - tech_change_cost - crop_change_cost
        self.y = y # !!! not generalizable Total yield (n_s, n_c, 1) [1e4 bu]
        self.rev = rev
        self.cost_e = cost_e
        self.cost_tech = cost_tech
        self.tech_change_cost = tech_change_cost
        self.crop_change_cost = crop_change_cost
        self.profit = profit

        return profit