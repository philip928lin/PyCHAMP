r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Jun 9, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""
import numpy as np
from dotmap import DotMap

class Finance():
    """
    A finance simulator.

    Attributes
    ----------
    config : dict or DotMap
        General info of the model.

    """
    def __init__(self, config):
        config = DotMap(config)
        self.cf = config.finance

        self.cost_e = None
        self.cost_tech = None
        self.tech_change_cost = None
        self.crop_change_cost = None
        self.profit = None

    def step(self, fields, wells):
        """
        Calculate the profit of the current step.

        Parameters
        ----------
        fields : DotMap
            A dictionary stored as a DotMap object that contains field objects.
        wells : DotMap
            A dictionary stored as a DotMap object that contains well objects.

        Returns
        -------
        profit : float
            [1e4$].

        """
        # Calulate profit and pumping cost
        y = sum([field.y for f, field in fields.items()])
        e = sum([well.e for w, well in wells.items()])

        cf = self.cf
        cost_tech = sum([cf.irr_tech_operational_cost[field.te] for f, field in fields.items()])

        tech_change_cost = 0
        crop_change_cost = 0
        for f, field in fields.items():
            # Assume crop_options are the same accross fields.
            crop_options = field.crop_options

            key = (field.pre_te, field.te)
            ctc = cf.irr_tech_change_cost.get(key)
            if ctc is None: ctc = 0
            tech_change_cost += ctc


            i_crop = field.i_crop
            pre_i_crop = field.pre_i_crop

            cc = (i_crop - pre_i_crop)[:,:,0]
            for s in range(cc.shape[0]):
                ccc = cc[s, :]
                fr = np.where(ccc == -1)[0]
                to = np.where(ccc == 1)[0]
                if fr.size != 0 & to.size != 0:
                    key = (crop_options[fr], crop_options[to])
                    ccc = cf.crop_change_cost.get(key)
                    if ccc is None: ccc = 0
                    crop_change_cost += ccc

        cost_e = e * cf.energy_price  # 1e4$
        cp = cf.crop_profit
        # Assume crop_options are the same accross fields.
        rev = sum([y[i,j,:] * cp[c] for i in range(y.shape[0]) \
                   for j, c in enumerate(crop_options)])[0]
        profit = rev - cost_e - cost_tech - tech_change_cost - crop_change_cost
        self.cost_e = cost_e
        self.cost_tech = cost_tech
        self.tech_change_cost = tech_change_cost
        self.crop_change_cost = crop_change_cost
        self.profit = profit
        return profit