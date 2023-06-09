r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

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
    crop_options : list, optional
        A list of crop type options. They must exist in the config. The
        default is ["corn", "sorghum", "soybean", "fallow"].

    """
    def __init__(self, config):
        config = DotMap(config)
        self.cf = config.finance

    def step(self, fields, wells):
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
        self.profit = profit
        return profit

    # def sim_step_old(self, e, y):
    #     """
    #     Simulate a single timestep.

    #     Parameters
    #     ----------
    #     e : float
    #         Total energy consumption [PJ].
    #     y : 3darray
    #         Crop yield with the dimension (n_s, n_c, 1) [1e4 bu].

    #     Returns
    #     -------
    #     profit : float
    #         Annual profit (Million).

    #     """
    #     ep = self.energy_price          # [1e6$/PJ]
    #     cp = self.crop_profit           # [$/bu]
    #     crop_options = self.crop_options

    #     cost_e = e * ep     # 1e6$
    #     rev = sum([y[i,j,:] * cp[c] * 1e-2 for i in range(y.shape[0]) \
    #                for j, c in enumerate(crop_options)])    # 1e6$
    #     profit = rev - cost_e
    #     profit = profit[0]
    #     self.profit = profit    # 1e6$
    #     return profit