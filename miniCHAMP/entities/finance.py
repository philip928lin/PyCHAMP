r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""

from dotmap import DotMap

class Finance():
    """
    An finance simulator.

    Attributes
    ----------
    config : dict or DotMap
        General info of the model.
    crop_options : list, optional
        A list of crop type options. They must exist in the config. The
        default is ["corn", "sorghum", "soybean", "fallow"].

    """
    def __init__(self, config,
                 crop_options=["corn", "sorghum", "soybean", "fallow"]):
        config = DotMap(config)
        self.energy_price = config.finance.energy_price
        self.crop_profit = config.finance.crop_profit
        self.crop_options = crop_options

    def sim_step(self, e, y):
        """
        Simulate a single timestep.

        Parameters
        ----------
        e : float
            Total energy consumption.
        y : 3darray
            Crop yield with the dimension (n_s, n_c, 1).

        Returns
        -------
        profit : float
            Annual profit.

        """
        ep = self.energy_price
        cp = self.crop_profit
        crop_options = self.crop_options

        cost_e = e * ep
        rev = sum([y[i,j,:] * cp[c] for i in range(y.shape[0]) \
                   for j, c in enumerate(crop_options)])
        profit = rev - cost_e
        profit = profit[0]
        self.profit = profit
        return profit