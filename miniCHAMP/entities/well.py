r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""
import numpy as np
from dotmap import DotMap

class Well():
    """
    A well simulator.

    Attributes
    ----------
    well_id : str or int
        Well id.
    config : dict or DotMap
        General info of the model.
    r : float
        Well radius.
    tr : float
        Transmissivity.
    sy : float
        Specific yield.
    l_wt : float
        Initial head for the lift from the water table to the ground
        surface at the start of the pumping season.
    eff_pump : float
        Pump efficiency. The default is 0.77.
    eff_well : float
        Well efficiency. The default is 0.5.
    aquifer_id : str or int, optional
        Aquifer id. The default is None.

    """
    def __init__(self, well_id, config, r, tr, sy, l_wt,
                 eff_pump=0.77, eff_well=0.5, aquifer_id=None):
        # for name_, value_ in vars().items():
        #     if name_ != 'self' and name_ != 'config':
        #         setattr(self, name_, value_)
        self.well_id, self.r, self.tr, self.sy, self.l_wt = \
            well_id, r, tr, sy, l_wt
        self.eff_pump, self.eff_well = eff_pump, eff_well
        self.aquifer_id = aquifer_id

        config = DotMap(config)
        self.rho = config.well.rho
        self.g = config.well.g

        self.t = 0

    def sim_step(self, v, dwl, q, l_pr):
        """
        Simulate a single timestep.

        Parameters
        ----------
        v : float
            Irrigation amount that will be withdraw from this well.
        dwl : float
            Groudwater level change.
        q : float
            Average daily pumping rate.
        l_pr : float
            The effective lift due to pressurization and of water and pipe
            losses necessary for the allocated irrigation system.

        Returns
        -------
        e : float
            Energy consumption.

        """
        # update groundwater level change from the last year
        self.l_wt += dwl
        l_wt = self.l_wt

        r, tr, sy = self.r, self.tr, self.sy
        eff_well, eff_pump = self.eff_well, self.eff_pump
        rho, g = self.rho, self.g

        cm_ha_2_m3 = 1000
        fpitr = 4 * np.pi * tr
        l_cd_l_wd = (1+eff_well) * q/fpitr \
                    * (-0.5772 - np.log(r**2*sy/fpitr)) * cm_ha_2_m3
        l_t = l_wt + l_cd_l_wd + l_pr
        e = rho * g * v * l_t / eff_pump * cm_ha_2_m3

        # record
        self.t += 1
        e = e[0]
        self.e = e
        return e