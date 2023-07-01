r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Jun 9, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""
import numpy as np

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
        Well radius [m].
    k : float
        Hydraulic conductivity [m/d]. This will be used to calculate
        transmissivity [m2/d] by multiply saturated thickness [m].
    st: float
        Aquifer saturated thickness [m].
    sy : float
        Specific yield.
    l_wt : float
        Initial head for the lift from the water table to the ground
        surface at the start of the pumping season [m].
    eff_pump : float
        Pump efficiency. The default is 0.77.
    eff_well : float
        Well efficiency. The default is 0.5.
    aquifer_id : str or int, optional
        Aquifer id. The default is None.

    """
    def __init__(self, well_id, config, r, k, st, sy, l_wt,
                 eff_pump=0.77, eff_well=0.5, aquifer_id=None, **kwargs):

        self.well_id, self.r, self.k, self.st, self.sy, self.l_wt = \
            well_id, r, k, st, sy, l_wt
        self.eff_pump, self.eff_well = eff_pump, eff_well
        self.aquifer_id = aquifer_id
        self.load_config(config)

        # Load other kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.tr = st * k    # Transmissivity

        self.t = 0

        # Container
        self.withdrawal = None # m-ha

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
        config_well = config["well"]
        self.rho = config_well["rho"]
        self.g = config_well["g"]

    def step(self, v, dwl, q, l_pr):
        """
        Simulate a single timestep.

        Parameters
        ----------
        v : float
            Irrigation amount that will be withdraw from this well [m-ha].
        dwl : float
            Groudwater level change [m].
        q : float
            Average daily pumping rate [m-ha].
        l_pr : float
            The effective lift due to pressurization and of water and pipe
            losses necessary for the allocated irrigation system [m].

        Returns
        -------
        e : float
            Energy consumption [PJ].

        """
        self.t +=1

        # update groundwater level change from the last year
        self.l_wt -= dwl
        self.st += dwl
        self.tr = self.st * self.k
        self.withdrawal = v
        l_wt = self.l_wt

        r, tr, sy = self.r, self.tr, self.sy
        eff_well, eff_pump = self.eff_well, self.eff_pump
        rho, g = self.rho, self.g

        m_ha_2_m3 = 10000
        fpitr = 4 * np.pi * tr
        l_cd_l_wd = (1+eff_well) * q/fpitr \
                    * (-0.5772 - np.log(r**2*sy/fpitr)) * m_ha_2_m3
        l_t = l_wt + l_cd_l_wd + l_pr
        e = rho * g * m_ha_2_m3 / eff_pump / 1e15 * v * l_t     # PJ

        # record
        self.e = e

        return e