r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Sep 6, 2023
"""
import numpy as np

class Well():
    """
    A simulator for a groundwater well.
    
    This class simulates energy consumption and other attributes of a well based
    on various parameters like well radius, hydraulic conductivity, and more.

    Attributes
    ----------
    well_id : str or int
        Unique identifier for the well.
    config : dict
        General configuration information for the model.
    r : float
        Radius of the well in meters.
    k : float
        Hydraulic conductivity in m/day. Used to calculate transmissivity as 
        T = k * saturated_thickness.
    st : float
        Saturated thickness of the aquifer in meters.
    sy : float
        Specific yield of the aquifer.
    l_wt : float
        Initial lift from the water table to the ground surface at the start of
        the pumping season in meters.
    eff_pump : float, optional
        Pumping efficiency. Default is 0.77.
    eff_well : float, optional
        Well efficiency. Default is 0.5.
    aquifer_id : Union[str, int], optional
        Identifier for the associated aquifer. Default is None.
    """

    def __init__(self, well_id, config, r, k, st, sy, l_wt,
                 eff_pump=0.77, eff_well=0.5, aquifer_id=None, **kwargs):
        """
        Initialize a Well object.

        Parameters are set as attributes. Additional keyword arguments can be 
        passed to set other attributes.

        Parameters
        ----------
        ...
        Same as class attributes.
        """
        #super().__init__(agt_id, mesa_model)
        # MESA required attributes
        self.unique_id = well_id
        self.agt_type = "Well"

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
        Load well-related configurations from the model's general configuration.

        Parameters
        ----------
        config : dict
            General configuration information for the model.

        Returns
        -------
        None.

        """
        config_well = config["well"]
        self.rho = config_well["rho"]
        self.g = config_well["g"]

    def step(self, withdrawal, dwl, pumping_rate, l_pr):
        """
        Simulate the well for a single time step.

        Parameters
        ----------
        withdrawal : float
            Volume of irrigation water to be withdrawn from this well in m-ha.
        dwl : float
            Change in groundwater level in meters.
        pumping_rate : float
            Average daily pumping rate in m-ha.
        l_pr : float
            The effective lift due to pressurization and of water and pipe
            losses necessary for the allocated irrigation system

        Returns
        -------
        float
            Energy consumption for the step in PJ (PetaJoules).
        """
        self.t +=1

        # Update saturated thickness and water table lift based on groundwater
        # level change
        self.l_wt -= dwl
        self.st += dwl
        self.tr = self.st * self.k # Update Transmissivity
        self.withdrawal = withdrawal
        l_wt = self.l_wt

        r, tr, sy = self.r, self.tr, self.sy
        eff_well, eff_pump = self.eff_well, self.eff_pump
        rho, g = self.rho, self.g

        # Calculate energy consumption
        m_ha_2_m3 = 10000
        fpitr = 4 * np.pi * tr
        l_cd_l_wd = (1+eff_well) * pumping_rate/fpitr \
                    * (-0.5772 - np.log(r**2*sy/fpitr)) * m_ha_2_m3
        l_t = l_wt + l_cd_l_wd + l_pr
        e = rho * g * m_ha_2_m3 / eff_pump / 1e15 * withdrawal * l_t     # PJ

        # Record energy consumption
        self.e = e

        return e