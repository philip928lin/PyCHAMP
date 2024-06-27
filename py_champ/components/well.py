# The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
# Email: chungyi@vt.edu
# Last modified on Dec 30, 2023
import mesa
import numpy as np


class Well(mesa.Agent):
    """
    This module is a well simulator.

    Parameters
    ----------
    unique_id : int
        A unique identifier for this agent.
    model
        The model instance to which this agent belongs.
    settings : dict
        A dictionary containing settings specific to a well, such as
        hydraulic properties and initial conditions.

        - 'r': Radius of the well [m].
        - 'k': Hydraulic conductivity of the aquifer [m/day].
        - 'sy': Specific yield of the aquifer [-].
        - 'rho': Density of water [kg/m³].
        - 'g': Acceleration due to gravity [m/s²].
        - 'eff_pump': Pump efficiency as a fraction [-].
        - 'eff_well': Well efficiency as a fraction [-].
        - 'pumping_capacity': Maximum pumping capacity of the well [m-ha/year].
        - 'init': Initial conditions, which include water table lift (l_wt [m]), saturated thickness (st [m]) and pumping_days (days).

        >>> # A sample settings dictionary
        >>> settings = {
        >>>     "r": None,
        >>>     "k": None,
        >>>     "sy": None,
        >>>     "rho": None,
        >>>     "g": None,
        >>>     "eff_pump": None,
        >>>     "eff_well": None,
        >>>     "aquifer_id": None,
        >>>     "pumping_capacity": None,
        >>>     "init":{
        >>>         "l_wt": None,
        >>>         "st": None,
        >>>         "pumping_days": None
        >>>         },
        >>>     }

    **kwargs
        Additional keyword arguments that can be dynamically set as well agent attributes.

    Attributes
    ----------
    agt_type : str
        The type of the agent, set to 'Well'.
    st : float
        The saturated thickness of the aquifer at the well location [m].
    l_wt : float
        The lift of the water table from its initial position [m].
    pumping_days : int
        Number of days the well pumps water [day].
    tr : float
        The transmissivity of the aquifer at the well location [m²/day].
    t : int
        The current time step, initialized to zero.
    e : float or None
        The energy consumption [PJ], initialized to None.
    withdrawal : float or None
        The volume of water withdrawn in meter-hectares [m-ha].

    Notes
    -----
    - Transmissivity 'tr' is calculated as the product of saturated thickness and hydraulic conductivity.
    """

    def __init__(self, unique_id, model, settings: dict, **kwargs):
        """Initialize a Well agent in the Mesa model."""
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, model)
        self.agt_type = "Well"
        # Load kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.load_settings(settings)
        self.st = self.init["st"]
        self.l_wt = self.init["l_wt"]
        self.pumping_days = self.init["pumping_days"]
        self.tr = self.st * self.k  # Transmissivity

        # Some other attributes
        self.t = 0
        self.e = None
        self.withdrawal = None  # m-ha

    def load_settings(self, settings: dict):
        """
        Load the well settings from the dictionary.

        Parameters
        ----------
        settings : dict
            A dictionary containing well settings. Expected keys include 'r', 'k', 'sy', 'rho',
            'g', 'eff_pump', 'eff_well', 'aquifer_id', 'pumping_capacity', and 'init'.
        """
        self.r = settings["r"]
        self.k = settings["k"]
        self.sy = settings["sy"]
        self.rho = settings["rho"]
        self.g = settings["g"]
        self.eff_pump = settings["eff_pump"]
        self.eff_well = settings["eff_well"]
        self.aquifer_id = settings["aquifer_id"]
        self.pumping_capacity = settings["pumping_capacity"]
        self.init = settings["init"]

    def step(
        self,
        withdrawal: float,
        dwl: float,
        pumping_rate: float,
        l_pr: float,
        pumping_days: int | None = None,
    ) -> float:
        """
        Perform a single step of well simulation, calculating the energy consumption.

        Parameters
        ----------
        withdrawal : float
            The amount of water withdrawn in this step [m-ha].
        dwl : float
            The change in the water level due to withdrawal [m].
        pumping_rate : float
            The rate at which water is being pumped [m-ha/day].
        l_pr : float
            effective loss due to pressurization of water and losses in the piping,
            dependent on the type of the irrigation system [m].
        pumping_days : int, optional
            Number of days the well is operational. If not specified, previous
            value is used.

        Returns
        -------
        float
            The energy consumption for this step [Petajoules, PJ].

        Notes
        -----
        The method calculates energy consumption based on several factors including withdrawal volume,
        water table lift, well and pump efficiency, and hydraulic properties of the aquifer.
        """
        self.t += 1
        # Only update pumping_days when it is given.
        if pumping_days is not None:
            self.pumping_days = pumping_days
        # Update saturated thickness and water table lift based on groundwater
        # level change
        self.l_wt -= dwl
        self.st += dwl
        tr_ = self.st * self.k  # Update Transmissivity
        # cannot divided by zero
        if tr_ < 0.001:
            self.tr = 0.001
        else:
            self.tr = tr_

        self.withdrawal = withdrawal
        l_wt = self.l_wt

        r, tr, sy = self.r, self.tr, self.sy
        eff_well, eff_pump = self.eff_well, self.eff_pump
        rho, g = self.rho, self.g
        pumping_days = self.pumping_days

        # Calculate energy consumption
        m_ha_2_m3 = 10000
        fpitr = 4 * np.pi * tr
        ftrd = 4 * tr * pumping_days

        l_wd_l_cd = (
            pumping_rate
            / fpitr
            * (-0.5772 - np.log(r**2 * sy / ftrd))
            * m_ha_2_m3
            / eff_well
        )

        l_t = l_wt + l_wd_l_cd + l_pr

        e = rho * g * m_ha_2_m3 / eff_pump / 1e15 * withdrawal * l_t  # PJ

        # Record energy consumption
        self.pumping_rate = pumping_rate
        self.e = e
        return e
