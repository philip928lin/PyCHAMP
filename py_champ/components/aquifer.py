# The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
# Email: chungyi@vt.edu
# Last modified on Dec 30, 2023
import warnings

import mesa


class Aquifer(mesa.Agent):
    """A class to represent the aquifer component in PyCHAMP based on the KGS-WBM model.


    Parameters
    ----------
    unique_id : str or int
        A unique identifier for this agent.
    model
        The model instance to which this agent belongs.
    settings : dict
        A dictionary containing the settings for the aquifer. Expected keys:
        'aq_a', 'aq_b', 'area', 'sy', and 'init'.

        - 'aq_a' and 'aq_b' are coefficients used in the static inflow calculation (inflow = None in step()).
        - 'area' is the area of the aquifer [ha]
        - 'sy' is the specific yield of the underlying aquifer and is used in the dynamic inflow calculation (inflow = float in step()) [-].
        - 'init' is a dictionary containing initial conditions such as the saturated thickness (st [m]) and initial water level change (dwl [m]).

        >>> # A sample settings dictionary
        >>> settings = {
        >>>     "aq_a": None,
        >>>     "aq_b": None,
        >>>     "area": None,
        >>>     "sy": None,
        >>>     "init": {
        >>>         "st": None,
        >>>         "dwl": None
        >>>         }
        >>>     }


    Attributes
    ----------
    agt_type : str
        The type of the agent, set to 'Aquifer'.
    st : float
        The initial saturated thickness of the aquifer [m].
    dwl_list : list
        A list to store changes in water level [m].
    dwl : float
        The initial change in water level [m].
    t : int
        The current time step, initialized to zero.
    withdrawal : float or None
        The current water withdrawal [m-ha], initialized to None.

    Notes
    -----
    The unit of water level is meters [m]. The area is expected in hectares [ha].

    For more details on the KGS-WBM model, refer to:

    Butler, J. J., Whittemore, D. O., Wilson, B. B., & Bohling, G. C. (2018).
    Sustainability of aquifers supporting irrigated agriculture: A case study
    of the High Plains aquifer in Kansas. Water International, 43(6), 815-828.
    https://doi.org/10.1080/02508060.2018.1515566

    """

    def __init__(self, unique_id, model, settings: dict):
        """Initialize an Aquifer agent in the Mesa model."""
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, model)
        self.agt_type = "Aquifer"

        self.load_settings(settings)

        self.st = self.init["st"]  # Initialize saturated thickness
        self.dwl_list = [
            self.init["dwl"]
        ]  # Initialize list to store changes in water level
        self.dwl = self.init["dwl"]  # Initialize change in water level

        self.t = 0  # Initialize time step to zero
        self.withdrawal = None  # Initialize withdrawal to None

    def load_settings(self, settings: dict):
        """
        Load settings for the aquifer simulation from a provided dictionary.

        Parameters
        ----------
        settings : dict
            The dictionary containing the settings for the aquifer as defined above.

        """
        # static inflow
        self.aq_a = settings.get("aq_a")
        self.aq_b = settings.get("aq_b")
        # dynamic inflow
        self.area = settings.get("area")
        self.sy = settings.get("sy")
        self.init = settings["init"]

    def step(self, withdrawal: float, inflow: float | None = None) -> float:
        """
        Perform a single step of the aquifer simulation.

        Parameters
        ----------
        withdrawal : float
            The amount of water withdrawn from the aquifer in this step [m-ha].
        inflow : float, optional
            The amount of inflow water into the aquifer [m-ha]. If None, a static
            inflow is assumed (implied by using 'aq_a' and 'aq_b').

        Returns
        -------
        float
            The change in water level in the aquifer in this step
            [m].

        Notes
        -----
        The method calculates the change in water level either based on static
        inflow, using 'aq_a' and 'aq_b' coefficients or dynamic inflow,
        applying 'sy' and 'area' of the aquifer.
        """
        self.t += 1

        if inflow is None:  # static
            dwl = self.aq_b - self.aq_a * withdrawal  # Calculate change in water level
        else:  # dynamic
            asy = self.area * self.sy
            dwl = inflow / asy - withdrawal / asy

        # Update class attributes based on the new information
        self.dwl_list.append(dwl)
        self.st += dwl
        self.dwl = dwl
        self.withdrawal = withdrawal

        # Check st is not negative
        if self.st < 0:
            warnings.warn(
                f"The saturated thickness is negative in aquifer {self.unique_id}.",
                stacklevel=2,
            )
        return dwl
