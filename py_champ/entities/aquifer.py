r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Sep 9, 2023
"""
import mesa

class Aquifer(mesa.Agent):
    """
    An aquifer simulator based on the KGS-WBM model.

    The class simulates changes in the aquifer's groundwater level
    based on water withdrawals. It uses coefficients `aq_a` and `aq_b` 
    to calculate the change in groundwater level.

    For more details on the model, refer to:
    Butler, J. J., Whittemore, D. O., Wilson, B. B., & Bohling, G. C. (2018).
    Sustainability of aquifers supporting irrigated agriculture: A case study
    of the High Plains aquifer in Kansas. Water International, 43(6), 815–828.
    https://doi.org/10.1080/02508060.2018.1515566

    Attributes
    ----------
    unique_id : str or int
        Unique identifier for the aquifer.
    aq_a : float
        KGS-WBM coefficient for groundwater level change 
        (aq_b - aq_a * withdrawal).
    aq_b : float
        KGS-WBM coefficient for groundwater level change 
        (aq_b - aq_a * withdrawal).
    ini_st : float
        Initial saturated thickness of the aquifer in meters. Default is 0.
    ini_dwl : float
        Initial change in groundwater level in meters. Default is 0.
    """

    def __init__(self, unique_id, mesa_model, ini_st=0, ini_dwl=0,
                 aq_a=None, aq_b=None, area=None, sy=None):
        """
        Initialize an Aquifer object.

        Parameters
        ----------
        unique_id : str or int
            Unique identifier for the aquifer.
        mesa_model : object
            Reference to the overarching MESA model instance.
        aq_a : float
            KGS-WBM coefficient for groundwater level change.
        aq_b : float
            KGS-WBM coefficient for groundwater level change.
        ini_st : float, optional
            Initial saturated thickness of the aquifer in meters. Default is 0.
        ini_dwl : float, optional
            Initial change in groundwater level in meters. Default is 0.
        """
        # MESA required attributes => (unique_id, model)
        super().__init__(unique_id, mesa_model)
        self.agt_type = "Aquifer"
        # for name_, value_ in vars().items():
        #     if name_ != 'self':
        #         setattr(self, name_, value_)
        self.unique_id = unique_id
        
        # Either   (static)
        self.aq_a = aq_a
        self.aq_b = aq_b
        # Or       (dynamic)
        self.area = None
        self.sy = None
        
        self.t = 0                  # Initialize time step to zero
        self.st = ini_st            # Initialize saturated thickness
        self.dwl_list = [ini_dwl]   # Initialize list to store changes in water level
        self.dwl = ini_dwl          # Initialize change in water level
        self.withdrawal = None      # Initialize withdrawal to None
        
    def step(self, withdrawal, inflow=None):
        """
        Simulate the aquifer for one time step based on the water withdrawal.

        Parameters
        ----------
        withdrawal : float
            The total volume of water to be withdrawn from the aquifer in m-ha.

        Returns
        -------
        float
            The change in groundwater level for the current time step in meters per year.

        Attributes Modified
        -------------------
        dwl : float
            Updated change in groundwater level.
        st : float
            Updated saturated thickness.
        t : int
            Updated time step.
        dwl_list : list of float
            Updated list of changes in water level.
        withdrawal : float
            Updated withdrawal amount.
        """      
        if inflow is None: # static
            dwl = self.aq_b - self.aq_a * withdrawal  # Calculate change in water level
        else: # dynamic
            asy = self.area * self.sy
            dwl = inflow/asy - withdrawal/asy
        
        # Update class attributes based on the new information
        self.withdrawal = withdrawal
        self.dwl_list.append(dwl)
        self.st += dwl
        self.t += 1
        self.dwl = dwl

        return dwl

