r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""

class Aquifer():
    """
    An aquifer simulator.

    Attributes
    ----------
    aquifer_id : str or int
        Aquifer id.
    sy : float
        Specific yield.
    area : float
        Area [ha].
    lag : int
        Vertical percolation time lag for infiltrated water to contribute
        to the groundwater level change [yr].
    ini_st: float
        Initial saturated thickness [m].
    ini_inflow : float, optional
        Initial inflow [m-ha]. This will be assigned for the first number of the
        "lag" years. The default is 0.
    ini_dwl : float, optional
        Initial groundwater level change [m]. The default is 0.
    """

    def __init__(self, aquifer_id, sy, area, lag,
                 ini_st=0, ini_inflow=0, ini_dwl=0):

        # for name_, value_ in vars().items():
        #     if name_ != 'self':
        #         setattr(self, name_, value_)
        self.aquifer_id, self.sy, self.area, self.lag = aquifer_id, sy, area, lag
        self.in_list = [ini_inflow]*lag
        self.t = 0
        self.st = ini_st
        self.dwl_list = [ini_dwl]
        self.dwl = ini_dwl

    def sim_step(self, inflow, v):
        """
        Simulate a single timestep.

        Parameters
        ----------
        inflow : float
            Inflow of the aquifer [m-ha].
        v : float
            Total water withdraw from the aquifer [m-ha].

        Returns
        -------
        dwl : float
            Groundwater level change [m/yr].

        """
        in_list = self.in_list
        sy, area = self.sy, self.area

        in_list.append(inflow)
        inflow_lag = in_list.pop(0)
        dwl = 1/(area * sy) * (inflow_lag - v)

        self.dwl_list.append(dwl)
        self.st += dwl
        self.t += 1
        self.dwl = dwl

        return dwl
