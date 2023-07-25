r"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on Jun 9, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""

class Aquifer():
    """
    An aquifer simulator.

    Butler, J. J., Whittemore, D. O., Wilson, B. B., & Bohling, G. C. (2018).
    Sustainability of aquifers supporting irrigated agriculture: A case study
    of the High Plains aquifer in Kansas. Water International, 43(6), 815–828.
    https://doi.org/10.1080/02508060.2018.1515566

    Attributes
    ----------
    aquifer_id : str or int
        Aquifer id.
    aq_a : float
        KGS-WBM coefficient.
    aq_b : float
        KGS-WBM coefficient.
    ini_st: float
        Initial saturated thickness [m].
    ini_dwl : float, optional
        Initial groundwater level change [m]. The default is 0.
    """

    def __init__(self, aquifer_id, aq_a, aq_b, ini_st=0, ini_dwl=0):

        # for name_, value_ in vars().items():
        #     if name_ != 'self':
        #         setattr(self, name_, value_)
        self.aquifer_id = aquifer_id
        self.aq_a, self.aq_b = aq_a, aq_b
        self.t = 0
        self.st = ini_st
        self.dwl_list = [ini_dwl]
        self.dwl = ini_dwl
        self.withdrawal = None

    def step(self, withdrawal):
        """
        Simulate a single timestep.

        Parameters
        ----------
        withdrawal : float
            The total water withdraw from the aquifer [m-ha].

        Returns
        -------
        dwl : float
            Groundwater level change [m/yr].

        """
        dwl = self.aq_b - self.aq_a * withdrawal

        self.withdrawal = withdrawal
        self.dwl_list.append(dwl)
        self.st += dwl
        self.t += 1
        self.dwl = dwl

        return dwl

# Archive
#     def step(self, inflow, withdrawal):
#         """
#         Simulate a single timestep.

#         Parameters
#         ----------
#         inflow : float
#             Inflow to the aquifer [m-ha].
#         withdrawal : float
#             The total water withdraw from the aquifer [m-ha].

#         Returns
#         -------
#         dwl : float
#             Groundwater level change [m/yr].

#         """
#         in_list = self.in_list
#         sy, area = self.sy, self.area

#         in_list.append(inflow)
#         inflow_lag = in_list.pop(0)
#         dwl = 1/(area * sy) * (inflow_lag - withdrawal)

#         self.dwl_list.append(dwl)
#         self.st += dwl
#         self.t += 1
#         self.dwl = dwl

#         return dwl