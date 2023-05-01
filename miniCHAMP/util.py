# -*- coding: utf-8 -*-
"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""
import numpy as np
from pandas import to_numeric

def dict_to_string(dictionary, prefix="", indentor="  ", level=2):
    """Ture a dictionary into a printable string.
    Parameters
    ----------
    dictionary : dict
        A dictionary.
    indentor : str, optional
        Indentor, by default "  ".

    Returns
    -------
    str
        A printable string.
    """
    def dict_to_string_list(dictionary, indentor="  ", count=1, string=[]):
        for key, value in dictionary.items():
            string.append(prefix + indentor * count + str(key))
            if isinstance(value, dict) and count < level:
                string = dict_to_string_list(value, indentor, count+1, string)
            elif isinstance(value, dict) is False and count == level:
                string[-1] += ":\t" + str(value)
            else:
                string.append(prefix + indentor * (count+1) + str(value))
        return string
    return "\n".join(dict_to_string_list(dictionary, indentor))


def cal_pet_Hamon(temp, lat, dz=None):
    """Calculate potential evapotranspiration (pet) with Hamon (1961) equation.

    Parameters
    ----------
    temp : numpy.ndarray
        Daily mean temperature [degC].
    lat : float
        Latitude [deg].
    dz : float, optional
        Altitude temperature adjustment [m], by default None.

    Returns
    -------
    numpy.ndarray
        Potential evapotranspiration [cm/day]

    Note
    ----
    The code is adopted from HydroCNHS (Lin et al., 2022).
    Lin, C. Y., Yang, Y. C. E., & Wi, S. (2022). HydroCNHS: A Python Package of
    Hydrological Model for Coupled Natural–Human Systems. Journal of Water
    Resources Planning and Management, 148(12), 06022005.
    """
    pdDatedateIndex = temp.index
    temp = temp.values.flatten()
    # Altitude temperature adjustment
    if dz is not None:
        # Assume temperature decrease 0.6 degC for every 100 m elevation.
        tlaps = 0.6
        temp = temp - tlaps*dz/100
    # Calculate Julian days
    # data_length = len(temp)
    # start_date = to_datetime(start_date, format="%Y/%m/%d")
    # pdDatedateIndex = date_range(start=start_date, periods=data_length,
    #                              freq="D")
    JDay = to_numeric(pdDatedateIndex.strftime('%j')) # convert to Julian days
    # Calculate solar declination [rad] from day of year (JDay) based on
    # equations 24 in ALLen et al (1998).
    sol_dec = 0.4093 * np.sin(2. * 3.141592654 / 365. * JDay - 1.39)
    lat_rad = lat*np.pi/180
    # Calculate sunset hour angle from latitude and solar declination [rad]
    # based on equations 25 in ALLen et al (1998).
    omega = np.arccos(-np.tan(sol_dec) * np.tan(lat_rad))
    # Calculate maximum possible daylight length [hr]
    dl = 24 / np.pi * omega
    # From Prudhomme(hess, 2013)
    # https://hess.copernicus.org/articles/17/1365/2013/hess-17-1365-2013-supplement.pdf
    # Slightly different from what we used to.
    pet = (dl / 12) ** 2 * np.exp(temp / 16)
    pet = np.array(pet/10)         # Convert from mm to cm
    pet[np.where(temp <= 0)] = 0   # Force pet = 0 when temperature is below 0.
    return pet      # [cm/day]