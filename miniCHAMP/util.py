# -*- coding: utf-8 -*-
"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on April 24, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.
"""

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