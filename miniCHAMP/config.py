"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.

Suggestion:
    1. We could adopt yaml format in the future.
"""
# =============================================================================
# config file collecting general info for miniCHAMP.
# =============================================================================

config = {
    "field": {
        "field_area": 50.,  # ha
        "area_split": 1,    # -- # 2 will be alright, starting from 3 it will be slow in process.
        "crop": {
            #           ymax, wmax,  a,    b,   c
            "corn":     [531, 93., -3.25, 5.81, -1.74],    # you have numerical issue here~~ try to change to $M
            "sorghum":  [267, 82.1, -3.47, 5.90, -1.57],
            "soybean":  [165, 89.2, -2.81, 5.00, -1.42],
            "fallow":   [0., 100., 0, 0, 0.],
            },
        "tech": {
            #                        a  b [m3 -> m-ha] Lpr[m] (McCarthy et al., 2020)
            "center pivot":      [0.0051, 0.268744, 28.12],
            "center pivot LEPA": [0.0058, 0.212206, 12.65]
            },
        "growth_period_ratio": {
            "corn":     1,
            "sorghum":  0.822,
            "soybean":  0.934,
            "fallow":   1,
            }
        },
    "well": {
        "rho": 1000.,   # kg/m3
        "g": 9.8016     # m/s2
        },
    "finance": {
        "energy_price": 27.77777778,     # $/PJ      # $0.10/kWh = $ 27.77777778 M/PJ (Aguilar et al., 2015)
        "crop_profit": {
            "corn":     6.10,   # $/bu  2023    Northwest KS (can vary from 2.5-~6; Aguilar et al., 2015)
            "sorghum":  6.16,   # $/bu  2023    Northwest KS
            "soybean":  12.87,  # $/bu  2023    Northwest KS
            "fallow":   0.}
        },
    "aquifer": {
        "lag": 5    # yr
        },
    "consumat": {
        "alpha":{
            "profit": 0.001,    # keep the potential numerical issue in mind!
            "yield_pct": 1},
        "satisfaction_threshold": 1,
        "uncertainty_threshold": 1
        },
    "gurobi":
        {"LogToConsole": 1}     # 0: no console output; 1: with console output
    }