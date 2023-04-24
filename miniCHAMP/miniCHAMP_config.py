"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on April 24, 2023

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
        "field_area": 50.,
        "area_split": 1,
        "crop": {
            #           ymax, wmax,  a,    b,   c
            "corn":     [5., 400., -1.75, 2.65, 0.],
            "sorghum":  [6., 500., -1.75, 2.65, 0.],
            "soybean":  [7., 600., -1.75, 2.65, 0.],
            "fallow":   [8., 700., -1.75, 2.65, 0.],
            },
        "tech": {
            #                        a  b [m3 -> m-ha] Lpr[m] (McCarthy et al., 2020)
            "center pivot":      [0.0051, 0.268744, 28.12],
            "center pivot LEPA": [0.0058, 0.212206, 12.65]
            }
        },
    "well": {
        "rho": 1000.,
        "g": 9.8016
        },
    "finance": {
        "energy_price": 0.,
        "crop_profit": {
            "corn":     20.,
            "sorghum":  10.,
            "soybean":  10.,
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
        }
    }