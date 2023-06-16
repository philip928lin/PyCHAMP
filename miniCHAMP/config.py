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
            "corn":     (455.0162437162218, 93.70410821239668, -3.792352328384122, 6.784495928053121, -2.0343557910784873),
            "sorghum":  (252.99787566746383, 82.1193601089726, -3.6572818850257454, 6.228651607058931, -1.651976389963492),
            "soybeans":  (133.65537966632013, 89.15443022069083, -3.4664151483341152, 6.184045213667842, -1.758066588119619),
            "fallow":   (0., 100., 0, 0, 0.),
            },
        "tech": {
            #                        a  b [m3 -> m-ha] Lpr[m] (McCarthy et al., 2020)
            "center pivot":      [0.0051, 0.268744, 28.12],
            "center pivot LEPA": [0.0058, 0.212206, 12.65]
            },
        "growth_period": ["4/1", "10/31"],
        "growth_period_ratio": {
            "corn":     1,
            "sorghum":  0.822,
            "soybeans":  0.934,
            "fallow":   1,
            }
        },
    "well": {
        "rho": 1000.,   # kg/m3
        "g": 9.8016     # m/s2
        },
    "finance": {
        "energy_price": 2777.777778,     # 1e4$/PJ      # $0.10/kWh = $ 2777.777778 1e4/PJ (Aguilar et al., 2015)
        "crop_profit": {
            "corn":     6.10,   # $/bu  2023    Northwest KS (can vary from 2.5-~6; Aguilar et al., 2015)
            "sorghum":  6.16,   # $/bu  2023    Northwest KS
            "soybeans":  12.87,  # $/bu  2023    Northwest KS
            "fallow":   0.},
        "irr_tech_operational_cost": {
            # [1e4$]
            "center pivot":         0,
            "center pivot LEPA":    0},
        "irr_tech_change_cost": {
            # if not specify, 0 is the default [1e4$]
            ("center pivot", "center pivot LEPA"): 0,
            ("center pivot LEPA", "center pivot"): 0},
        "crop_change_cost": {
            # if not specify, 0 is the default.
            # This is a fixed cost per unit area crop change. [1e4$]
            ("corn", "sorghum"):    0,
            ("corn", "soybeans"):    0,
            ("corn", "fallow"):     0,
            ("sorghum", "corn"):    0,
            ("sorghum", "soybeans"): 0,
            ("sorghum", "fallow"):  0,
            ("soybeans", "corn"):    0,
            ("soybeans", "sorghum"): 0,
            ("soybeans", "fallow"):  0,
            ("fallow", "corn"):     0,
            ("fallow", "sorghum"):  0,
            ("fallow", "soybeans"):  0}
        },
    "aquifer": {
        "lag": 5    # yr
        },
    "consumat": {
        "alpha":{
            "profit": 0.1,    # keep the potential numerical issue in mind!
            "yield_pct": 1},
        "satisfaction_threshold": 1,
        "uncertainty_threshold": 1
        },
    "gurobi":
        {"LogToConsole": 1}     # 0: no console output; 1: with console output
    }