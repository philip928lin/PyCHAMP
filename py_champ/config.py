"""
The code is developed by Chung-Yi Lin at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.

Suggestion:
    1. We could adopt yaml format in the future.
    
old format
"corn":     [520.179, 79.4827, -4.0874, 6.4036, -1.635],
"sorghum":  [293.1682, 61.8959, -2.0082, 2.9031, -0.4178],
"soybeans": [168.6044, 72.9901, -3.3069, 5.0755, -1.0811],
"wheat":    [148.9766, 69.4979, -1.5776, 2.2792, 0.0215],
"fallow":   [0., 100., 0, 0, 0.],

"corn":     [491.6036, 77.7756, -3.9677, 6.1283, -1.5745, 0.1151],
"sorghum":  [293.1682, 60.152, -2.267, 3.2361, -0.4907, 0.4],
"soybeans": [168.6044, 68.7955, -3.9376, 5.6797, -1.1822, 0.1186],
"wheat":    [148.9766, 69.4979, -2.2698, 3.0036, -0.0893, 0.4294],
"""
# =============================================================================
# config file collecting general info for miniCHAMP.
# =============================================================================

config = {
    "field": {
        "field_area": 50.,  # [ha]  Field size for a grid.
        "area_split": 1,    # [--]  Num of decision set for a field. The solving process will slow down starting from 3.
        "crop": {       # (ymax [bu], wmax[cm], a, b, c, min_yield_pct) fitted from the Risk Management Agency (RMS)
                        # If min_yield_pct is not given, default is zero.
            # 2008 - 2022 data
            "corn":     [439.2302, 77.7756, -3.2904, 5.9082, -1.6521, 0.1294],
            "sorghum":  [178.2362, 60.152, -1.9572, 3.5132, -0.5766, 0.6246],
            "soybeans": [134.7049, 68.7955, -2.1413, 3.8485, -0.7292, 0.1316],
            "wheat":    [113.7038, 69.4979, -1.5576, 2.5687, -0.0591, 0.4498],
            "fallow":   [0., 100., 0, 0, 0., 0],
            },
        "tech": {   # (a [m3 -> m-ha], b [m3 -> m-ha], Lpr [m]) (McCarthy et al., 2020)
            "center pivot":      [0.0051, 0.268744, 28.12],
            "center pivot LEPA": [0.0058, 0.212206, 12.65]
            },
        "growing_season": {
            "corn":     ["5/1", "10/3"],
            "sorghum":  ["6/2", "11/3"],
            "soybeans": ["6/2", "10/15"],
            "wheat":    ["10/3", "6/27"]
            #["5/1", "10/15"],  # (Deines et al., 2021)
            },
        # "growth_period_ratio": {    # If we want to incorporate wheat, explicit dates are required for prec_aw.
        #     "corn":     1,
        #     "sorghum":  0.822,
        #     "soybeans": 0.934,
        #     "fallow":   1,
        #     }
        },
    "well": {
        "rho": 1000.,   # [kg/m3]
        "g": 9.8016     # [m/s2]
        },
    "finance": {
        "energy_price": 2777.777778,    # [1e4$/PJ] $0.10/kWh = $ 2777.777778 1e4/PJ (Aguilar et al., 2015)
        "crop_profit": {
            "corn":     5.394667,       # $/bu  2007    KFMA (can vary from 2.5-~6; Aguilar et al., 2015)
            "sorghum":  6.598655566,    # $/bu  2007    KFMA
            "soybeans": 13.3170448,     # $/bu  2007    KFMA
            "wheat":    8.28157881,
            "fallow":   0.
            },
        "irr_tech_operational_cost": {  # [1e4$]
            "center pivot":         1.876,
            "center pivot LEPA":    1.876
            },
        "irr_tech_change_cost": {   # If not specify, 0 is the default.
            ("center pivot", "center pivot LEPA"): 0,
            ("center pivot LEPA", "center pivot"): 0
            },
        "crop_change_cost": {   # [1e4$] If not specify, 0 is the default. This is a fixed cost per unit area crop change.
            # ("corn", "sorghum"):     0,
            # ("corn", "soybeans"):    0,
            # ("corn", "fallow"):      0,
            # ("sorghum", "corn"):     0,
            # ("sorghum", "soybeans"): 0,
            # ("sorghum", "fallow"):   0,
            # ("soybeans", "corn"):    0,
            # ("soybeans", "sorghum"): 0,
            # ("soybeans", "fallow"):  0,
            # ("fallow", "corn"):      0,
            # ("fallow", "sorghum"):   0,
            # ("fallow", "soybeans"):  0
            }
        },
    "aquifer": {
        "lag": 5    # [yr] Vertical infiltration
        },
    "consumat": {
        "alpha": {  # [0-1] Sensitivity factor for the "satisfication" calculation.
            "profit":    1,
            "yield_pct": 1
            },
        "scale": {  # Normalize "need" for "satisfication" calculation.
            "profit":    0.23 * 50, # Use corn 1e4$*bu*ha
            "yield_pct": 1
            },
        "satisfaction_threshold": 1,    # [0-1]
        "uncertainty_threshold":  1     # [0-1]
        },
    "gurobi": {
        "LogToConsole": 1,  # 0: no console output; 1: with console output.
        "Presolve": -1      # Options are Auto (-1; default), Aggressive (2), Conservative (1), Automatic (-1), or None (0).
        }
    }