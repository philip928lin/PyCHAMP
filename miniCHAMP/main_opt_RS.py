import miniCHAMP as mc
# import os
# os.chdir(r"C:\Users\Philip\Documents\GitHub\miniCHAMP\miniCHAMP")
# from opt_model import OptModel

config = {
    "field": {
        "field_area": 50.,  # ha
        "area_split": 1,    # --
        "crop": {
            #           ymax, wmax,  a,    b,   c
            "corn":     [521, 103., -4.19, 7.21, -2.23],    # you have numerical issue here~~ try to change to $M
            "sorghum":  [264, 98.7, -4.19, 7.21, -2.23],
            "soybean":  [163, 101., -4.19, 7.21, -2.23], #-1.75, 2.65, 0.
            "fallow":   [0., 100., 0, 0, 0.],
            },
        "tech": {
            #                        a  b [m3 -> m-ha] Lpr[m] (McCarthy et al., 2020)
            "center pivot":      [0.0051, 0.268744, 28.12],
            "center pivot LEPA": [0.0058, 0.212206, 12.65]
            }
        },
    "well": {
        "rho": 1000.,   # kg/m3
        "g": 9.8016     # m/s2
        },
    "finance": {
        #"energy_price": 27.77777778,     # $/PJ      # $0.10/kWh = $ 27.77777778 M/PJ (Aguilar et al., 2015)
        "energy_price": 2777.777778,     # $/PJ      # $0.10/kWh = $ 27.77777778 M/PJ (Aguilar et al., 2015)
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
            "profit": 0.01,    # keep the potential numerical issue in mind!
            "yield_pct": 1},
        "satisfaction_threshold": 1,
        "uncertainty_threshold": 1
        },
    "gurobi":
        {"LogToConsole": 1}     # 0: no console output; 1: with console output
    }

sur_ele = 882.57
bedrock_depth = 75.06
avg_st = 17.82
l_wt = bedrock_depth - avg_st


# I think it is because min and max that cause such a long time to solve => Nope
# Maybe the number become too small? change to 1e4 instead?


m = mc.opt_model.OptModel()
#m = OptModel()
m.setup_ini_model(config=config, horizon=5, eval_metric="profit")
m.setup_constr_field(field_id="f1", prec=50)
m.setup_constr_well(well_id="w1", dwl=0.55, st=avg_st, l_wt=l_wt, r=0.4064,
                    k=79.1, sy=0.061, eff_pump=0.77, eff_well=0.5)
m.setup_constr_finance()
m.setup_obj()
m.finish_setup()
m.model.tune()
#%%
m.solve()
dm_sols = m.sols


r"""
Explored 5980 nodes (100058 simplex iterations) in 10.77 seconds (3.84 work units)
Thread count was 8 (of 8 available processors)

Solution count 10: 0.0871675 0.0871675 0.0871675 ... 0.0871524

Optimal solution found (tolerance 1.00e-04)
Best objective 8.716752847747e-02, best bound 8.717313179933e-02, gap 0.0064%

Explored 6535 nodes (121904 simplex iterations) in 8.15 seconds (4.20 work units)
Thread count was 8 (of 8 available processors)

Solution count 10: 0.0871716 0.0871655 0.0870541 ... 0.0867699
No other solutions better than 0.0871716

Optimal solution found (tolerance 1.00e-04)
Best objective 8.717164591546e-02, best bound 8.717164591546e-02, gap 0.0000%

Explored 7719 nodes (152904 simplex iterations) in 9.80 seconds (3.74 work units)
Thread count was 8 (of 8 available processors)

Solution count 10: 0.0871669 0.087153 0.0871527 ... 0.086763

Optimal solution found (tolerance 1.00e-04)
Best objective 8.716685685158e-02, best bound 8.717522170328e-02, gap 0.0096%

Explored 5980 nodes (100057 simplex iterations) in 7.68 seconds (3.84 work units)
Thread count was 8 (of 8 available processors)

Solution count 10: 0.0871675 0.0871675 0.0871675 ... 0.0871524

Optimal solution found (tolerance 1.00e-04)
Best objective 8.716752847747e-02, best bound 8.717313179933e-02, gap 0.0064%

Explored 819 nodes (8394 simplex iterations) in 1.26 seconds (0.52 work units)
Thread count was 8 (of 8 available processors)

Solution count 3: 0.999875 0.999792 0.999785

Optimal solution found (tolerance 1.00e-04)
Best objective 9.998753505142e-01, best bound 9.999375123088e-01, gap 0.0062%
"""
