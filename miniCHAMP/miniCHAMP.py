# -*- coding: utf-8 -*-
"""
The code is developed by Chung-Yi Lin, postdoc at Virginia Tech, in April 2023.
Email: chungyi@vt.edu
Last modified on May 1, 2023

WARNING: This code is not yet published, please do not distributed the code
without permission.

To do:
    Build agentpy see if agents can run in parallel (OptModel might not be pickable)
    Unpickable issue: https://stackoverflow.com/questions/70128335/what-is-the-proper-way-to-make-an-object-with-unpickable-fields-pickable
    Identify similar agent set
# https://numpydoc.readthedocs.io/en/latest/example.html
# https://realpython.com/python-doctest/
# How to turn off assertion: https://realpython.com/python-assert-statement/

Discard this file
"""
# import pandas as pd
# import numpy as np
# from dotmap import DotMap   # Dot-like dict structure
# import miniCHAMP as mc

# # Sample data
# temp = pd.DataFrame()
# temp["temp"] = np.random.uniform(15, 25, 365)
# temp.index = pd.date_range(start="2001/1/1", periods=365)

# fields_dict = {"f1": {"te": "center pivot",
#                       "lat": 39.4,
#                       "dz": None,
#                       "rain_fed_option": False}}


# wells_dict = {"w1": {"r": 0.05,
#                     "tr": 1,
#                     "sy": 1,
#                     "l_wt": 10,
#                     "eff_pump": 0.77,
#                     "eff_well": 0.5,
#                     "aquifer_id": "ac1",
#                     "pumping_capacity": None}}


# aquifers_dict = {"ac1": {"sy": 1,
#                          "area": 250,
#                          "lag": 5,
#                          "init": {
#                              "inflow": 0,
#                              "dwl": 0.5}}}

# agent_dict = {"horizon": 5,
#               "eval_metric": "profit",
#               "risk_attitude_prec": 30,
#               "n_dwl": 5,
#               "comparable_agt_ids": [],
#               "alphas": None,
#               "init": {
#                   "te": "center pivot",
#                   "crop_type": "corn"}}


# acdict = DotMap(aquifers_dict)
# aquifers = DotMap()
# for ac, v in acdict.items():
#     aquifers[ac] = mc.Aquifer(aquifer_id=ac, sy=v.sy, area=v.area, lag=v.lag,
#                           ini_inflow=v.init.inflow, ini_dwl=v.init.dwl)

# #%%
# # A quick demo of Farmer agent
# config = mc.config
# agt = mc.Farmer()
# agt.setup(agt_id="agt1", config=config, agent_dict=agent_dict,
#           fields_dict=fields_dict, wells_dict=wells_dict,
#           prec_dict={"f1": 10}, temp_dict={"f1": temp}, aquifers=aquifers,
#           crop_options=["corn", "sorghum", "soybean", "fallow"],
#           tech_options=["center pivot", "center pivot LEPA"])

# agt.sim_step(prec_dict={"f1": 10}, temp_dict={"f1": temp})
# agt.finance.profit
# agt.satisfaction
# agt.uncertainty




