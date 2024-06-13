#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:57:36 2024

@author: wayne
"""

import numpy
import pandas
import mesa
import scipy
import gurobipy
from py_champ.models.sd6_model import SD6Model
from py_champ.entities import field

import os
from os import chdir, getcwd
import dill
import os
import pandas as pd

wd=getcwd() # set working directory
chdir(wd)

file_path =  wd + "\\Inputs_SD6.pkl"

with open(file_path, "rb") as f:
    (aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict,
     prec_aw_step, crop_price_step, shared_config) = dill.load(f)
    
data = pd.read_csv(os.path.join(wd, "Data_SD6.csv"), index_col=["year"])
prec_avg = pd.read_csv(os.path.join(wd, "Inputs\\Prec_avg.csv"), index_col=[0]).iloc[1:, :]
# crop type for the simulation
crop_options = ["corn", "sorghum", "soybeans", "wheat", "fallow"]

# type of irrigation technology available
tech_options = ["center pivot LEPA"]

# number of splits for each field
area_split = 1

# seed for model replicability and comparison
seed = 12345

# calibrated parameters for simulation
pars = {'perceived_risk': 0.7539,
 'forecast_trust': 0.8032,
 'sa_thre': 0.1421,
 'un_thre': 0.0773}

m = SD6Model(
    pars=pars,
    crop_options=crop_options,
    tech_options=tech_options,
    area_split=area_split,
    aquifers_dict=aquifers_dict,
    fields_dict=fields_dict,
    wells_dict=wells_dict,
    finances_dict=finances_dict,
    behaviors_dict=behaviors_dict,
    prec_aw_step=prec_aw_step,
    init_year=2007,
    end_year=2022,
    lema_options=(True, 'wr_LEMA_5yr', 2013),
    fix_state=None,
    show_step=True,
    seed=seed,
    shared_config=shared_config,
    # kwargs
    crop_price_step=crop_price_step
    )

for i in range(15):
        m.step()
        
# read outputs for attributes related to different agent types
df_farmers, df_fields, df_wells, df_aquifers = SD6Model.get_dfs(m)

# read system level outputs. For e.g., ratios of crop types, irrigation technology, rainfed or irrigated field for the duration of the simulation
df_sys = SD6Model.get_df_sys(m, df_farmers, df_fields, df_wells, df_aquifers)
metrices = m.get_metrices(df_sys, data) # same length

# !pip install seaborn
# !pip install adjustText

import seaborn
# import adjustText

from plot_EMS import (plot_cali_gwrc, plot_crop_ratio, reg_prec_withdrawal)

# Plot results
plot_cali_gwrc(df_sys.reindex(data.index), data, metrices, prec_avg, stochastic=[], savefig=None)
plot_crop_ratio(df_sys.reindex(data.index), data, metrices, prec_avg, savefig=None)

reg_prec_withdrawal(prec_avg, df_sys.reindex(data.index), df_sys_nolema=None, data=data, 
                    df_sys_list=None, df_sys_nolema_list=None, dot_labels=True, obv_dots=False, savefig=None)

# Plot results
plot_cali_gwrc(df_sys.reindex(data.index), data, metrices, prec_avg, stochastic=[], savefig=None)
plot_crop_ratio(df_sys.reindex(data.index), data, metrices, prec_avg, savefig=None)
reg_prec_withdrawal(prec_avg, df_sys.reindex(data.index), df_sys_nolema=None, data=data, 
                    df_sys_list=None, df_sys_nolema_list=None, savefig=None)






