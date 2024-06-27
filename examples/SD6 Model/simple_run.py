import os

wd = r"C:\Users\CL\OneDrive\VT\Proj_DIESE\Code"
import sys

sys.setrecursionlimit(10000)
import dill
from py_champ.models.sd6_model import SD6Model

# %%
# =============================================================================
# Run simulation
# =============================================================================
# Load data
wd = r"Add your working directory"
with open(os.path.join(wd, "Inputs_SD6.pkl"), "rb") as f:
    (
        aquifers_dict,
        fields_dict,
        wells_dict,
        finances_dict,
        behaviors_dict,
        prec_aw_step,
        crop_price_step,
        shared_config,
    ) = dill.load(f)

crop_options = ["corn", "sorghum", "soybeans", "wheat", "fallow"]
tech_options = ["center pivot LEPA"]
area_split = 1
seed = 3

pars = {
    "perceived_risk": 0.7539013390415119,
    "forecast_trust": 0.8032197483934305,
    "sa_thre": 0.14215821111637678,
    "un_thre": 0.0773514357873846,
}

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
    lema_options=(True, "wr_LEMA_5yr", 2013),
    fix_state=None,
    show_step=True,
    seed=seed,
    shared_config=shared_config,
    # kwargs
    crop_price_step=crop_price_step,
)

for i in range(15):
    m.step()

m.end()

# %%
# =============================================================================
# Analyze results
# =============================================================================
df_farmers, df_fields, df_wells, df_aquifers = SD6Model.get_dfs(m)
df_sys = SD6Model.get_df_sys(m, df_farmers, df_fields, df_wells, df_aquifers)

df_sys["GW_st"].plot()
df_sys["withdrawal"].plot()
df_sys[["corn", "sorghum", "soybeans", "wheat", "fallow"]].plot()
df_sys[["Imitation", "Social comparison", "Repetition", "Deliberation"]].plot()
