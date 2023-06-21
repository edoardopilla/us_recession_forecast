### PAPER REPLICATION
# import necessary libraries
import pandas as pd

import statsmodels as sm
from statsmodels.discrete.discrete_model import Probit

import matplotlib
matplotlib.use("Qt5Agg", force = True)
import matplotlib.pyplot as plt

# read data
ntfs = pd.read_excel(io = "ntfs_d.xlsx")
ntfs_new = pd.read_excel(io = "ntfs_new_d.xlsx")

rec_q = pd.read_excel(io = "rec_dummy_q4.xlsx")
rec_q_new = pd.read_excel(io = "rec_dummy_new_q4.xlsx")

long_spread = pd.read_excel(io = "10y_2y_d.xlsx")
medium_spread = pd.read_excel(io = "10y_1y_d.xlsx")
short_spread = pd.read_excel(io = "10y_3m_d.xlsx")

# read vanilla recession dataframe for graph comparison from 1972 to 2018
rec_df = pd.read_excel(io = "rec_dummy_q.xlsx")

# read vanilla recession dataframe for graph comparison from 1981 to 2023
rec_df_new = pd.read_excel(io = "rec_dummy_new_q.xlsx")

# create quarterly dataframe
ntfs = ntfs.set_index("date").resample("QS").mean()
ntfs["date"] = ntfs.index
ntfs = ntfs.reset_index(drop = True)[["date", "spread"]]

ntfs_new = ntfs_new.set_index("date").resample("QS").mean()
ntfs_new["date"] = ntfs_new.index
ntfs_new = ntfs_new.reset_index(drop = True)[["date", "spread"]]

long_spread = long_spread.set_index("date").resample("QS").mean()
long_spread["date"] = long_spread.index
long_spread = long_spread.reset_index(drop = True)[["date", "spread"]]

medium_spread = medium_spread.set_index("date").resample("QS").mean()
medium_spread["date"] = medium_spread.index
medium_spread = medium_spread.reset_index(drop = True)[["date", "spread"]]

short_spread = short_spread.set_index("date").resample("QS").mean()
short_spread["date"] = short_spread.index
short_spread = short_spread.reset_index(drop = True)[["date", "spread"]]

# merge dataframes
merged_df = ntfs.merge(right = rec_q, how = "inner", on = "date")

merged_df_new = ntfs_new.merge(right = long_spread, how = "inner", on = "date")
merged_df_new = merged_df_new.merge(right = rec_q_new, how = "inner", on = "date")

merged_df_medium = ntfs_new.merge(right = medium_spread, how = "inner", on = "date")
merged_df_medium = merged_df_medium.merge(right = rec_q_new, how = "inner", on = "date")

merged_df_short = ntfs_new.merge(right = short_spread, how = "inner", on = "date")
merged_df_short = merged_df_short.merge(right = rec_q_new, how = "inner", on = "date")

# initialize and fit probit model to data for near-term forward spread from 1972 to 2018
probit_model_ntfs = sm.discrete.discrete_model.Probit(endog = merged_df["rec"].values, exog = merged_df["spread"].values)
model_res_ntfs = probit_model_ntfs.fit()

# initialize and fit probit model to data for near-term forward spread from 1981 to 2023
probit_model_ntfs_new = sm.discrete.discrete_model.Probit(endog = merged_df_new["rec"].values, exog = merged_df_new["spread_x"].values)
model_res_ntfs_new = probit_model_ntfs_new.fit()

# initialize and fit probit model to data for long-term yield spread from 1981 to 2023
probit_model_long = sm.discrete.discrete_model.Probit(endog = merged_df_new["rec"].values, exog = merged_df_new["spread_y"].values)
model_res_long = probit_model_long.fit()

# create dictionary to host repeated predictions for model comparison
df_dct_pred = {"long": merged_df_new, "medium": merged_df_medium, "short": merged_df_short}

for k, v in df_dct_pred.items():
    probit_model = sm.discrete.discrete_model.Probit(endog = v["rec"].values, exog = v[["spread_x", "spread_y"]].values)
    model_res = probit_model.fit()
    print(model_res.get_margeff().summary())
    df_dct_pred[k] = model_res.predict()

# extract marginal effects for the different models outside the loop
print(model_res_ntfs.get_margeff().summary())
print(model_res_ntfs_new.get_margeff().summary())
print(model_res_long.get_margeff().summary())

# predict recession probabilities for the models outside the loop
model_pred_ntfs = model_res_ntfs.predict()
model_pred_ntfs_new = model_res_ntfs_new.predict()
model_pred_long = model_res_long.predict()

# define above dictionary values for repeated graphs below
model_pred_ntfs_long = df_dct_pred["long"]
model_pred_ntfs_medium = df_dct_pred["medium"]
model_pred_ntfs_short = df_dct_pred["short"]

# plot predicted values against recession indicators from 1972 to 2018
pred_df = pd.DataFrame()
pred_df["date"] = merged_df["date"]
pred_df["fit_ntfs"] = model_pred_ntfs
pred_df["rec"] = rec_df["rec"]
pred_df.plot(x = "date", y = ["rec", "fit_ntfs"], kind = "line", figsize = (10, 6))

plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Estimated recession probabilities")

# plot predicted values against recession indicators from 1981 to 2023 for individual regressors
pred_df_new = pd.DataFrame()
pred_df_new["date"] = merged_df_new["date"]
pred_df_new["fit_ntfs"] = model_pred_ntfs_new
pred_df_new["fit_long"] = model_pred_long
pred_df_new["rec"] = rec_df_new["rec"]
pred_df_new.plot(x = "date", y = ["rec", "fit_ntfs", "fit_long"], kind = "line", figsize = (10, 6))

plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Estimated recession probabilities")
plt.legend(loc = (.75, .6))

# create dataframe dictionary for repeated graphs of joint regressors from 1981 to 2023
df_dct = {"model_pred_ntfs_long": merged_df_new, "model_pred_ntfs_medium": merged_df_medium,
          "model_pred_ntfs_short": merged_df_short}

for k, v in df_dct.items():
    pred_df = pd.DataFrame()
    pred_df["date"] = v["date"]
    pred_df["fit"] = eval(k)
    pred_df["rec"] = rec_df_new["rec"]
    pred_df.plot(x = "date", y = ["rec", "fit"], kind = "line", figsize = (10, 6))

    plt.xlabel("Time")
    plt.ylabel("Probability")
    plt.title("Estimated recession probabilities")
    plt.legend(loc = (.75, .6))