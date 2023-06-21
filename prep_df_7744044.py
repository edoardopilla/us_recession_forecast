### PREPARE DATAFRAME
# import necessary libraries
import numpy as np
import pandas as pd

# read spread data
short_spread_df = pd.read_excel(io="10y_3m_m.xlsx")
long_spread_df = pd.read_excel(io="10y_2y_m.xlsx")

# read macro data
ip_df = pd.read_excel(io="ip_m.xlsx")
nonfarm_df = pd.read_excel(io="nonfarm_m.xlsx")

# read recession data
rec_df = pd.read_excel(io="rec_dummy_m12.xlsx")

# convert timestamps to dates
short_spread_df["date"] = pd.to_datetime(short_spread_df["date"]).dt.date
long_spread_df["date"] = pd.to_datetime(long_spread_df["date"]).dt.date

ip_df["date"] = pd.to_datetime(ip_df["date"]).dt.date
nonfarm_df["date"] = pd.to_datetime(nonfarm_df["date"]).dt.date

rec_df["date"] = pd.to_datetime(rec_df["date"]).dt.date

# convert values to decimals
short_spread_df["spread"] = short_spread_df["spread"] / 100
long_spread_df["spread"] = long_spread_df["spread"] / 100

# scale macro data
ip_df["index"] = np.log(ip_df["index"]) / 1000
nonfarm_df["nonfarm"] = np.log(nonfarm_df["nonfarm"]) / 1000

# merge rate dataframes
merged_df = short_spread_df.merge(right=long_spread_df, how="inner", on="date")
merged_df = merged_df.merge(right=ip_df, how="inner", on="date")
merged_df = merged_df.merge(right=nonfarm_df, how="inner", on="date")
merged_df = merged_df.merge(right=rec_df, how="inner", on="date")