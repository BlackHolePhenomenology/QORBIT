import pandas as pd
import numpy as np

from values.constants import M_sun,c_val,G_val

#read the binary data csv file
df = pd.read_csv("../../data/binary_data.csv",skipinitialspace=True)

#binary data
name_binary = np.asarray(df["Name"])
M_binary = np.asarray(df["Mass"], dtype=np.float64)
M_err_binary = np.asarray(df["unc_M"])
f_U_val_binary = np.asarray(df["f_U"])
f_U_err_binary = np.asarray(df["unc_f_U"])
f_L_val_binary = np.asarray(df["f_L"])
f_L_err_binary = np.asarray(df["unc_f_L"])