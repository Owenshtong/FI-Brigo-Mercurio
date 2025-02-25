#### part 2 ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import direct
from scipy.optimize import differential_evolution
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import part1
import part2


# (1)

y_observed = pd.read_csv("yield_curves.csv", index_col=0, na_values=["NA", "N/A", "", " na"], parse_dates=True)
y_observed = y_observed[[" ZC025YR"," ZC050YR", " ZC100YR", " ZC500YR", " ZC1000YR"]]
y_observed = y_observed.dropna()
y_observed = y_observed.iloc[::12, :]
tau = [.25,.5,1,5,10]
y_observed.columns = tau


# (3)
# Backout rt path
opt_x = [.1349, .0061, .0506, -.04253,.0053, 5.3 * 10e-5, 2.5 * 10e-5, .0377, .0438]
LLH, rt_path, ME = part2.L_sum(*opt_x[:3], opt_x[4], opt_x[4:], y_observed , tau)
y_hat = y_observed + ME



# r_observed = pd.DataFrame(nss_coeff.BETA0 + nss_coeff.BETA1)
fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
# ax.scatter(y_observed.index, r_observed, marker = "o", s = 5, label = "Observed")
ax.scatter(y_observed.index, rt_path, marker = "o", s = 5, label = "Estimated")
plt.title("Short rate")
plt.ylabel("Yield")
plt.legend()
plt.show()




