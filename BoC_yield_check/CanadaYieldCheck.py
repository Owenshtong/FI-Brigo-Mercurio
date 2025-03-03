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
import part2_toy
import copy

# (1)
y_observed = pd.read_csv("yield_curves.csv", index_col=0, na_values=["NA", "N/A", "", " na"], parse_dates=True)
y_observed = y_observed[[" ZC025YR"," ZC050YR", " ZC100YR", " ZC500YR", " ZC1000YR"]]
y_observed = y_observed.dropna()
y_observed = y_observed.iloc[::12, :]
tau = [.25,.5,1,5,10]
y_observed.columns = tau




def neg_L_sum(pack):
    LLh, _, _=  part2.L_sum(
        *pack[0:4], pack[4:], y_observed, tau
    )
    nLLh = -1 * LLh
    print(nLLh)
    print(pack)
    return nLLh
k_bond = [(0.12,0.15)]
theta_bond = [(0,.1)]
sigma_bond = [(0, .1)]
lambda_bond = [(-1,-0.01)]
h_bond = [(1e-5, 1e-2)]
bond = k_bond+theta_bond+sigma_bond+lambda_bond+h_bond*5


x0 = [0.12, 0.01, 0.16, -0.4] + [1e-3] * 5
# x0 = opt_x
# x0 = [0.125]*9
opt_pack = minimize(neg_L_sum,
                    x0 = x0,
                    bounds = bond,
                    method="Powell",
                    # constraints=[{'type': 'ineq', 'fun': lambda x: x[3] - x[0]/x[1]},
                    #              {'type': 'ineq', 'fun': lambda x: -2 * x[0] * x[1] + x[2]}]
                    options={'maxiter': 1000,  "adptive" : True},
                   )

opt_x = opt_pack["x"]


# (3)
# Backout rt path
opt_x = [1.20057581e-01, 5.77190039e-02, 9.19559829e-02, -6.60116682e-01, 5.78915475e-05, 5.78915475e-05, 5.78915475e-05, 5.78915546e-05, 5.78915475e-05]
# opt_x = [0.1,0.1,0.25,-100000] + [0.00015]*5
LLH, rt_path, ME = part2.L_sum(*opt_x[:3], opt_x[4], opt_x[4:], y_observed , tau)
LLH, rt_path_Q, ME = part2_toy.L_sum(*opt_x[:3], opt_x[4], opt_x[4:], y_observed , tau)
y_hat = y_observed - ME

# r_observed = pd.DataFrame(nss_coeff.BETA0 + nss_coeff.BETA1)
fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
# ax.scatter(y_observed.index, r_observed, marker = "o", s = 5, label = "Observed")
# ax.scatter(y_observed.index, rt_path_Q, marker = "o", s = 5, label = "Q")
ax.scatter(y_observed.index, rt_path, marker = "o", s = 5, label = "P")
# ax.scatter(y_observed.index, np.array(rt_path) - rt_path_Q, marker = "o", s = 5, label = "P")
plt.title("Short rate")
plt.ylabel("Yield")
plt.legend()
plt.show()

#
# for i in range(5):
#     fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
#     ax.plot(y_observed.index,y_hat.iloc[:,i], marker = "D", markersize = 2, label = "Estimated")
#     # ax.plot(y_observed.index, y_observed.iloc[:,i], marker = "D", markersize = 2, label = "Observed")
#     plt.title(r"$\tau = " + str(tau[i]) + "$")
#     plt.ylabel("Yield")
#     plt.legend()
#     plt.show()

fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
for i in range(5):
    ax.plot(y_observed.index, y_hat.iloc[:, i], marker="D", markersize=2, label="Estimated")
# ax.plot(y_observed.index, y_observed.iloc[:,i], marker = "D", markersize = 2, label = "Observed")
plt.title(r"$\tau = " + str(tau[i]) + "$")
plt.ylabel("Yield")
plt.legend()
plt.show()

#####  Report

def fisher(pack, y, tau, epsilon =0.00001):
    # pack in order kappa theta sigma h1to7
    # y is the yield
    dig = []
    for i in range(4):
        pack_min = copy.copy(pack)
        pack_plu = copy.copy(pack)

        pack_min[i] += epsilon
        pack_plu[i] += epsilon
        print(part2.L_sum(*pack_plu[:4], pack_plu[4:], y, tau)[0])
        print(part2.L_sum(*pack[:4], pack[4:], y, tau)[0])
        print(part2.L_sum(*pack[:4], pack[4:], y, tau)[0])

        _2nd_prox = -1 * (part2.L_sum(*pack_plu[:4], pack_plu[4:], y, tau)[0]\
                    - 2 *  part2.L_sum(*pack[:4], pack[4:], y, tau)[0] \
                    +  part2.L_sum(*pack_min[:4], pack_min[4:], y, tau)[0]) / (epsilon**2)
        dig.append(_2nd_prox)

    return np.diag(dig)


THETA_opt = opt_x[:4] # kappa theta sigma lambda

LLH, rt_path, ME = part2.L_sum(*opt_x[:3], opt_x[4], opt_x[4:11], y_observed , tau)
likelihood = LLH # Likelihood
ME_mean = ME.describe().loc["mean"] # mean of measurement error
# ME_var = ME.describe().loc["std"]**2 # var of measurement error (similar to h1 to h7)
ME_var = opt_x[4:] # h1 2 3 4 5 6 7 the var of measurement

fisher = fisher(opt_x,y_observed, tau)
sig = np.sqrt(np.linalg.inv(fisher)) # variance of parameters
