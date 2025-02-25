#### part 2 ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import direct
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from scipy.optimize import dual_annealing
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import part1
import part2

# (1)
nss_coeff = pd.read_excel("TP1 data 60201 W2025.xlsx", index_col=0)/100
tau = [0.25, 0.5, 1,3,5,10,30]
nss_vect = np.vectorize(part1.nss, excluded=["t"])
vault = []
for i in nss_coeff.index:
    nss_pack = nss_coeff.loc[i]
    yi = nss_vect(tau, *nss_pack)
    vault.extend(list(yi))
y_observed = pd.DataFrame(np.reshape(vault, (len(nss_coeff),7)), index=nss_coeff.index, columns=tau)
summ_stat = round(y_observed.describe().iloc[[1,2,4,5,6],] ,4)

# (2)

l4 = [7.000e-01, 2.752e-02, 5.209e-02, -2.000e+00, 1.000e-05, 1.000e-05, 6.723e-05, 1.855e-04, 2.121e-04, 2.333e-04, 2.491e-04] #-7780
l5 = [9.000e-01, 2.771e-02, 5.501e-02, -1.815e+00, 1.000e-05, 1.000e-05, 6.532e-05, 1.814e-04, 2.094e-04, 2.323e-04, 2.489e-04] #-7801
l6 = [5.150e-01, 2.750e-02, 1.247e-01, -6.666e-01, 1.927e-05, 1.000e-07, 4.611e-05, 1.668e-04, 1.983e-04, 2.251e-04, 2.463e-04] #-7810
l7 = [6.990e-01, 2.740e-02, 1.001e-01, -6.002e-01, 1.737e-05, 1.000e-06, 4.790e-05, 1.672e-04, 1.993e-04, 2.265e-04, 2.468e-04] #-7820
l8 = [8.990e-01, 2.768e-02, 1.008e-01, -6.035e-01, 1.649e-05, 1.000e-06, 4.720e-05, 1.663e-04, 1.993e-04, 2.269e-04, 2.469e-04] #-7820
l9 =[8.990e-01  ,2.768e-02,  1.008e-01, -6.035e-01,  1.656e-05, 9.503e-07,  4.711e-05,  1.663e-04,  1.993e-04,  2.269e-04, 2.469e-04]#-7820
l10 = [2.789e+00 , 2.891e-02 , 8.438e-02 ,-1.998e+00  ,1.058e-07, 9.480e-06  ,6.632e-05 , 1.790e-04  ,2.097e-04 ,2.344e-04, 2.515e-04] # -7970.51
l_best = [2.788e+00 ,2.893e-02,  8.416e-02, -1.000e+01,  1.070e-07, 9.479e-06,  6.631e-05,  1.790e-04,  2.095e-04,  2.340e-04,  2.512e-04] #-7971
l_toy = [2.795e+00 ,2.895e-02,  8.408e-02, -29.000,  1.070e-07, 9.479e-06,  6.631e-05,  1.790e-04,  2.095e-04,  2.340e-04,  2.512e-04] #-7971

def neg_L_sum(pack):
    LLh, _, _=  part2.L_sum(
        *pack[0:4], pack[4:], y_observed, tau
    )
    nLLh = -1 * LLh
    print(nLLh)
    # print(pack)
    return nLLh

# bond_kts = [(.0001,0.5)] * 3
k_bond = [(0.1,0.5)]
theta_bond = [(0,0.1)]
sigma_bond = [(0.1, 1)]
lambda_bond = [(-0.8, -.1)]
h_bond = [(1e-7, 1e-2)]
bond = k_bond+theta_bond+sigma_bond+lambda_bond+h_bond*7

x0 = opt_x
# cons = [{'type': 'ineq', 'fun': 2 * x[0] * x[1] - x[2]**2}]
opt_pack = minimize(neg_L_sum,
                    x0 =x0,
                    bounds = bond,
                    method="L-BFGS-B",
                    # constraints=[{'type': 'ineq', 'fun': lambda x: -2 * x[0] * x[1] + x[2]**2}]
                   )
# opt_pack = differential_evolution(neg_L_sum, bounds=bond)
# opt_pack = dual_annealing(neg_L_sum, x0 = x0, bounds=bond)
opt_x = opt_pack["x"]

#####  Report
THETA_opt = opt_x[:4] # kappa theta sigma lambda


LLH, rt_path, ME = part2.L_sum(*opt_x[:3], opt_x[4], opt_x[4:11], y_observed , tau)
likelihood = LLH # Likelihood
ME_mean = ME.describe().loc["mean"] # mean of measurement error
# ME_var = ME.describe().loc["std"]**2 # var of measurement error (similar to h1 to h7)
ME_var = opt_x[4:] # h1 2 3 4 5 6 7 the var of measurement

fisher = part2.fisher(opt_x,y_observed, tau)
sig = np.sqrt(-1 * np.linalg.inv(fisher)) # variance of parameters

# (3)
# Backout rt path
opt_x = l_best
LLH, rt_path, ME = part2.L_sum(*opt_x[:3], opt_x[4], opt_x[4:11], y_observed , tau)
y_hat = y_observed + ME

for i in range(7):
    fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
    ax.plot(y_observed.index,y_hat.iloc[:,i], marker = "D", markersize = 2, label = "Estimated")
    ax.plot(y_observed.index, y_observed.iloc[:,i], marker = "D", markersize = 2, label = "Observed")
    plt.title(r"$\tau = " + str(tau[i]) + "$")
    plt.ylabel("Yield")
    plt.legend()
    plt.show()


r_observed = pd.DataFrame(nss_coeff.BETA0 + nss_coeff.BETA1)
fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
ax.scatter(y_observed.index, r_observed, marker = "o", s = 5, label = "Observed")
ax.scatter(y_observed.index, rt_path, marker = "o", s = 5, label = "Estimated")
plt.title("Short rate")
plt.ylabel("Yield")
plt.legend()
plt.show()




