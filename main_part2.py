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

## (1)
nss_coeff = pd.read_excel("TP1 data 60201 W2025.xlsx", index_col=0)/100
nss_coeff.TAU1 = nss_coeff.TAU1 * 100
nss_coeff.TAU2 = nss_coeff.TAU2 * 100


tau = [0.25, 0.5, 1,3,5,10,30]
nss_vect = np.vectorize(part1.nss, excluded=["t"])
vault = []
for i in nss_coeff.index:
    nss_pack = nss_coeff.loc[i]
    yi = nss_vect(tau, *nss_pack)
    vault.extend(list(yi))
y_observed = pd.DataFrame(np.reshape(vault, (len(nss_coeff),7)), index=nss_coeff.index, columns=tau)

summ_stat = round(y_observed.describe().iloc[[1,2,4,5,6],] ,4)
r_observed = pd.DataFrame(nss_coeff.BETA0 + nss_coeff.BETA1)

## (2)
def neg_L_sum(pack):
    LLh, _, _=  part2.L_sum(
        *pack[0:4], pack[4:], y_observed, tau
    )
    nLLh = -1 * LLh
    print(nLLh)
    print(pack)

    return nLLh

k_bond = [(.01, .3)]
theta_bond = [(0, .04)]
sigma_bond = [(0, .2)]
lambda_bond = [(-1, -.01)]
h_bond = [(1e-7, 1e-2)]
bond = k_bond + theta_bond + sigma_bond + lambda_bond + h_bond*7

x0 = [.1, .02, .1, -.1] + [10e-3]*7
opt_pack = minimize(neg_L_sum,
                    x0 = x0,
                    bounds = bond,
                    method="nelder-mead",
                    # options={'maxiter': 1000},
                    constraints=({'type': 'ineq', 'fun': lambda x: x[2]**2 - 2*x[0]*x[1]},
                                 {'type': 'ineq', 'fun': lambda x: x[3] - x[0]/x[2]}),
                   )
opt_x = opt_pack["x"]


################################# THE OPTIMAL SOLUTION ############################
opt_x  =[1.133e-01, 2.915e-02, 5.731e-02, -1.864e-02, 1.558e-05, 4.498e-06, 4.259e-07, 1.314e-05, 1.786e-05, 1.359e-05, 4.976e-05]
###################################################################################
LLH, rt_path, ME = part2.L_sum(*opt_x[:4], opt_x[4:11], y_observed , tau)
rt_path = pd.DataFrame(rt_path, index = y_observed.index)
y_hat = y_observed - ME

# Plot the rt series
fig, ax = plt.subplots(figsize=(10, 5), dpi = 100)
ax.plot(y_observed.index, r_observed, color = "red", marker  = "o", markersize=2, linestyle = "-.", label = "Observed")
ax.plot(y_observed.index, rt_path,color = "black",marker  = "o",markersize=2, label = "Estimated")
plt.title("Short rate paths over time")
plt.ylabel("Yield")
plt.legend(facecolor='none', edgecolor='none')
plt.gcf()
plt.savefig("plot/rt.pdf")
plt.show()

# Plot different yield return over time
for i in range(7):
    fig, ax = plt.subplots(figsize=(10, 5), dpi = 200)
    ax.plot(y_observed.index,y_hat.iloc[:,i], marker = "D", markersize = 2, label = "Estimated")
    ax.plot(y_observed.index, y_observed.iloc[:,i], marker = "D", markersize = 2, label = "Observed")
    plt.title(r"$\tau = " + str(tau[i]) + "$")
    plt.ylabel("Yield")
    plt.legend()
    plt.gcf()
    plt.savefig("plot/y_"+str(tau[i])+".pdf")
    plt.show()

#  Report satistics
THETA_opt = opt_x[:4] # kappa theta sigma lambda

LLH, rt_path, ME = part2.L_sum(*opt_x[:3], opt_x[4], opt_x[4:11], y_observed , tau)
likelihood = LLH # Likelihood
ME_mean = ME.describe().loc["mean"] # mean of measurement error
ME_var = opt_x[4:] # h1 2 3 4 5 6 7 the var of measurement

fisher = part2.fisher(opt_x,y_observed, tau)
sig = np.sqrt(-1 * np.linalg.inv(fisher)) # variance of parameters


