# Functions for part2
import numpy as np
import part1
import pandas as pd
import copy as copy
# (1)
a_vect = np.vectorize(part1.a_cir, excluded=["t1","kappa", "theta", "sigma","gamma", "theta"])
b_vect = np.vectorize(part1.b_cir, excluded=["t1","kappa", "gamma"])
def Lt(kappa, theta, sigma, h1_to_7,
       rt1, Pt1, yt, tau):
    # first 5 are parameters h1_to_7 is a list of 7
    # r_t1 is previous estimated short rate
    # P_t1 is the previous estimated r_t1 variance
    # yt is a list of yield observed at time t
    # tau is the list of maturity of yt observed

    T = len(tau) # number of maturity
    tau = np.array(tau)
    yt = np.array(yt).reshape((T,1))

    H = np.diag(h1_to_7)
    gamma =  np.sqrt(kappa**2 + 2 * sigma**2)

    B = np.reshape(b_vect(t1 = 0, t2 = tau, kappa = kappa,
               gamma = gamma) * (1 / tau),
                   (T,1))

    A = np.reshape(
        np.log(a_vect(t1 = 0, t2 = tau, kappa = kappa,
               gamma = gamma, sigma = sigma,theta = theta)) * (-1 / tau),
                   (T, 1))

    delta = len(h1_to_7)
    C = theta * (1 - np.exp(-kappa * delta/365))
    D = np.exp(-kappa * delta/365)

    phi = rt1 * (sigma**2 / kappa) * (D - D**2) + theta *  (sigma**2 / kappa) * .5 * (1 - D)**2

    # Hat r and P
    rhat_t = C + D * rt1
    Phat_t = D**2 * Pt1 + phi

    # vt
    vt = H + Phat_t * B @ B.T
    # vt = np.diag(np.diag(vt)) # Try?

    # Gamma t
    Lambda_t = yt - A - rhat_t * B
    # print(Lambda_t)

    # Kalman gain (a row vector)
    kgt = Phat_t * B.T @ np.linalg.inv(vt)

    # Output 1:  log llh
    Lt = -.5 * (np.log(np.linalg.det(vt)) + Lambda_t.T @ np.linalg.inv(vt) @ Lambda_t)
    Lt = Lt[0,0]

    # Output 2: updated rt
    rt = rhat_t + kgt @ Lambda_t

    # Output 3: updated Pt
    Pt = Phat_t - kgt @ B * Phat_t

    # Output 4: Meausrement error

    return Lt, Pt, rt, Lambda_t

def L_sum(kappa, theta, sigma, lamb, h1_to_7, df_yield ,tau):
    T = len(df_yield)
    n = len(h1_to_7)

    # Initialize by un-conditional expectation
    rt1 = kappa * theta / (kappa - sigma * lamb)
    Pt1 = sigma**2 * kappa * theta / (2 * (kappa - sigma * lamb)**2)

    L_acc = 0
    rt_vault = []
    Lambda_vault = []
    for i in df_yield.index:
        yt = df_yield.loc[i]
        Llh_t, Pt, rt, Lambda_t = Lt(kappa, theta, sigma, h1_to_7, rt1, Pt1, yt, tau)
        rt1, Pt1 = rt, Pt
        L_acc += Llh_t
        rt_vault.append(rt1[0,0])

        Lambda_t = Lambda_t.reshape((1, n))
        Lambda_vault.extend(list(Lambda_t[0]))
    Lambda_vault = pd.DataFrame(np.array(Lambda_vault).reshape((T,n)),
                                index=df_yield.index, columns=tau)



    return L_acc, rt_vault, Lambda_vault

def fisher(pack, y, tau, epsilon = 10e-10):
    # pack in order kappa theta sigma h1to7
    # y is the yield
    dig = []
    for i in range(4):
        pack_min = copy.copy(pack)
        pack_plu = copy.copy(pack)

        pack_min[i] += epsilon
        pack_plu[i] += epsilon

        _2nd_prox = (L_sum(*pack_plu[:4], pack_plu[4:], y, tau)[0]\
                    - 2 *  L_sum(*pack[:4], pack[4:], y, tau)[0] \
                    +  L_sum(*pack_min[:4], pack_min[4:], y, tau)[0]) / (epsilon **2)
        dig.append(_2nd_prox)

    return np.diag(dig)
