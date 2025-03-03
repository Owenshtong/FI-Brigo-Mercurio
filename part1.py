# Functions for Part1 #
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import ncx2


# (1)
def _P_interpolation(t, t0, t1, p0, p1):
    """
    :param t: t \in [t0,t1] to be interpolated
    :param t0: right end on t
    :param t1: left end on t
    :param p0: fun value at t0
    :param p1: fun value at t1
    :return: interpolated p(t)
    """
    return (t1 - t0)**(-1) * (p0 * (t1 - t) + p1 * (t -t0))

def _root_equation(p_v):
    """
    :param p_v: pd.series of discount factor
    :return: a function of target discount
    """

    T0 = p_v.index.max() # Interpolate after this envolves the root
    P0 = p_v[T0]
    def _to_be_solved(x,c,T):
        """
        :param x: the discount factor to be solved
        :param c: coupon rate
        :param T: maturity of the coupon bearing bond
        """
        coupon = 100 * c/2
        acc = 0
        for i in range(int(T0/0.5)): # number of semi-years
            t_year = (1+i) * 0.5
            lt = max(p_v.index[p_v.index < t_year])
            ut = min(p_v.index[p_v.index >= t_year])
            lp = p_v[lt]
            up = p_v[ut]
            acc += coupon * _P_interpolation(t_year, lt, ut, lp, up)
        for i in range(int(T0/0.5), int(T/0.5)):
            t_year = (1 + i) * 0.5
            acc += coupon * _P_interpolation(t_year, T0, T, P0, x)
        acc += 100 * x
        return acc - 100
    return  _to_be_solved

# (2)
def nss(t,a,b,c,d, tau, theta):
    f1 = 1
    f2 = (1 - np.exp(-t / tau))/(t/tau)
    f3 = f2 - np.exp(-t/tau)
    f4 = (1 - np.exp(-t / theta))/(t/theta)- np.exp(-t/theta)
    return a*f1 + b*f2 + c*f3 + d*f4
def yield_plt(lb, ub, prarm):
    x = np.linspace(lb, ub, 500)
    nss_vect = np.vectorize(nss, excluded=["a",'b','c',"d", "tau", "theta"])
    y = nss_vect(x, *prarm)
    plt.plot(x,y, s=5)
    plt.show()

# (3)
# P(0,t) ZCB from the nss curve
def P(t,a,b,c,d, tau, theta):
    return np.exp(-1 * nss(t,a,b,c,d, tau, theta) * t)

# proportional forward rate, t2 > t1
def F(Pt1, Pt2, t1, t2):
    return ((Pt1 / Pt2) - 1) * (1 / (t2 - t1))

# Black formula for market cap
def bms_cap(K ,T ,sigma, pack, tau = .25, M = 1):
    """
    :param K: strike
    :param T: ANNUAL Maturity
    :param sigma: combined IV
    :param pack: the NSS parameters
    :param tau: Settlement period
    :param M: Nominal amount
    :return: The market cap price
    """
    n = T / tau

    cap = 0
    for j in range(2, int(n) + 1):
        # example: 1 year cap has 3 settlement. Start with 2.
        Pj_1 = P(tau * (j-1), *pack)
        Pj = P(tau * j, *pack)
        Fj_1 = F(Pj_1, Pj, tau * (j-1), tau * j)
        vj = sigma * np.sqrt(tau * (j-1))

        d1 = (1/vj) * (np.log(Fj_1 / K) + vj**2 / 2)
        d2 = d1 - vj

        cap += M * tau * Pj * (Fj_1 * norm.cdf(d1) - K * norm.cdf(d2))
    return cap

#instantaneous forward rate from NSS
def nss_f(t,a,b,c,d, tau, theta):
    return a + b*np.exp(-1 * t/tau) + \
        c* (t/tau) * np.exp(-1 * t/tau) + d * t/theta * np.exp(-1 * t/theta)

# Cap ATM K level
def K_atm(T, nss_para, tau = 0.25):
    # nss_para is the list of parameters of NSS curve in order of a,b,c,d,tau, theta
    n = T/tau
    acc = 0
    for i in range(2, int(n + 1)):
        acc += tau * P(tau * i, *nss_para)
    return (1 - P(T, *nss_para)) / acc


### BrigoMercurio universe ###
# The single put written on zcb
def BM_put_ZCB(t, K, Tg, T, kappa, theta, sigma, rt, gamma, x0, pack_nss):
    # Market price of P
    Pmt, PmTg, PmT =  P(t, *pack_nss), P(Tg, *pack_nss), P(T, *pack_nss)
    ft = nss_f(t,*pack_nss)

    # constants
    PbmtT = P_bm(t, T, Pmt, PmT, ft, kappa, theta, sigma, rt, gamma, x0)
    PbmtTg = P_bm(t, Tg, Pmt, PmTg, ft, kappa, theta, sigma, rt, gamma, x0)

    # a & b
    a = lambda t, tt: a_cir(t, tt, kappa, theta, sigma, gamma)
    b = lambda t, tt: b_cir(t, tt, kappa, theta)

    acir_TgT, acir_0T, acir_0Tg, bcir_TgT, bcir_0T, bcir_0Tg = a(Tg, T), a(0,T), a(0,Tg), b(Tg, T), b(0,T), b(0,Tg)

    # greeks
    rhat = (1 / bcir_TgT) * (
            np.log(acir_TgT/K) - np.log((PmTg * acir_0T * np.exp(-1 * bcir_0T * x0)) / (PmT * acir_0Tg * np.exp(-1 * bcir_0Tg* x0)))
    )
    phi = 2 * gamma / (sigma**2 * (np.exp(gamma * (T - t)) - 1))
    psi = (gamma + kappa)/sigma**2
    xi = phi + psi + bcir_TgT
    shift_t = shift(t, ft, kappa, theta, gamma, x0)

    # print(rhat, phi, psi, shift_t)

    # probabilities
    chi1 = ncx2.cdf(2 * xi * rhat,
                    df = 4 * kappa * theta / sigma**2,
                    nc = (2 * phi**2 * (rt - shift_t) * np.exp(gamma * (Tg - t)))/xi)
    chi2 = ncx2.cdf(2 * (phi + psi) * rhat,
                    df=4 * kappa * theta / sigma ** 2,
                    nc=(2 * phi ** 2 * (rt - shift_t) * np.exp(gamma * (Tg - t))) / (phi + psi))

    # Put premium
    g = K * PbmtTg * (1 - chi2) - PbmtT * (1 - chi1)
    return g * (1 / K)


def BrigoMercurio_cap(t, K, T,
                      kappa, theta, sigma, rt, gamma, x0,
                      pack_nss ,
                      tau = 0.25):
    """
    :param tau: reset period
    :param t: o.oo3 (1 day or around)
    :param K: The strike for all puts
    :param T: Maturity of the CAP
    :return: Price of the cap
    """
    n = T/tau
    acc = 0
    for j in range(2, int(n) + 1):
        Tg, T = (j-1) * tau, j * tau
        put_j = BM_put_ZCB(t, K, Tg, T,   kappa, theta, sigma, rt, gamma, x0, pack_nss)
        acc += put_j
    return acc


def a_cir(t1, t2, kappa, theta, sigma, gamma):
    numer = 2 * gamma * np.exp(((kappa + gamma)/2) * (t2 - t1))
    denom = (kappa + gamma) * (np.exp(gamma * (t2 - t1)) - 1) + 2 * gamma
    return  (numer / denom) ** (2 * kappa * theta / sigma**2)

def b_cir(t1, t2, kappa, gamma):
    numer = 2 * (np.exp(gamma * (t2 - t1)) - 1)                                  ## Check -1 inside or outside
    denom = (kappa + gamma) * (np.exp(gamma * (t2 - t1)) - 1) + 2 * gamma
    return numer / denom

def a_bm(t1, t2, Pmt1, Pmt2, ft, kappa, theta, sigma, gamma, x0):
    a = lambda t, tt: a_cir(t,tt, kappa, theta, sigma, gamma)
    b = lambda t, tt: b_cir(t,tt, kappa, theta)

    numerator   = Pmt2 * a(0,t1) * np.exp(-1 * b(0,t1) * x0) * a(t1,t2) * np.exp(b(t1,t2) * ft)
    denominator = Pmt1 * a(0,t2) * np.exp(-1 * b(0,t2) * x0)

    return numerator / denominator

def b_bm(t1, t2, kappa, gamma):
    return b_cir(t1, t2, kappa, gamma)

def P_bm(t1, t2, Pmt1, Pmt2, ft, kappa, theta, sigma, rt, gamma, x0):
    a = a_bm(t1, t2, Pmt1, Pmt2, ft, kappa, theta, sigma, gamma, x0)
    b = b_bm(t1, t2, kappa, gamma)
    return a * np.exp(-1 * b * rt)

def shift(t, fm_0t, kappa, theta, gamma, x0):
    # fm_0t is market instantaneous forward rate. From nss or other smoothed yield
    denominator = (kappa + gamma) * (np.exp(gamma * t) - 1) + 2 * gamma
    term1 = fm_0t
    term2 = kappa * theta * (np.exp(gamma * t)-1) / denominator
    term3 = x0 * 4 * gamma**2 * np.exp(gamma * t) / denominator**2
    return term1 + term2 + term3





def CIR_yield(t1,t2,rt, kappa, theta, sigma):
    gamma = np.sqrt(kappa**2 + 2 * sigma**2)
    a = a_cir(t1, t2, kappa, theta, sigma,gamma)
    b = b_cir(t1, t2, kappa, gamma)
    return - 1/(t2-t1) * (np.log(a) - b * rt)


