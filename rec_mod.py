import numpy as np
import time_mod
import interpolation
from constants import m_H, epsilon_0, m_e, alpha, k_b, hbar, s_o_l, sigma_T
from scipy.integrate import odeint, cumtrapz
from matplotlib.mlab import find
import sys

def compute_ne(x, bg_params, quantity='ne'):
    """ Returns the electron density evaluated at x = ln a. 
    
    x must be an array, while bg_params is a params.BGParams instance. 
    Quantity can be 'ne' or 'xe', returning either the electron density or
    the electron fraction.
    
    """
    T_b = bg_params.T_0 * np.exp(-x)
    n_b = bg_params.omega_b * bg_params.rho_c / m_H * np.exp(-3 * x)
    fac = 1.5 * (np.log(0.5 * m_e * T_b * k_b / np.pi)) - epsilon_0 / (T_b * k_b) - 3 * np.log(hbar) - np.log(n_b)
    saha_factor_divbynb = np.exp(fac)
    frac = 0.5 * (-saha_factor_divbynb + np.sqrt(saha_factor_divbynb * saha_factor_divbynb + 4 * saha_factor_divbynb))
    #Just a small correction since some of the numbers come out slightly higher     #than 1
    frac[frac > 1] = 1

    #Do the proper Peebles equation where the Saha equation does not apply
    peebles_inds = find(frac < 0.99)
    if not all(np.diff(peebles_inds) == 1):
        raise ValueError("Bigger than 1 difference in peebles indices")

    x_solve = x[np.append(peebles_inds[0] - 1, peebles_inds)]
    def peebles_rhs(frac, x_in):
        T_b_in = bg_params.T_0 * np.exp(-x_in)
        H_in = time_mod.get_H(x_in, bg_params)

        #The various terms, physical constants and everything. Have to work in
        #log space sometimes to avoid numerical issues
        phi2 = 0.448 * np.log(epsilon_0 / (T_b_in * k_b))
        alpha2 = 64 * np.sqrt(epsilon_0 * np.pi / (27 * k_b * T_b_in)) * alpha ** 2 * hbar ** 2 / (s_o_l * m_e ** 2) * phi2
        logbeta = np.log(alpha2) + 1.5 * (np.log(m_e * k_b * T_b_in) - np.log(2 * np.pi * hbar **2)) - epsilon_0 / (k_b * T_b_in)
        logb2 = logbeta + 0.75 * epsilon_0 / (k_b * T_b_in)
        n_b = bg_params.omega_b * bg_params.rho_c / m_H * np.exp(-3 * x_in)
        n1s = (1 - frac) * n_b
        lam_al = H_in * (3 * epsilon_0 / (hbar * s_o_l)) ** 3 / ((8 * np.pi) ** 2 * n1s)
        lam_2s1s = 8.227

        cr = (lam_2s1s + lam_al) / (lam_2s1s + lam_al + np.exp(logb2))
        res = cr / H_in * (np.exp(logbeta) * (1 - frac) - n_b * alpha2 * frac ** 2)
        return res

    #Assignment is quite messy here since peebles_inds have one less element
    #than x_solve (the initial value one). Initial value is 
    #frac[peebles_inds[0] - 1], which is the last of the Saha-valid numbers
    frac[peebles_inds[0]-1:peebles_inds[-1] + 1] = odeint(peebles_rhs, y0=frac[peebles_inds[0] - 1], t=x_solve)[:, 0]

    if quantity == 'xe':
        return frac
    elif quantity == 'ne':
        return frac * n_b

class RecSolver(object):
    """ Class to handle tau, visibility function and electron density computations. 
    
    tau_init could in principle be something else than zero, but as of now,
    it is assumed in the code that tau_init is the value of tau today, and that
    it is zero.
    
    """

    def __init__(self,
                 bg_params,
                 a_start=None,
                 a_end=None,
                 a_npoints=None,
                 x_array=None,
                 tau_init=0):
        if x_array is None:
            self.x_array = np.linspace(np.log(a_start), np.log(a_end), a_npoints)
        else:
            self.x_array = x_array
        self.a_array = np.exp(self.x_array)
        self.bg_params = bg_params
        if tau_init != 0:
            raise NotImplementedError('tau_init must currently be zero')
        self.tau_init = tau_init

        #Precompute info for splining
        self.ne_array = compute_ne(self.x_array, self.bg_params)
        self.tau_der_array = -self.ne_array * s_o_l * sigma_T * np.exp(self.x_array) / time_mod.get_H_p(self.x_array, self.bg_params)
        #Reversing the array because we're starting at today and
        #integrating backwards
        self.tau_array = np.append(np.array([self.tau_init]), cumtrapz(self.tau_der_array[::-1], self.x_array[::-1]))[::-1]
        #Typically, this won't matter, and it avoids log issues
        self.tau_array[-1] = self.tau_array[-2]
        self.g_array = - self.tau_der_array * np.exp(-self.tau_array)

        #Set up splines
        self.ne_spline = interpolation.cubic_spline(self.x_array, np.log(self.ne_array))
        self.tau_spline = interpolation.cubic_spline(self.x_array, np.log(self.tau_array))
        self.tau_der_spline = interpolation.cubic_spline(self.x_array, np.log(-self.tau_der_array))

        self.g_spline = interpolation.cubic_spline(self.x_array, self.g_array)
        #Spline the second derivative of the log of g
        self.g2_spline = interpolation.cubic_spline(self.x_array, self.g_spline)

    def get_ne(self, x):
        """ Uses a precomputed spline to find the electron density at x. """

        return np.exp(interpolation.splint(self.x_array, np.log(self.ne_array), self.ne_spline, x))

    def get_tau(self, x):
        """ Uses a precomputed spline to find tau at x """

        #Hack to work with the ode solver that sometimes wants an x value
        #later than today
        if isinstance(x, np.ndarray):
            x[x > self.x_array[-1]] = self.x_array[-1]
        else:
            if x > self.x_array[-1]:
                x = self.x_array[-1]
        return np.exp(interpolation.splint(self.x_array, np.log(self.tau_array), self.tau_spline, x))

    def get_tau_primed(self, x):
        """ Uses a spline of the derivative to find dtau/dx at x """

        #Hack to work with the ode solver that sometimes wants an x value
        #later than today
        if isinstance(x, np.ndarray):
            x[x > self.x_array[-1]] = self.x_array[-1]
        else:
            if x > self.x_array[-1]:
                x = self.x_array[-1]

        return -np.exp(interpolation.splint(self.x_array, np.log(-self.tau_der_array), self.tau_der_spline, x))

    def get_tau_double_primed(self, x):
        """ Uses information in the tau primed spline to splint the derivative of this again """

        #Since what we have information about is log(-dtau/dx), we have to return
        # -dtau/dx * d log(-dtau/dx)/dx in order to actually return d2tau/dx2
        return -self.get_tau_primed(x) * interpolation.splint_deriv(self.x_array, np.log(-self.tau_der_array), self.tau_der_spline, x)

    def get_g(self, x):
        """ Uses precomputed information to compute the visibility function. """

        return interpolation.splint(self.x_array, self.g_array, self.g_spline, x)

    def get_g_primed(self, x):
        """ Uses precomputed information to compute the derivative of the visibility function. """ 

        return interpolation.splint_deriv(self.x_array, self.g_array, self.g_spline, x)

    def get_g_double_primed(self, x):
        """ Uses precomputed information to compute the double derivative of the visibility function. """

        return interpolation.splint(self.x_array, self.g_spline, self.g2_spline, x)
