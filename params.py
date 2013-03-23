from constants import G_grav
import numpy as np

class BGParams(object):
    """ Stores parameters necessary for the background evolution of the universe. """

    def __init__(self, 
                 H_0,
                 omega_b, 
                 omega_m,
                 omega_r,
                 omega_nu,
                 omega_lambda,
                 T_0):
        self.H_0 = H_0
        self.omega_b = omega_b
        self.omega_m = omega_m
        self.omega_r = omega_r
        self.omega_nu = omega_nu
        self.omega_lambda = omega_lambda
        self.T_0 = T_0

        #Derived parameters
        self.rho_c = 0.375 * self.H_0 ** 2 / np.pi / G_grav 

