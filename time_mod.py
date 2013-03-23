import numpy as np
from scipy.integrate import cumtrapz
#import scipy.interpolate
import interpolation
from constants import s_o_l

def get_a(x):
    """ Returns the scale factor for a corresponding x value """

    return np.exp(x)

def get_H(x, bg_params):
    """ Calculates the Hubble parameter as a function of x = ln a.

    bg_params is a params.BGParams instance.

    """
    a = np.exp(x)

    return bg_params.H_0 * np.sqrt((bg_params.omega_b + bg_params.omega_m) * (a ** -3) + (bg_params.omega_r + bg_params.omega_nu) * (a ** -4) + bg_params.omega_lambda)

def get_H_p(x, bg_params):
    """ Calculates the scaled Hubble parameter as a function of x = ln a.

    bg_params is a params.BGParams instance.

    """

    return np.exp(x) * get_H(x, bg_params)

def get_dH_p(x, bg_params):
    """ Calculates the derivative of the scaled Hubble parameter wrt. x = ln a """
    a = np.exp(x)

    return get_H_p(x, bg_params) - 0.5 * bg_params.H_0 * ((3 * (bg_params.omega_m + bg_params.omega_b) * (a ** -2) + 4 * (bg_params.omega_r + bg_params.omega_nu) * (a ** -3)) * ((bg_params.omega_m + bg_params.omega_b) * (a ** -3) + (bg_params.omega_r + bg_params.omega_nu) * (a ** -4) + bg_params.omega_lambda) ** -0.5)

def get_xgrid(numpoint_recomb, 
              numpoint_postrecomb, 
              z_start_recomb, 
              z_end_recomb):
    """ Calculates a grid of x values given number of points during and after recombination. """

    x_start_recomb = -np.log(1 + z_start_recomb)
    x_end_recomb = -np.log(1 + z_end_recomb)
    x_today = 0
    return np.append(np.linspace(x_start_recomb,
                                x_end_recomb,
                                numpoint_recomb,
                                endpoint=False), #Endpoint is the same as start point for the next array
                    np.linspace(x_end_recomb,
                                x_today,
                                numpoint_postrecomb))
def get_kandxmeshgrids(numpoint_recomb,
                       numpoint_postrecomb,
                       z_start_recomb,
                       z_end_recomb,
                       numpoint_k,
                       k_start,
                       k_end):
    """ Makes meshgrids for k and x, where the spacing in x is non-uniform. """
    
    x_start_recomb = -np.log(1 + z_start_recomb)
    x_end_recomb = -np.log(1 + z_end_recomb)
    x_today = 0
    numpoint_x = numpoint_recomb + numpoint_postrecomb
    dx = (x_today - x_end_recomb) / numpoint_postrecomb
    kgrid = (np.mgrid[k_start:k_end:complex(0, numpoint_k),0:1:complex(0, numpoint_x)])[0]
    xgrid1 = (np.mgrid[0:1:complex(0, numpoint_k), x_start_recomb:x_end_recomb:complex(0, numpoint_recomb)])[1]
    xgrid2 = (np.mgrid[0:1:complex(0, numpoint_k), x_end_recomb + dx:x_today:complex(0, numpoint_postrecomb)])[1]
    xgrid = np.append(xgrid1, xgrid2, axis=1)
    return kgrid, xgrid


class EtaSolver(object):
    """ Object to handle eta computations. 

    Upon initialization, either takes start-and-endpoints for a log-a 
    array, or the array itself. Also needs background parameters in the form
    of a params.BGParams instance. The start-and-endpoints are for the values
    of a, not for log(a).
    
    """

    def __init__(self, 
                 bg_params, 
                 a_start=None, 
                 a_end=None, 
                 a_npoints=None, 
                 x_array=None,
                 eta_init=0):
        if x_array is None:
            self.x_array = np.linspace(np.log(a_start), np.log(a_end), a_npoints)
        else:
            self.x_array = x_array
        self.a_array = np.exp(self.x_array)
        self.eta_array = None
        self.eta_spline = None
        self.bg_params = bg_params
        if eta_init != 0:
            raise NotImplementedError('Eta_init must currently be zero')
        self.eta_init = eta_init

    def get_eta(self, x):
        """ Using input arrays, produces eta for a given value of x = ln a. """

        if self.eta_spline is None:
            if self.eta_array is None:
                #Initialize the eta array based on the scale factor values. 
                #We need to append to an array with a single entry because 
                #cumtrapz returns an array with one less value than the input 
                #array
                self.eta_array = np.append(np.array([self.eta_init]), s_o_l * cumtrapz(1 / (get_H_p(self.x_array, self.bg_params)), self.x_array))
            self.eta_spline = interpolation.cubic_spline(self.x_array, self.eta_array)
#            self.eta_splint = scipy.interpolate.interp1d(self.a_array, self.eta_array, kind='cubic')
        if isinstance(x, np.ndarray):
            x[x > self.x_array[-1]] = self.x_array[-1]
        elif isinstance(x, float):
            if x > self.x_array[-1]:
                x = self.x_array[-1]
        return interpolation.splint(self.x_array, self.eta_array, self.eta_spline, x)
