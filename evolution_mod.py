import numpy as np
import time_mod
import rec_mod
from constants import s_o_l
from scipy.integrate import odeint
import sys
import ode_mod
import matplotlib.pyplot as plt

def evolve_boltzmann(x, k, bg_params, rec_info, eta_info, lmax, lmax_nu, x_end_rec):
    """ x and k should be n_k * n_x grids """
    n_k = x.shape[0]
    n_x = x.shape[1]
    ind = get_tight_coupling_index(x, k, rec_info, bg_params, x_end_rec)
    #Don't necessarily have to do this for every x, could evaluate for one k
    #and copy, but this step is only done once, so very little to save here
    divH_p = 1 / time_mod.get_H_p(x, bg_params)
    divtau_p = 1 / rec_info.get_tau_primed(x)
    ck = s_o_l * k
    a = np.exp(x)
    H_0 = bg_params.H_0
    omega_nu = bg_params.omega_nu
    omega_r = bg_params.omega_r
    divf_nu = (omega_nu + omega_r) / omega_nu

    theta = np.zeros((lmax + 1, n_k, n_x))
    theta_p = np.zeros((lmax + 1, n_k, n_x))
    n = np.zeros((lmax_nu + 1, n_k, n_x))
    #Set up initial conditions
    phi = np.ones((n_k, n_x))
    delta = 1.5 * phi
    delta_b = delta
    v = 0.5 * ck * divH_p * phi
    v_b = v
    theta[0] = 0.5 * phi
    theta[1] = -ck * divH_p * phi / 6
    n[0] = 0.5 * phi
    n[1] = -ck * divH_p / 6 * phi
    n[2] = - ck ** 2 * a ** 2 * phi / (12 * H_0 ** 2 * omega_nu * (2.5 * divf_nu + 1))
    for l in xrange(3, lmax_nu + 1):
        n[l] = ck * divH_p / (2 * l + 1) * n[l-1]

    #Just some handy indices (valid after tight coupling)
    theta_p_start = lmax + 3
    n_start = 2 * lmax + 4
    n_end = 2 * lmax + lmax_nu + 5
#    print ind
#    print k[:, 0]
#    sys.exit()

    #Solve the actual equations
    #TODO - could this be done without the for loop?
    for k_i in xrange(n_k):
        print k_i
        #if k_i == 5:
        #    sys.exit()
        #First evolve until tight coupling is no longer valid
        x_tight_coupling = x[k_i, :ind[k_i]]
        y0 = np.array([])
        #This can also be done better, I am sure
        for el in (phi[k_i, 0], v_b[k_i, 0], theta[:2, k_i, 0], n[:, k_i, 0], delta_b[k_i, 0], v[k_i, 0], delta[k_i, 0]):
            y0 = np.append(y0, el)
        #Do the integration
        print 'tight_coupling'
        out = ode_mod.Output(nsave=-1, force_stepsize=True, xarr=x_tight_coupling)
        tc_solver = ode_mod.OdeInt(y0, x_tight_coupling[0], x_tight_coupling[-1], atol=1e-3, rtol=1e-3, h1=0.01, hmin=0.0, out=out, derivs=einstein_boltzmann_tight_coupling(k[k_i, 0], rec_info, eta_info, lmax, lmax_nu, bg_params))
        tc_solver.integrate()
        res = tc_solver.out.ysave
        print 'nok', tc_solver.nok
        print 'nbad', tc_solver.nbad

        phi[k_i, 0:ind[k_i]] = res[0, :]
        v_b[k_i, 0:ind[k_i]] = res[1, :]
        theta[:2, k_i, 0:ind[k_i]] = res[2:4, :]
        n[:, k_i, 0:ind[k_i]] = res[4:lmax_nu + 5, :]
        delta_b[k_i, 0:ind[k_i]] = res[lmax_nu + 5, :]
        v[k_i, 0:ind[k_i]] = res[lmax_nu + 6, :]
        delta[k_i, 0:ind[k_i]] = res[lmax_nu + 7, :]

        #Set up the derived quantities during tight coupling. Alternative here
        #is to wait to after the loop, but since ind will be k-dependent, this
        #means copying more than strictly necessary
        theta[2, k_i, 0:ind[k_i]] = -8 * ck[k_i, 0:ind[k_i]] * divH_p[k_i, 0:ind[k_i]] * divtau_p[k_i, 0:ind[k_i]] / 15 * theta[1, k_i, 0:ind[k_i]]
        for l in xrange(3, lmax + 1):
            theta[l, k_i, 0:ind[k_i]] = - l * ck[k_i, 0:ind[k_i]] / (2 * l + 1) * divH_p[k_i, 0:ind[k_i]] * divtau_p[k_i, 0:ind[k_i]] * theta[l-1, k_i, 0:ind[k_i]]
        theta_p[0, k_i, 0:ind[k_i]] = 1.25 * theta[2, k_i, 0:ind[k_i]]
        theta_p[1, k_i, 0:ind[k_i]] = -0.25 * ck[k_i, 0:ind[k_i]] * divH_p[k_i, 0:ind[k_i]] * divtau_p[k_i, 0:ind[k_i]] * theta[2, k_i, 0:ind[k_i]]
        theta_p[2, k_i, 0:ind[k_i]] = 0.25 * theta[2, k_i, 0:ind[k_i]]
        for l in xrange(3, lmax + 1):
            theta_p[l, k_i, 0:ind[k_i]] = - l / (2 * l + 1) * ck[k_i, 0:ind[k_i]] * divH_p[k_i, 0:ind[k_i]] * divtau_p[k_i, 0:ind[k_i]] * theta_p[l-1, k_i, 0:ind[k_i]]

        #Now evolve after tight coupling
        x_after_tc = x[k_i, ind[k_i] - 1:]
        y0 = np.array([])
        #Use last step of tight coupling as initial conditions for current step
        for el in (phi[k_i, ind[k_i]-1], v_b[k_i, ind[k_i]-1], theta[:,k_i, ind[k_i]-1], theta_p[:,k_i, ind[k_i]-1], n[:,k_i, ind[k_i]-1], delta_b[k_i, ind[k_i]-1], v[k_i, ind[k_i]-1], delta[k_i, ind[k_i]-1]):
            y0 = np.append(y0, el)
        print 'post'
        print 'y0', y0
        out = ode_mod.Output(nsave=-1, force_stepsize=True, xarr=x_after_tc)
        tc_solver = ode_mod.OdeInt(y0, x_after_tc[0], x_after_tc[-1], atol=1e-3, rtol=1e-3, h1=0.01, hmin=0.0, out=out, derivs=einstein_boltzmann(k[k_i, 0], rec_info, eta_info, lmax, lmax_nu, bg_params), stepper='sie')
        tc_solver.integrate()
        res = tc_solver.out.ysave
        print 'nok', tc_solver.nok
        print 'nbad', tc_solver.nbad


        phi[k_i, ind[k_i]-1:] = res[0, :]
        v_b[k_i, ind[k_i]-1:] = res[1, :]
        theta[:,k_i, ind[k_i]-1:] = res[2:theta_p_start, :]
        theta_p[:, k_i, ind[k_i] - 1:] = res[theta_p_start:n_start, :]
        n[:, k_i, ind[k_i] - 1:] = res[n_start:n_end, :]
        delta_b[k_i, ind[k_i] - 1:] = res[n_end, :]
        v[k_i, ind[k_i] - 1:] = res[n_end + 1, :]
        delta[k_i, ind[k_i] - 1:] = res[n_end + 2, :]
        np.savetxt('k%02d_phi.dat' % k_i, phi[k_i, :])
        np.savetxt('k%02d_v_b.dat' % k_i, v_b[k_i, :])
        for l in range(lmax + 1):
            np.savetxt('k%02d_theta%02d.dat' % (k_i, l), theta[l, k_i])
            np.savetxt('k%02d_theta_p%02d.dat' % (k_i, l), theta_p[l, k_i])
        for l in range(lmax_nu + 1):
            np.savetxt('k%02d_n%02d.dat' % (k_i, l), n[l, k_i])
        np.savetxt('k%02d_delta_b.dat' % k_i, delta_b[k_i, :])
        np.savetxt('k%02d_v.dat' % k_i, v[k_i, :])
        np.savetxt('k%02d_delta.dat' % k_i, delta[k_i, :])

        np.savetxt('k%02d_ckdivH_p.dat' % k_i, (ck * divH_p)[k_i, :])
        np.savetxt('k%02d_12H0divk2a2.dat' % k_i, (12 * H_0 ** 2 / ck ** 2 / a ** 2)[k_i, :])
        np.savetxt('k%02d_psi.dat' % k_i, (-phi - 12 * H_0 ** 2 / ck ** 2 / a ** 2 * (omega_r * theta[2] + omega_nu * n[2]))[k_i])

    return (phi, v_b, theta, theta_p, n, delta_b, v, delta)

def get_tight_coupling_index(x, k, rec_info, bg_params, x_end_rec):
    """ Here, x and k are assumed to be an n_k * n_x grid of values """

    n_k = x.shape[0]
    n_x = x.shape[1]
    #We only evaluate the first column of the x grid and copy the result to
    #the other columns since it is independent of k
    tau_p = np.zeros(x.shape)
    tau_p[0] = rec_info.get_tau_primed(x[0])
    tau_p[1:] = np.array((n_k - 1) * [tau_p[0]])
    #This is evaluated on a n_x * n_k grid
    ck_byH_ptau_p = s_o_l * k / (tau_p * time_mod.get_H_p(x, bg_params))
    #Again, we find the index just for the first tau array.
    ind = np.searchsorted(np.abs(tau_p[0])[::-1], 10)
    ind = n_x - ind
    #This also is independent of k, so we do this step first
    if x_end_rec < x[0, ind]:
        ind = np.searchsorted(x[0, :], x_end_rec)
    #Here, there is a k dependence so we make a list comprehension that loops
    #through all ks
    ind = np.array([ind if ck_byH_ptau_p[i, ind] <= 0.1 else np.searchsorted(ck_byH_p_tau_p[i, :], 0.1) for i in xrange(n_k)])
    #Ind should now be an n_k array
    return ind

#TODO: Rewrite Hubble-getting routines to be classes instead

class einstein_boltzmann_tight_coupling(object):
    """Class that makes it possible to call this without any other arguments than x and y"""
    def __init__(self, k, rec_info, eta_info, lmax, lmax_nu, bg_params):
        self.omega_r = bg_params.omega_r
        self.omega_b = bg_params.omega_b
        self.omega_nu = bg_params.omega_nu
        self.omega_m = bg_params.omega_m
        self.l_array = np.arange(max((lmax + 1), (lmax_nu + 1)))
        self.delta_l_2 = np.zeros(max((lmax + 1), (lmax_nu + 1)))
        self.delta_l_2[2] = 1
        self.div3 = 1 / 3.0
        self.ck = s_o_l * k
        self.k = k
        self.H_0 = bg_params.H_0
        self.eta_info = eta_info
        self.rec_info = rec_info
        self.bg_params = bg_params
        self.lmax = lmax
        self.lmax_nu = lmax_nu
        self.n_start = 4
        self.n_end = self.lmax_nu + 5

    def __call__(self, x, y):
        """ Returns dy/dx, where y is the vector of all the Einstein-Boltzmann equations, during tight coupling """
        #Set up useful aliases
        res = np.zeros(len(y))
        diva = np.exp(-x)
        tau_p = self.rec_info.get_tau_primed(x)
        tau_dp = self.rec_info.get_tau_double_primed(x)
        divH_p = 1 / time_mod.get_H_p(x, self.bg_params)
        dH_p = time_mod.get_dH_p(x, self.bg_params)
        diveta = 1 / self.eta_info.get_eta(x)
        n_start = self.n_start
        n_end = self.n_end
        lmax = self.lmax
        ck = self.ck
        lmax_nu = self.lmax_nu
        k = self.k
        H_0 = self.H_0
        div3 = self.div3
        delta_l_2 = self.delta_l_2
        l_array = self.l_array
        omega_r = self.omega_r
        omega_b = self.omega_b
        omega_nu = self.omega_nu
        omega_m = self.omega_m

        phi = y[0]
        v_b = y[1]
        theta = y[2:n_start]
        n = y[n_start:n_end]
        delta_b = y[n_end]
        v = y[n_end + 1]
        delta = y[n_end + 2]
        theta_2 = -8 * ck * divH_p / (15 * tau_p) * theta[1]

        #Start calculations
        R = 4 * omega_r * div3 * diva / omega_b
        psi = -phi - 12 * H_0 ** 2 * (omega_r * theta_2 + omega_nu * n[2]) * diva ** 2 / ck ** 2
    
        #Phi
        res[0] = psi - ck ** 2 * phi * divH_p ** 2 * div3 + H_0 ** 2 * 0.5 * divH_p ** 2 * (omega_m * delta * diva + omega_b * diva * delta_b + 4 * omega_r * diva ** 2 * theta[0] + 4 * omega_nu * diva ** 2 * n[0])
        #Theta0
        res[2] = -ck * divH_p * theta[1] - res[0]
    
        q = (-((1 - 2*R) * tau_p + (1 + R) * tau_dp) * (3 * theta[1] + v_b) - ck * divH_p * psi + (1 - dH_p * divH_p) * ck * divH_p * (-theta[0] + 2 * theta_2) - ck * divH_p * res[2]) / ((1 + R) * tau_p + dH_p * divH_p - 1)
        #vb
        res[1] = (-v_b - ck * divH_p * psi + R * (q + ck * divH_p*(-theta[0] + 2 * theta_2) - ck * divH_p * psi)) / (1 + R)
    #    print 'vb', res[1]
        #Theta1
        res[3] = div3 * (q - res[1])
        #Neutrino multipoles
        res[n_start] = -ck * divH_p * n[1] - res[0]
        res[n_start + 1] = ck * divH_p * div3 * n[0] - 2 * ck * divH_p * div3 * n[2] + ck * divH_p * div3 * psi
        res[n_start+2:n_end-1] = ck * divH_p * l_array[2:lmax_nu] / (2 * l_array[2:lmax_nu] + 1) * n[1:lmax_nu-1] - (l_array[2:lmax_nu] + 1) * ck * divH_p * (2 * l_array[2:lmax_nu] + 1) * n[3:lmax_nu + 1]
        res[n_end - 1] = ck * divH_p * n[lmax_nu-1] - (lmax_nu + 1) * n[lmax_nu] * divH_p * diveta
        #db, v, d
        res[n_end] = ck * divH_p * v_b - 3 * res[0]
        res[n_end + 1] = -v - ck * divH_p * psi
        res[n_end + 2] = ck * divH_p * v - 3 * res[0]
    #    print 'v', res[n_end + 1]
    
        return res
    
#def einstein_boltzmann(y, x, k, rec_info, eta_info, lmax, lmax_nu, bg_params):
class einstein_boltzmann(object):
    """Handles post-tight-coupling equation"""

    def __init__(self, k, rec_info, eta_info, lmax, lmax_nu, bg_params):
        self.bg_params = bg_params
        self.l_array = np.arange(max((lmax + 1), (lmax_nu + 1)))
        self.delta_l_2 = np.zeros(max((lmax + 1), (lmax_nu + 1)))
        self.delta_l_2[2] = 1
        self.div3 = 1/3.0
        self.ck = s_o_l * k
        self.divck = 1 / self.ck
        self.k = k
        self.rec_info = rec_info
        self.eta_info = eta_info
        self.lmax = lmax
        self.lmax_nu = lmax_nu
        self.theta_p_start = lmax + 3
        self.n_start = 2 * lmax + 4
        self.n_end = 2 * lmax + lmax_nu + 5
    
    def __call__(self, x, y):
        """ Returns dy/dx, where y is the vector of all the Einstein-Boltzmann equations """

        #Set up useful aliases
        res = np.zeros(len(y))
        diva = np.exp(-x)
        omega_r = self.bg_params.omega_r
        omega_b = self.bg_params.omega_b
        omega_nu = self.bg_params.omega_nu
        omega_m = self.bg_params.omega_m
        #print x
        tau_p = self.rec_info.get_tau_primed(x)
        divH_p = 1 / time_mod.get_H_p(x, self.bg_params)
        l_array = self.l_array
        delta_l_2 = self.delta_l_2
        diveta = 1 / self.eta_info.get_eta(x)
        div3 = self.div3
        ck = self.ck
        H_0 = self.bg_params.H_0
        lmax = self.lmax
        lmax_nu = self.lmax_nu

        theta_p_start = self.theta_p_start
        n_start = self.n_start
        n_end = self.n_end

        phi = y[0]
        v_b = y[1]
        theta = y[2:theta_p_start]
        theta_p = y[theta_p_start:n_start]
        n = y[n_start:n_end]
        delta_b = y[n_end]
        v = y[n_end + 1]
        delta = y[n_end + 2]

        #Begin calculations
        R = 4 * omega_r * div3 * diva / omega_b
        PI = theta[2] + theta_p[0] + theta_p[2]
        psi = -phi - 12 * H_0 ** 2 * (omega_r * theta[2] + omega_nu * n[2]) * diva ** 2 / ck ** 2
        #phi
        res[0] = psi - ck ** 2 * phi * divH_p ** 2 * div3 + H_0 ** 2 * 0.5 * divH_p ** 2 * (omega_m * delta * diva + omega_b * diva * delta_b + 4 * omega_r * diva ** 2 * theta[0] + 4 * omega_nu * diva ** 2 * n[0])
        #vb
        res[1] = -v_b - ck * divH_p * psi + tau_p * R * (3 * theta[1] + v_b)
#        print 'res', res[1]
#        print 'vb', v_b
#        print 'ck', ck
#        print 'divH_p', divH_p
#        print 'psi', psi
#        print 'tau_p', tau_p
#        print 'R', R
#        print 'threethetaplusvb', 3 * theta[1] + v_b
#        print 'theta1', theta[1]
        #theta0
        res[2] = -ck * divH_p * theta[1] - res[0]
        #theta1
        res[3] = ck * div3 * divH_p * theta[0] - 2 * ck * div3 * divH_p * theta[2] + ck * div3 * divH_p * psi + tau_p * (theta[1] + div3 * v_b)
#        print 'theta0', theta[0]
#        print 'theta2', theta[2]
#        print 'theta1plusonethirdvb', theta[1] + div3 * v_b
#        print 'dphi', res[0]
#        print 'ckdivHtheta1', ck*divH_p * theta[1]
        #theta2-lmax-1
        res[4:theta_p_start-1] = ck * divH_p * l_array[2:lmax] / (2 * l_array[2:lmax] + 1) * theta[1:lmax-1] - (l_array[2:lmax] + 1) * ck * divH_p / (2 * l_array[2:lmax] + 1) * theta[3:lmax+1] + tau_p * (theta[2:lmax] - 0.1 * PI * delta_l_2[2:lmax])
#        print 'dtheta2', res[4]
#        print 'theta3', theta[3]
#        print 'thetap2', theta_p[2]
#        print 'thetap0', theta_p[0]
#        print 'pi', PI
        #thetalmax
        res[theta_p_start - 1] = ck * divH_p * theta[lmax - 1] - (lmax + 1) * divH_p * diveta * theta[lmax] + tau_p * theta[lmax]
        #Theta_p
        res[theta_p_start] = -ck * divH_p * theta_p[1] + tau_p * (theta_p[0] - 0.5 * PI)
        res[theta_p_start+1:n_start-1] = ck * divH_p * l_array[1:lmax] / (2 * l_array[1:lmax] + 1) * theta_p[0:lmax-1] - (l_array[1:lmax] + 1) * ck * divH_p / (2 * l_array[1:lmax] + 1) * theta_p[2:lmax+1] + tau_p * (theta_p[1:lmax] - 0.1 * PI * delta_l_2[1:lmax])
        res[n_start - 1] = ck * divH_p * theta_p[lmax - 1] - (lmax + 1) * divH_p * diveta * theta_p[lmax] + tau_p * theta_p[lmax]

        #Neutrino multipoles
        res[n_start] = -ck * divH_p * n[1] - res[0]
        res[n_start + 1] = ck * divH_p * div3 * n[0] - 2 * ck * divH_p * div3 * n[2] + ck * divH_p * div3 * psi
        res[n_start+2:n_end-1] = ck * divH_p * l_array[2:lmax_nu] / (2 * l_array[2:lmax_nu] + 1) * n[1:lmax_nu-1] - (l_array[2:lmax_nu] + 1) * ck * divH_p * (2 * l_array[2:lmax_nu] + 1) * n[3:lmax_nu + 1]
        res[n_end - 1] = ck * divH_p * n[lmax_nu-1] - (lmax_nu + 1) * n[lmax_nu] * divH_p * diveta
        #db, v, d
        res[n_end] = ck * divH_p * v_b - 3 * res[0]
        res[n_end + 1] = -v - ck * divH_p * psi
        res[n_end + 2] = ck * divH_p * v - 3 * res[0]

        return res

    def jacobian(self, x, y):
        diva = np.exp(-x)
        omega_r = self.bg_params.omega_r
        omega_b = self.bg_params.omega_b
        omega_nu = self.bg_params.omega_nu
        omega_m = self.bg_params.omega_m
        tau_p = self.rec_info.get_tau_primed(x)
        divH_p = 1 / time_mod.get_H_p(x, self.bg_params)
        l_array = self.l_array
        delta_l_2 = self.delta_l_2
        diveta = 1 / self.eta_info.get_eta(x)
        div3 = self.div3
        ck = self.ck
        ckdivH_p = ck * divH_p
        divck = self.divck
        H_0 = self.bg_params.H_0
        lmax = self.lmax
        lmax_nu = self.lmax_nu
    
        theta_start = 2
        theta_p_start = self.theta_p_start
        n_start = self.n_start
        n_end = self.n_end
    
        R = 4 * omega_r * div3 * diva / omega_b
        numvars = len(y)
        dpi = np.zeros(numvars)
        dpi[theta_start + 2] = 1
        dpi[theta_p_start] = 1
        dpi[theta_p_start+2] = 1
        dpsi = np.zeros(numvars)
        dpsi[0] = -1
        dpsi[theta_start + 2] = -12 * H_0 ** 2 * diva ** 2 * divck ** 2 * omega_r
        dpsi[n_start + 2] = -12 * H_0 ** 2 * diva ** 2 * divck ** 2 * omega_nu
        res = np.zeros((numvars, numvars))
        #Dtheta
        res[0, 0] = - ckdivH_p ** 2 * div3
        res[0, theta_start] = 2 * (H_0 * divH_p * diva) ** 2 * omega_r
        res[0, n_start] = 2 * (H_0 * divH_p * diva) ** 2 * omega_nu
        res[0, n_end] = 0.5 * diva * (H_0 * divH_p) ** 2 * omega_m
        res[0, n_end + 2] = 0.5 * diva * (H_0 * divH_p) ** 2 * omega_b
        res[0, :] += dpsi
        #Dv_b
        res[1, 1] = -1 + tau_p * R
        res[1, theta_start + 1] = 3 * tau_p * R
        res[1, :] -= ckdivH_p * dpsi
        #Dtheta0
        res[theta_start, theta_start + 1] = -ckdivH_p
        res[theta_start, :] -= res[0, :]
        #Dtheta1
        res[theta_start + 1, 1] = tau_p * div3
        res[theta_start + 1, theta_start] = ckdivH_p * div3
        res[theta_start + 1, theta_start + 1] = tau_p
        res[theta_start + 1, theta_start + 2] = -2 * ckdivH_p * div3
        res[theta_start + 1, :] -= ckdivH_p * div3 * dpsi
        #Dtheta2
        res[theta_start + 2, theta_start + 1] = 0.4 * ckdivH_p
        res[theta_start + 2, theta_start + 2] = tau_p
        res[theta_start + 2, theta_start + 3] = - 0.6 * ckdivH_p
        res[theta_start + 2, :] -= 0.1 * tau_p * dpi
        #Dthetal
        base_array = np.array([l_array * ckdivH_p / (2 * l_array + 1), [tau_p] * len(l_array), - (l_array + 1) * ckdivH_p / (2 * l_array + 1)]).T
        base_matrix = np.array([[base_array[l, i-l+1] if abs(l-i) < 2 else 0 for i in xrange(2, lmax + 1)] for l in xrange(3, lmax)])
        res[theta_start + 3:theta_p_start - 1, theta_start + 2:theta_p_start] = base_matrix
        #Dthetalmax
        res[theta_p_start-1, theta_p_start - 2] = -ckdivH_p
        res[theta_p_start - 1, theta_p_start - 1] = -(lmax + 1) * divH_p * diveta + tau_p
        #Dtheta_p0
        res[theta_p_start, theta_p_start] = tau_p
        res[theta_p_start, theta_p_start + 1] = -ckdivH_p
        res[theta_p_start, :] -= 0.5 * tau_p * dpi
        #Dtheta_pl
        base_matrix = np.array([[base_array[l, i-l+1] if abs(l-i) < 2 else 0 for i in xrange(0, lmax+1)] for l in xrange(1, lmax)])
        res[theta_p_start + 1:n_start - 1, theta_p_start:n_start] = base_matrix
        #For theta2 we have the pi contribution
        res[theta_p_start + 2, :] -= 0.1 * tau_p * dpi
        #Dtheta_plmax
        res[n_start - 1, n_start - 2] = ckdivH_p
        res[n_start - 1, n_start - 1] = - (lmax + 1) * divH_p * diveta + tau_p
        #N0
        res[n_start, n_start + 1] = -ckdivH_p
        res[n_start, :] -= res[0, :]
        #N1
        res[n_start + 1, n_start] = ckdivH_p * div3
        res[n_start + 1, n_start + 2] = -2 * ckdivH_p * div3
        res[n_start + 1, :] += ckdivH_p * div3 * dpsi
        #Nl
        base_array = np.array([l_array * ckdivH_p / (2 * l_array + 1), np.zeros(len(l_array)), - (l_array + 1) * ckdivH_p / (2 * l_array + 1)]).T
        base_matrix = np.array([[base_array[l, i-l+1] if abs(l-i) == 1 else 0 for i in xrange(1, lmax_nu + 1)] for l in xrange(2, lmax_nu)])
        res[n_start + 2:n_end - 1, n_start + 1:n_end] = base_matrix
        #Nlmax
        res[n_end-1, n_end-2] = ckdivH_p
        res[n_end-1, n_end-1] = - (lmax_nu + 1) * divH_p * diveta
        #delta_b
        res[n_end, 1] = ckdivH_p
        res[n_end, :] -= 3 * res[0, :]
        #v
        res[n_end + 1, n_end + 1] = -1
        res[n_end + 1, :] -= ckdivH_p * dpsi
        #delta
        res[n_end + 2, n_end + 1] = ckdivH_p
        res[n_end + 2, :] -= 3 * res[0, :]
    
        return res
