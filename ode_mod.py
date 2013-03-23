import numpy as np
import sys
#import scipy.linalg

class Output(object):
    def __init__(self, nsave=None, force_stepsize=False, xarr=None):
        self.do_output = True
        self.force_stepsize = force_stepsize
        if nsave is not None:
            if not force_stepsize:
                self.nsave = nsave
                self.count_dense = 0
                self.count = 0
                self.xsave_dense = np.empty(0)
                self.xsave = np.empty(0)
                self.dense = self.nsave > 0
            else:
                self.xarr = xarr
                self.dense = False
                self.xsave = np.empty(0)
                self.count = 0
                self.nsave = nsave
        else:
            self.xsave = np.empty(0)
            self.do_output = False
            self.dense = False
            self.count = 0

    def init(self, neqn, xlo, xhi):
        self.nvar = neqn
        if not self.do_output: return
        if self.dense:
            self.ysave_dense = np.empty((self.nvar, 0))
            self.ysave = np.empty((self.nvar, 0))
            self.x1 = xlo
            self.x2 = xhi
            self.xout = self.x1
            self.dxout = (self.x2 - self.x1) / self.nsave
        else:
            if self.force_stepsize:
                self.ysave = np.empty((self.nvar, 0))
                if self.xarr is None:
                    self.xarr = np.linspace(xlo, xhi, self.nsave)
            else:
                self.ysave = np.empty((self.nvar, 0))

    def save_dense(self, s, xout, h):
        self.ysave_dense = np.append(self.ysave_dense, np.reshape(s.dense_out(xout, h), (self.nvar, 1)), axis=1)
        self.xsave_dense = np.append(self.xsave_dense, xout)
        self.count_dense += 1

    def save(self, x, y):
        if not self.do_output: return
        self.ysave = np.append(self.ysave, np.reshape(y, (len(y), 1)), axis=1)
        self.xsave = np.append(self.xsave, x)
        self.count += 1

    def out(self, nstp, x, y, s, h):
        if not self.dense:
            raise ValueError("Dense output not set in Output!")
        if nstp == -1:
            self.ysave_dense = np.append(self.ysave_dense, np.reshape(y, (self.nvar, 1)), axis=1)
            self.xsave_dense = np.append(self.xsave_dense, x) 
            self.count_dense += 1
            self.xout += self.dxout
        else:
            self.save(x, y)
            while (x - self.xout) * (self.x2 - self.x1) > 0.0:
                self.save_dense(s, self.xout, h)
                self.xout += self.dxout

class OdeInt(object):
    def __init__(self, ystart, x1, x2, atol, rtol, h1, hmin, out, derivs, stepper='BS'):
        self.maxstep = 500000
        self.nvar = len(ystart)
        self.ystart = ystart
        self.nok = 0
        self.nbad = 0
        self.x1 = x1
        self.x2 = x2
        self.hmin = hmin
        self.out = out
        self.dense = out.dense
        self.force_stepsize = out.force_stepsize
        if self.dense and self.force_stepsize:
            print 'Warning - using both dense and force_stepsize makes little sense'
        self.derivs = derivs
        if stepper == 'BS':
            self.s = StepperBS(self.ystart, self.x1, atol, rtol, self.dense)
        elif stepper == 'sie':
            self.s = StepperSie(self.ystart, self.x1, atol, rtol, self.dense)
        self.eps = sys.float_info.epsilon
        if self.force_stepsize:
            if self.out.xarr is None:
                self.h = (self.x2 - self.x1) / (self.out.nsave - 1.0)
            else:
                self.h = self.out.xarr[1] - self.out.xarr[0]
        else:
            self.h = h1
        if x1>x2 and not self.force_stepsize:
            self.h = -self.h
        self.out.init(self.s.neqn, x1, x2)

    def integrate(self):
        self.s.dydx = self.derivs(self.s.x, self.s.y)
        if self.dense:
            self.out.out(-1, self.s.x, self.s.y, self.s, self.h)
        else:
            self.out.save(self.s.x, self.s.y)
        arrcount = 1
        for nstp in xrange(self.maxstep):
            laststep = False
            print 'x', self.s.x
            if not self.force_stepsize:
                if (self.s.x+self.h*1.0001 - self.x2) * (self.x2 - self.x1) > 0:
                    self.h = self.x2 - self.s.x
                    laststep = True
            if self.force_stepsize and self.h == self.s.hnext:
                self.s.step(self.h, self.derivs, allow_step_increase=False, last_step=laststep)
            else:
                self.s.step(self.h, self.derivs, last_step=laststep)
            if self.s.hdid == self.h:
                self.nok += 1
            else:
                self.nbad += 1
            if self.dense:
                self.out.out(nstp, self.s.x, self.s.y, self.s, self.s.hdid)
            else:
                if self.force_stepsize:
#                    if self.s.hdid == self.h:
                    if self.s.x == self.out.xarr[arrcount]:
                        #Means that we successfully integrated to the current
                        #array point, and can save it
                        self.out.save(self.s.x, self.s.y)
                        arrcount += 1
                else:
                    self.out.save(self.s.x, self.s.y)
            if (self.s.x - self.x2) * (self.x2 - self.x1) >= 0:
                self.ystart = self.s.y
                if self.out.do_output:
                    if self.dense and np.abs(self.out.xsave_dense[self.out.count_dense-1] - self.x2) > 100.0 * np.abs(self.x2) * self.eps:
                        self.out.out(-1, self.s.x, self.s.y, self.s, self.s.hdid)
                    elif not self.dense and np.abs(self.out.xsave[self.out.count-1] - self.x2) > 100.0 * np.abs(self.x2) * self.eps:
                        self.out.save(self.s.x, self.s.y)
                return self.ystart
            if np.abs(self.s.hnext) <= self.hmin:
                raise ValueError("Step size too small in Odeint")
            if not self.force_stepsize:
                self.h = self.s.hnext
            else:
                self.h = min(self.out.xarr[arrcount] - self.s.x, self.s.hnext)
        raise ValueError("Too many steps in routine Odeint")

#class OdeInt_grid(object):
#    """Handles the simultaneous integration of ODEs on a grid where the first dimension is the number of different systems we have, and the second dimension is the number of equations per system. The 'derivs' function should be able to accept a mask of which grid values should be returned. ystart, x1 and x2 should all be (nsys, nvar)."""
#    def __init__(self, ystart, x1, x2, atol, rtol, h1, hmin, out, derivs, stepper='BS'):
#        self.maxstep = 500000
#        self.gridshape = ystart.shape
#        self.nsys = self.gridshape[0]
#        self.nvar = self.gridshape[1]
#        self.ystart = ystart
#        self.nok = np.zeros(self.nsys)
#        self.nbad = np.zeros(self.nsys)
#        self.x1 = x1
#        self.x2 = x2
#        self.hmin = hmin
#        self.out = out
#        self.dense = out.dense
#        self.force_stepsize = out.force_stepsize
#        if self.dense and self.force_stepsize:
#            print 'Warning - using both dense and force_stepsize makes little sense'
#        self.derivs = derivs
#        if stepper == 'BS':
#            self.s = StepperBS_grid(self.ystart, self.x1, atol, rtol, self.dense)
#        elif stepper == 'sie':
#            self.s = StepperSie_grid(self.ystart, self.x1, atol, rtol, self.dense)
#        self.eps = np.array([sys.float_info.epsilon] * self.nsys)
#        if self.force_stepsize:
#            if self.out.xarr is None:
#                self.h = (self.x2 - self.x1) / (self.out.nsave - 1.0)
#            else:
#                self.h = self.out.xarr[:, 1] - self.out.xarr[:, 0]
#        else:
#            self.h = h1
#        #FOR NOW, assumes that we either go forward or backward for all systems
##        if not self.force_stepsize:
##            self.h = np.array([-self.h[i] if x1[i] > x2[i] else self.h[i]for i in self.nsys])
#        if x1[0] > x2[0] and not self.force_stepsize:
#            self.h = -self.h
#        self.out.init(self.s.neqn, x1, x2)
#
#    def integrate(self):
#        self.s.dydx = self.derivs(self.s.x, self.s.y)
#        if self.dense:
#            self.out.out(-1, self.s.x, self.s.y, self.s, self.h)
#        else:
#            self.out.save(self.s.x, self.s.y)
#        arrcount = np.ones(self.nsys, dtype='int')
#        laststep = np.zeros(self.nsys, dtype='bool')
#        finished = np.zeros(self.nsys, dtype='bool')
#        notfinished = ~finished
##        trackfinished = np.arange(self.nsys)
#        self.ystart = np.empty((self.nsys, self.nvars))
#        for nstp in xrange(self.maxstep):
#            laststep[:] = False
#            print 'x', self.s.x
#            if not self.force_stepsize:
#                filterr = self.s.x + self.h * 1.0001 - self.x2[notfinished] * (self.x2[notfinished] - self.x1[notfinished]) > 0
#                self.h[filterr] = self.x2[notfinished][filterr] - self.s.x[filterr]
#                laststep[filterr] = True
#            if self.force_stepsize:
#                filterr = self.h == self.s.hnext
#                self.s.step(self.h, self.derivs, allow_step_increase= ~filterr, last_step=laststep)
#            else:
#                self.s.step(self.h, self.derivs, last_step=laststep)
#            self.nok[notfinished][self.s.hdid == self.h] += 1
#            self.nbad[notfinished][self.s.hdid != self.h] += 1
#            if self.dense:
#                self.out.out(nstp, self.s.x, self.s.y, self.s, self.s.hdid)
#            else:
#                if self.force_stepsize:
#                    filterr = self.s.x == self.out.xarr[notfinished, arrcount]
#                    if any(filterr):
#                        self.out.save(self.s.x, self.s.y, filterr)
#                        arrcount[filterr] += 1
#                else:
#                    self.out.save(self.s.x, self.s.y)
#            filterr = (self.s.x - self.x2[notfinished]) * (self.x2[notfinished] - self.x1[notfinished]) >= 0
#            self.ystart[notfinished][filterr] = self.s.y[filterr]
#            if self.out.do_output and any(filterr):
#                nfilter = filterr and np.abs(self.out.xsave_sense[:, self.out.count_dense-1] - self.x2[notfinished]) > 100.0 * np.abs(self.x2[notfinished]) * self.eps[notfinished]
#                if self.dense and any(nfilter):
#                    self.out.out(-1, self.s.x, self.s.y, self.s, self.s.hdid, nfilter)
#                elif not self.dense and any(nfilter):
#                    self.out.save(self.s.x, self.s.y, nfilter)
#            #Update finished arrays
#            if any(filterr):
#                finished[notfinished] = filterr.copy()
#                notfinished = ~finished
#                notfilter = ~filterr
#                self.h = self.h[notfilter].copy()
#                laststep = laststep[notfilter].copy()
#                arrcount = arrcount[notfilter].copy()
#                self.s.update_systems(notfilter)
#            if all(finished):
#                return self.ystart
#            if any(np.abs(self.s.hnext) <= self.hmin):
#                raise ValueError("Step size too small in Odeint")
#            if not self.force_stepsize:
#                self.h = self.s.hnext.copy()
#            else:
#                self.h = np.minimum(self.out.xarr[notfinished, arrcount] - self.s.x, self.s.hnext)
#        raise ValueError("Too many steps in routine Odeint")

class StepperBase(object):
    def __init__(self, y, x, atol, rtol, dense):
        self.x = x
        self.y = y
        self.atol = atol
        self.rtol = rtol
        self.dense = dense
        self.n = len(y)
        self.neqn = self.n

class StepperBS(StepperBase):
    """ Class that handles the Bulirsch-Stoer algorithm for solving ODEs. """

    def __init__(self, y, x, atol, rtol, dense):
        self.first_step = True
#        self.allow_step_increase = True
        self.reject = False
        self.prev_reject = False
        self.stepfac1 = 0.65
        self.stepfac2 = 0.94
        self.stepfac3 = 0.02
        self.stepfac4 = 4.0
        StepperBase.__init__(self, y, x, atol, rtol, dense)
        self.kmaxx = 8
        self.imaxx = self.kmaxx + 1
        self.table = np.empty((self.kmaxx, self.n))
        self.ysave = np.empty((self.imaxx, self.n))
        self.fsave = np.empty((0, self.n))
        self.dens = np.empty((2 * self.imaxx + 5) * self.n)
        self.eps = sys.float_info.epsilon
        if self.dense:
            self.nseq = np.array([4 * i + 2 for i in xrange(self.imaxx)])
        else:
            self.nseq = np.array([2 * (i + 1) for i in xrange(self.imaxx)])
        self.cost = np.cumsum(self.nseq) + 1
        self.hnext = -1.0e99
        logfact = -np.log10(max(1.0e-12, rtol)) * 0.6 + 0.5
        self.k_targ = max(1, min(self.kmaxx-1, int(logfact)))
        self.coeff = np.array([[1.0 / ((float(self.nseq[k]) / self.nseq[l]) ** 2 - 1) for l in xrange(k)] for k in xrange(self.imaxx)])
        self.coeff = np.array([[1.0 / ((float(self.nseq[k]) / self.nseq[l]) ** 2 - 1) if l < k else 0 for l in xrange(self.imaxx)] for k in xrange(self.imaxx)])
        self.errfac = np.empty(2 * self.imaxx + 1)
        for i in xrange(len(self.errfac)):
            ip5 = i + 5
            self.errfac[i] = 1.0 / (ip5 * ip5)
            e = 0.5 * np.sqrt(float(i + 1) / ip5)
            for j in xrange(i + 1):
                self.errfac[i] *= e / (j + 1)
        self.ipoint = np.empty(self.imaxx + 1, dtype=int)
        self.ipoint[0] = 0
        for i in xrange(1, self.imaxx + 1):
            njadd = 4 * i - 2
            if self.nseq[i-1] > njadd: njadd += 1
            self.ipoint[i] = self.ipoint[i-1] + njadd

    def step(self, htry, derivs, allow_step_increase=True, last_step=False):
#        self.allow_step_increase = allow_step_increase
        hopt = np.empty(self.imaxx)
        work = np.empty(self.imaxx)
        h = htry
        self.forward = h > 0
        ysav = self.y.copy()
        if self.reject:
            self.prev_reject = True
            last_step = False
        self.reject = False
        firstk = True
        hnew = np.abs(h)
        interp_error = True
        while interp_error:
            while firstk or self.reject:
                if self.forward:
    #                if self.allow_step_increase:
                    if allow_step_increase:
                        h = hnew
                    else:
                        h = min(hnew, h)
                else:
    #                if self.allow_step_increase:
                    if allow_step_increase:
                        h = -hnew
                    else:
                        h = -min(hnew, h)
                firstk = False
                self.reject = False
                if np.abs(h) <= np.abs(self.x) * self.eps:
                    raise ValueError("Step size underflow in StepperBS")
                for k in xrange(self.k_targ + 2):
                    yseq = self.modified_midpoint(ysav, h, k, derivs)
                    if k == 0:
                        self.y = yseq
                    else:
                        self.table[k-1] = yseq
                        self.polyextr(k)
                        scale = self.atol + self.rtol * np.max(np.append(np.abs(np.reshape(ysav, (1, self.n))), np.abs(np.reshape(self.y, (1, self.n))), axis=0), axis=0)
                        err = np.sqrt(np.mean((np.abs(self.y - self.table[0]) / scale) ** 2))
#                        print 'errfirst', err
                        expo = 1.0 / (2 * k + 1)
                        facmin = self.stepfac3 ** expo
                        if err == 0:
                            fac = 1.0 / facmin
                        else:
                            fac = self.stepfac2 * (err / self.stepfac1) ** -expo
                            fac = max(facmin / self.stepfac4, min(1.0 / facmin, fac))
                        hopt[k] = np.abs(h * fac)
                        work[k] = self.cost[k] / hopt[k]
#                        if (self.first_step or not self.allow_step_increase) and err <= 1.0:
#                        if (self.first_step or not allow_step_increase) and err <= 1.0:
                        if (self.first_step or last_step) and err <= 1.0:
                            khit = k
                            break
#                        if k == self.k_targ - 1 and not self.prev_reject and not self.first_step and self.allow_step_increase:
                        if k == self.k_targ - 1 and not self.prev_reject and not self.first_step and not last_step:
                            if err <= 1.0:
                                khit = k
                                break
                            elif err > (float(self.nseq[self.k_targ]) * self.nseq[self.k_targ + 1] / (self.nseq[0] * self.nseq[0])) ** 2:
                                self.reject = True
                                self.k_targ = k
                                if self.k_targ > 1 and work[k-1] < 0.8 * work[k]:
                                    self.k_targ -= 1
                                hnew = hopt[self.k_targ]
                                khit = k
                                break
                        if k == self.k_targ:
                            if err <= 1.0:
                                khit = k
                                break
                            elif err > (float(self.nseq[k+1]) / self.nseq[0]) ** 2:
                                self.reject = True
                                if self.k_targ > 1 and work[k-1] < 0.8 * work[k]:
                                    self.k_targ -= 1
                                hnew = hopt[self.k_targ]
                                khit = k
                                break
                        if k == self.k_targ + 1:
                            if err > 1.0:
                                self.reject = True
                                if self.k_targ > 1 and work[self.k_targ - 1] < 0.8 * work[self.k_targ]:
                                    self.k_targ -= 1
                                hnew = hopt[self.k_targ]
                            khit = k
                            break
                #K loop is done
                if self.reject:
                    self.prev_reject = True
            #Inner while loop is done
            dydxnew = derivs(self.x + h, self.y)
            if self.dense:
                err = self.prepare_dense(h, dydxnew, ysav, scale, khit)
#                print 'err', err
                hopt_int = h / max(err ** (1.0 / (2 * k + 3)), 0.01)
                if err > 10.0:
                    hnew = np.abs(hopt_int)
                    self.reject = True
                    self.prev_reject = True
                    #Do the whole shebang again
                    interp_error = True
                else:
                    interp_error = False
            else:
                interp_error = False
        #Setting up everything for next step
        self.dydx = dydxnew
        self.xold = self.x
        self.x += h
        self.hdid = h
        self.first_step = False
        if khit == 1:
            kopt = 2
        elif khit <= self.k_targ:
            kopt = khit
            if work[khit - 1] < 0.8 * work[khit]:
                kopt = khit - 1
            elif work[khit] < 0.9 * work[khit - 1]:
                kopt = min(khit + 1, self.kmaxx - 1)
        else:
            kopt = khit - 1
            if khit > 2 and work[khit - 2] < 0.8 * work[khit - 1]:
                kopt = khit -2
            if work[khit] < 0.9 * work[kopt]:
                kopt = min(k, self.kmaxx - 1)
        if self.prev_reject:
            self.k_targ = min(kopt, khit)
            hnew = min(np.abs(h), hopt[self.k_targ])
            self.prev_reject = False
        else:
            if kopt <= khit:
                hnew = hopt[kopt]
            else:
                if khit < self.k_targ and work[khit] < 0.9 * work[khit - 1]:
                    hnew = hopt[khit] * self.cost[kopt + 1] / self.cost[khit]
                else:
                    hnew = hopt[khit] * self.cost[kopt] / self.cost[khit]
            self.k_targ = kopt
        if self.dense:
            hnew = min(hnew, np.abs(hopt_int))
        if self.forward:
            self.hnext = hnew
        else:
            self.hnext = -hnew

    def modified_midpoint(self, y, htot, k, derivs):
        nstep = self.nseq[k]
        h = htot / nstep
        ym = y
        yn = y + h * self.dydx
        xnew = self.x + h
        #yend stores derivatives
        yend = derivs(xnew, yn)
        h2 = 2.0 * h
        for i in xrange(1, nstep):
            if self.dense and 2 * i == nstep:
                self.ysave[k] = yn
            if self.dense and np.abs(i - nstep) <= 2 * (2 * k + 1):
                self.fsave = np.append(self.fsave, np.reshape(yend, (1, len(yend))), axis=0)
            swap = ym + h2 * yend
            ym = yn
            yn = swap
            xnew += h
            yend = derivs(xnew, yn)
        if self.dense and nstep <= 2 * (2 * k + 1):
            self.fsave = np.append(self.fsave, np.reshape(yend, (1, len(yend))), axis=0)
        return 0.5 * (ym + yn + h * yend)

    def polyextr(self, k):
        for j in xrange(k-1, 0, -1):
            self.table[j-1] = self.table[j] + self.coeff[k, j] * (self.table[j] - self.table[j-1])
        self.y = self.table[0] + self.coeff[k][0] * (self.table[0] - self.y)

    def prepare_dense(self, h, dydxnew, ysav, scale, k):
        self.mu = 2 * k - 1
        self.dens[0:self.n] = ysav.copy()
        self.dens[self.n:2*self.n] = h * self.dydx
        self.dens[2*self.n:3*self.n] = self.y
        self.dens[3*self.n:4*self.n] = dydxnew * h
        for j in xrange(1, k + 1):
            dblenj = float(self.nseq[j])
            for l in xrange(j, 0, -1):
                factor = (dblenj / self.nseq[l-1]) ** 2 - 1.0
                self.ysave[l-1] = self.ysave[l] + (self.ysave[l] - self.ysave[l-1]) / factor
        self.dens[4*self.n:5*self.n] = self.ysave[0]
        for kmi in xrange(1, self.mu+1):
            kbeg = (kmi - 1) / 2
            for kk in xrange(kbeg, k+1):
                facnj = (0.5 * self.nseq[kk]) ** (kmi - 1)
                ipt = self.ipoint[kk + 1] - 2 * kk + kmi - 3
                self.ysave[kk] = self.fsave[ipt] * facnj
            for j in xrange(kbeg + 1, k + 1):
                dblenj = float(self.nseq[j])
                for l in xrange(j, kbeg, -1):
                    factor = (dblenj / self.nseq[l-1]) ** 2 - 1.0
                    self.ysave[l-1] = self.ysave[l] + (self.ysave[l] - self.ysave[l-1]) / factor
            self.dens[(kmi + 4) * self.n:(kmi + 5) * self.n] = self.ysave[kbeg] * h
            if kmi == self.mu: continue
            for kk in xrange(kmi/2, k + 1):
                lbeg = self.ipoint[kk + 1] - 1
                lend = self.ipoint[kk] + kmi
                if kmi == 1: lend += 2
                for l in xrange(lbeg, lend - 1, -2):
                    self.fsave[l] -= self.fsave[l-2]
                if kmi == 1:
                    l = lend - 2
                    self.fsave[l] -= self.dydx
            for kk in xrange(kmi / 2, k + 1):
                lbeg = self.ipoint[kk + 1] - 2
                lend = self.ipoint[kk] + kmi + 1
                for l in xrange(lbeg, lend - 1, -2):
                    self.fsave[l] -= self.fsave[l-2]
        self.dens = self.dense_interp(self.n, self.dens, self.mu)
        error = 0.0
        if self.mu >= 1:
            error = np.sqrt(np.mean((self.dens[(self.mu+4)*self.n:(self.mu+5)*self.n] / scale) ** 2)) * self.errfac[self.mu - 1]
        return error

    def dense_interp(self, neqs, y, imit):
        res = y.copy()
        a = np.empty((31, neqs))
        y0 = y[:neqs]
        y1 = y[2*neqs:3*neqs]
        yp0 = y[neqs:2*neqs]
        yp1 = y[3*neqs:4*neqs]
        ydiff = y1-y0
        aspl = -yp1 + ydiff
        bspl = yp0 - ydiff
        res[neqs:2*neqs] = ydiff
        res[2*neqs:3*neqs] = aspl
        res[3*neqs:4*neqs] = bspl
        if imit >= 0:
            ph0 = (y0 + y1) * 0.5 + 0.125 * (aspl + bspl)
            ph1 = ydiff + (aspl-bspl) * 0.25
            ph2 = -(yp0 - yp1)
            ph3 = 6.0 * (bspl-aspl)
            if imit >= 1:
                a[1] = 16.0 * (y[5*neqs:6*neqs] - ph1)
                if imit >= 3:
                    a[3] = 16.0 * (y[7*neqs:8*neqs] - ph3 + 3*a[1])
                    for im in xrange(5, imit + 1, 2):
                        fac1 = im * (im - 1) * 0.5
                        fac2 = fac1 * (im-2) * (im-3) * 2.0
                        a[im] = 16.0 * (y[(im+4)*neqs:(im+5)*neqs] + fac1 * a[im-2] - fac2 * a[im-4])
            a[0] = (y[4*neqs:5*neqs] - ph0) * 16.0
            if imit >= 2:
                a[2] = (y[neqs*6:neqs*7] - ph2 + a[0]) * 16.0
                for im in xrange(4, imit+1, 2):
                    fac1 = im * (im -1) * 0.5
                    fac2 = im * (im -1) * (im - 2) * (im-3)
                    a[im] = (y[neqs*(im+4):neqs*(im+5)] + a[im-2]*fac1 - a[im-4]*fac2)*16.0
            res[neqs*4:neqs*(imit+5)] = a[0:imit+1].flatten()
        return res

    def dense_out(self, x, h):
        theta = (x - self.xold) / h
        theta1 = 1.0 - theta
        yinterp = self.dens[:self.n] + theta * (self.dens[self.n:2*self.n] + theta1*(self.dens[2*self.n:3*self.n]*theta + self.dens[3*self.n:4*self.n] * theta1))
        if self.mu < 0:
            return yinterp
        theta05 = theta - 0.5
        t4 = (theta * theta1) ** 2
        c = self.dens[self.n*(self.mu+4):self.n*(self.mu+5)]
        for j in xrange(self.mu, 0, -1):
            c = self.dens[self.n*(j+3):self.n*(j+4)] + c*theta05/j
        yinterp += t4*c
        return yinterp

class StepperSie(StepperBase):
    def __init__(self, y, x, atol, rtol, dense):
        self.stepfac1 = 0.6
        self.stepfac2 = 0.93
        self.stepfac3 = 0.1
        self.stepfac4 = 4.0
        self.stepfac5 = 0.5
        self.kfac1 = 0.7
        self.kfac2 = 0.9
        StepperBase.__init__(self, y, x, atol, rtol, dense)
        self.kmax = 12
        self.imax = self.kmax + 1
        self.nseq = np.empty(self.imax, dtype=np.int)
        self.cost = np.empty(self.imax)
        self.table = np.empty((self.kmax, self.n))
        self.dfdy = np.empty((self.n, self.n))
        self.dfdx = np.empty(self.n)
        self.calcjac = False
#        self.a = np.empty((self.n, self.n))
        self.coeff = np.empty((self.imax, self.imax))
        self.fsave = np.empty((0, self.n))
        self.dens = np.empty((self.imax + 2) * self.n)
        self.factrl = np.empty(self.imax)
        self.costfunc = 1.0
        self.costjac = 5.0
        self.costlu = 1.0
        self.costsolve = 1.0
        self.eps = sys.float_info.epsilon
        self.jac_redo = min(1.0e-4, rtol)
        self.theta = 2.0 * self.jac_redo
        self.nseq[0] = 2
        self.nseq[1] = 3
        for i in xrange(2, self.imax):
            self.nseq[i] = 2 * self.nseq[i-2]
        self.cost[0] = self.costjac + self.costlu + self.nseq[0] * (self.costfunc + self.costsolve)
        for k in xrange(self.kmax):
            self.cost[k + 1] = self.cost[k] + (self.nseq[k + 1] - 1) * (self.costfunc + self.costsolve) + self.costlu
        self.hnext = -1.0e99
        logfact = -np.log10(self.rtol + self.atol) * 0.6 + 0.5
        self.k_targ = max(1, min(self.kmax - 1, int(logfact)))
        self.coeff[:, :] = np.array([[1.0/(float(self.nseq[k]) / self.nseq[l] - 1.0) if l < k else 0 for l in xrange(self.imax)] for k in xrange(self.imax)])
        self.factrl[0] = 1.0
        for k in xrange(self.imax - 1):
            self.factrl[k+1] = (k + 1) * self.factrl[k]
        self.first_step = True
        self.reject = False
        self.prev_reject = False
    
    def step(self, htry, derivs, allow_step_increase=True, last_step=False):
        hopt = np.empty(self.imax)
        work = np.empty(self.imax)
        work[0] = 1.e30
        h = htry
        self.forward = h > 0
        ysav = self.y.copy()
        if self.reject:
            self.prev_reject = True
            last_step = False
            self.theta = 2.0 * self.jac_redo
        scale = self.atol + self.rtol * np.abs(self.y)
        self.reject = False
        firstk = True
        hnew = np.abs(h)
        compute_jac = True
        while firstk or self.reject:
            if compute_jac:
                compute_jac = False
                if self.theta > self.jac_redo and not self.calcjac:
                    self.dfdy = derivs.jacobian(self.x, self.y)
                    self.calcjac = True
            if self.forward:
#                if self.allow_step_increase:
                if allow_step_increase:
                    h = hnew
                else:
                    h = min(hnew, h)
            else:
#                if self.allow_step_increase:
                if allow_step_increase:
                    h = -hnew
                else:
                    h = -min(hnew, h)
            firstk = False
            self.reject = False
            if np.abs(h) <= np.abs(self.x) * self.eps:
                raise ValueError("Step size underflow in StepperSie")
            self.fsave = np.empty((0, self.n))
#            print 'k_targ', self.k_targ
            for k in xrange(self.k_targ + 2):
#                print 'ysav', ysav
                success, yseq = self.semi_implicit_euler(ysav, h, k, scale, derivs)
                if not success:
                    self.reject = True
                    hnew = np.abs(h) * self.stepfac5
                    khit = k
                    break
                #print 'self.x + h', self.x + h
                #print 'yseq', yseq
#                sys.exit()
                if k == 0:
                    self.y = yseq
                else:
                    self.table[k-1] = yseq
                    self.polyextr(k)
                    err = 0.0
                    scale = self.atol + self.rtol * np.abs(ysav)
                    #print 'table', self.table[0]
                    #print 'y', self.y
                    err = np.sqrt(np.mean(((self.y - self.table[0]) / scale) ** 2))
                    #print 'err', err
                    if err > 1.0 / self.eps or (k > 1 and err >= errold):
                        self.reject = True
                        hnew = np.abs(h) * self.stepfac5
                        khit = k
                        break
                    errold = max(4.0*err, 1.0)
                    expo = 1.0 / (k + 1)
                    facmin = self.stepfac3 ** expo
                    if err == 0.0:
                        fac = 1.0 / facmin
                    else:
                        fac = self.stepfac2 / (err / self.stepfac1) ** expo
                        fac = max(facmin / self.stepfac4, min(1.0 / facmin, fac))
                    hopt[k] = np.abs(h * fac)
                    work[k] = self.cost[k] / hopt[k]
                    if (self.first_step or last_step) and err <= 1.0:
                        khit = k
                        break
                    if k == self.k_targ - 1 and not self.reject and not self.first_step and not last_step:
                        if err <= 1.0:
                            khit = k
                            break
                        elif err > self.nseq[self.k_targ] * self.nseq[self.k_targ+1] * 4.0:
                            self.reject = True
                            self.k_targ = k
                            if self.k_targ > 1 and work[k-1] < self.kfac1 * work[k]:
                                self.k_targ -= 1
                            hnew = hopt[self.k_targ]
                            khit = k
                            break
                    if k == self.k_targ:
                        if err <= 1.0:
                            khit = k
                            break
                        elif err > self.nseq[k+1] * 2.0:
                            self.reject = True
                            if self.k_targ > 1 and work[k-1] < self.kfac1 * work[k]:
                                self.k_targ -= 1
                            hnew = hopt[self.k_targ]
                            khit = k
                            break
                    if k == self.k_targ + 1:
                        if err > 1.0:
                            self.reject = True
                            if self.k_targ > 1 and work[self.k_targ-1] < self.kfac1 * work[self.k_targ]:
                                self.k_targ -= 1
                            hnew = hopt[self.k_targ]
                        khit = k
                        break
            if self.reject:
                self.prev_reject = True
                if not self.calcjac:
                    theta = 2.0 * jac_redo
                    compute_jac = True
        self.calcjac = False
        if self.dense:
            self.prepare_dense(h, ysav, scale, khit, err)
        self.xold = self.x
        self.x += h
#        print self.x
#        print self.y
#        sys.exit()
        self.hdid = h
        self.first_step = False
        if khit == 1:
            kopt = 2
        elif khit <= self.k_targ:
            kopt = khit
            if work[khit - 1] < self.kfac1 * work[khit]:
                kopt = khit - 1
            elif work[khit] < self.kfac2 * work[khit]:
                kopt = min(khit + 1, self.kmax -1)
        else:
            kopt = khit - 1
            if khit > 2 and work[khit-2] < self.kfac1 * work[khit-1]:
                kopt = khit-2
            if work[khit] < self.kfac2 * work[kopt]:
                kopt = min(khit, self.kmax - 1)
        if self.prev_reject:
            self.k_targ = min(kopt, khit)
            hnew = min(np.abs(h), hopt[self.k_targ])
            self.prev_reject = False
        else:
            if kopt <= khit:
                hnew = hopt[kopt]
            else:
                if khit < self.k_targ and work[khit] < self.kfac2 * work[khit-1]:
                    hnew = hopt[khit] * self.cost[kopt+1] / self.cost[kopt]
                else:
                    hnew = hopt[khit] * self.cost[kopt] / self.cost[kopt]
            self.k_targ = kopt
        if self.forward:
            self.hnext = hnew
        else:
            self.hnext = -hnew

    def semi_implicit_euler(self, y, htot, k, scale, derivs):
        from scipy.linalg import lu_solve
        from scipy.linalg import lu_factor

        #print 'y', y
        #print 'k', k
        #print 'size', len(self.nseq)
        nstep = self.nseq[k]
        h = htot / nstep
        a = -self.dfdy
        a += np.identity(self.n) / h
        alu = lu_factor(a, overwrite_a=True)
        xnew = self.x + h
        dell = derivs(xnew, y)
        #print 'xnew', xnew
        #print 'dell', dell
        #print 'ydell', y
        #print 'dell', dell
        #print 'xnew', xnew
        #print 'y', y
        #print 'alu', alu
        #print 'a', a
        #print 'h', h
#        sys.exit()
#        print 'xnew', xnew
#        print 'y', y
#        print 'dellbefore', dell
        ytemp = y.copy()
        dell = lu_solve(alu, dell, overwrite_b=True)
        if self.dense and nstep == k + 1:
            print 'SKJERALDRI'
            self.fsave = np.append(self.fsave, np.reshape(dell, (1, self.n), axis=0))
#            ytemp = np.zeros(self.n)
        for nn in xrange(1, nstep):
            ytemp += dell
            xnew += h
            yend = derivs(xnew, ytemp)
            if nn == 1 and k <= 1:
                del1 = np.sqrt(np.sum((dell/scale) ** 2))
                dytemp = derivs(self.x + h, ytemp)
                dell = dytemp - dell / h
                dell = lu_solve(alu, dell, overwrite_b=True)
                del2 = np.sqrt(np.sum((dell/scale) ** 2))
                theta = del2 / max(1.0, del1)
                if theta > 1.0:
                    return False, yend
            dell = lu_solve(alu, yend, overwrite_b=False)
            if self.dense and nn >= nstep-k-1:
                self.fsave = np.append(self.fsave, np.reshape(dell, (1, self.n)), axis=0)
        yend = ytemp + dell
        #print 'yend', yend
        #print 'xph', self.x + htot
        return True, yend

    def polyextr(self, k):
        for j in xrange(k-1, 0, -1):
            self.table[j-1] = self.table[j] + self.coeff[k, j] * (self.table[j] - self.table[j-1])
        self.y = self.table[0] + self.coeff[k][0] * (self.table[0] - self.y)

    def prepare_dense(self, h, ysav, scale, k, error):
        self.kright = k
        self.dens[:self.n] = ysav.copy()
        self.dens[self.n:2*self.n] = self.y
        for klr in xrange(self.kright):
            if klr >= 1:
                for kk in xrange(klr, k+1):
                    lbeg = ((kk + 3) * kk) / 2
                    lend = lbeg - kk + 1
                    for l in xrange(lbeg, lend-1, -1):
                        self.fsave[l] -= self.fsave[l-1]
            for kk in xrange(klr, k+1):
                facnj = float(self.nseq[kk])
                ipt = ((kk + 3) * kk) / 2
                krn = (kk + 2) * self.n
                self.dens[krn:krn + self.n] = self.fsave[ipt]*facnj
            for j in xrange(klr+1, k+1):
                dblenj = float(self.nseq[j])
                for l in xrange(j, klr, -1):
                    factor = dblenj / self.nseq[l-1] - 1.0
                    krn = np.arange((l+2) * self.n, (l+3) * self.n)
                    self.dens[krn-self.n] = self.dens[krn] + (self.dens[krn] - self.dens[krn-self.n]) / factor
        for inn in xrange(self.n):
            for j in xrange(1, self.kright+2):
                ii = self.n*j + inn
                self.dens[ii] -= self.dens[ii-self.n]

    def dense_out(self, x, h):
        theta = x - self.xold
        k = self.kright
        yinterp = self.dens[(k+1)*self.n:(k+2)*self.n].copy()
        for j in xrange(1, k+1):
            yinterp = self.dens[(k+1-j)*self.n:(k+2-j)*self.n] + yinterp * (theta-1.0)
        return self.dens[:self.n] + yinterp * theta

class StepperBase_grid(object):
    def __init__(self, y, x, atol, rtol, dense):
        self.x = x
        self.y = y
        self.atol = atol
        self.rtol = rtol
        self.dense = dense
        shape = np.shape(y)
        self.n = shape[1]
        self.nsys = shape[0]
        self.neqn = self.n

class StepperSie_grid(StepperBase_grid):
    def __init__(self, y, x, atol, rtol, dense):
        self.stepfac1 = 0.6
        self.stepfac2 = 0.93
        self.stepfac3 = 0.1
        self.stepfac4 = 4.0
        self.stepfac5 = 0.5
        self.kfac1 = 0.7
        self.kfac2 = 0.9
        StepperBase.__init__(self, y, x, atol, rtol, dense)
        self.kmax = 12
        self.imax = self.kmax + 1
        self.nseq = np.empty(self.imax, dtype=np.int)
        self.cost = np.empty(self.imax)
        self.table = np.empty((self.nsys, self.kmax, self.n))
        self.dfdy = np.empty((self.nsys, self.n, self.n))
        self.dfdx = np.empty((self.nsys, self.n))
        self.calcjac = np.zeros(self.nsys, dtype='bool')
#        self.a = np.empty((self.n, self.n))
        self.coeff = np.empty((self.imax, self.imax))
        self.fsave = np.empty((0, self.nsys, self.n))
        self.dens = np.empty((self.nsys, (self.imax + 2) * self.n))
        self.factrl = np.empty(self.imax)
        self.costfunc = 1.0
        self.costjac = 5.0
        self.costlu = 1.0
        self.costsolve = 1.0
        self.eps = sys.float_info.epsilon
        self.jac_redo = min(1.0e-4, rtol)
        self.theta = np.array([2.0 * self.jac_redo] * self.nsys)
        self.nseq[0] = 2
        self.nseq[1] = 3
        for i in xrange(2, self.imax):
            self.nseq[i] = 2 * self.nseq[i-2]
#        print 'nseq', self.nseq
#        sys.exit()
        self.cost[0] = self.costjac + self.costlu + self.nseq[0] * (self.costfunc + self.costsolve)
        for k in xrange(self.kmax):
            self.cost[k + 1] = self.cost[k] + (self.nseq[k + 1] - 1) * (self.costfunc + self.costsolve) + self.costlu
        self.hnext = np.array([-1.0e99] * self.nsys)
        logfact = -np.log10(self.rtol + self.atol) * 0.6 + 0.5
        self.k_targ = max(1, min(self.kmax - 1, int(logfact)))
        self.k_targ = np.array([self.k_targ] * self.nsys)
        self.coeff[:, :] = np.array([[1.0/(float(self.nseq[k]) / self.nseq[l] - 1.0) if l < k else 0 for l in xrange(self.imax)] for k in xrange(self.imax)])
#        print self.coeff
#        sys.exit()
        self.factrl[0] = 1.0
        for k in xrange(self.imax - 1):
            self.factrl[k+1] = (k + 1) * self.factrl[k]
        self.first_step = np.ones(self.nsys, dtype='bool')
#        self.allow_step_increase = True
        self.reject = np.zeros(self.nsys, dtype='bool')
        self.prev_reject = np.zeros(self.nsys, dtype='bool')

    def update_systems(self, filterr):
        self.nsys = newnsys
        self.x = self.x[filterr]
        self.y = self.y[filterr]
        self.hnext = self.h[filterr]
        self.hdid = self.hdid[filterr]
        self.table = self.table[filterr]
        self.dfdy = self.dfdy[filterr]
        self.dfdx = self.dfdx[filterr]
        self.calcjac = self.calcjac[filterr]
        self.fsave = self.fsave[:, filterr]
        self.dens = self.dens[filterr]
        self.first_step = self.first_step[filterr]
        self.reject = self.reject[filterr]
        self.prev_reject = self.prev_reject[filterr]
        self.theta = self.theta[filterr]
    
    def step(self, htry, derivs, allow_step_increase=np.ones(self.nsys, dtype='bool'), last_step=np.zeros(self.nsys, dtype='bool')):
        hopt = np.empty((self.imax, self.nsys))
        work = np.empty(self.imax)
        work[0] = 1.e30
        errold = np.zeros(self.nsys)
        h = htry.copy()
        self.forward = h > 0
        ysav = self.y.copy()
        self.prev_reject[self.reject] = True
        self.last_step[self.reject] = False
        self.theta[self.reject] = 2.0 * self.jac_redo
        scale = self.atol + self.rtol * np.abs(self.y)
        self.reject[:] = False
        firstk = np.ones(self.nsys, dtype='bool')
        hnew = np.abs(h)
        compute_jac = np.ones(self.nsys, dtype='bool')
        khit = np.empty(self.nsys, dtype='int')
        while any(firstk) or any(self.reject):
            topfilter = firstk & self.reject
            subfilter1 = topfilter & compute_jac
            compute_jac[subfilter1] = False
            subsubfilter1 = subfilter1 & self.theta > self.jac_redo & ~self.calcjac
            self.dfdy[subsubfilter1] = derivs.jacobian(self.x, self.y, subsubfilter1)
            self.calcjac[subsubfilter1] = True

            subfilter1 = topfilter & self.forward

            subsubfilter1 = subfilter1 & allow_step_increase
            h[subsubfilter1] = hnew[subsubfilter1]
            subsubfilter1 = subfilter1 & ~allow_step_increase
            h[subsubfilter1] = np.minimum(hnew[subsubfilter1], h[subsubfilter1])

            subfilter1 = topfilter & ~self.forward
            subsubfilter1 = subfilter1 & allow_step_increase
            h[subsubfilter1] = -hnew[subsubfilter1]
            subsubfilter1 = subfilter1 & ~allow_step_increase
            h[subsubfilter1] = -np.minimum(hnew[subsubfilter1], h[subsubfilter1])
            firstk[topfilter] = False
            self.reject[topfilter] = False
            if any(np.abs(h[topfilter] <= np.abs(self.x[topfilter]) * self.eps)):
                raise ValueError("Step size underflow in StepperSie")
            #This is rather mysterious to me
            self.fsave = np.empty((0, self.nsys, self.n))
            for k in xrange(np.max(self.k_targ) + 2):
                topfilter2 = topfilter & self.k_targ + 2 > k
                if not any(topfilter2):
                    break
                success, yseq = self.semi_implicit_euler(ysav, h, k, scale, derivs, topfilter2)
                subfilter1 = topfilter2 & ~success
                self.reject[subfilter1] = True
                hnew[subfilter1] = np.abs(h[subfilter1]) * self.stepfac5
                khit[subfilter1] = k
                if not any(success):
                    break
                topfilter3 = topfilter2 & success
                if not any(topfilter3):
                    break
                if k == 0:
                    self.y[topfilter3] = yseq[topfilter3]
                else:
                    self.table[k-1, topfilter3] = yseq[topfilter3]
                    self.polyextr(k, topfilter3)
                    scale[topfilter3] = self.atol + self.rtol * np.abs(ysav[topfilter3])
                    err = np.zeros(len(topfilter3))
                    err[topfilter3] = np.sqrt(np.mean(((self.y[topfilter3] - self.table[0, topfilter3]) / scale[topfilter3]) ** 2))
                    if k > 1:
                        subfilter1 = topfilter3 & (err >= errold | err > 1.0 / self.eps)
                    else:
                        subfilter1 = topfilter3 & (err > 1.0 / self.eps)
                    self.reject[subfilter1] = True
                    hnew[subfilter1] = np.abs(h[subfilter1]) * self.stepfac5
                    khit[subfilter1] = k
                    if all(subfilter1[topfilter3]):
                        break
                    topfilter4 = topfilter3 & ~subfilter1
                    if not any(topfilter4):
                        break
                    errold[topfilter4] = max(4.0*err[topfilter4], 1.0)
                    expo = 1.0 / (k + 1)
                    facmin = self.stepfac3 ** expo
                    fac = np.zeros(len(topfilter4))
                    subfilter1 = topfilter4 & err == 0.0
                    fac[subfilter1] = 1.0 / facmin
                    subfilter1 = topfilter4 & err != 0.0
                    fac[subfilter1] = self.stepfac2 / (err[subfilter1] / self.stepfac1) ** expo
                    fac[subfilter1] = np.maximum(facmin / self.stepfac5, np.minimum(1.0 / facmin, fac[subfilter1]))
                    hopt[k, topfilter4] = np.abs(h[topfilter4] * fac)
                    work[k, topfilter4] = self.cost[k] / hopt[k, topfilter4]
                    subfilter1 = topfilter4 & (self.first_step | last_step) & err <= 1.0
                    khit[subfilter1] = k
                    if all(subfilter1[topfilter4]):
                        break
                    topfilter5 = topfilter4 & ~subfilter1
                    if not any(topfilter5):
                        break
                    subfilter1 = topfilter5 & self.k_targ - 1 == k & ~self.reject & ~self.first_step & ~last_step
                    subsubfilter1 = subfilter1 & err <= 1.0
                    khit[subsubfilter1] = k
                    if all(subsubfilter1[topfilter5]):
                        break
                    subsubfilter1 = subfilter1 & ~subsubfilter1 & err > self.nseq[self.k_targ] * self.nseq[self.k_targ] * 4.0
                    self.reject[subsubfilter1] = True
                    self.k_targ[subsubfilter1] = k
                    self.k_targ[subsubfilter1 & self.k_targ > 1 & work[k-1] < self.kfac1 * work[k]] -= 1
                    hnew[subsubfilter1] = hopt[self.k_targ][subsubfilter1, subsubfilter1]
                    khit[subsubfilter1] = k
                    if all(subsubfilter1[topfilter5]):
                        break
                    subfilter1 = topfilter5 & self.k_targ == k
                    subsubfilter1 = subfilter1 & err <= 1.0
                    khit[subsubfilter1] = k
                    if all(subsubfilter1[topfilter5]):
                        break
                    subsubfilter1 = subfilter1 & ~subsubfilter1 & err > self.nseq[k+1] * 2.0
                    self.reject[subsubfilter1] = True
                    self.k_targ[subsubfilter1 & work[k-1] < self.kfac * work[k]] -= 1
                    hnew[subsubfilter1] = hopt[self.k_targ][subsubfilter1, subsubfilter1]
                    khit[subsubfilter1] = k
                    if all(subsubfilter1[topfilter5]):
                        break
                    subfilter1 = topfilter5 & self.k_targ + 1  == k
                    subsubfilter1 = subfilter1 & err > 1.0
                    self.reject[subsubfilter1] = True
                    self.k_targ[subsubfilter1 & self.k_targ > 1 & work[k-1] < self.kfac1 * work[k]] -= 1
                    hnew[subsubfilter1] = hopt[self.k_targ][subsubfilter1, subsubfilter1]
                    khit[subfilter1] = k
                    if all(subfilter1[topfilter5]):
                        break
            subfilter1 = topfilter & self.reject
            self.prev_reject[subfilter1] = True
            subsubfilter1 = subfilter1 & ~self.calcjac
            theta[subsubfilter1] = 2.0 * jac_redo[subsubfilter1]
            compute_jac[subsubfilter1] = True
        self.calcjac[:] = False
        if self.dense:
            self.prepare_dense(h, ysav, scale, khit, err)
        self.xold = self.x.copy()
        self.x += h
        self.hdid = h.copy()
        self.first_step[:] = False
        kopt[khit == 1] = 2
        filterr = ~(khit == 1) & khit <= self.k_targ
        kopt[filterr] = khit[filterr]
        subfilter = filterr & np.diagonal(work[khit - 1]) < self.kfac1 * np.diagonal(work[khit])
        kopt[subfilter] = khit[subfilter] - 1
        subfilter = filterr & ~subfilter & np.diagonal(work[khit]) < self.kfac2 * np.diagonal(work[khit])
        kopt[subfilter] = np.minimum(khit[subfilter] + 1, self.kmax - 1)
        filterr = ~(khit <= self.k_targ)
        kopt[filterr] = khit[filterr] - 1
        subfilter = filterr & khit > 2 & np.diagonal(work[khit-2]) < self.kfac1 * np.diagonal(work[khit-1])
        kopt[subfilter] = khit[subfilter] - 2
        subfilter = filterr & np.diagonal(work[khit]) < self.kfac2 * np.diagonal(work[kopt])
        kopt[subfilter] = np.minimum(khit[subfilter], self.kmax - 1)
        filterr = self.prev_reject
        self.k_targ[filterr] = np.minimum(kopt[filterr], khit[filterr])
        hnew[filterr] = np.minimum(np.abs(h[filterr]), hopt[self.k_targ][filterr, filterr])
        self.prev_reject[filterr] = False
        filterr = ~filterr
        subfilter = filterr & kopt <= khit
        hnew[subfilter] = hopt[kopt][subfilter, subfilter]
        subfilter = filterr & ~subfilter
        if khit == 1:
            kopt = 2
        elif khit <= self.k_targ:
            kopt = khit
            if work[khit - 1] < self.kfac1 * work[khit]:
                kopt = khit - 1
            elif work[khit] < self.kfac2 * work[khit]:
                kopt = min(khit + 1, self.kmax -1)
        else:
            kopt = khit - 1
            if khit > 2 and work[khit-2] < self.kfac1 * work[khit-1]:
                kopt = khit-2
            if work[khit] < self.kfac2 * work[kopt]:
                kopt = min(khit, self.kmax - 1)
        if self.prev_reject:
            self.k_targ = min(kopt, khit)
            hnew = min(np.abs(h), hopt[self.k_targ])
            self.prev_reject = False
        else:
            if kopt <= khit:
                hnew = hopt[kopt]
            else:
                if khit < self.k_targ and work[khit] < self.kfac2 * work[khit-1]:
                    hnew = hopt[khit] * self.cost[kopt+1] / self.cost[kopt]
                else:
                    hnew = hopt[khit] * self.cost[kopt] / self.cost[kopt]
            self.k_targ = kopt
        if self.forward:
            self.hnext = hnew
        else:
            self.hnext = -hnew

    def semi_implicit_euler(self, y, htot, k, scale, derivs):
        from scipy.linalg import lu_solve
        from scipy.linalg import lu_factor

        #print 'y', y
        #print 'k', k
        #print 'size', len(self.nseq)
        nstep = self.nseq[k]
        h = htot / nstep
        a = -self.dfdy
        a += np.identity(self.n) / h
        alu = lu_factor(a, overwrite_a=True)
        xnew = self.x + h
        dell = derivs(xnew, y)
        #print 'xnew', xnew
        #print 'dell', dell
        #print 'ydell', y
        #print 'dell', dell
        #print 'xnew', xnew
        #print 'y', y
        #print 'alu', alu
        #print 'a', a
        #print 'h', h
#        sys.exit()
#        print 'xnew', xnew
#        print 'y', y
#        print 'dellbefore', dell
        ytemp = y.copy()
        dell = lu_solve(alu, dell, overwrite_b=True)
        if self.dense and nstep == k + 1:
            print 'SKJERALDRI'
            self.fsave = np.append(self.fsave, np.reshape(dell, (1, self.n), axis=0))
#            ytemp = np.zeros(self.n)
        for nn in xrange(1, nstep):
            ytemp += dell
            xnew += h
            yend = derivs(xnew, ytemp)
            if nn == 1 and k <= 1:
                del1 = np.sqrt(np.sum((dell/scale) ** 2))
                dytemp = derivs(self.x + h, ytemp)
                dell = dytemp - dell / h
                dell = lu_solve(alu, dell, overwrite_b=True)
                del2 = np.sqrt(np.sum((dell/scale) ** 2))
                theta = del2 / max(1.0, del1)
                if theta > 1.0:
                    return False, yend
            dell = lu_solve(alu, yend, overwrite_b=False)
            if self.dense and nn >= nstep-k-1:
                self.fsave = np.append(self.fsave, np.reshape(dell, (1, self.n)), axis=0)
        yend = ytemp + dell
        #print 'yend', yend
        #print 'xph', self.x + htot
        return True, yend

    def polyextr(self, k):
        for j in xrange(k-1, 0, -1):
            self.table[j-1] = self.table[j] + self.coeff[k, j] * (self.table[j] - self.table[j-1])
        self.y = self.table[0] + self.coeff[k][0] * (self.table[0] - self.y)

    def prepare_dense(self, h, ysav, scale, k, error):
        self.kright = k
        self.dens[:self.n] = ysav.copy()
        self.dens[self.n:2*self.n] = self.y
        for klr in xrange(self.kright):
            if klr >= 1:
                for kk in xrange(klr, k+1):
                    lbeg = ((kk + 3) * kk) / 2
                    lend = lbeg - kk + 1
                    for l in xrange(lbeg, lend-1, -1):
                        self.fsave[l] -= self.fsave[l-1]
            for kk in xrange(klr, k+1):
                facnj = float(self.nseq[kk])
                ipt = ((kk + 3) * kk) / 2
                krn = (kk + 2) * self.n
                self.dens[krn:krn + self.n] = self.fsave[ipt]*facnj
            for j in xrange(klr+1, k+1):
                dblenj = float(self.nseq[j])
                for l in xrange(j, klr, -1):
                    factor = dblenj / self.nseq[l-1] - 1.0
                    krn = np.arange((l+2) * self.n, (l+3) * self.n)
                    self.dens[krn-self.n] = self.dens[krn] + (self.dens[krn] - self.dens[krn-self.n]) / factor
        for inn in xrange(self.n):
            for j in xrange(1, self.kright+2):
                ii = self.n*j + inn
                self.dens[ii] -= self.dens[ii-self.n]

    def dense_out(self, x, h):
        theta = x - self.xold
        k = self.kright
        yinterp = self.dens[(k+1)*self.n:(k+2)*self.n].copy()
        for j in xrange(1, k+1):
            yinterp = self.dens[(k+1-j)*self.n:(k+2-j)*self.n] + yinterp * (theta-1.0)
        return self.dens[:self.n] + yinterp * theta


def mmid(rhs, x0, y0, length, stepnum):
    #y0 is the value of rhs evaluated at x0
    h = length / stepnum
    h2 = 2 * h
    zm1 = y0
    z1 = zm1 + h * rhs(x0, y0)
    znew = zm1 + h2 * rhs(x0 + h, z1)
    for i in xrange(2, stepnum):
        zm1 = z1
        z1 = znew
        znew = zm1 + h2 * rhs(x0 + i * h, z1)
    return 0.5 * (znew + z1 + h * rhs(x0 + length, znew))

