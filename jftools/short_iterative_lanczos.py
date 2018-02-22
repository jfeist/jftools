import numpy as np
from numpy import einsum, empty, zeros, vdot, log10, exp
from numpy.linalg import norm
from scipy.linalg import eig_banded
import warnings

try:
    import qutip
    have_qutip = True
except:
    have_qutip = False

class normdotndarray(np.ndarray):
    """extension of numpy array that supports interface for lanczos_timeprop"""
    def norm(self):
        return norm(self)
    def dot(self,other):
        return vdot(self,other)

def calc_coeff(step, a_band, HT, coeff):
    vals, vecs = eig_banded(a_band[:,:step],lower=False,overwrite_a_band=False)
    # expH(j,k) = vecs(j,i) * exp(vals(i)) * delta(i,l) * transpose(vecs(l,k))
    # expH(j,k) = vecs(j,i) * exp(vals(i)) * vecs(k,i)
    coeff[:step] = einsum('ji,i,i->j',vecs,vecs[0],exp(-1j*HT*vals))
    coeff[step:] = 0.
    return coeff

class lanczos_timeprop:
    def __init__(self,H,maxsteps,target_convg,debug=0,do_full_order=False):
        if have_qutip and isinstance(H,qutip.Qobj):
            H = H.data

        if not callable(H):
            # time-independent operator
            # assume it supports .dot for matrix-vector multiplication
            def Hfun(t,phi,Hphi):
                Hphi[:] = H.dot(phi)
                return Hphi
            self.Hfun = Hfun
        else:
            self.Hfun = H

        self.maxsteps = maxsteps
        self.target_convg = target_convg
        self.prefacs = empty(maxsteps+1)
        # this is the array that will hold the banded matrix
        self.a_band = empty((2,maxsteps+1))
        self.debug = debug
        self.do_full_order = do_full_order

        self.curr_coeff = zeros(maxsteps+1,dtype=complex)
        self.prev_coeff = self.curr_coeff.copy()

    def propagate(self,phi0,ts,maxHT=None):
        ts = np.asarray(ts)
        assert ts.ndim==1, 'ts must be a 1d array'

        get_phi_out = lambda x: x.copy()
        if isinstance(phi0,np.ndarray):
            get_phi_out = lambda x: x.view(np.ndarray).copy()
            phi0 = phi0.view(normdotndarray)
        elif have_qutip and isinstance(phi0,qutip.Qobj):
            outdims = phi0.dims.copy()
            outtype = phi0.type
            get_phi_out = lambda x: qutip.Qobj(x,dims=outdims,type=outtype)
            phi0 = phi0.full().view(normdotndarray)

        self.phia = [phi0.copy() for _ in range(self.maxsteps+1)]

        ids = np.array([id(x) for x in self.phia])

        tt = ts[0]
        phis = [get_phi_out(phi0)]
        for tf in ts[1:]:
            while tt<tf:
                HT = tf-tt
                if maxHT is not None:
                    HT = min(HT,maxHT)
                HT_done = self._step(tt,HT)
                tt += HT_done
            phis.append(get_phi_out(self.phia[0]))

        if not np.all(ids==np.array([id(x) for x in self.phia])):
            warnings.warn('self.phia have not been updated in-place!')

        return phis

    def _step(self,t,HT):
        # create local variables that use the class storage locations
        beta, alpha = self.a_band
        phia = self.phia
        prefacs = self.prefacs
        curr_coeff = self.curr_coeff
        prev_coeff = self.prev_coeff
        debug = self.debug
        Hfun = self.Hfun

        HT_done = HT

        # initialize norm of starting vector
        phinorm = phia[0].norm()
        prefacs[0] = 1. / phinorm

        # set current solution vector to zero so that it
        # doesn't converge at first step
        curr_coeff[:] = 0.

        for step in range(1, self.maxsteps+1):
            # set |phia(step)> to H|phia(step-1)>
            phia[step] = Hfun(t,phia[step-1],phia[step])
            prefacs[step] = prefacs[step-1]
            phinorm = prefacs[step] * phia[step].norm()
            # phinorm = sqrt(<q(step-1)|H H|q(step-1)>)
            # build lanczos-matrix
            # it's tridiagonal, so we only need two steps in the loop
            # start with step-1 for numerical reasons - this should ensure better orthogonality by
            # removing the potentially largest part (by a significant amount) first
            dotpr = prefacs[step-1] * prefacs[step] * phia[step-1].dot(phia[step])
            # don't use too stringent a criterion here, as the dot product does not contain squares
            if abs(dotpr.imag) > 1e-9 and debug>3:
                print('imaginary part of dotpr !=0:', dotpr.imag)
            alpha[step-1] = dotpr.real

            phia[step] -= alpha[step-1] * prefacs[step-1]/prefacs[step] * phia[step-1]

            # phinorm     = sqrt(<q(step-1)|H H|q(step-1)>)
            # alpha(step) = <q(step-1)| H |q(step-1)>
            # abs(phinorm**2 - alpha(step)**2) is deltaH**2, i.e. a measure of how close q(step-1) is
            #   to being an eigenvector of H. if this is a small number, we take another
            #   gram-schmidt step to ensure that we have good orthogonality
            if abs(phinorm**2-alpha[step-1]**2) < 0.1:
                dotpr = prefacs[step-1] * prefacs[step] * phia[step-1].dot(phia[step])
                phia[step] -= dotpr * prefacs[step-1]/prefacs[step] * phia[step-1]
            if step >= 2:
                phia[step] -= beta[step-1] * prefacs[step-2] / prefacs[step] * phia[step-2]

            # ************ normalize phia(step) to get q_step ***************
            # be careful here: beta should be the norm of the
            # current q_step == prefac * |phi>,
            # i.e. beta = prefac * sqrt(<phi|phi>)
            # after that, we set prefac to _normalize_ the vector |phi>,
            # i.e. to prefac = 1.d0 / sqrt(<phi|phi>)
            phinorm = phia[step].norm()
            beta[step] = prefacs[step] * phinorm
            prefacs[step] = 1. / phinorm
            if abs(log10(prefacs[step])) > 4.:
                phia[step] *= prefacs[step]
                prefacs[step] = 1.
            if abs(beta[step]) < 1e-2 and debug > 2:
                print('WARNING! beta[%d]=%g is very small - there seems to be a linearly dependent vector!'%(step,beta[step]))
            if debug > 1:
                # check if new vector is orthogonal to all others
                for ii in range(step):
                    dotpr = prefacs[ii] * prefacs[step] * phia[ii].dot(phia[step])
                if abs(dotpr) > 1e-12:
                    print('WARNING! vectors not orthogonal. dotpr(%d,%d) = %g'%(ii,step,dotpr))

            # check convergence
            prev_coeff[:] = curr_coeff[:]
            calc_coeff(step, self.a_band, HT_done, curr_coeff)
            convg = norm(curr_coeff-prev_coeff)
            if debug > 6: print('prev_coeff:', prev_coeff)
            if debug > 6: print('curr_coeff:', curr_coeff)
            if debug > 5: print('convg:', convg)

            if not self.do_full_order and convg < self.target_convg:
                break

        if debug > 8:
            print(alpha[0:step])
            print(beta[1:step])

        # if convergence was reached in lanczos_loop, convg < target_convg, and this loop is never entered
        while convg > self.target_convg:
            # error (~convg) should be O(HT**maxsteps)
            # convg = a * HT**maxsteps
            # target_convg = a * HT_new**maxsteps
            # target_convg/convg = (HT_new/HT)**maxsteps
            # -> HT_new = HT * (target_convg/convg)**(1/maxsteps)
            scale = 0.95 * (self.target_convg/convg)**(1./step)
            # the 0.95d0 is to get convergence when we're very close
            # and scale would be almost unity
            # to prevent going to much too small steps when very far from convergence, decrease
            # step size by at most one half
            scale = max(0.5,scale)
            if debug > 3:
                print('scales HT with scale =', scale)
            HT_done = HT_done * scale
            calc_coeff(step-1, self.a_band, HT_done, prev_coeff)
            calc_coeff(step  , self.a_band, HT_done, curr_coeff)
            convg = norm(curr_coeff-prev_coeff)
            if debug > 6: print('prev_coeff:', abs(prev_coeff)**2)
            if debug > 6: print('curr_coeff:', abs(curr_coeff)**2)
            if debug > 5: print('convg:', convg)

        if debug > 6 and HT_done!=HT:
            print('did not converge in %d iterations, step size decreased from %g to %g'%(self.maxsteps,HT,HT_done))

        # build the new vector
        # do NOT include the prefactor prefac[0] into phi - we do not want to normalize it
        phia[0] *= curr_coeff[0]
        cc = curr_coeff*prefacs/prefacs[0]
        for ii in range(1,step+1):
            phia[0] += cc[ii] * phia[ii]

        return HT_done

def sesolve_lanczos(H,phi0,ts,maxsteps,target_convg,maxHT=None,debug=0,do_full_order=False):
    prop = lanczos_timeprop(H,maxsteps,target_convg,debug,do_full_order)
    return prop.propagate(phi0,ts,maxHT)
