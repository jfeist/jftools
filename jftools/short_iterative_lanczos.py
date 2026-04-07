import warnings

import numpy as np
from numpy import einsum, empty, exp, log10, vdot, zeros
from numpy.linalg import norm
from scipy.linalg import eigh_tridiagonal
from scipy import sparse as sp

try:
    from .short_iterative_lanczos_cython import _lanczos_timeprop_cython

    have_cython_backend = True
except ImportError:
    _lanczos_timeprop_cython = None
    have_cython_backend = False

try:
    from .short_iterative_lanczos_numba import _lanczos_timeprop_numba

    have_numba_backend = True
except ImportError:
    _lanczos_timeprop_numba = None
    have_numba_backend = False

try:
    import qutip

    have_qutip = True
except ImportError:
    have_qutip = False


class normdotndarray(np.ndarray):
    """extension of numpy array that supports interface for lanczos_timeprop"""

    def norm(self):
        return norm(self)

    def dot(self, other):
        return vdot(self, other)


def calc_coeff(step, alpha, beta, HT, coeff):
    alpha_step = alpha[:step]
    beta_step = beta[: step - 1]
    vals, vecs = eigh_tridiagonal(alpha_step, beta_step)
    # expH(j,k) = vecs(j,i) * exp(vals(i)) * delta(i,l) * transpose(vecs(l,k))
    # expH(j,k) = vecs(j,i) * exp(vals(i)) * vecs(k,i)
    coeff[:step] = einsum("ji,i,i->j", vecs, vecs[0], exp(-1j * HT * vals))
    coeff[step:] = 0.0
    return coeff


def _qobj_to_matrix(H):
    data = H.data
    if hasattr(data, "as_scipy"):
        return data.as_scipy()
    if hasattr(data, "to_array"):
        return data.to_array()
    return np.asarray(H)


def _matvec(H, phi):
    if hasattr(H, "dot"):
        return H.dot(phi)
    return H @ phi


def _as_hfun(H):
    """Return an in-place Hfun(t, phi, Hphi) callable for H."""
    if callable(H):
        return H

    H_f = H.dot if hasattr(H, "dot") else H.__matmul__

    def Hfun(t, phi, Hphi):
        Hphi[:] = H_f(phi)

    return Hfun


def _qobj_state_io(phi0):
    outdims = phi0.dims
    outshape = phi0.full().shape

    def get_phi_out(x):
        arr = np.asarray(x).reshape(outshape)
        return qutip.Qobj(arr, dims=outdims)

    phi0_arr = np.asarray(phi0.full()).reshape(-1)
    return phi0_arr, get_phi_out


class _lanczos_timeprop_reference:
    def __init__(self, H, maxsteps, target_convg, debug=0, do_full_order=False):
        if have_qutip and isinstance(H, qutip.Qobj):
            H = _qobj_to_matrix(H)
        self.Hfun = _as_hfun(H)

        self.maxsteps = maxsteps
        self.target_convg = target_convg
        self.prefacs = empty(maxsteps + 1)
        self.alpha = empty(maxsteps + 1)
        self.beta = empty(maxsteps)
        self.debug = debug
        self.do_full_order = do_full_order

        self.curr_coeff = zeros(maxsteps + 1, dtype=complex)
        self.prev_coeff = self.curr_coeff.copy()
        self.breakdown_tol = 1e-14

    def propagate(self, phi0, ts, maxHT=None):
        phi0 = np.asarray(phi0).view(normdotndarray)

        self.phia = [phi0.copy() for _ in range(self.maxsteps + 1)]

        ids = np.array([id(x) for x in self.phia])

        tt = ts[0]
        phis = [phi0.view(np.ndarray).copy()]
        for tf in ts[1:]:
            while tt < tf:
                HT = tf - tt
                if maxHT is not None:
                    HT = min(HT, maxHT)
                HT_done = self._step(tt, HT)
                tt += HT_done
            phis.append(self.phia[0].view(np.ndarray).copy())

        if not np.all(ids == np.array([id(x) for x in self.phia])):
            warnings.warn("self.phia have not been updated in-place!")

        return phis

    def _step(self, t, HT):
        # create local variables that use the class storage locations
        alpha = self.alpha
        beta = self.beta
        phia = self.phia
        prefacs = self.prefacs
        curr_coeff = self.curr_coeff
        prev_coeff = self.prev_coeff
        debug = self.debug
        Hfun = self.Hfun
        max_lanczos_steps = min(self.maxsteps, phia[0].shape[0])

        HT_done = HT

        # initialize norm of starting vector
        phinorm = phia[0].norm()
        prefacs[0] = 1.0 / phinorm

        # set current solution vector to zero so that it
        # doesn't converge at first step
        curr_coeff[:] = 0.0

        convg = np.inf
        exact_complete = False

        for step in range(1, max_lanczos_steps + 1):
            # set |phia(step)> to H|phia(step-1)>
            Hfun(t, phia[step - 1], phia[step])
            prefacs[step] = prefacs[step - 1]
            phinorm = prefacs[step] * phia[step].norm()
            # phinorm = sqrt(<q(step-1)|H H|q(step-1)>)
            # build lanczos-matrix
            # it's tridiagonal, so we only need two steps in the loop
            # start with step-1 for numerical reasons - this should ensure better orthogonality by
            # removing the potentially largest part (by a significant amount) first
            dotpr = prefacs[step - 1] * prefacs[step] * phia[step - 1].dot(phia[step])
            # don't use too stringent a criterion here, as the dot product does not contain squares
            if abs(dotpr.imag) > 1e-9 and debug > 3:
                print("imaginary part of dotpr !=0:", dotpr.imag)
            alpha[step - 1] = dotpr.real

            phia[step] -= alpha[step - 1] * prefacs[step - 1] / prefacs[step] * phia[step - 1]

            # phinorm     = sqrt(<q(step-1)|H H|q(step-1)>)
            # alpha(step) = <q(step-1)| H |q(step-1)>
            # abs(phinorm**2 - alpha(step)**2) is deltaH**2, i.e. a measure of how close q(step-1) is
            #   to being an eigenvector of H. if this is a small number, we take another
            #   gram-schmidt step to ensure that we have good orthogonality
            if abs(phinorm**2 - alpha[step - 1] ** 2) < 0.1:
                dotpr = prefacs[step - 1] * prefacs[step] * phia[step - 1].dot(phia[step])
                phia[step] -= dotpr * prefacs[step - 1] / prefacs[step] * phia[step - 1]
            if step >= 2:
                phia[step] -= beta[step - 2] * prefacs[step - 2] / prefacs[step] * phia[step - 2]

            # ************ normalize phia(step) to get q_step ***************
            # be careful here: beta should be the norm of the
            # current q_step == prefac * |phi>,
            # i.e. beta = prefac * sqrt(<phi|phi>)
            # after that, we set prefac to _normalize_ the vector |phi>,
            # i.e. to prefac = 1.d0 / sqrt(<phi|phi>)
            phinorm = phia[step].norm()
            beta[step - 1] = prefacs[step] * phinorm
            if phinorm <= self.breakdown_tol:
                prefacs[step] = 1.0
                phia[step][:] = 0.0
                exact_complete = True
            else:
                prefacs[step] = 1.0 / phinorm
                if abs(log10(prefacs[step])) > 4.0:
                    phia[step] *= prefacs[step]
                    prefacs[step] = 1.0
                if abs(beta[step - 1]) < 1e-2 and debug > 2:
                    print("WARNING! beta[%d]=%g is very small - there seems to be a linearly dependent vector!" % (step, beta[step - 1]))
                if debug > 1:
                    # check if new vector is orthogonal to all others
                    for ii in range(step):
                        dotpr = prefacs[ii] * prefacs[step] * phia[ii].dot(phia[step])
                    if abs(dotpr) > 1e-12:
                        print("WARNING! vectors not orthogonal. dotpr(%d,%d) = %g" % (ii, step, dotpr))

            # check convergence
            prev_coeff[:] = curr_coeff[:]
            calc_coeff(step, alpha, beta, HT_done, curr_coeff)
            convg = norm(curr_coeff - prev_coeff)
            if debug > 6:
                print("prev_coeff:", prev_coeff)
            if debug > 6:
                print("curr_coeff:", curr_coeff)
            if debug > 5:
                print("convg:", convg)

            if exact_complete or step == max_lanczos_steps:
                break

            if not self.do_full_order and convg < self.target_convg:
                break

        if debug > 8:
            print(alpha[0:step])
            print(beta[: step - 1])

        # if convergence was reached in lanczos_loop, convg < target_convg, and this loop is never entered
        while (not exact_complete) and convg > self.target_convg:
            # error (~convg) should be O(HT**maxsteps)
            # convg = a * HT**maxsteps
            # target_convg = a * HT_new**maxsteps
            # target_convg/convg = (HT_new/HT)**maxsteps
            # -> HT_new = HT * (target_convg/convg)**(1/maxsteps)
            scale = 0.95 * (self.target_convg / convg) ** (1.0 / step)
            # the 0.95d0 is to get convergence when we're very close
            # and scale would be almost unity
            # to prevent going to much too small steps when very far from convergence, decrease
            # step size by at most one half
            scale = max(0.5, scale)
            if debug > 3:
                print("scales HT with scale =", scale)
            HT_done = HT_done * scale
            calc_coeff(step - 1, alpha, beta, HT_done, prev_coeff)
            calc_coeff(step, alpha, beta, HT_done, curr_coeff)
            convg = norm(curr_coeff - prev_coeff)
            if debug > 6:
                print("prev_coeff:", abs(prev_coeff) ** 2)
            if debug > 6:
                print("curr_coeff:", abs(curr_coeff) ** 2)
            if debug > 5:
                print("convg:", convg)

        if debug > 6 and HT_done != HT:
            print("did not converge in %d iterations, step size decreased from %g to %g" % (self.maxsteps, HT, HT_done))

        # build the new vector
        # do NOT include the prefactor prefac[0] into phi - we do not want to normalize it
        phia[0] *= curr_coeff[0]
        cc = curr_coeff * prefacs / prefacs[0]
        for ii in range(1, step + 1):
            phia[0] += cc[ii] * phia[ii]

        return HT_done


def _is_dense_matrix(H):
    return isinstance(H, np.ndarray) and H.ndim == 2 and H.shape[0] == H.shape[1]


def _is_csr_matrix(H):
    if sp.isspmatrix_csr(H):
        return True
    csr_array = getattr(sp, "csr_array", None)
    return csr_array is not None and isinstance(H, csr_array)


def _is_numba_sum_operator(H):
    if not isinstance(H, (tuple, list)) or len(H) < 2:
        return False
    for term in H[1:]:
        if not isinstance(term, (tuple, list)) or len(term) != 2 or not callable(term[1]):
            return False
    return True


def _select_backend(H, backend):
    backend = backend.strip().lower()

    if backend == "python":
        return "python"

    if backend == "numba":
        if not have_numba_backend:
            raise ValueError("backend='numba' requested but Numba backend is not available.")
        return "numba"

    if backend == "cython":
        if not have_cython_backend:
            raise ValueError("backend='cython' requested but Cython backend extension is not available.")
        return "cython"

    if backend != "auto":
        raise ValueError("Unknown backend value '%s'. Valid values are 'python', 'numba', 'cython', 'auto'." % backend)

    if have_numba_backend and (_is_dense_matrix(H) or _is_csr_matrix(H) or _is_numba_sum_operator(H)):
        return "numba"
    if have_cython_backend:
        return "cython"
    if have_numba_backend:
        return "numba"
    return "python"


class lanczos_timeprop:
    def __init__(self, H, maxsteps, target_convg, debug=0, do_full_order=False, backend="auto"):
        if have_qutip and isinstance(H, qutip.Qobj):
            H = _qobj_to_matrix(H)
        self.backend = _select_backend(H, backend)
        if self.backend == "numba":
            self._impl = _lanczos_timeprop_numba(H, maxsteps, target_convg, debug, do_full_order)
        elif self.backend == "cython":
            if not (_is_dense_matrix(H) or _is_csr_matrix(H)):
                H = _as_hfun(H)

            self._impl = _lanczos_timeprop_cython(H, maxsteps, target_convg, debug, do_full_order)
        else:
            self._impl = _lanczos_timeprop_reference(H, maxsteps, target_convg, debug, do_full_order)

    def propagate(self, phi0, ts, maxHT=None):
        ts = np.asarray(ts, dtype=float)
        assert ts.ndim == 1, "ts must be a 1d array"

        def get_phi_out(x):
            return np.asarray(x)

        if isinstance(phi0, np.ndarray):
            phi0_arr = np.asarray(phi0, dtype=np.complex128).reshape(-1)
        elif have_qutip and isinstance(phi0, qutip.Qobj):
            phi0_arr, get_phi_out = _qobj_state_io(phi0)
            phi0_arr = np.asarray(phi0_arr, dtype=np.complex128).reshape(-1)
        else:
            raise TypeError("lanczos_timeprop only supports numpy.ndarray and qutip.Qobj states")

        out = self._impl.propagate(phi0_arr, ts, maxHT)
        return [get_phi_out(x) for x in out]

    def _step(self, t, HT):
        return self._impl._step(t, HT)

    def __getattr__(self, name):
        return getattr(self._impl, name)


def sesolve_lanczos(H, phi0, ts, maxsteps, target_convg, maxHT=None, debug=0, do_full_order=False, backend="auto"):
    prop = lanczos_timeprop(H, maxsteps, target_convg, debug, do_full_order, backend)
    return prop.propagate(phi0, ts, maxHT)
