import numpy as np
from numba import njit, objmode
from numba.core import types
from numba.extending import overload
from numba_lapack import dstevd, zgemm
from scipy import sparse as sp

_CALLABLE_OPERATOR_REGISTRY = {}

def _register_callable_operator(H):
    handle = id(H)
    _CALLABLE_OPERATOR_REGISTRY[handle] = H
    return handle


@njit(cache=True)
def _vdot_numba(a, b):
    out = 0.0 + 0.0j
    for idx in range(a.shape[0]):
        out += np.conjugate(a[idx]) * b[idx]
    return out


@njit(cache=True)
def _vnorm_numba(a):
    out = 0.0
    for idx in range(a.shape[0]):
        out += a[idx].real * a[idx].real + a[idx].imag * a[idx].imag
    return np.sqrt(out)


@njit(cache=True)
def _coeff_diff_norm_numba(a, b, n):
    out = 0.0
    for idx in range(n):
        delta = a[idx] - b[idx]
        out += delta.real * delta.real + delta.imag * delta.imag
    return np.sqrt(out)


@njit(cache=True)
def _axpy_numba(scale, x, y):
    for idx in range(x.shape[0]):
        y[idx] += scale * x[idx]


def _apply_H_operator_numba(H, t, x, y):
    raise NotImplementedError


@overload(_apply_H_operator_numba)
def _overload_apply_H_operator_numba(H, t, x, y):
    if isinstance(H, types.BaseTuple) and len(H) == 3:
        def impl(H, t, x, y):
            data, indices, indptr = H
            for row in range(indptr.shape[0] - 1):
                acc = 0.0 + 0.0j
                for pos in range(indptr[row], indptr[row + 1]):
                    acc += data[pos] * x[indices[pos]]
                y[row] = acc

    elif isinstance(H, types.Array):
        def impl(H, t, x, y):
            charN = np.uint8(ord("N"))
            zgemm(charN, charN, H.shape[0], 1, H.shape[1], 1.0 + 0.0j, H,
                  H.shape[0], x, x.shape[0], 0.0 + 0.0j, y, y.shape[0])

    elif isinstance(H, types.Integer):
        def impl(H, t, x, y):
            with objmode():
                _CALLABLE_OPERATOR_REGISTRY[H](t, x, y)

    else:
        raise TypeError("Numba Lanczos operator must be dense array, CSR tuple, or callable handle")

    return impl


@njit(cache=True)
def _calc_coeff_numba(step, HT, coeff, scratch):
    alpha, beta, _, _, _, _, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info = scratch
    coeff[:] = 0.0 + 0.0j
    if step <= 0:
        return

    diag_work[:step] = alpha[:step]
    offdiag_work[:step - 1] = beta[:step - 1]

    lapack_info[0] = 0
    dstevd(np.uint8(ord("V")), step, diag_work, offdiag_work, eigvecs, eigvecs.shape[0],
           lapack_work, lapack_work.shape[0], lapack_iwork, lapack_iwork.shape[0], lapack_info)
    if lapack_info[0] != 0:
        raise ValueError("dstevd failed in Numba Lanczos backend")

    for idx in range(step):
        factor = eigvecs[0, idx] * np.exp(-1j * HT * diag_work[idx])
        for jdx in range(step):
            coeff[jdx] += eigvecs[jdx, idx] * factor


@njit(cache=True)
def _step_numba(operator, t, HT, config, scratch):
    maxsteps, target_convg, do_full_order, breakdown_tol = config
    alpha, beta, prefacs, curr_coeff, prev_coeff, phia, _, _, _, _, _, _ = scratch
    max_lanczos_steps = min(maxsteps, phia.shape[1])
    HT_done = HT

    phinorm = _vnorm_numba(phia[0])
    prefacs[0] = 1.0 / phinorm
    curr_coeff[:] = 0.0 + 0.0j

    convg = 1.0e300
    exact_complete = False
    step_count = 0

    for step in range(1, max_lanczos_steps + 1):
        step_count = step
        _apply_H_operator_numba(operator, t, phia[step - 1], phia[step])
        prefacs[step] = prefacs[step - 1]
        phinorm = prefacs[step] * _vnorm_numba(phia[step])

        dotpr = prefacs[step - 1] * prefacs[step] * _vdot_numba(phia[step - 1], phia[step])
        alpha[step - 1] = dotpr.real

        scale = -alpha[step - 1] * prefacs[step - 1] / prefacs[step]
        _axpy_numba(scale, phia[step - 1], phia[step])

        if abs(phinorm * phinorm - alpha[step - 1] * alpha[step - 1]) < 0.1:
            dotpr = prefacs[step - 1] * prefacs[step] * _vdot_numba(phia[step - 1], phia[step])
            scale = -dotpr * prefacs[step - 1] / prefacs[step]
            _axpy_numba(scale, phia[step - 1], phia[step])

        if step >= 2:
            scale = -beta[step - 2] * prefacs[step - 2] / prefacs[step]
            _axpy_numba(scale, phia[step - 2], phia[step])

        phinorm = _vnorm_numba(phia[step])
        beta[step - 1] = prefacs[step] * phinorm
        if phinorm <= breakdown_tol:
            prefacs[step] = 1.0
            phia[step] = 0.
            exact_complete = True
        else:
            prefacs[step] = 1.0 / phinorm
            if abs(np.log10(prefacs[step])) > 4.0:
                phia[step] *= prefacs[step]
                prefacs[step] = 1.0

        prev_coeff[:] = curr_coeff
        _calc_coeff_numba(step, HT_done, curr_coeff, scratch)
        convg = _coeff_diff_norm_numba(curr_coeff, prev_coeff, step)

        if exact_complete or step == max_lanczos_steps:
            break
        if (not do_full_order) and convg < target_convg:
            break

    while (not exact_complete) and convg > target_convg:
        scale = 0.95 * (target_convg / convg) ** (1.0 / step_count)
        if scale < 0.5:
            scale = 0.5
        HT_done *= scale
        _calc_coeff_numba(step_count - 1, HT_done, prev_coeff, scratch)
        _calc_coeff_numba(step_count, HT_done, curr_coeff, scratch)
        convg = _coeff_diff_norm_numba(curr_coeff, prev_coeff, step_count)

    phia[0] *= curr_coeff[0]
    for idx in range(1, step_count):
        scale = curr_coeff[idx] * prefacs[idx] / prefacs[0]
        _axpy_numba(scale, phia[idx], phia[0])

    return HT_done


@njit(cache=True)
def _propagate_numba(operator, ts, use_maxht, maxht_value, config, scratch, out):
    _, _, _, _, _, phia, _, _, _, _, _, _ = scratch
    tt = ts[0]
    out[0, :] = phia[0, :]
    for out_idx in range(1, ts.shape[0]):
        tf = ts[out_idx]
        while tt < tf:
            HT = tf - tt
            if use_maxht and HT > maxht_value:
                HT = maxht_value
            HT_done = _step_numba(operator, tt, HT, config, scratch)
            if not np.isfinite(HT_done) or HT_done <= 0.0:
                raise ValueError("Numba Lanczos backend produced a non-positive propagation step")
            tt += HT_done
        out[out_idx, :] = phia[0, :]


def _normalize_static_operator(H):
    if sp.isspmatrix_csr(H) or (hasattr(sp, "csr_array") and isinstance(H, sp.csr_array)):
        H_csr = sp.csr_matrix(H).astype(np.complex128)
        return (
            H_csr.shape[0],
            (H_csr.data, H_csr.indices.astype(np.int64), H_csr.indptr.astype(np.int64))
        )

    H_dense = np.asfortranarray(H, dtype=np.complex128)
    if H_dense.ndim != 2 or H_dense.shape[0] != H_dense.shape[1]:
        raise TypeError("Numba Lanczos backend only supports static dense numpy arrays and scipy CSR matrices")

    return (H_dense.shape[0], H_dense)


def _allocate_scratch(maxsteps, dim):
    return (
        np.empty(maxsteps + 1, dtype=np.float64),  # alpha diagonal terms
        np.empty(maxsteps, dtype=np.float64),  # beta off-diagonal terms
        np.empty(maxsteps + 1, dtype=np.float64),  # normalization prefactors
        np.empty(maxsteps + 1, dtype=np.complex128),  # current expansion coefficients
        np.empty(maxsteps + 1, dtype=np.complex128),  # previous expansion coefficients
        np.empty((maxsteps + 1, dim), dtype=np.complex128),  # Lanczos basis vectors
        np.empty((maxsteps, maxsteps), dtype=np.float64, order="F"),  # tridiagonal eigenvectors (Fortran order)
        np.empty(maxsteps, dtype=np.float64),  # dstevd diagonal workspace
        np.empty(maxsteps - 1, dtype=np.float64),  # dstevd off-diagonal workspace
        np.empty(1 + 4 * maxsteps + maxsteps * maxsteps, dtype=np.float64),  # dstevd work array
        np.empty(3 + 5 * maxsteps, dtype=np.int32),  # dstevd integer work array
        np.zeros(1, dtype=np.int32),  # dstevd info flag
    )


class _lanczos_timeprop_numba:
    def __init__(self, H, maxsteps, target_convg, debug=0, do_full_order=False):
        self.maxsteps = maxsteps
        self.target_convg = target_convg
        self.debug = debug
        self.do_full_order = do_full_order
        self.breakdown_tol = 1e-14
        self.config = (maxsteps, target_convg, do_full_order, self.breakdown_tol)
        self.H = H

        if callable(H):
            self.operator = _register_callable_operator(H)
            self.dim = None
            self.scratch = None
        else:
            self.dim, self.operator = _normalize_static_operator(H)
            self.scratch = _allocate_scratch(maxsteps, self.dim)

    def __del__(self):
        if callable(self.H) and self.operator in _CALLABLE_OPERATOR_REGISTRY:
            del _CALLABLE_OPERATOR_REGISTRY[self.operator]

    def propagate(self, phi0, ts, maxHT=None):
        phi0 = np.asarray(phi0, dtype=np.complex128)
        if phi0.ndim != 1:
            raise ValueError("phi0 must be a 1d complex array")

        if self.dim is None:
            if not callable(self.H):
                raise ValueError("Internal error: non-callable operator should have dimension set")
            self.dim = phi0.shape[0]
            self.scratch = _allocate_scratch(self.maxsteps, self.dim)
        else:
            if phi0.shape[0] != self.dim:
                raise ValueError("State dimension does not match Hamiltonian dimension")

        ts = np.asarray(ts, dtype=np.float64)
        out = np.empty((ts.shape[0], self.dim), dtype=np.complex128)
        self.scratch[5].fill(0.0)
        self.scratch[5][0, :] = phi0

        use_maxht = maxHT is not None
        maxht_value = 0.0 if maxHT is None else float(maxHT)

        _propagate_numba(self.operator, ts, use_maxht, maxht_value, self.config, self.scratch, out)

        return out