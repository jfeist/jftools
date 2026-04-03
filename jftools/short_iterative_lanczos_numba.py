import numpy as np
from numba import njit
from numba_lapack import dstevd, zgemm
from scipy import sparse as sp


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


@njit(cache=True)
def _scal_numba(scale, x):
    for idx in range(x.shape[0]):
        x[idx] *= scale


@njit(cache=True)
def _zero_vec_numba(x):
    for idx in range(x.shape[0]):
        x[idx] = 0.0 + 0.0j


@njit(cache=True)
def _csr_matvec_numba(data, indices, indptr, x, y):
    for row in range(indptr.shape[0] - 1):
        acc = 0.0 + 0.0j
        for pos in range(indptr[row], indptr[row + 1]):
            acc += data[pos] * x[indices[pos]]
        y[row] = acc


@njit(cache=True)
def _dense_gemm_matvec_numba(H, x, y):
    charN = np.uint8(ord("N"))
    zgemm(charN, charN, H.shape[0], 1, H.shape[1], 1.0 + 0.0j, H,
          H.shape[0], x, x.shape[0], 0.0 + 0.0j, y, y.shape[0])


@njit(cache=True)
def _calc_coeff_numba(step, alpha, beta, HT, coeff, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info):
    coeff[:] = 0.0 + 0.0j
    if step <= 0:
        return

    diag_work[:step] = alpha[:step]
    offdiag_work[:step - 1] = beta[:step - 1]

    # work = lapack_work[:1 + 4 * step + step * step]
    # iwork = lapack_iwork[:3 + 5 * step]
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
def _step_dense_numba(H, HT, maxsteps, target_convg, do_full_order, breakdown_tol, alpha, beta, prefacs, curr_coeff, prev_coeff, phia, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info):
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
        _dense_gemm_matvec_numba(H, phia[step - 1], phia[step])
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
            _zero_vec_numba(phia[step])
            exact_complete = True
        else:
            prefacs[step] = 1.0 / phinorm
            if abs(np.log10(prefacs[step])) > 4.0:
                _scal_numba(prefacs[step], phia[step])
                prefacs[step] = 1.0

        prev_coeff[:] = curr_coeff
        _calc_coeff_numba(step, alpha, beta, HT_done, curr_coeff, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info)
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
        _calc_coeff_numba(step_count - 1, alpha, beta, HT_done, prev_coeff, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info)
        _calc_coeff_numba(step_count, alpha, beta, HT_done, curr_coeff, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info)
        convg = _coeff_diff_norm_numba(curr_coeff, prev_coeff, step_count)

    _scal_numba(curr_coeff[0], phia[0])
    for idx in range(1, step_count):
        scale = curr_coeff[idx] * prefacs[idx] / prefacs[0]
        _axpy_numba(scale, phia[idx], phia[0])

    return HT_done


@njit(cache=True)
def _step_csr_numba(data, indices, indptr, HT, maxsteps, target_convg, do_full_order, breakdown_tol, alpha, beta, prefacs, curr_coeff, prev_coeff, phia, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info):
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
        _csr_matvec_numba(data, indices, indptr, phia[step - 1], phia[step])
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
            _zero_vec_numba(phia[step])
            exact_complete = True
        else:
            prefacs[step] = 1.0 / phinorm
            if abs(np.log10(prefacs[step])) > 4.0:
                _scal_numba(prefacs[step], phia[step])
                prefacs[step] = 1.0

        prev_coeff[:] = curr_coeff
        _calc_coeff_numba(step, alpha, beta, HT_done, curr_coeff, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info)
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
        _calc_coeff_numba(step_count - 1, alpha, beta, HT_done, prev_coeff, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info)
        _calc_coeff_numba(step_count, alpha, beta, HT_done, curr_coeff, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info)
        convg = _coeff_diff_norm_numba(curr_coeff, prev_coeff, step_count)

    _scal_numba(curr_coeff[0], phia[0])
    for idx in range(1, step_count):
        scale = curr_coeff[idx] * prefacs[idx] / prefacs[0]
        _axpy_numba(scale, phia[idx], phia[0])

    return HT_done


@njit(cache=True)
def _propagate_dense_numba(H, ts, use_maxht, maxht_value, maxsteps, target_convg, do_full_order, breakdown_tol, alpha, beta, prefacs, curr_coeff, prev_coeff, phia, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info, out):
    tt = ts[0]
    out[0, :] = phia[0, :]
    for out_idx in range(1, ts.shape[0]):
        tf = ts[out_idx]
        while tt < tf:
            HT = tf - tt
            if use_maxht and HT > maxht_value:
                HT = maxht_value
            HT_done = _step_dense_numba(H, HT, maxsteps, target_convg, do_full_order, breakdown_tol, alpha, beta, prefacs, curr_coeff, prev_coeff, phia, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info)
            if not np.isfinite(HT_done) or HT_done <= 0.0:
                raise ValueError("Numba Lanczos backend produced a non-positive propagation step")
            tt += HT_done
        out[out_idx, :] = phia[0, :]


@njit(cache=True)
def _propagate_csr_numba(data, indices, indptr, ts, use_maxht, maxht_value, maxsteps, target_convg, do_full_order, breakdown_tol, alpha, beta, prefacs, curr_coeff, prev_coeff, phia, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info, out):
    tt = ts[0]
    out[0, :] = phia[0, :]
    for out_idx in range(1, ts.shape[0]):
        tf = ts[out_idx]
        while tt < tf:
            HT = tf - tt
            if use_maxht and HT > maxht_value:
                HT = maxht_value
            HT_done = _step_csr_numba(data, indices, indptr, HT, maxsteps, target_convg, do_full_order, breakdown_tol, alpha, beta, prefacs, curr_coeff, prev_coeff, phia, eigvecs, diag_work, offdiag_work, lapack_work, lapack_iwork, lapack_info)
            if not np.isfinite(HT_done) or HT_done <= 0.0:
                raise ValueError("Numba Lanczos backend produced a non-positive propagation step")
            tt += HT_done
        out[out_idx, :] = phia[0, :]


class _lanczos_timeprop_numba:
    def __init__(self, H, maxsteps, target_convg, debug=0, do_full_order=False):
        if callable(H):
            raise TypeError("Numba Lanczos backend only supports static dense numpy arrays and scipy CSR matrices")

        self.maxsteps = maxsteps
        self.target_convg = target_convg
        self.debug = debug
        self.do_full_order = do_full_order
        self.breakdown_tol = 1e-14

        if sp.isspmatrix_csr(H) or (hasattr(sp, "csr_array") and isinstance(H, sp.csr_array)):
            self.kind = "csr"
            H_csr = sp.csr_matrix(H).astype(np.complex128)
            self.dim = H_csr.shape[0]
            self.H_data = np.asarray(H_csr.data, dtype=np.complex128)
            self.H_indices = np.asarray(H_csr.indices, dtype=np.int64)
            self.H_indptr = np.asarray(H_csr.indptr, dtype=np.int64)
            self.H_dense = None
        else:
            H_dense = np.asfortranarray(H, dtype=np.complex128)
            if H_dense.ndim != 2 or H_dense.shape[0] != H_dense.shape[1]:
                raise TypeError("Numba Lanczos backend only supports static dense numpy arrays and scipy CSR matrices")
            self.kind = "dense"
            self.dim = H_dense.shape[0]
            self.H_dense = H_dense
            self.H_data = None
            self.H_indices = None
            self.H_indptr = None

        self.alpha = np.empty(maxsteps + 1, dtype=np.float64)
        self.beta = np.empty(maxsteps, dtype=np.float64)
        self.prefacs = np.empty(maxsteps + 1, dtype=np.float64)
        self.curr_coeff = np.empty(maxsteps + 1, dtype=np.complex128)
        self.prev_coeff = np.empty(maxsteps + 1, dtype=np.complex128)
        self.phia = np.empty((maxsteps + 1, self.dim), dtype=np.complex128)
        self.eigvecs = np.asfortranarray(np.empty((maxsteps, maxsteps), dtype=np.float64))
        self.diag_work = np.empty(maxsteps, dtype=np.float64)
        self.offdiag_work = np.empty(maxsteps - 1, dtype=np.float64)
        self.lapack_work = np.empty(1 + 4 * maxsteps + maxsteps * maxsteps, dtype=np.float64)
        self.lapack_iwork = np.empty(3 + 5 * maxsteps, dtype=np.int32)
        self.lapack_info = np.zeros(1, dtype=np.int32)


    def propagate(self, phi0, ts, maxHT=None):
        phi0 = np.asarray(phi0, dtype=np.complex128)
        if phi0.ndim != 1:
            raise ValueError("phi0 must be a 1d complex array")
        if phi0.shape[0] != self.dim:
            raise ValueError("State dimension does not match Hamiltonian dimension")

        ts = np.asarray(ts, dtype=np.float64)
        out = np.empty((ts.shape[0], self.dim), dtype=np.complex128)
        self.phia.fill(0.0)
        self.phia[0, :] = phi0

        use_maxht = maxHT is not None
        maxht_value = 0.0 if maxHT is None else float(maxHT)

        if self.kind == "dense":
            _propagate_dense_numba(
                self.H_dense,
                ts,
                use_maxht,
                maxht_value,
                self.maxsteps,
                self.target_convg,
                self.do_full_order,
                self.breakdown_tol,
                self.alpha,
                self.beta,
                self.prefacs,
                self.curr_coeff,
                self.prev_coeff,
                self.phia,
                self.eigvecs,
                self.diag_work,
                self.offdiag_work,
                self.lapack_work,
                self.lapack_iwork,
                self.lapack_info,
                out,
            )
        else:
            _propagate_csr_numba(
                self.H_data,
                self.H_indices,
                self.H_indptr,
                ts,
                use_maxht,
                maxht_value,
                self.maxsteps,
                self.target_convg,
                self.do_full_order,
                self.breakdown_tol,
                self.alpha,
                self.beta,
                self.prefacs,
                self.curr_coeff,
                self.prev_coeff,
                self.phia,
                self.eigvecs,
                self.diag_work,
                self.offdiag_work,
                self.lapack_work,
                self.lapack_iwork,
                self.lapack_info,
                out,
            )

        return out