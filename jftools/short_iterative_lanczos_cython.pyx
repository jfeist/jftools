# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np
import time
cimport numpy as cnp
from cython cimport view
from libc.math cimport sqrt, log10, fabs, pow, cos, sin
from scipy import sparse as sp
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack

ctypedef cnp.complex128_t c128
ctypedef cnp.float64_t f64
ctypedef cnp.int64_t i64


cdef inline c128 _vdot(c128[::1] a, c128[::1] b) noexcept:
    cdef Py_ssize_t i, n = a.shape[0]
    cdef c128 s = 0.0 + 0.0j
    for i in range(n):
        s += a[i].conjugate() * b[i]
    return s


cdef inline double _vnorm(c128[::1] a) noexcept:
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double s = 0.0
    for i in range(n):
        s += (a[i].real * a[i].real + a[i].imag * a[i].imag)
    return sqrt(s)


cdef inline double _coeff_diff_norm(c128[::1] a, c128[::1] b) noexcept:
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double s = 0.0
    cdef c128 d
    for i in range(n):
        d = a[i] - b[i]
        s += (d.real * d.real + d.imag * d.imag)
    return sqrt(s)


cdef void _csr_matvec(c128[::1] data, i64[::1] indices, i64[::1] indptr,
                      c128[::1] x, c128[::1] y) noexcept nogil:
    cdef Py_ssize_t i, jj, n = indptr.shape[0] - 1
    cdef c128 acc
    for i in range(n):
        acc = 0.0 + 0.0j
        for jj in range(indptr[i], indptr[i + 1]):
            acc += data[jj] * x[indices[jj]]
        y[i] = acc


cdef class CythonLanczosPropagator:
    cdef int maxsteps
    cdef double target_convg
    cdef int debug
    cdef bint do_full_order
    cdef bint is_csr
    cdef bint profile_enabled
    cdef bint use_numpy_matmul
    cdef bint use_numpy_dot
    cdef int dim

    cdef object H_dense
    cdef object H_data
    cdef object H_indices
    cdef object H_indptr

    cdef object alpha
    cdef object beta
    cdef object prefacs
    cdef object curr_coeff
    cdef object prev_coeff
    cdef object phia
    cdef f64[::1] coeff_d_buf
    cdef f64[::1] coeff_e_buf
    cdef f64[:, ::1] coeff_z_buf
    cdef f64[::1] coeff_work_buf
    cdef int[::1] coeff_iwork_buf
    cdef double profile_total
    cdef double profile_matvec
    cdef double profile_update
    cdef double profile_coeff
    cdef double profile_reconstruct
    cdef Py_ssize_t profile_step_calls
    cdef Py_ssize_t profile_coeff_calls

    def __cinit__(self, H, int maxsteps, double target_convg, int debug=0, bint do_full_order=False, bint profile_enabled=False):
        self.maxsteps = maxsteps
        self.target_convg = target_convg
        self.debug = debug
        self.do_full_order = do_full_order
        cdef str dense_backend = "numpy_matmul"

        self.profile_enabled = profile_enabled
        self.use_numpy_matmul = dense_backend in ("numpy", "numpy_matmul", "matmul")
        self.use_numpy_dot = dense_backend in ("numpy_dot", "dot")
        self.profile_total = 0.0
        self.profile_matvec = 0.0
        self.profile_update = 0.0
        self.profile_coeff = 0.0
        self.profile_reconstruct = 0.0
        self.profile_step_calls = 0
        self.profile_coeff_calls = 0

        if sp.isspmatrix_csr(H) or (hasattr(sp, "csr_array") and isinstance(H, sp.csr_array)):
            self.is_csr = True
            self.dim = H.shape[0]
            H_csr = sp.csr_matrix(H).astype(np.complex128)
            self.H_data = np.asarray(H_csr.data, dtype=np.complex128)
            self.H_indices = np.asarray(H_csr.indices, dtype=np.int64)
            self.H_indptr = np.asarray(H_csr.indptr, dtype=np.int64)
        else:
            self.is_csr = False
            self.dim = np.asarray(H).shape[0]
            if self.use_numpy_matmul or self.use_numpy_dot:
                self.H_dense = np.ascontiguousarray(H, dtype=np.complex128)
            else:
                self.H_dense = np.asfortranarray(H, dtype=np.complex128)

        # Allocate working arrays as NumPy arrays (can be used for both C-level and Python-level operations)
        self.alpha = np.zeros(maxsteps + 1, dtype=np.float64)
        self.beta = np.zeros(maxsteps + 1, dtype=np.float64)
        self.prefacs = np.zeros(maxsteps + 1, dtype=np.float64)
        self.curr_coeff = np.zeros(maxsteps + 1, dtype=np.complex128)
        self.prev_coeff = np.zeros(maxsteps + 1, dtype=np.complex128)

        # Allocate LAPACK work buffers once and reuse in _calc_coeff.
        self.coeff_d_buf = view.array(shape=(maxsteps,), itemsize=sizeof(f64), format="d")
        self.coeff_e_buf = view.array(shape=(maxsteps,), itemsize=sizeof(f64), format="d")
        self.coeff_z_buf = view.array(shape=(maxsteps, maxsteps), itemsize=sizeof(f64), format="d", mode="c")
        self.coeff_work_buf = view.array(shape=(1 + 4 * maxsteps + maxsteps * maxsteps,), itemsize=sizeof(f64), format="d")
        self.coeff_iwork_buf = view.array(shape=(3 + 5 * maxsteps,), itemsize=sizeof(int), format="i")

    cpdef list propagate(self, cnp.ndarray[c128, ndim=1] phi0, cnp.ndarray[f64, ndim=1] ts, object maxHT=None):
        cdef int n = phi0.shape[0]
        cdef double tt, tf, HT, HT_done
        cdef double t0
        cdef list out = []

        if self.dim != n:
            raise ValueError("State dimension does not match Hamiltonian dimension")

        if self.profile_enabled:
            t0 = time.perf_counter()

        self.phia = np.empty((self.maxsteps + 1, n), dtype=np.complex128)
        self.phia[:] = phi0

        tt = ts[0]
        out.append(self.phia[0].copy())

        for tf in ts[1:]:
            while tt < tf:
                HT = tf - tt
                if maxHT is not None and HT > maxHT:
                    HT = maxHT
                HT_done = self._step(HT)
                tt += HT_done
            out.append(self.phia[0].copy())

        if self.profile_enabled:
            self.profile_total += time.perf_counter() - t0

        return out

    cdef void _calc_coeff(self,
                          int step,
                          double HT,
                          c128[::1] coeff) except *:
        # dstevd overwrites d/e and writes eigenvectors to z with leading dimension ldz.
        # z is allocated once as C-order (maxsteps x maxsteps), so ldz must be maxsteps.
        cdef int n = step
        cdef int ldz = <int>self.coeff_z_buf.shape[1]
        cdef int lwork = <int>self.coeff_work_buf.shape[0]
        cdef int liwork = <int>self.coeff_iwork_buf.shape[0]
        cdef int info = 0
        cdef char jobz = b'V'
        cdef int i, j
        cdef c128 fac
        cdef double ang
        cdef f64[::1] alpha_v = self.alpha
        cdef f64[::1] beta_v = self.beta

        self.coeff_d_buf[:n] = alpha_v[:n]
        self.coeff_e_buf[:n - 1] = beta_v[1:n]

        lapack.dstevd(&jobz, &n,
                      &self.coeff_d_buf[0], &self.coeff_e_buf[0],
                      &self.coeff_z_buf[0, 0], &ldz,
                      &self.coeff_work_buf[0], &lwork,
                      &self.coeff_iwork_buf[0], &liwork,
                      &info)
        if info != 0:
            raise RuntimeError(f"dstevd failed with info={info}")

        for i in range(step + 1):
            coeff[i] = 0.0
        for i in range(n):
            ang = -HT * self.coeff_d_buf[i]
            fac = self.coeff_z_buf[i, 0] * (cos(ang) + 1j * sin(ang))
            for j in range(n):
                coeff[j] = coeff[j] + self.coeff_z_buf[i, j] * fac

    cdef double _step(self, double HT):
        cdef int step, ii, jj, n
        cdef double HT_done = HT
        cdef double phinorm, convg, scale
        cdef double t0
        cdef c128 dotpr
        cdef c128 s
        cdef c128 alpha_blas, beta_blas
        cdef int incx, incy, lda, m, ncol, vec_n
        cdef char trans

        cdef c128[:, ::1] phia_v = self.phia
        cdef cnp.ndarray[c128, ndim=2, mode='fortran'] H_dense_f
        cdef f64[::1] alpha_v = self.alpha
        cdef f64[::1] beta_v = self.beta
        cdef f64[::1] prefacs_v = self.prefacs
        cdef c128[::1] curr_v = self.curr_coeff
        cdef c128[::1] prev_v = self.prev_coeff

        n = phia_v.shape[1]
        vec_n = n
        if not self.is_csr:
            m = n
            ncol = n
            incx = 1
            incy = 1
            if not (self.use_numpy_matmul or self.use_numpy_dot):
                H_dense_f = self.H_dense
                trans = 'N'
                lda = n
                alpha_blas = 1.0 + 0.0j
                beta_blas = 0.0 + 0.0j

        phinorm = _vnorm(phia_v[0])
        prefacs_v[0] = 1.0 / phinorm
        self.profile_step_calls += 1
        for ii in range(self.maxsteps + 1):
            curr_v[ii] = 0.0 + 0.0j

        for step in range(1, self.maxsteps + 1):
            if self.profile_enabled:
                t0 = time.perf_counter()
            if self.is_csr:
                _csr_matvec(self.H_data, self.H_indices, self.H_indptr, phia_v[step - 1], phia_v[step])
            else:
                if self.use_numpy_matmul:
                    self.phia[step, :] = self.H_dense @ self.phia[step - 1, :]
                elif self.use_numpy_dot:
                    self.phia[step, :] = self.H_dense.dot(self.phia[step - 1, :])
                else:
                    blas.zgemv(&trans, &m, &ncol, &alpha_blas, &H_dense_f[0, 0], &lda, &phia_v[step - 1, 0], &incx, &beta_blas, &phia_v[step, 0], &incy)
            if self.profile_enabled:
                self.profile_matvec += time.perf_counter() - t0

            if self.profile_enabled:
                t0 = time.perf_counter()
            prefacs_v[step] = prefacs_v[step - 1]
            phinorm = prefacs_v[step] * _vnorm(phia_v[step])

            dotpr = prefacs_v[step - 1] * prefacs_v[step] * _vdot(phia_v[step - 1], phia_v[step])
            alpha_v[step - 1] = dotpr.real

            s = alpha_v[step - 1] * prefacs_v[step - 1] / prefacs_v[step]
            if not self.is_csr:
                s = -s
                blas.zaxpy(&vec_n, &s, &phia_v[step - 1, 0], &incx, &phia_v[step, 0], &incy)
            else:
                for jj in range(n):
                    phia_v[step, jj] -= s * phia_v[step - 1, jj]

            if fabs(phinorm * phinorm - alpha_v[step - 1] * alpha_v[step - 1]) < 0.1:
                dotpr = prefacs_v[step - 1] * prefacs_v[step] * _vdot(phia_v[step - 1], phia_v[step])
                s = dotpr * prefacs_v[step - 1] / prefacs_v[step]
                if not self.is_csr:
                    s = -s
                    blas.zaxpy(&vec_n, &s, &phia_v[step - 1, 0], &incx, &phia_v[step, 0], &incy)
                else:
                    for jj in range(n):
                        phia_v[step, jj] -= s * phia_v[step - 1, jj]

            if step >= 2:
                s = beta_v[step - 1] * prefacs_v[step - 2] / prefacs_v[step]
                if not self.is_csr:
                    s = -s
                    blas.zaxpy(&vec_n, &s, &phia_v[step - 2, 0], &incx, &phia_v[step, 0], &incy)
                else:
                    for jj in range(n):
                        phia_v[step, jj] -= s * phia_v[step - 2, jj]

            phinorm = _vnorm(phia_v[step])
            beta_v[step] = prefacs_v[step] * phinorm
            prefacs_v[step] = 1.0 / phinorm

            if fabs(log10(prefacs_v[step])) > 4.0:
                s = prefacs_v[step]
                if not self.is_csr:
                    blas.zscal(&vec_n, &s, &phia_v[step, 0], &incx)
                else:
                    for jj in range(n):
                        phia_v[step, jj] *= s
                prefacs_v[step] = 1.0
            if self.profile_enabled:
                self.profile_update += time.perf_counter() - t0

            self.prev_coeff[:] = self.curr_coeff
            if self.profile_enabled:
                t0 = time.perf_counter()
            self._calc_coeff(step, HT_done, curr_v)
            if self.profile_enabled:
                self.profile_coeff += time.perf_counter() - t0
                self.profile_coeff_calls += 1
            convg = _coeff_diff_norm(curr_v, prev_v)
            if (not self.do_full_order) and convg < self.target_convg:
                break

        while convg > self.target_convg:
            scale = 0.95 * pow(self.target_convg / convg, 1.0 / step)
            if scale < 0.5:
                scale = 0.5
            HT_done = HT_done * scale
            if self.profile_enabled:
                t0 = time.perf_counter()
            self._calc_coeff(step - 1, HT_done, prev_v)
            self._calc_coeff(step, HT_done, curr_v)
            if self.profile_enabled:
                self.profile_coeff += time.perf_counter() - t0
                self.profile_coeff_calls += 2
            convg = _coeff_diff_norm(curr_v, prev_v)

        if self.profile_enabled:
            t0 = time.perf_counter()
        s = curr_v[0]
        if not self.is_csr:
            blas.zscal(&vec_n, &s, &phia_v[0, 0], &incx)
        else:
            for jj in range(n):
                phia_v[0, jj] *= s
        for ii in range(1, step + 1):
            s = curr_v[ii] * prefacs_v[ii] / prefacs_v[0]
            if not self.is_csr:
                blas.zaxpy(&vec_n, &s, &phia_v[ii, 0], &incx, &phia_v[0, 0], &incy)
            else:
                for jj in range(n):
                    phia_v[0, jj] += s * phia_v[ii, jj]
        if self.profile_enabled:
            self.profile_reconstruct += time.perf_counter() - t0

        return HT_done

    cpdef dict get_profile_stats(self):
        return {
            "total": self.profile_total,
            "matvec": self.profile_matvec,
            "update": self.profile_update,
            "coeff": self.profile_coeff,
            "reconstruct": self.profile_reconstruct,
            "step_calls": int(self.profile_step_calls),
            "coeff_calls": int(self.profile_coeff_calls),
            "dense_matvec_backend": "numpy_matmul" if self.use_numpy_matmul else ("numpy_dot" if self.use_numpy_dot else "scipy_blas"),
        }


def sesolve_lanczos_cython(H,
                           cnp.ndarray[c128, ndim=1] phi0,
                           cnp.ndarray[f64, ndim=1] ts,
                           int maxsteps,
                           double target_convg,
                           object maxHT=None,
                           int debug=0,
                           bint do_full_order=False,
                           bint profile_enabled=False):
    cdef CythonLanczosPropagator prop = CythonLanczosPropagator(H, maxsteps, target_convg, debug, do_full_order, profile_enabled)
    return prop.propagate(phi0, ts, maxHT)
