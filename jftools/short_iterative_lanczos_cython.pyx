# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

import numpy as np
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
    cdef int n = a.shape[0]
    cdef int inc = 1
    return blas.zdotc(&n, &a[0], &inc, &b[0], &inc)


cdef inline double _vnorm(c128[::1] a) noexcept:
    cdef int n = a.shape[0]
    cdef int inc = 1
    return blas.dznrm2(&n, &a[0], &inc)


cdef inline double _coeff_diff_norm(c128[::1] a, c128[::1] b, int n) noexcept:
    cdef Py_ssize_t i
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
    cdef int dim

    cdef c128[::1, :] H_dense_f
    cdef c128[::1] H_data
    cdef i64[::1] H_indices
    cdef i64[::1] H_indptr

    cdef f64[::1] alpha
    cdef f64[::1] beta
    cdef f64[::1] prefacs
    cdef c128[::1] curr_coeff
    cdef c128[::1] prev_coeff
    cdef c128[:, ::1] phia
    cdef f64[::1] coeff_d_buf
    cdef f64[::1] coeff_e_buf
    cdef f64[:, ::1] coeff_z_buf
    cdef f64[::1] coeff_work_buf
    cdef int[::1] coeff_iwork_buf

    def __cinit__(self, H, int maxsteps, double target_convg, int debug=0, bint do_full_order=False):
        self.maxsteps = maxsteps
        self.target_convg = target_convg
        self.debug = debug
        self.do_full_order = do_full_order

        if sp.isspmatrix_csr(H) or (hasattr(sp, "csr_array") and isinstance(H, sp.csr_array)):
            self.is_csr = True
            self.dim = H.shape[0]
            H_csr = sp.csr_matrix(H).astype(np.complex128)
            self.H_data = np.asarray(H_csr.data, dtype=np.complex128)
            self.H_indices = np.asarray(H_csr.indices, dtype=np.int64)
            self.H_indptr = np.asarray(H_csr.indptr, dtype=np.int64)
        else:
            self.is_csr = False
            self.H_dense_f = np.asfortranarray(H, dtype=np.complex128)
            self.dim = self.H_dense_f.shape[0]

        # Allocate working arrays as Cython-owned buffers.
        self.phia = view.array(shape=(maxsteps + 1, self.dim), itemsize=sizeof(c128), format="Zd", mode="c")
        self.alpha = view.array(shape=(maxsteps + 1,), itemsize=sizeof(f64), format="d")
        self.beta = view.array(shape=(maxsteps + 1,), itemsize=sizeof(f64), format="d")
        self.prefacs = view.array(shape=(maxsteps + 1,), itemsize=sizeof(f64), format="d")
        self.curr_coeff = view.array(shape=(maxsteps + 1,), itemsize=sizeof(c128), format="Zd")
        self.prev_coeff = view.array(shape=(maxsteps + 1,), itemsize=sizeof(c128), format="Zd")

        # Allocate LAPACK work buffers once and reuse in _calc_coeff.
        self.coeff_d_buf = view.array(shape=(maxsteps,), itemsize=sizeof(f64), format="d")
        self.coeff_e_buf = view.array(shape=(maxsteps,), itemsize=sizeof(f64), format="d")
        self.coeff_z_buf = view.array(shape=(maxsteps, maxsteps), itemsize=sizeof(f64), format="d", mode="c")
        self.coeff_work_buf = view.array(shape=(1 + 4 * maxsteps + maxsteps * maxsteps,), itemsize=sizeof(f64), format="d")
        self.coeff_iwork_buf = view.array(shape=(3 + 5 * maxsteps,), itemsize=sizeof(int), format="i")

    cpdef cnp.ndarray propagate(self, cnp.ndarray[c128, ndim=1] phi0, cnp.ndarray[f64, ndim=1] ts, object maxHT=None):
        cdef int n = phi0.shape[0]
        cdef double tt, tf, HT, HT_done
        cdef int its
        cdef cnp.ndarray[c128, ndim=2] out
        cdef c128[:, ::1] out_v
        cdef c128[::1] phi0_v = phi0

        if self.dim != n:
            raise ValueError("State dimension does not match Hamiltonian dimension")

        self.phia[0, :] = phi0_v

        tt = ts[0]
        out = np.empty((len(ts), n), dtype=np.complex128)
        out_v = out
        out_v[0, :] = phi0_v
        its = 1

        for tf in ts[1:]:
            while tt < tf:
                HT = tf - tt
                if maxHT is not None and HT > maxHT:
                    HT = maxHT
                HT_done = self._step(HT)
                tt += HT_done
            out_v[its, :] = self.phia[0, :]
            its += 1

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

        self.coeff_d_buf[:n] = self.alpha[:n]
        self.coeff_e_buf[:n - 1] = self.beta[1:n]

        lapack.dstevd(&jobz, &n,
                      &self.coeff_d_buf[0], &self.coeff_e_buf[0],
                      &self.coeff_z_buf[0, 0], &ldz,
                      &self.coeff_work_buf[0], &lwork,
                      &self.coeff_iwork_buf[0], &liwork,
                      &info)
        if info != 0:
            raise RuntimeError(f"dstevd failed with info={info}")

        coeff[:step + 1] = 0.0
        for i in range(n):
            ang = -HT * self.coeff_d_buf[i]
            fac = self.coeff_z_buf[i, 0] * (cos(ang) + 1j * sin(ang))
            for j in range(n):
                coeff[j] += self.coeff_z_buf[i, j] * fac

    cdef double _step(self, double HT):
        cdef int step, ii, n
        cdef double HT_done = HT
        cdef double phinorm, convg, scale
        cdef c128 dotpr
        cdef c128 s
        cdef c128 alpha_blas, beta_blas
        cdef int incx, incy, lda, m, ncol, vec_n, one
        cdef char trans

        cdef c128[:, ::1] phia_v = self.phia
        cdef f64[::1] alpha_v = self.alpha
        cdef f64[::1] beta_v = self.beta
        cdef f64[::1] prefacs_v = self.prefacs
        cdef c128[::1] curr_v = self.curr_coeff
        cdef c128[::1] prev_v = self.prev_coeff

        n = phia_v.shape[1]
        vec_n = n
        incx = incy = one = 1
        if not self.is_csr:
            m = ncol = lda = n
            trans = 'N'
            alpha_blas = 1.0 + 0.0j
            beta_blas = 0.0 + 0.0j

        phinorm = _vnorm(phia_v[0])
        prefacs_v[0] = 1.0 / phinorm
        for ii in range(self.maxsteps + 1):
            curr_v[ii] = 0.0 + 0.0j

        for step in range(1, self.maxsteps + 1):
            if self.is_csr:
                _csr_matvec(self.H_data, self.H_indices, self.H_indptr, phia_v[step - 1], phia_v[step])
            else:
                blas.zgemm(&trans, &trans, &m, &one, &ncol, &alpha_blas, &self.H_dense_f[0, 0], &lda, &phia_v[step - 1, 0], &ncol, &beta_blas, &phia_v[step, 0], &m)

            prefacs_v[step] = prefacs_v[step - 1]
            phinorm = prefacs_v[step] * _vnorm(phia_v[step])

            dotpr = prefacs_v[step - 1] * prefacs_v[step] * _vdot(phia_v[step - 1], phia_v[step])
            alpha_v[step - 1] = dotpr.real

            s = -alpha_v[step - 1] * prefacs_v[step - 1] / prefacs_v[step]
            blas.zaxpy(&vec_n, &s, &phia_v[step - 1, 0], &incx, &phia_v[step, 0], &incy)

            if fabs(phinorm * phinorm - alpha_v[step - 1] * alpha_v[step - 1]) < 0.1:
                dotpr = prefacs_v[step - 1] * prefacs_v[step] * _vdot(phia_v[step - 1], phia_v[step])
                s = -dotpr * prefacs_v[step - 1] / prefacs_v[step]
                blas.zaxpy(&vec_n, &s, &phia_v[step - 1, 0], &incx, &phia_v[step, 0], &incy)

            if step >= 2:
                s = -beta_v[step - 1] * prefacs_v[step - 2] / prefacs_v[step]
                blas.zaxpy(&vec_n, &s, &phia_v[step - 2, 0], &incx, &phia_v[step, 0], &incy)

            phinorm = _vnorm(phia_v[step])
            beta_v[step] = prefacs_v[step] * phinorm
            prefacs_v[step] = 1.0 / phinorm

            if fabs(log10(prefacs_v[step])) > 4.0:
                s = prefacs_v[step]
                blas.zscal(&vec_n, &s, &phia_v[step, 0], &incx)
                prefacs_v[step] = 1.0

            self.prev_coeff[:] = self.curr_coeff
            self._calc_coeff(step, HT_done, curr_v)
            convg = _coeff_diff_norm(curr_v, prev_v, step + 1)
            if (not self.do_full_order) and convg < self.target_convg:
                break

        while convg > self.target_convg:
            scale = 0.95 * pow(self.target_convg / convg, 1.0 / step)
            if scale < 0.5:
                scale = 0.5
            HT_done = HT_done * scale
            self._calc_coeff(step - 1, HT_done, prev_v)
            self._calc_coeff(step, HT_done, curr_v)
            convg = _coeff_diff_norm(curr_v, prev_v, step + 1)

        s = curr_v[0]
        blas.zscal(&vec_n, &s, &phia_v[0, 0], &incx)
        for ii in range(1, step + 1):
            s = curr_v[ii] * prefacs_v[ii] / prefacs_v[0]
            blas.zaxpy(&vec_n, &s, &phia_v[ii, 0], &incx, &phia_v[0, 0], &incy)

        return HT_done
