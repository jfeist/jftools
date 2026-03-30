import time

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, expm_multiply

import jftools

try:
    import qutip
except ImportError:
    qutip = None


def _make_chain_hamiltonian(n):
    offdiag = -0.5 * np.ones(n - 1, dtype=float)
    return diags([offdiag, offdiag], offsets=[-1, 1], shape=(n, n), format="csr")


def _normalized_random_state(n, seed=2):
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return phi / np.linalg.norm(phi)


def _run_sil(H, phi0, ts, maxHT=0.1):
    return jftools.short_iterative_lanczos.sesolve_lanczos(H, phi0, ts, maxsteps=14, target_convg=1e-12, maxHT=maxHT)


def test_general():
    assert jftools.__all__ == ["shade_color", "tic", "toc", "ipynbimport_install", "unroll_phase", "interp_cmplx", "plotcolored", "fedvr", "short_iterative_lanczos"]


def test_shade_color():
    # Test shade_color function
    color = jftools.shade_color("blue", 50)
    assert color == (0.5, 0.5, 1.0)  # Expected RGBA value

    color = jftools.shade_color("blue", -50)
    assert color == (0.0, 0.0, 0.5)  # Expected RGBA value


def test_tic_toc():
    # Test tic and toc functions
    jftools.tic()
    # Perform some time-consuming operation
    time.sleep(0.1)
    elapsed_time = jftools.toc()
    assert elapsed_time > 0.1  # Check that elapsed time is at least the time slept


def test_ipynbimport_install():
    # Test ipynbimport_install function
    # jftools.ipynbimport_install('numpy')
    # assert 'numpy' in sys.modules  # Check that numpy is imported
    pass


def test_unroll_phase():
    # Test unroll_phase function
    unrolled_phase = jftools.unroll_phase([1, 2, 3, -3, -2])
    assert np.allclose(unrolled_phase, np.array([1.0, 2.0, 3.0, -3 + 2 * np.pi, -2 + 2 * np.pi]))

    unrolled_phase = jftools.unroll_phase(np.array([1, 2, 3, -3, -2], dtype=float))
    assert np.allclose(unrolled_phase, np.array([1.0, 2.0, 3.0, -3 + 2 * np.pi, -2 + 2 * np.pi]))

    unrolled_phase = jftools.unroll_phase([-2, -2.9, -3.5 + 2 * np.pi, -3.9 + 2 * np.pi, -5.5 + 2 * np.pi, -7 + 2 * np.pi, -10 + 4 * np.pi])
    assert np.allclose(unrolled_phase, np.array([-2, -2.9, -3.5, -3.9, -5.5, -7, -10]))


def test_interp_cmplx():
    # Test interp_cmplx function
    x = np.array([1, 2, 3, 4])
    y = np.array([1j, 2j, 3j, 4j])
    intp = jftools.interp_cmplx(x, y)
    assert np.allclose(intp(2.5), 2.5j)  # Check that complex interpolation is correct


def test_plotcolored():
    # Test plotcolored function
    # x = np.linspace(0, 1, 100)
    # y = np.sin(2 * np.pi * x)
    # jftools.plotcolored(x, y, color='red')
    # Manually check that the plot is displayed correctly
    pass


def test_fedvr():
    # Gauss-Legendre quadrature
    r, wt = jftools.fedvr.gaussq("legendre", 5, [])
    assert np.allclose(r, [-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664], atol=1e-15, rtol=1e-15)
    assert np.allclose(wt, [0.23692688505618908, 0.47862867049936647, 0.5688888888888889, 0.47862867049936647, 0.23692688505618908], atol=1e-15, rtol=1e-15)

    # Gauss-Legendre-Lobatto quadrature
    r, wt = jftools.fedvr.gaussq("legendre", 5, [-1, 1])
    assert np.allclose(r, [-1.0, -0.6546536707079771, 0, 0.6546536707079771, 1.0], atol=1e-15, rtol=1e-15)
    assert np.allclose(wt, [0.1, 0.5444444444444444, 0.7111111111111111, 0.5444444444444444, 0.1], atol=1e-15, rtol=1e-15)


def test_short_iterative_lanczos_dense_and_sparse_against_expm_multiply():
    n = 18
    H_sparse = _make_chain_hamiltonian(n)
    H_dense = H_sparse.toarray()
    phi0 = _normalized_random_state(n)
    ts = np.linspace(0.0, 0.4, 5)

    prop_dense = jftools.short_iterative_lanczos.lanczos_timeprop(H_dense, maxsteps=14, target_convg=1e-12)
    prop_sparse = jftools.short_iterative_lanczos.lanczos_timeprop(H_sparse, maxsteps=14, target_convg=1e-12)
    assert prop_dense.backend in ("python", "cython")
    assert prop_sparse.backend in ("python", "cython")

    phis_dense = _run_sil(H_dense, phi0, ts)
    phis_sparse = _run_sil(H_sparse, phi0, ts)

    for t, phi_dense, phi_sparse in zip(ts, phis_dense, phis_sparse):
        phi_ref = expm_multiply(-1j * t * H_dense, phi0)
        assert np.allclose(phi_dense, phi_ref, rtol=5e-9, atol=5e-10)
        assert np.allclose(phi_sparse, phi_ref, rtol=5e-9, atol=5e-10)


def test_short_iterative_lanczos_3x3_hamiltonian_dense_sparse():
    H_dense = np.array(
        [
            [0.3, -0.2, 0.0],
            [-0.2, 0.1, -0.15],
            [0.0, -0.15, -0.4],
        ],
        dtype=complex,
    )
    H_sparse = diags(
        [
            np.array([-0.2, -0.15], dtype=float),
            np.array([0.3, 0.1, -0.4], dtype=float),
            np.array([-0.2, -0.15], dtype=float),
        ],
        offsets=[-1, 0, 1],
        shape=(3, 3),
        format="csr",
    ).astype(complex)
    phi0 = np.array([1.0 + 0.0j, 0.2 - 0.1j, -0.3 + 0.4j], dtype=complex)
    phi0 = phi0 / np.linalg.norm(phi0)
    ts = np.array([0.0, 0.1, 0.25, 0.5], dtype=float)

    dense_out = _run_sil(H_dense, phi0, ts, maxHT=0.05)
    sparse_out = _run_sil(H_sparse, phi0, ts, maxHT=0.05)

    for t, phi_dense, phi_sparse in zip(ts, dense_out, sparse_out):
        phi_ref = expm_multiply(-1j * t * H_dense, phi0)
        assert np.allclose(phi_dense, phi_ref, rtol=5e-10, atol=5e-12)
        assert np.allclose(phi_sparse, phi_ref, rtol=5e-10, atol=5e-12)


@pytest.mark.skipif(qutip is None, reason="qutip not installed")
def test_short_iterative_lanczos_3x3_hamiltonian_qutip_inputs():
    H_sparse = diags(
        [
            np.array([-0.2, -0.15], dtype=float),
            np.array([0.3, 0.1, -0.4], dtype=float),
            np.array([-0.2, -0.15], dtype=float),
        ],
        offsets=[-1, 0, 1],
        shape=(3, 3),
        format="csr",
    ).astype(complex)
    H_q = qutip.Qobj(H_sparse)
    phi0_np = np.array([1.0 + 0.0j, 0.2 - 0.1j, -0.3 + 0.4j], dtype=complex)
    phi0_np = phi0_np / np.linalg.norm(phi0_np)
    phi0_q = qutip.Qobj(phi0_np)
    ts = np.array([0.0, 0.2, 0.4], dtype=float)

    out_np = _run_sil(H_q, phi0_np, ts, maxHT=0.05)
    out_q = _run_sil(H_q, phi0_q, ts, maxHT=0.05)

    assert isinstance(out_np[-1], np.ndarray)
    assert isinstance(out_q[-1], qutip.Qobj)
    assert out_q[-1].dims == phi0_q.dims
    assert np.allclose(out_q[-1].full().ravel(), out_np[-1], rtol=5e-10, atol=5e-12)


def test_short_iterative_lanczos_time_dependent_callable_regression():
    n = 20
    H0 = _make_chain_hamiltonian(n).toarray().astype(complex)
    x = np.linspace(-1.0, 1.0, n)
    V = np.diag(x)
    phi0 = _normalized_random_state(n, seed=8)
    ts = np.linspace(0.0, 0.6, 7)

    def Hfun(t, phi, Hphi):
        Hphi[:] = H0.dot(phi) + np.sin(0.7 * t) * V.dot(phi)
        return Hphi

    phis_coarse = _run_sil(Hfun, phi0, ts, maxHT=0.1)
    phis_fine = _run_sil(Hfun, phi0, ts, maxHT=0.05)

    for phi in phis_coarse:
        assert np.isclose(np.linalg.norm(phi), 1.0, rtol=5e-8, atol=5e-10)
    rel_err = np.linalg.norm(phis_coarse[-1] - phis_fine[-1]) / np.linalg.norm(phis_fine[-1])
    assert rel_err < 1e-2


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("backend", ["python", "cython"])
def test_short_iterative_lanczos_small_dense_diagonal_exact(backend, n):
    if backend == "cython" and not jftools.short_iterative_lanczos.have_cython_backend:
        pytest.skip("cython backend not available")

    diag_vals = np.linspace(-0.7, 1.1, n)
    H = np.diag(diag_vals).astype(complex)
    phi0 = _normalized_random_state(n, seed=20 + n)
    ts = np.linspace(0.0, 0.6, 5)

    out = jftools.short_iterative_lanczos.sesolve_lanczos(
        H,
        phi0,
        ts,
        maxsteps=8,
        target_convg=1e-13,
        maxHT=0.05,
        backend=backend,
    )

    for t, phi in zip(ts, out):
        phi_ref = np.exp(-1j * diag_vals * t) * phi0
        assert np.allclose(phi, phi_ref, rtol=5e-11, atol=5e-13)


@pytest.mark.parametrize("backend", ["python", "cython"])
def test_short_iterative_lanczos_small_callable_diagonal_exact(backend):
    if backend == "cython" and not jftools.short_iterative_lanczos.have_cython_backend:
        pytest.skip("cython backend not available")

    n = 3
    h0_diag = np.array([0.8, -0.6, 0.2], dtype=float)
    hi_diag = np.array([0.4, 0.1, -0.3], dtype=float)
    omega = 2.3
    phi0 = _normalized_random_state(n, seed=31)
    ts = np.linspace(0.0, 0.6, 5)

    H0 = np.diag(h0_diag).astype(complex)
    HI = np.diag(hi_diag).astype(complex)

    def Hfun(t, phi, Hphi):
        Hphi[:] = H0.dot(phi)
        Hphi[:] += np.cos(omega * t) * HI.dot(phi)
        return Hphi

    out = jftools.short_iterative_lanczos.sesolve_lanczos(
        Hfun,
        phi0,
        ts,
        maxsteps=8,
        target_convg=1e-13,
        maxHT=2e-4,
        backend=backend,
    )

    for t, phi in zip(ts, out):
        theta = h0_diag * t + (hi_diag / omega) * np.sin(omega * t)
        phi_ref = np.exp(-1j * theta) * phi0
        assert np.allclose(phi, phi_ref, rtol=2e-4, atol=1e-6)


@pytest.mark.parametrize("backend", ["python", "cython"])
def test_short_iterative_lanczos_full_basis_completion_stops_cleanly(backend):
    if backend == "cython" and not jftools.short_iterative_lanczos.have_cython_backend:
        pytest.skip("cython backend not available")

    n = 3
    H = np.array(
        [
            [0.3, -0.2, 0.05],
            [-0.2, 0.1, -0.15],
            [0.05, -0.15, -0.4],
        ],
        dtype=complex,
    )
    phi0 = _normalized_random_state(n, seed=42)
    ts = np.array([0.0, 0.2, 0.4], dtype=float)

    out = jftools.short_iterative_lanczos.sesolve_lanczos(
        H,
        phi0,
        ts,
        maxsteps=10,
        target_convg=1e-30,
        maxHT=0.4,
        do_full_order=True,
        backend=backend,
    )

    for t, phi in zip(ts, out):
        phi_ref = expm_multiply(-1j * t * H, phi0)
        assert np.allclose(phi, phi_ref, rtol=5e-11, atol=5e-13)


@pytest.mark.skipif(qutip is None, reason="qutip not installed")
def test_short_iterative_lanczos_qutip_h_and_state_compatibility():
    n = 16
    H_sparse = _make_chain_hamiltonian(n)
    H_qobj = qutip.Qobj(H_sparse)
    ts = np.linspace(0.0, 0.3, 4)

    phi_np = _normalized_random_state(n, seed=4)
    prop_qobj = jftools.short_iterative_lanczos.lanczos_timeprop(H_qobj, maxsteps=14, target_convg=1e-12)
    assert prop_qobj.backend in ("python", "cython")

    out_np = _run_sil(H_qobj, phi_np, ts)
    assert isinstance(out_np[-1], np.ndarray)

    phi_qobj = qutip.Qobj(phi_np)
    out_qobj = _run_sil(H_qobj, phi_qobj, ts)
    assert isinstance(out_qobj[-1], qutip.Qobj)
    assert out_qobj[-1].dims == phi_qobj.dims
    assert np.allclose(out_qobj[-1].full().ravel(), out_np[-1], rtol=5e-9, atol=5e-10)


def test_short_iterative_lanczos_unknown_backend_raises():
    H = _make_chain_hamiltonian(8)
    with pytest.raises(ValueError, match="Unknown backend value"):
        jftools.short_iterative_lanczos.lanczos_timeprop(H, maxsteps=8, target_convg=1e-12, backend="made_up_backend")


def test_short_iterative_lanczos_explicit_python_allowed_for_callable():
    def Hfun(t, phi, Hphi):
        Hphi[:] = phi
        return Hphi

    prop = jftools.short_iterative_lanczos.lanczos_timeprop(Hfun, maxsteps=8, target_convg=1e-12, backend="python")
    assert prop.backend == "python"


def test_short_iterative_lanczos_explicit_cython_callable_needs_no_dim():
    if not jftools.short_iterative_lanczos.have_cython_backend:
        pytest.skip("cython backend not available")

    n = 12
    H_dense = _make_chain_hamiltonian(n).toarray().astype(complex)
    phi0 = _normalized_random_state(n, seed=11)
    ts = np.linspace(0.0, 0.3, 4)

    def Hfun(t, phi, Hphi):
        Hphi[:] = H_dense.dot(phi)

    prop = jftools.short_iterative_lanczos.lanczos_timeprop(Hfun, maxsteps=14, target_convg=1e-12, backend="cython")
    assert prop.backend == "cython"

    out_cython = prop.propagate(phi0, ts, maxHT=0.05)
    out_python = jftools.short_iterative_lanczos.sesolve_lanczos(
        Hfun,
        phi0,
        ts,
        maxsteps=14,
        target_convg=1e-12,
        maxHT=0.05,
        backend="python",
    )

    for phi_cython, phi_python in zip(out_cython, out_python):
        assert np.allclose(phi_cython, phi_python, rtol=5e-10, atol=5e-12)


def test_short_iterative_lanczos_explicit_cython_unavailable_no_fallback(monkeypatch):
    sil_mod = jftools.short_iterative_lanczos
    H = _make_chain_hamiltonian(8)
    monkeypatch.setattr(sil_mod, "have_cython_backend", False)
    with pytest.raises(ValueError, match="backend='cython' requested but Cython backend extension is not available"):
        sil_mod.lanczos_timeprop(H, maxsteps=8, target_convg=1e-12, backend="cython")


def test_short_iterative_lanczos_auto_prefers_cython_for_linear_operator():
    if not jftools.short_iterative_lanczos.have_cython_backend:
        pytest.skip("cython backend not available")

    n = 10
    H_dense = _make_chain_hamiltonian(n).toarray().astype(complex)
    phi0 = _normalized_random_state(n, seed=13)
    ts = np.linspace(0.0, 0.25, 4)

    H_linop = LinearOperator(
        shape=(n, n),
        matvec=lambda x: H_dense.dot(x),
        dtype=np.complex128,
    )
    def Hfun(t, phi, Hphi):
        Hphi[:] = H_linop(phi)

    prop_auto = jftools.short_iterative_lanczos.lanczos_timeprop(
        Hfun, maxsteps=12, target_convg=1e-12, backend="auto"
    )
    assert prop_auto.backend == "cython"

    out_auto = prop_auto.propagate(phi0, ts, maxHT=0.05)
    out_python = jftools.short_iterative_lanczos.sesolve_lanczos(
        Hfun,
        phi0,
        ts,
        maxsteps=12,
        target_convg=1e-12,
        maxHT=0.05,
        backend="python",
    )

    for phi_auto, phi_py in zip(out_auto, out_python):
        assert np.allclose(phi_auto, phi_py, rtol=5e-9, atol=5e-11)
