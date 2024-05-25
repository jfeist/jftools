import time
import numpy as np
import jftools


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


def test_short_iterative_lanczos():
    # Test short_iterative_lanczos function
    # Add your test for the short_iterative_lanczos function here
    pass
