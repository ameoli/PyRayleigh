import math

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest
from scipy import special

from pyrayleigh.constants import SPEED_OF_LIGHT
from pyrayleigh.modes import normalized_frequency
from pyrayleigh.modes.step_index import (LP11_CUTOFF, _cladding_bessel, _core_bessel,
                                         _lp01_characteristic)

CORE_RADIUS = 4.1e-6
CORE_INDEX = 1.444745306273739
CLADDING_INDEX = 1.44

# The same 70-digit mode references used by PyOFDR's Marcuse tests.
MODES = [(1310e-9, 1.6193640806061295, 1.6344193265961299),
          (1550e-9, 1.5089422143265536, 1.226518140343568),
          (1625e-9, 1.4759008044820001, 1.1233815646353687)]


@pytest.fixture
def jax_numpy():
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    try:
        yield jnp
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture(params=["numpy", "strict", "jax"])
def xp(request):
    if request.param == "numpy":
        return numpy_api
    if request.param == "strict":
        return strict
    return request.getfixturevalue("jax_numpy")


def test_normalized_frequency_matches_the_pyofdr_mode_references(xp):
    frequency = xp.asarray([SPEED_OF_LIGHT/wavelength for wavelength, _, _ in MODES])
    v = normalized_frequency(frequency, CORE_RADIUS, CORE_INDEX, CLADDING_INDEX)
    expected = [math.hypot(u, w) for _, u, w in MODES]
    assert v.dtype == xp.float64
    assert type(v) is type(frequency)
    np.testing.assert_allclose(np.asarray(v), expected, rtol=2e-14)


def test_normalized_frequency_broadcasts_radius_and_frequency(xp):
    frequency = xp.asarray([1.8e14, 1.9e14, 2e14])
    radius = xp.asarray([[4e-6], [3e-6]], dtype=xp.float32)
    v = normalized_frequency(frequency, radius, CORE_INDEX, CLADDING_INDEX)
    expected = (2.*np.pi*np.asarray(frequency)*np.asarray(radius, dtype=np.float64)/SPEED_OF_LIGHT
                 *math.sqrt((CORE_INDEX - CLADDING_INDEX)*(CORE_INDEX + CLADDING_INDEX)))
    np.testing.assert_allclose(np.asarray(v), expected, rtol=1e-14)
    assert v.dtype == xp.float64
    assert normalized_frequency(xp.zeros((0,)), radius, CORE_INDEX, CLADDING_INDEX).shape == (2, 0)


def test_native_core_bessel_series_matches_scipy_through_cutoff(xp):
    u = np.linspace(0., LP11_CUTOFF, 101)
    squared = xp.asarray((u*u).tolist())
    j0, uj1 = _core_bessel(squared, xp)
    assert j0.dtype == uj1.dtype == xp.float64
    assert type(j0) is type(squared)
    np.testing.assert_allclose(np.asarray(j0), special.j0(u), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(np.asarray(uj1), u*special.j1(u), rtol=1e-13, atol=1e-15)


def test_native_cladding_series_matches_scipy_at_weak_and_strong_confinement(xp):
    log_w = np.r_[np.linspace(-300., -3., 101), np.linspace(-3., np.log(LP11_CUTOFF), 101)]
    w = np.exp(log_w)
    native = xp.asarray(log_w.tolist())
    k0, wk1 = _cladding_bessel(native, xp)
    assert k0.dtype == wk1.dtype == xp.float64
    assert type(k0) is type(native)
    np.testing.assert_allclose(np.asarray(k0), special.k0(w), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(wk1), w*special.k1(w), rtol=1e-13)


def test_characteristic_broadcasts_and_matches_the_cross_multiplied_equation(xp):
    log_w = xp.asarray([[-4.], [-2.]])
    v = xp.asarray([.8, 1.5, 2.4])
    result = _lp01_characteristic(log_w, v)
    w = np.exp(np.asarray(log_w))
    u = np.sqrt((np.asarray(v) - w)*(np.asarray(v) + w))
    expected = u*special.j1(u)*special.k0(w) - w*special.k1(w)*special.j0(u)
    assert result.dtype == xp.float64
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-13)
    assert _lp01_characteristic(xp.zeros((0,)), 1.).shape == (0,)


@pytest.mark.parametrize("wavelength,u,w", MODES)
def test_pyofdr_reference_modes_satisfy_the_native_characteristic(xp, wavelength, u, w):
    v = normalized_frequency(SPEED_OF_LIGHT/wavelength, CORE_RADIUS, CORE_INDEX, CLADDING_INDEX, xp=xp)
    residual = _lp01_characteristic(math.log(w), v)
    np.testing.assert_allclose(np.asarray(residual), 0., rtol=0., atol=5e-14)


def test_characteristic_brackets_resolved_lp01_modes_without_singular_endpoints(xp):
    v = xp.asarray([.1, .25, .5, 1., 2., 2.4048])
    low = _lp01_characteristic(-300., v)
    high = _lp01_characteristic(xp.log(v), v)
    assert np.all(np.asarray(low) > 0.)
    assert np.all(np.asarray(high) < 0.)
    assert np.all(np.isfinite(np.asarray(low)))
    assert np.all(np.isfinite(np.asarray(high)))


def test_mixed_guide_data_need_an_explicit_namespace(jax_numpy):
    frequency = np.asarray(1.9e14)
    radius = jax_numpy.asarray(CORE_RADIUS)
    with pytest.raises(TypeError):
        normalized_frequency(frequency, radius, CORE_INDEX, CLADDING_INDEX)
    result = normalized_frequency(frequency, radius, CORE_INDEX, CLADDING_INDEX, xp=jax_numpy)
    assert type(result) is type(radius)
    with pytest.raises(TypeError):
        _lp01_characteristic(np.asarray(-1.), jax_numpy.asarray(2.))
    result = _lp01_characteristic(np.asarray(-1.), jax_numpy.asarray(2.), xp=jax_numpy)
    assert type(result) is type(radius)


def test_public_and_characteristic_inputs_are_real():
    with pytest.raises(TypeError):
        normalized_frequency(1.9e14, CORE_RADIUS, CORE_INDEX + 1j, CLADDING_INDEX)
    with pytest.raises(TypeError):
        _lp01_characteristic(-1j, 2.)


def test_jax_normalized_frequency_keeps_all_guide_derivatives(jax_numpy):
    import jax
    def v(parameters):
        f_scale, a_scale, n1, n2 = parameters
        return normalized_frequency(1.9e14*f_scale, CORE_RADIUS*a_scale, n1, n2)
    parameters = jax_numpy.asarray([1., 1., CORE_INDEX, CLADDING_INDEX])
    difference = (CORE_INDEX - CLADDING_INDEX)*(CORE_INDEX + CLADDING_INDEX)
    expected = 2.*math.pi*1.9e14*CORE_RADIUS/SPEED_OF_LIGHT*math.sqrt(difference)
    slope = np.array([1., 1., CORE_INDEX/difference, -CLADDING_INDEX/difference])
    second = np.diag([-1., -1., 1./difference - 2.*CORE_INDEX**2/difference**2,
                      -1./difference - 2.*CLADDING_INDEX**2/difference**2])
    second[2, 3] = second[3, 2] = 2.*CORE_INDEX*CLADDING_INDEX/difference**2
    np.testing.assert_allclose(jax.jit(v)(parameters), expected, rtol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.grad(v))(parameters), expected*slope, rtol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.hessian(v))(parameters),
                               expected*(np.outer(slope, slope) + second), rtol=1e-11, atol=1e-13)


def test_jax_core_series_has_finite_derivatives_at_u_zero(jax_numpy):
    import jax
    def pair(squared):
        return jax_numpy.stack(_core_bessel(squared, jax_numpy))
    squared = jax_numpy.asarray(0.)
    np.testing.assert_array_equal(jax.jit(pair)(squared), [1., 0.])
    np.testing.assert_array_equal(jax.jit(jax.jacfwd(pair))(squared), [-.25, .5])
    np.testing.assert_array_equal(jax.jit(jax.jacfwd(jax.jacfwd(pair)))(squared), [.03125, -.125])


def test_jax_cladding_series_derivatives_match_bessel_identities(jax_numpy):
    import jax
    def pair(log_w):
        return jax_numpy.stack(_cladding_bessel(log_w, jax_numpy))
    z = np.array([-300., -80., -5., -.2, math.log(LP11_CUTOFF)])
    w = np.exp(z)
    k0, wk1 = special.k0(w), w*special.k1(w)
    first = np.column_stack((-wk1, -w*w*k0))
    second = np.column_stack((w*w*k0, w*w*(wk1 - 2.*k0)))
    native = jax_numpy.asarray(z.tolist())
    np.testing.assert_allclose(jax.jit(jax.vmap(jax.jacfwd(pair)))(native),
                               first, rtol=1e-11, atol=1e-300)
    np.testing.assert_allclose(jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(pair))))(native),
                               second, rtol=1e-10, atol=1e-300)


def test_jax_characteristic_gradient_and_hessian_match_scipy_identities(jax_numpy):
    import jax
    def residual(parameters):
        v, log_w = parameters
        return _lp01_characteristic(log_w, v)
    v, w = 2., .8
    s = v*v - w*w
    u = math.sqrt(s)
    a, b, c, d = special.j0(u), u*special.j1(u), special.k0(w), w*special.k1(w)
    da, db = -b/(2.*s), a/2.
    d2a, d2b = -a/(4.*s) + b/(2.*s*s), -b/(4.*s)
    dc, dd = -d, -w*w*c
    d2c, d2d = w*w*c, w*w*(d - 2.*c)
    ds, dz = np.array([2.*v, -2.*w*w]), np.array([0., 1.])
    d2s = np.diag([2., -4.*w*w])
    first = (db*c - d*da)*ds + (b*dc - dd*a)*dz
    second = ((d2b*c - d*d2a)*np.outer(ds, ds) + (db*c - d*da)*d2s
              + (db*dc - dd*da)*(np.outer(ds, dz) + np.outer(dz, ds))
              + (b*d2c - d2d*a)*np.outer(dz, dz))
    parameters = jax_numpy.asarray([v, math.log(w)])
    np.testing.assert_allclose(jax.jit(residual)(parameters), b*c - d*a, rtol=1e-13)
    np.testing.assert_allclose(jax.jit(jax.grad(residual))(parameters), first, rtol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.hessian(residual))(parameters), second, rtol=1e-11)
