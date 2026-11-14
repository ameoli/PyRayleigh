import math

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest
from scipy import special
from scipy.optimize import brentq

from pyrayleigh.constants import SPEED_OF_LIGHT
from pyrayleigh.modes import StepIndexMode, solve_lp01_mode
from pyrayleigh.modes.step_index import LP11_CUTOFF, _lp01_characteristic, _lp01_log_w

CORE_RADIUS = 4.1e-6
CORE_INDEX = 1.444745306273739
CLADDING_INDEX = 1.44
MODES = [(1310e-9, 1.6193640806061295, 1.6344193265961299, 6918202.195238498),
          (1550e-9, 1.5089422143265536, 1.226518140343568, 5844942.291210868),
          (1625e-9, 1.4759008044820001, 1.1233815646353687, 5574606.425580748)]


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


def reference_root(v):
    def residual(log_w):
        w = math.exp(log_w)
        u = math.sqrt(max(0., (v - w)*(v + w)))
        return u*special.j1(u)*special.k0(w) - w*special.k1(w)*special.j0(u)
    return brentq(residual, -300., math.log(v), xtol=5e-14, rtol=1e-15)


def reference_root_derivatives(v):
    log_w = reference_root(v)
    w = math.exp(log_w)
    s = (v - w)*(v + w)
    u = math.sqrt(s)
    a, b, c, d = special.j0(u), u*special.j1(u), special.k0(w), w*special.k1(w)
    da, db = -b/(2.*s), a/2.
    d2a, d2b = -a/(4.*s) + b/(2.*s*s), -b/(4.*s)
    dc, dd = -d, -w*w*c
    d2c, d2d = w*w*c, w*w*(d - 2.*c)
    ds, dz = np.array([2.*v, -2.*w*w]), np.array([0., 1.])
    first = (db*c - d*da)*ds + (b*dc - dd*a)*dz
    second = ((d2b*c - d*d2a)*np.outer(ds, ds) + (db*c - d*da)*np.diag([2., -4.*w*w])
              + (db*dc - dd*da)*(np.outer(ds, dz) + np.outer(dz, ds))
              + (b*d2c - d2d*a)*np.outer(dz, dz))
    slope = -first[0]/first[1]
    curvature = -(second[0, 0] + 2.*second[0, 1]*slope + second[1, 1]*slope*slope)/first[1]
    return log_w, slope, curvature


@pytest.mark.parametrize("wavelength,u,w,beta", MODES)
def test_solver_matches_pyofdr_high_precision_mode_references(xp, wavelength, u, w, beta):
    frequency = xp.asarray(SPEED_OF_LIGHT/wavelength)
    mode = solve_lp01_mode(frequency, CORE_RADIUS, CORE_INDEX, CLADDING_INDEX)
    assert isinstance(mode, StepIndexMode)
    assert (mode.name, mode.azimuthal_order, mode.radial_order) == ("LP01", 0, 1)
    assert type(mode.U) is type(frequency)
    assert mode.U.dtype == mode.W.dtype == mode.effective_index.dtype == xp.float64
    np.testing.assert_allclose(np.asarray(mode.U), u, rtol=2e-12)
    np.testing.assert_allclose(np.asarray(mode.W), w, rtol=2e-12)
    np.testing.assert_allclose(np.asarray(mode.propagation_constant), beta, rtol=2e-12)
    np.testing.assert_allclose(np.asarray(mode.b), w*w/(u*u + w*w), rtol=2e-12)


def test_solver_brackets_weak_confinement_and_near_cutoff_modes(xp):
    v = xp.asarray([.082, .1, .25, .5, 1., 1.9, 2.4048])
    roots = _lp01_log_w(v, xp)
    expected = [reference_root(value) for value in np.asarray(v)]
    np.testing.assert_allclose(np.asarray(roots), expected, rtol=0., atol=2e-12)
    np.testing.assert_allclose(np.asarray(_lp01_characteristic(roots, v)), 0., atol=2e-14)


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_solver_keeps_native_shapes_and_mode_identities(xp, shape):
    frequency = xp.full(shape, SPEED_OF_LIGHT/1550e-9, dtype=xp.float64)
    mode = solve_lp01_mode(frequency, CORE_RADIUS, CORE_INDEX, CLADDING_INDEX)
    for name in ("frequency", "core_radius", "core_index", "cladding_index", "V", "U", "W", "b",
                  "effective_index"):
        value = getattr(mode, name)
        assert value.shape == shape
        assert value.dtype == xp.float64
        assert type(value) is type(frequency)
    np.testing.assert_allclose(np.asarray(mode.U*mode.U + mode.W*mode.W), np.asarray(mode.V*mode.V),
                               rtol=1e-14)
    assert np.all(np.asarray(mode.W) > 0.)
    assert np.all(np.asarray(mode.W) < np.asarray(mode.V))
    assert np.all(np.asarray(mode.effective_index) >= CLADDING_INDEX)
    assert np.all(np.asarray(mode.effective_index) < CORE_INDEX)


def test_solver_broadcasts_guide_parameters_and_optical_frequencies(xp):
    frequency = xp.asarray([SPEED_OF_LIGHT/row[0] for row in MODES])
    radius = xp.asarray([[CORE_RADIUS], [.8*CORE_RADIUS]])
    mode = solve_lp01_mode(frequency, radius, CORE_INDEX, CLADDING_INDEX)
    assert mode.W.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(mode.frequency), np.broadcast_to(np.asarray(frequency), (2, 3)))
    np.testing.assert_allclose(np.asarray(mode.core_radius), np.broadcast_to(np.asarray(radius), (2, 3)))
    assert np.all(np.isfinite(np.asarray(mode.W)))


def test_solver_owns_its_input_arrays():
    frequency, radius = np.array([1.9e14]), np.array(CORE_RADIUS)
    mode = solve_lp01_mode(frequency, radius, CORE_INDEX, CLADDING_INDEX)
    frequency[:] = 1.
    radius[...] = 1.
    np.testing.assert_array_equal(mode.frequency, [1.9e14])
    np.testing.assert_array_equal(mode.core_radius, [CORE_RADIUS])


def test_out_of_range_roots_remain_native_nan(xp):
    values = xp.asarray([0., -.1, .01, LP11_CUTOFF, 3., float("nan"), float("inf")])
    result = _lp01_log_w(values, xp)
    assert np.all(np.isnan(np.asarray(result)))


@pytest.mark.parametrize("bad", [
    {"frequency": 0.}, {"frequency": float("inf")}, {"frequency": 1e10},
    {"frequency": 1e15}, {"core_radius": 0.}, {"core_index": CLADDING_INDEX},
    {"core_index": 1.43}, {"core_index": 1.6}, {"cladding_index": .9},
])
def test_unsupported_guides_have_nan_derived_fields(bad):
    inputs = dict(frequency=SPEED_OF_LIGHT/1550e-9, core_radius=CORE_RADIUS,
                   core_index=CORE_INDEX, cladding_index=CLADDING_INDEX)
    mode = solve_lp01_mode(**{**inputs, **bad})
    for name in ("V", "U", "W", "b", "effective_index", "propagation_constant"):
        assert np.isnan(getattr(mode, name))


def test_mixed_supported_and_unsupported_queries_keep_valid_modes(xp):
    frequency = xp.asarray([SPEED_OF_LIGHT/1550e-9, 1e15, 1e10])
    mode = solve_lp01_mode(frequency, CORE_RADIUS, CORE_INDEX, CLADDING_INDEX)
    np.testing.assert_allclose(np.asarray(mode.W)[0], MODES[1][2], rtol=2e-12)
    assert np.all(np.isnan(np.asarray(mode.W)[1:]))


def test_mixed_namespaces_need_explicit_selection(jax_numpy):
    frequency, radius = np.array(1.9e14), jax_numpy.asarray(CORE_RADIUS)
    with pytest.raises(TypeError):
        solve_lp01_mode(frequency, radius, CORE_INDEX, CLADDING_INDEX)
    mode = solve_lp01_mode(frequency, radius, CORE_INDEX, CLADDING_INDEX, xp=jax_numpy)
    assert type(mode.W) is type(radius)


def test_solver_requires_real_parameters():
    with pytest.raises(TypeError):
        solve_lp01_mode(1.9e14, CORE_RADIUS, CORE_INDEX + 1j, CLADDING_INDEX)


def test_jax_root_derivatives_match_implicit_scipy_derivatives(jax_numpy):
    import jax
    def root(v):
        return _lp01_log_w(v, jax_numpy)
    points = [.1, .5, 1.9, 2.4048]
    expected = np.array([reference_root_derivatives(v) for v in points])
    native = jax_numpy.asarray(points)
    for column, function in enumerate((root, jax.grad(root), jax.hessian(root))):
        actual = jax.jit(jax.vmap(function))(native)
        np.testing.assert_allclose(np.asarray(actual), expected[:, column], rtol=2e-10, atol=1e-12)


def test_jax_propagation_constant_keeps_all_guide_gradients_and_hessians(jax_numpy):
    import jax
    frequency = SPEED_OF_LIGHT/1550e-9
    wave_number = 2.*math.pi*frequency/SPEED_OF_LIGHT
    v0 = wave_number*CORE_RADIUS*math.sqrt((CORE_INDEX - CLADDING_INDEX)*(CORE_INDEX + CLADDING_INDEX))
    z0, dz, d2z = reference_root_derivatives(v0)
    def beta(parameters):
        f_scale, a_scale, n1, n2 = parameters
        mode = solve_lp01_mode(frequency*f_scale, CORE_RADIUS*a_scale, n1, n2)
        return mode.propagation_constant/wave_number
    def expected(parameters):
        f_scale, a_scale, n1, n2 = parameters
        v = wave_number*f_scale*CORE_RADIUS*a_scale*jax_numpy.sqrt((n1 - n2)*(n1 + n2))
        offset = v - v0
        log_w = z0 + dz*offset + .5*d2z*offset*offset
        w = jax_numpy.exp(log_w)
        index = jax_numpy.sqrt(n2*n2 + (w/v)**2*(n1 - n2)*(n1 + n2))
        return f_scale*index
    parameters = jax_numpy.asarray([1., 1., CORE_INDEX, CLADDING_INDEX])
    for transform in (lambda f: f, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(beta))(parameters), transform(expected)(parameters),
                                   rtol=2e-10, atol=1e-12)
