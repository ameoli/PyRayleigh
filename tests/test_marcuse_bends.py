import math

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest
from scipy import special

from pyrayleigh import (BendLossTable, Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, Probe, Section, SpatialProfile, compute_response)
from pyrayleigh.constants import SPEED_OF_LIGHT
from pyrayleigh.modes import marcuse_bend_attenuation, marcuse_bend_loss_db

A, N1, N2 = 4.1e-6, 1.444745306273739, 1.44
WAVELENGTH = 1550e-9
FREQUENCY = SPEED_OF_LIGHT/WAVELENGTH
U, W, BETA = 1.5089422143265536, 1.226518140343568, 5844942.291210868

# Numerical references at 70 digits, also used by PyOFDR.
LOSSES = [(1310e-9, .005, 1., 1., 46.19300056287784),
           (1310e-9, .010, 1., 1.28, .05920915994712783),
           (1310e-9, .020, 2.5, 1.28, .0000026041057348797894),
           (1550e-9, .005, 1., 1., 157.99011139659696),
           (1550e-9, .010, 1., 1.28, 3.356204768598211),
           (1550e-9, .020, 2.5, 1.28, .014797705525512475),
           (1625e-9, .005, 1., 1., 205.23729191688077),
           (1625e-9, .010, 1., 1.28, 8.210017522028247),
           (1625e-9, .020, 2.5, 1.28, .10227218252576663)]


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


def loss(radius=.01, turns=1., wavelength=WAVELENGTH, correction=1.28, xp=None):
    return marcuse_bend_loss_db(radius, turns, wavelength, A, N1, N2, correction, xp=xp)


def test_total_loss_and_power_rate_match_pyofdr_high_precision_references(xp):
    reference = np.array(LOSSES)
    wavelength, radius, turns, correction = [xp.asarray(reference[:, index].tolist())
                                            for index in range(4)]
    total = marcuse_bend_loss_db(radius, turns, wavelength, A, N1, N2, correction)
    rate = marcuse_bend_attenuation(SPEED_OF_LIGHT/wavelength, A, N1, N2, radius, correction)
    expected_rate = reference[:, 4]*math.log(10.)/(20.*math.pi*reference[:, 1]*reference[:, 2])
    assert total.dtype == rate.dtype == xp.float64
    assert type(total) is type(radius)
    np.testing.assert_allclose(np.asarray(total), reference[:, 4], rtol=2e-10)
    np.testing.assert_allclose(np.asarray(rate), expected_rate, rtol=2e-10)


def test_radius_correction_changes_the_rate_without_stretching_the_arc(xp):
    corrected = loss(radius=.01, correction=1.28, xp=xp)
    uncorrected = loss(radius=.0128, correction=1., xp=xp)
    np.testing.assert_allclose(np.asarray(corrected), np.asarray(uncorrected)/1.28, rtol=1e-12)
    rate = marcuse_bend_attenuation(FREQUENCY, A, N1, N2, .01, 1.28, xp=xp)
    same_rate = marcuse_bend_attenuation(FREQUENCY, A, N1, N2, .0128, 1., xp=xp)
    np.testing.assert_allclose(np.asarray(rate), np.asarray(same_rate), rtol=1e-13)


def test_loss_broadcasts_wavelength_radius_and_fractional_turns(xp):
    radius = xp.asarray([[.005], [.01]], dtype=xp.float32)
    wavelength = xp.asarray([1310e-9, 1550e-9, 1625e-9])
    total = loss(radius, .75, wavelength)
    twice = loss(radius, 1.5, wavelength)
    assert total.shape == (2, 3)
    assert total.dtype == xp.float64
    assert type(total) is type(radius)
    np.testing.assert_allclose(np.asarray(twice), 2.*np.asarray(total), rtol=1e-13)
    assert np.all(np.diff(np.asarray(total), axis=0) < 0.)
    assert np.all(np.diff(np.asarray(total), axis=1) > 0.)
    empty = loss(.01, 1., xp.zeros((0,)))
    assert empty.shape == (0,)


@pytest.mark.parametrize("scale", [.001, 1000.])
def test_uniform_length_scaling_preserves_total_db(xp, scale):
    base = loss(xp=xp)
    scaled = marcuse_bend_loss_db(.01*scale, 1., WAVELENGTH*scale, A*scale, N1, N2, xp=xp)
    np.testing.assert_allclose(np.asarray(scaled), np.asarray(base), rtol=2e-11)


def test_weak_confinement_is_finite_without_forming_k1_directly(xp):
    wavelength = 2.*math.pi*A*math.sqrt((N1 - N2)*(N1 + N2))/.1
    rate = marcuse_bend_attenuation(SPEED_OF_LIGHT/wavelength, A, N1, N2, .01, xp=xp)
    assert np.isfinite(np.asarray(rate)) and 0. < float(rate) < 1e-30


def test_log_total_loss_survives_underflow_of_the_per_metre_rate(xp):
    radius, turns, correction = 1.2, 1e100, 1.28
    rate = marcuse_bend_attenuation(FREQUENCY, A, N1, N2, radius, correction, xp=xp)
    np.testing.assert_array_equal(np.asarray(rate), 0.)
    total = loss(radius, turns, correction=correction, xp=xp)
    gamma, kappa = W/A, U/A
    log_rate = (.5*math.log(math.pi) + 2.*math.log(kappa) - math.log(2.)
                 - 1.5*math.log(gamma) - 2.*math.log(math.hypot(U, W))
                 - .5*math.log(radius*correction) - 2.*math.log(special.k1(W))
                 - (2./3.)*(gamma/BETA)**2*gamma*radius*correction)
    expected = math.exp(log_rate + math.log(10./math.log(10.))
                          + math.log(2.*math.pi*radius) + math.log(turns))
    assert float(total) > 0.
    np.testing.assert_allclose(np.asarray(total), expected, rtol=3e-11, atol=0.)
    np.testing.assert_array_equal(np.asarray(loss(radius=100., xp=xp)), 0.)


def test_marcuse_rates_enter_the_existing_optical_response(xp):
    radius, turns = .01, .75
    length = 2.*math.pi*radius*turns
    rates = marcuse_bend_attenuation(FREQUENCY, A, N1, N2, xp.asarray([.02, .01, .005]))
    table = BendLossTable([50., 100., 200.], [FREQUENCY], rates[:, None])
    guide = GuideOptics(FREQUENCY, 1.4682, 1.4682, bend_loss=table)
    fiber = Fiber("spool", length, 293.15, FREQUENCY - 1e9, FREQUENCY + 1e9,
                  [Section(0., length, guide, 0., 0.)], [PointReflector(length, .1, 0., .8, .7)])
    state = FiberState(curvature_x=Field([FieldTerm(SpatialProfile(
        [0., length], xp.asarray([1./radius, 1./radius])))]))
    result = compute_response(fiber, Probe(xp.asarray(FREQUENCY)), state)
    total = .75*LOSSES[4][-1]
    np.testing.assert_allclose(np.abs(np.asarray(result.transmission))**2,
                               .64*10.**(-total/10.), rtol=2e-10)
    np.testing.assert_allclose(np.abs(np.asarray(result.reflection)),
                               .1*10.**(-total/10.), rtol=2e-10)


@pytest.mark.parametrize("bad", [
    {"radius": 0.}, {"radius": float("inf")}, {"radius": A},
    {"turns": 0.}, {"turns": -1.}, {"turns": float("nan")},
    {"wavelength": 0.}, {"wavelength": 1e-3}, {"wavelength": 1000e-9},
    {"core_radius": 0.}, {"core_index": 1.43}, {"core_index": 1.6},
    {"radius_correction": 0.}, {"radius_correction": 1e-5},
])
def test_unsupported_inputs_return_nan(bad):
    inputs = dict(radius=.01, turns=1., wavelength=WAVELENGTH, core_radius=A,
                   core_index=N1, cladding_index=N2)
    assert np.isnan(marcuse_bend_loss_db(**{**inputs, **bad}))


def test_invalid_bends_do_not_hide_valid_bends_in_the_same_query(xp):
    radius = xp.asarray([.01, 0., .02])
    result = loss(radius)
    np.testing.assert_allclose(np.asarray(result)[[0, 2]], [LOSSES[4][-1], LOSSES[5][-1]/2.5], rtol=2e-10)
    assert np.isnan(np.asarray(result)[1])


def test_overflowing_total_loss_is_native_nan(xp):
    with np.errstate(over="ignore"):
        total = loss(turns=1e308, xp=xp)
    assert np.isnan(np.asarray(total))


def test_mixed_namespaces_require_explicit_selection(jax_numpy):
    radius = jax_numpy.asarray(.01)
    wavelength = np.asarray(WAVELENGTH)
    with pytest.raises(TypeError):
        loss(radius, wavelength=wavelength)
    result = loss(radius, wavelength=wavelength, xp=jax_numpy)
    assert type(result) is type(radius)
    with pytest.raises(TypeError):
        marcuse_bend_attenuation(np.asarray(FREQUENCY), A, N1, N2, radius)
    rate = marcuse_bend_attenuation(np.asarray(FREQUENCY), A, N1, N2, radius, xp=jax_numpy)
    assert type(rate) is type(radius)


def test_marcuse_parameters_are_real():
    with pytest.raises(TypeError):
        loss(radius=.01j)
    with pytest.raises(TypeError):
        marcuse_bend_attenuation(FREQUENCY, A, N1 + 1j, N2, .01)


def test_jax_radius_turn_and_correction_gradients_and_hessian(jax_numpy):
    import jax
    def total(parameters):
        radius, turns, correction = parameters
        return loss(radius, turns, correction=correction)
    radius, turns, correction = .01, 1., 1.28
    h = (2./3.)*(W/A)**3/BETA**2
    expected = LOSSES[4][-1]
    slope = np.array([.5/radius - h*correction, 1./turns, -.5/correction - h*radius])
    second = np.diag([-.5/radius**2, -1./turns**2, .5/correction**2])
    second[0, 2] = second[2, 0] = -h
    parameters = jax_numpy.asarray([radius, turns, correction])
    np.testing.assert_allclose(jax.jit(total)(parameters), expected, rtol=2e-10)
    np.testing.assert_allclose(jax.jit(jax.grad(total))(parameters), expected*slope, rtol=2e-10)
    np.testing.assert_allclose(jax.jit(jax.hessian(total))(parameters),
                               expected*(np.outer(slope, slope) + second), rtol=3e-10, atol=1e-10)


def test_jax_spectral_and_guide_derivatives_match_implicit_mode_and_bessel_identities(jax_numpy):
    import jax
    v0 = 2.*math.pi*FREQUENCY*A/SPEED_OF_LIGHT*math.sqrt((N1 - N2)*(N1 + N2))
    s, z0 = (v0 - W)*(v0 + W), math.log(W)
    u = math.sqrt(s)
    a, b, c, d = special.j0(u), u*special.j1(u), special.k0(W), W*special.k1(W)
    da, db = -b/(2.*s), a/2.
    d2a, d2b = -a/(4.*s) + b/(2.*s*s), -b/(4.*s)
    dc, dd = -d, -W*W*c
    d2c, d2d = W*W*c, W*W*(d - 2.*c)
    ds, dz_axis = np.array([2.*v0, -2.*W*W]), np.array([0., 1.])
    first = (db*c - d*da)*ds + (b*dc - dd*a)*dz_axis
    second = ((d2b*c - d*d2a)*np.outer(ds, ds) + (db*c - d*da)*np.diag([2., -4.*W*W])
              + (db*dc - dd*da)*(np.outer(ds, dz_axis) + np.outer(dz_axis, ds))
              + (b*d2c - d2d*a)*np.outer(dz_axis, dz_axis))
    dz = -first[0]/first[1]
    d2z = -(second[0, 0] + 2.*second[0, 1]*dz + second[1, 1]*dz*dz)/first[1]
    def total(parameters):
        f_scale, a_scale, n1, n2 = parameters
        return marcuse_bend_loss_db(.01, 1., WAVELENGTH/f_scale, A*a_scale, n1, n2)
    def expected(parameters):
        f_scale, a_scale, n1, n2 = parameters
        core = A*a_scale
        wave_number = 2.*math.pi*FREQUENCY*f_scale/SPEED_OF_LIGHT
        v = wave_number*core*jax_numpy.sqrt((n1 - n2)*(n1 + n2))
        offset = v - v0
        z = z0 + dz*offset + .5*d2z*offset*offset
        w = jax_numpy.exp(z)
        transverse = jax_numpy.sqrt((v - w)*(v + w))
        q = d + dd*(z - z0) + .5*d2d*(z - z0)**2
        beta = jax_numpy.sqrt((wave_number*n2)**2 + (w/core)**2)
        gamma = w/core
        rate = (math.sqrt(math.pi)*(transverse/core)**2
                *jax_numpy.exp(-(2./3.)*(gamma/beta)**2*gamma*.0128)
                /(2.*gamma**1.5*v*v*math.sqrt(.0128)*(q/w)**2))
        return rate*(2.*math.pi*.01)*10./math.log(10.)
    parameters = jax_numpy.asarray([1., 1., N1, N2])
    for transform in (lambda f: f, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(total))(parameters), transform(expected)(parameters),
                                   rtol=2e-8, atol=1e-9)


def test_jax_underflowed_loss_has_finite_zero_radius_derivatives(jax_numpy):
    import jax
    def total(radius):
        return loss(radius)
    radius = jax_numpy.asarray(100.)
    np.testing.assert_array_equal(jax.jit(total)(radius), 0.)
    np.testing.assert_array_equal(jax.jit(jax.grad(total))(radius), 0.)
    np.testing.assert_array_equal(jax.jit(jax.hessian(total))(radius), 0.)
