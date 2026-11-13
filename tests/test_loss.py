from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import Fiber, GuideOptics, RayleighRealization, Section, optical_path, units
from pyrayleigh.propagation.loss import FiberLoss

REFERENCE = 193414489032258.06
GUIDE = GuideOptics(REFERENCE, 1.4682, 1.4682)


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


def make_fiber(rates=(.2, .6)):
    parts = [Section(0., 1., GUIDE, rates[0], 1e-8),
             Section(1., 3., GUIDE, rates[1], 1e-8)]
    return Fiber("spool-1", 3., 293.15, REFERENCE - 50e9, REFERENCE + 50e9, parts)


def test_piecewise_loss_integrates_exact_overlaps_and_clips_positions(xp):
    rates = [xp.asarray(value, dtype=xp.float64) for value in (.2, .6)]
    path = optical_path(make_fiber(rates))
    position = xp.asarray([-1., 0., .5, 1., 1.5, 3., 4.], dtype=xp.float64)
    frequency = xp.asarray([REFERENCE - 1e9, REFERENCE, REFERENCE + 1e9])
    actual = path.attenuation_integral(position, frequency)
    assert actual.dtype == xp.float64
    expected = np.repeat(np.array([0., 0., .1, .2, .5, 1.4, 1.4])[:, None], 3, axis=1)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-14, atol=0.)


@pytest.mark.parametrize("position_shape,frequency_shape", [
    ((), (2, 3)), ((2, 1), (3,)), ((0,), (2,)), ((2,), (0,)),
])
def test_loss_shape_is_position_shape_plus_frequency_shape(xp, position_shape, frequency_shape):
    position = xp.full(position_shape, .5, dtype=xp.float64)
    frequency = xp.full(frequency_shape, REFERENCE, dtype=xp.float64)
    actual = FiberLoss(make_fiber()).attenuation_integral(position, frequency)
    assert actual.shape == position_shape + frequency_shape
    assert actual.dtype == xp.float64
    np.testing.assert_allclose(np.asarray(actual), np.full(actual.shape, .1), rtol=1e-14, atol=0.)


def test_one_way_and_round_trip_field_and_power_factors(xp):
    path = optical_path(make_fiber())
    frequency = xp.asarray([[REFERENCE - 1e9, REFERENCE + 1e9]], dtype=xp.float64)
    start, end = np.float64(.5), np.float64(2.)
    one_way = path.transmission(start, end, frequency)
    round_trip = path.round_trip(start, end, frequency)
    assert one_way.dtype == round_trip.dtype == xp.complex128
    phase = path.phase_factor(start, end, frequency)
    np.testing.assert_allclose(np.asarray(one_way), np.exp(-.35)*np.asarray(phase),
                               rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.abs(np.asarray(one_way))**2, np.full((1, 2), np.exp(-.7)),
                               rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.abs(np.asarray(round_trip))**2, np.full((1, 2), np.exp(-1.4)),
                               rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.asarray(round_trip), np.asarray(one_way)**2, rtol=1e-14, atol=0.)


def test_uniform_db_per_km_loss_matches_pyofdr_round_trip_convention(xp):
    alpha = units.attenuation_from_db_per_km(.2, xp=xp)
    fiber = Fiber("kilometre", 1000., 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                  [Section(0., 1000., GUIDE, alpha, 1e-8)])
    path = optical_path(fiber)
    one_way = path.transmission(0., 1000., REFERENCE)
    round_trip = path.round_trip(0., 1000., REFERENCE)
    np.testing.assert_allclose(np.abs(np.asarray(one_way)), .9772372209558107, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.abs(np.asarray(round_trip)), .954992586021436, rtol=1e-14, atol=0.)


def test_zero_length_transmission_is_unity_and_reversed_paths_are_rejected(xp):
    path = optical_path(make_fiber())
    frequency = xp.asarray([REFERENCE, REFERENCE + 1e9])
    for method in (path.transmission, path.round_trip):
        np.testing.assert_array_equal(np.asarray(method(1., 1., frequency)), [1. + 0j, 1. + 0j])
        with pytest.raises(ValueError):
            method(2., 1., frequency)


@pytest.mark.parametrize("rates", [(0., 0.), (.2, .6)])
def test_stored_rayleigh_keeps_full_distributed_attenuation(xp, rates):
    fiber = make_fiber(rates)
    sample = RayleighRealization([.01j, .02j], [.5, 2.], [0., 1., 3.])
    reference = optical_path(fiber)
    attached = optical_path(replace(fiber, rayleigh=sample))
    frequency = xp.asarray(REFERENCE, dtype=xp.float64)
    before = reference.attenuation_integral([0., 1., 3.], frequency)
    after = attached.attenuation_integral([0., 1., 3.], frequency)
    np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    np.testing.assert_array_equal(np.asarray(attached.transmission(0., 3., frequency)),
                                  np.asarray(reference.transmission(0., 3., frequency)))


def test_numpy_rates_and_jax_queries_need_explicit_selection(jax_numpy):
    path = optical_path(make_fiber((np.array(.2), np.array(.6))))
    frequency = jax_numpy.asarray(REFERENCE, dtype=jax_numpy.float64)
    assert path.phase_factor(0., 2., frequency).dtype == jax_numpy.complex128
    for method, args in ((path.attenuation_integral, (2., frequency)),
                          (path.transmission, (.5, 2., frequency)),
                          (path.round_trip, (.5, 2., frequency))):
        with pytest.raises(TypeError):
            method(*args)
        actual = method(*args, xp=jax_numpy)
        assert type(actual) is type(frequency)


def test_loss_requires_a_fiber():
    with pytest.raises(TypeError, match="Fiber"):
        FiberLoss(None)


def test_jax_cumulative_loss_position_derivative_is_the_local_rate(jax_numpy):
    import jax
    loss = FiberLoss(make_fiber())
    def cumulative(position):
        return loss.attenuation_integral(position, REFERENCE)
    position = jax_numpy.asarray(1.5, dtype=jax_numpy.float64)
    np.testing.assert_allclose(jax.jit(jax.grad(cumulative))(position), .6, rtol=1e-14, atol=0.)
    np.testing.assert_array_equal(jax.jit(jax.grad(jax.grad(cumulative)))(position), 0.)


@pytest.mark.parametrize("method_name,multiple", [("transmission", 1.), ("round_trip", 2.)])
def test_jax_power_derivatives_in_section_attenuation(jax_numpy, method_name, multiple):
    import jax
    def power(rates):
        path = optical_path(make_fiber(rates))
        field = getattr(path, method_name)(.5, 2., REFERENCE)
        return jax_numpy.real(jax_numpy.conj(field)*field)
    rates = jax_numpy.asarray([.2, .6], dtype=jax_numpy.float64)
    value = jax.jit(power)(rates)
    gradient = jax.jit(jax.grad(power))(rates)
    hessian = jax.jit(jax.hessian(power))(rates)
    expected = np.exp(-multiple*.7)
    weights = multiple*np.array([.5, 1.])
    np.testing.assert_allclose(value, expected, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(gradient, -weights*expected, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(hessian, np.outer(weights, weights)*expected, rtol=1e-13, atol=0.)
