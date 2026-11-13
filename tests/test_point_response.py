import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Fiber, GuideOptics, PointReflector, RayleighRealization,
                        Section, lossless_reflector, optical_path, point_response)
from pyrayleigh.constants import SPEED_OF_LIGHT

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


def make_fiber(elements=(), attenuation=.2, length=3., rayleigh=None):
    return Fiber("spool-1", length, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                  [Section(0., length, GUIDE, attenuation, 1e-8)], elements, rayleigh)


def test_single_point_uses_local_reflection_and_distributed_round_trip(xp):
    point = lossless_reflector(np.float64(1.25), .04, .3, xp=xp)
    fiber = make_fiber([point])
    frequency = xp.asarray([[REFERENCE - 1e9, REFERENCE, REFERENCE + 1e9]])
    actual = point_response(fiber, frequency)
    phase = optical_path(fiber).phase_factor(0., point.position, frequency, round_trip=True)
    expected = .2*np.exp(.3j - .25)*np.asarray(phase)
    assert actual.shape == (1, 3)
    assert actual.dtype == xp.complex128
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-14, atol=0.)


def test_preceding_transmissions_keep_both_complex_port_coefficients(xp):
    points = []
    for position, values in ((1., (.2, 0., .5j, .4j)),
                             (2., (.3, 0., .8, .7)), (3., (.1, 0., .1, .1))):
        coefficients = [xp.asarray(value, dtype=xp.complex128) for value in values]
        points.append(PointReflector(position, *coefficients))
    fiber = make_fiber([points[2], points[0], points[1]], attenuation=.07)
    frequency = xp.asarray([REFERENCE - 1e9, REFERENCE, REFERENCE + 1e9])
    path = optical_path(fiber)
    expected = (.2*np.asarray(path.round_trip(0., 1., frequency))
                - .06*np.asarray(path.round_trip(0., 2., frequency))
                - .0112*np.asarray(path.round_trip(0., 3., frequency)))
    actual = point_response(fiber, frequency)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-13, atol=0.)


def test_opaque_first_point_reflects_and_blocks_later_returns(xp):
    first = lossless_reflector(0., 1., xp=xp)
    later = lossless_reflector(3., .25, xp=xp)
    frequency = xp.full((2, 3), REFERENCE, dtype=xp.float64)
    response = point_response(make_fiber([later, first]), frequency)
    np.testing.assert_array_equal(np.asarray(response), np.ones((2, 3), dtype=complex))


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_no_points_give_zero_even_when_rayleigh_is_stored(xp, shape):
    sample = RayleighRealization([.01j], [1.], [0., 3.])
    frequency = xp.full(shape, REFERENCE, dtype=xp.float64)
    response = point_response(make_fiber(rayleigh=sample), frequency)
    assert response.shape == shape
    assert response.dtype == xp.complex128
    np.testing.assert_array_equal(np.asarray(response), np.zeros(shape, dtype=complex))


def test_fixed_numpy_point_and_jax_frequency_need_explicit_selection(jax_numpy):
    fiber = make_fiber([lossless_reflector(1., .04)])
    frequency = jax_numpy.asarray(REFERENCE, dtype=jax_numpy.float64)
    with pytest.raises(TypeError):
        point_response(fiber, frequency)
    actual = point_response(fiber, frequency, xp=jax_numpy)
    assert actual.dtype == jax_numpy.complex128
    np.testing.assert_allclose(np.abs(np.asarray(actual)), .2*np.exp(-.2), rtol=1e-14, atol=0.)


def test_frequency_is_real_even_for_an_empty_element_list():
    with pytest.raises(TypeError):
        point_response(make_fiber(), [REFERENCE + 1j])


def test_jax_reflectance_and_loss_derivatives_include_interference(jax_numpy):
    import jax
    length = 1_000_000.25*SPEED_OF_LIGHT/(GUIDE.effective_index*REFERENCE)
    def power(parameters):
        reflectance, optical_depth = parameters
        first = lossless_reflector(0., reflectance)
        last = lossless_reflector(length, .09, xp=jax_numpy)
        fiber = make_fiber([first, last], attenuation=optical_depth/length, length=length)
        field = point_response(fiber, REFERENCE)
        return jax_numpy.real(jax_numpy.conj(field)*field)
    parameters = jax_numpy.asarray([.04, .2], dtype=jax_numpy.float64)
    value = jax.jit(power)(parameters)
    gradient = jax.jit(jax.grad(power))(parameters)
    hessian = jax.jit(jax.hessian(power))(parameters)
    r, depth = np.asarray(parameters)
    b = .3*np.exp(-depth)
    field = np.sqrt(r) - (1. - r)*b
    field_gradient = np.array([1./(2.*np.sqrt(r)) + b, (1. - r)*b])
    field_hessian = np.array([[-1./(4.*r**1.5), -b], [-b, -(1. - r)*b]])
    expected_hessian = 2.*np.outer(field_gradient, field_gradient) + 2.*field*field_hessian
    np.testing.assert_allclose(value, field**2, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(gradient, 2.*field*field_gradient, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-12, atol=1e-14)
