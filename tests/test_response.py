import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, Probe, RayleighRealization, Response,
                        Section, SpatialProfile, TimeHistory, compute_response,
                        optical_path)
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


def make_fiber(sample=None, elements=(), attenuation=0.):
    return Fiber("spool-1", 2., 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                  [Section(0., 2., GUIDE, attenuation, 1e-8)], elements, sample)


def test_coherent_reflection_and_forward_point_transmissions(xp):
    points = []
    for position, values in ((0., (.1, 0., .5j, .4j)),
                             (2., (0., 0., .8, .6))):
        points.append(PointReflector(position, *[
            xp.asarray(value, dtype=xp.complex128) for value in values]))
    sample = RayleighRealization(xp.asarray([.01]), [0.], [0., 2.])
    fiber = make_fiber(sample, points)
    probe = Probe(xp.asarray([REFERENCE, REFERENCE + 1e9]))
    response = compute_response(fiber, probe)
    assert response.probe is probe
    assert response.reflection.dtype == xp.complex128
    assert response.transmission.dtype == xp.complex128
    np.testing.assert_allclose(np.asarray(response.reflection), [.098, .098],
                               rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.abs(np.asarray(response.reflection))**2, .009604,
                               rtol=1e-14, atol=0.)
    expected = .4j*np.asarray(optical_path(fiber).transmission(0., 2., probe.frequency))
    np.testing.assert_allclose(np.asarray(response.transmission), expected,
                               rtol=1e-14, atol=0.)


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_empty_fiber_response_shapes_and_distributed_loss(xp, shape):
    probe = Probe(xp.full(shape, REFERENCE, dtype=xp.float64))
    response = compute_response(make_fiber(attenuation=.2), probe)
    assert response.shape == shape
    assert response.reflection.shape == shape
    assert response.transmission.shape == shape
    assert response.reflection.dtype == xp.complex128
    assert response.transmission.dtype == xp.complex128
    np.testing.assert_array_equal(np.asarray(response.reflection), np.zeros(shape))
    np.testing.assert_allclose(np.abs(np.asarray(response.transmission))**2,
                               np.full(shape, np.exp(-.4)), rtol=1e-14, atol=0.)


def test_static_state_keeps_probe_times_and_changes_both_paths(xp):
    strain = SpatialProfile([0., 2.], xp.asarray([1e-6, 1e-6]))
    state = FiberState(axial_strain=Field([FieldTerm(strain)]))
    point = PointReflector(2., .1, 0., .8, .7)
    fiber = make_fiber(elements=[point], attenuation=.2)
    probe = Probe(xp.asarray([REFERENCE, REFERENCE]), xp.asarray([0., 1.]))
    response = compute_response(fiber, probe, state)
    path = optical_path(fiber, state)
    assert response.probe is probe
    np.testing.assert_array_equal(np.asarray(response.probe.time), [0., 1.])
    np.testing.assert_allclose(np.asarray(response.reflection),
                               .1*np.asarray(path.round_trip(0., 2., probe.frequency)),
                               rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.asarray(response.transmission),
                               .8*np.asarray(path.transmission(0., 2., probe.frequency)),
                               rtol=1e-14, atol=0.)
    assert np.asarray(response.reflection)[0] == np.asarray(response.reflection)[1]


def test_manual_response_keeps_native_arrays_and_promotes_dtype(xp):
    probe = Probe(xp.asarray([REFERENCE, REFERENCE]))
    reflection = xp.asarray([.1, .2])
    response = Response(probe, reflection=reflection, transmission=xp.asarray([1j, .5j]))
    assert response.probe is probe
    assert type(response.reflection) is type(reflection)
    assert type(response.transmission) is type(reflection)
    assert response.reflection.dtype == xp.complex128
    assert response.transmission.dtype == xp.complex128
    np.testing.assert_array_equal(np.asarray(response.reflection), [.1, .2])
    np.testing.assert_array_equal(np.asarray(response.transmission), [1j, .5j])


def test_manual_response_owns_copies():
    reflection = np.array([.1j, .2j])
    transmission = np.array([.8, .7])
    response = Response(Probe([REFERENCE, REFERENCE]),
                        reflection=reflection, transmission=transmission)
    reflection[:] = 0.
    transmission[:] = 0.
    np.testing.assert_array_equal(response.reflection, [.1j, .2j])
    np.testing.assert_array_equal(response.transmission, [.8, .7])


@pytest.mark.parametrize("reflection,transmission", [
    (0., [1., 1.]), ([0., 0.], 1.), ([[0., 0.]], [1., 1.]),
    ([0., 0.], []),
])
def test_manual_response_requires_exact_probe_shape(reflection, transmission):
    with pytest.raises(ValueError, match="probe's shape"):
        Response(Probe([REFERENCE, REFERENCE]),
                 reflection=reflection, transmission=transmission)


def test_response_requires_an_explicit_probe():
    with pytest.raises(TypeError, match="Probe"):
        compute_response(make_fiber(), [REFERENCE])
    with pytest.raises(TypeError, match="Probe"):
        Response([REFERENCE], reflection=[0.], transmission=[1.])


def test_dynamic_state_requires_probe_times_even_for_a_constant_history():
    field = Field([FieldTerm(SpatialProfile([0., 2.], [1e-6, 1e-6]),
                             TimeHistory([0.], [0.]))])
    state = FiberState(axial_strain=field)
    with pytest.raises(ValueError, match="probe.time"):
        compute_response(make_fiber(), Probe([REFERENCE]), state)


def test_numpy_probe_and_jax_sample_require_explicit_selection(jax_numpy):
    sample = RayleighRealization(jax_numpy.asarray([.01j]), [0.], [0., 2.])
    probe = Probe(REFERENCE)
    fiber = make_fiber(sample)
    with pytest.raises(TypeError):
        compute_response(fiber, probe)
    response = compute_response(fiber, probe, xp=jax_numpy)
    assert response.probe is probe
    assert type(response.reflection) is type(sample.coefficients)
    assert type(response.transmission) is type(sample.coefficients)
    np.testing.assert_allclose(np.asarray(response.reflection), .01j, rtol=1e-14, atol=0.)


def test_jax_response_keeps_reflection_and_transmission_derivatives(jax_numpy):
    import jax
    def total_power(parameters):
        amplitude, forward = parameters
        sample = RayleighRealization(jax_numpy.reshape(amplitude, (1,)), [0.], [0., 2.])
        point = PointReflector(0., .1, 0., forward, .7)
        response = compute_response(make_fiber(sample, [point]),
                                    Probe(jax_numpy.asarray(REFERENCE)))
        return (jax_numpy.abs(response.reflection)**2
                + jax_numpy.abs(response.transmission)**2)
    parameters = jax_numpy.asarray([.01, .8], dtype=jax_numpy.float64)
    value = jax.jit(total_power)(parameters)
    gradient = jax.jit(jax.grad(total_power))(parameters)
    hessian = jax.jit(jax.hessian(total_power))(parameters)
    amplitude, forward = np.asarray(parameters)
    reflection = .1 + .7*amplitude*forward
    derivative = np.array([.7*forward, .7*amplitude])
    expected_gradient = 2.*reflection*derivative + [0., 2.*forward]
    expected_hessian = (2.*np.outer(derivative, derivative)
                        + 2.*reflection*np.array([[0., .7], [.7, 0.]])
                        + np.array([[0., 0.], [0., 2.]]))
    np.testing.assert_allclose(value, reflection**2 + forward**2, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-12, atol=0.)


@pytest.mark.parametrize("shape", [(), (2, 2), (0,), (0, 2)])
def test_varying_state_matches_separate_frozen_samples(xp, shape):
    static = FieldTerm(SpatialProfile([0., 1., 1., 2.],
                                      xp.asarray([1e-6, 1e-6, -1e-6, -1e-6])))
    moving = FieldTerm(SpatialProfile([0., 2.], xp.asarray([0., 2e-6])),
                       TimeHistory([0., 2.], xp.asarray([1., 3.])))
    thermal = FieldTerm(SpatialProfile([0., 2.], xp.asarray([1., 2.])),
                        TimeHistory([0., 2.], xp.asarray([3., 1.])))
    state = FiberState(axial_strain=Field([static, moving]),
                       temperature_change=Field([thermal]))
    sample = RayleighRealization(xp.asarray([.01j, -.02j]), [.25, 1.5], [0., 1., 2.])
    fiber = make_fiber(sample, [PointReflector(1., .2, 0., .8j, .7j)], attenuation=.2)
    times = np.linspace(-.5, 2.5, int(np.prod(shape))).reshape(shape)
    probe = Probe(xp.asarray(REFERENCE + times*1e9), xp.asarray(times))
    response = compute_response(fiber, probe, state)
    expected_r = np.empty(shape, dtype=np.complex128)
    expected_t = np.empty(shape, dtype=np.complex128)
    for index in np.ndindex(shape):
        frozen = state.at_time(probe.time[index])
        expected = compute_response(fiber, Probe(probe.frequency[index]), frozen)
        expected_r[index] = np.asarray(expected.reflection)
        expected_t[index] = np.asarray(expected.transmission)
    assert response.probe is probe
    assert not state.is_static
    for name, expected in (("reflection", expected_r), ("transmission", expected_t)):
        actual = getattr(response, name)
        assert actual.shape == shape
        assert actual.dtype == xp.complex128
        assert type(actual) is type(probe.frequency)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-13, atol=0.)
    np.testing.assert_array_equal(np.asarray(sample.coefficients), [.01j, -.02j])


@pytest.mark.parametrize("empty", [False, True])
def test_jax_history_and_numpy_probe_need_explicit_selection(jax_numpy, empty):
    profile = SpatialProfile([0., 2.], jax_numpy.asarray([1e-6, 1e-6]))
    history = TimeHistory([0., 2.], jax_numpy.asarray([1., 3.]))
    state = FiberState(axial_strain=Field([FieldTerm(profile, history)]))
    probe = Probe([], []) if empty else Probe([REFERENCE], [.5])
    fiber = make_fiber(elements=[PointReflector(1., .1, 0., .8, .7)])
    with pytest.raises(TypeError):
        compute_response(fiber, probe, state)
    response = compute_response(fiber, probe, state, xp=jax_numpy)
    assert response.probe is probe
    for name in ("reflection", "transmission"):
        actual = getattr(response, name)
        assert actual.shape == probe.shape
        assert actual.dtype == jax_numpy.complex128
        assert type(actual) is type(profile.values)


def test_jax_per_sample_response_keeps_history_profile_and_time_derivatives(jax_numpy):
    import jax
    length = SPEED_OF_LIGHT/(4.*GUIDE.effective_index*REFERENCE)
    weights = jax_numpy.asarray([1., 2.])
    def objective(parameters):
        a, b, time = parameters
        profile = SpatialProfile([0., length], jax_numpy.stack((a, a))*.01)
        history = TimeHistory([0., 2.], jax_numpy.stack((b, 3.*b)))
        state = FiberState(axial_strain=Field([FieldTerm(profile, history)]))
        sample = RayleighRealization(jax_numpy.asarray([.1]), [0.], [0., length])
        fiber = Fiber("short", length, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                      [Section(0., length, GUIDE, 0., 1e-8)],
                      [PointReflector(length, .2, 0., .8, .7)], sample)
        probe = Probe(jax_numpy.full((2,), REFERENCE, dtype=jax_numpy.float64),
                       jax_numpy.stack((time, 2.*time)))
        response = compute_response(fiber, probe, state)
        return jax_numpy.sum(weights*(jax_numpy.abs(response.reflection)**2
                                      + .1*jax_numpy.real(response.transmission)))
    parameters = jax_numpy.asarray([1., 2., .4], dtype=jax_numpy.float64)
    value = jax.jit(objective)(parameters)
    gradient = jax.jit(jax.grad(objective))(parameters)
    hessian = jax.jit(jax.hessian(objective))(parameters)
    a, b, time = np.asarray(parameters)
    scale = np.pi*GUIDE.strain_phase_factor*.01
    expected_value = 0.
    expected_gradient = np.zeros(3)
    expected_hessian = np.zeros((3, 3))
    for weight, multiple in ((1., 1.), (2., 2.)):
        u = 1. + multiple*time
        phase = scale*a*b*u
        derivative = scale*np.array([b*u, a*u, a*b*multiple])
        second = scale*np.array([[0., u, b*multiple], [u, 0., a*multiple],
                                  [b*multiple, a*multiple, 0.]])
        first_factor = .04*np.sin(phase) - .04*np.cos(phase/2.)
        second_factor = .04*np.cos(phase) + .02*np.sin(phase/2.)
        expected_value += weight*(.05 - .04*np.cos(phase) - .08*np.sin(phase/2.))
        expected_gradient += weight*first_factor*derivative
        expected_hessian += weight*(second_factor*np.outer(derivative, derivative)
                                     + first_factor*second)
    np.testing.assert_allclose(value, expected_value, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-10, atol=1e-14)
