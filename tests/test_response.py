import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, Probe, RayleighRealization, Response,
                        Section, SpatialProfile, TimeHistory, compute_response,
                        optical_path)

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


def test_dynamic_state_is_rejected_even_with_probe_times():
    field = Field([FieldTerm(SpatialProfile([0., 2.], [1e-6, 1e-6]),
                             TimeHistory([0., 1.], [0., 1.]))])
    state = FiberState(axial_strain=field)
    with pytest.raises(NotImplementedError):
        compute_response(make_fiber(), Probe([REFERENCE], [0.]), state)


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
