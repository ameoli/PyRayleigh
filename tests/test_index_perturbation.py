from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, Probe, RayleighRealization, Section,
                        SpatialProfile, TimeHistory, compute_response, optical_path)
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


def make_fiber(profile=None, length=2.):
    return Fiber("spool-1", length, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                  [Section(0., length, GUIDE, .2, 1e-8)], index_perturbation=profile)


def test_fiber_retains_the_profile_when_rayleigh_is_generated():
    profile = SpatialProfile([0., 2.], [1e-6, 2e-6])
    fiber = make_fiber(profile)
    generated = fiber.with_rayleigh(7, cell_length=.5)
    assert fiber.index_perturbation is profile
    assert generated.index_perturbation is profile
    plain = make_fiber().with_rayleigh(7, cell_length=.5)
    np.testing.assert_array_equal(generated.rayleigh.coefficients, plain.rayleigh.coefficients)


@pytest.mark.parametrize("profile", [1e-6, [1e-6, 1e-6], Field()])
def test_index_perturbation_requires_a_spatial_profile(profile):
    with pytest.raises(TypeError, match="SpatialProfile"):
        make_fiber(profile)


def test_index_integrals_across_steps_and_sections_add_to_state_lengths(xp):
    guides = [GuideOptics(REFERENCE, 1.45, 1.47, -2.2e-26),
              GuideOptics(REFERENCE + 5e9, 1.46, 1.48, -1.8e-26)]
    profile = SpatialProfile([-.5, .75, .75, 2.5],
                              xp.asarray([2e-6, 2e-6, -1e-6, -1e-6]))
    fiber = replace(make_fiber(profile), sections=[
        Section(0., 1., guides[0], .2, 1e-8), Section(1., 2., guides[1], .2, 1e-8)])
    strain = SpatialProfile([0., 2.], xp.asarray([1e-6, 1e-6]))
    state = FiberState(axial_strain=Field([FieldTerm(strain)]))
    path = optical_path(fiber, state)
    frequency = xp.asarray([REFERENCE - 1e10, REFERENCE, REFERENCE + 1e10])
    expected_phase, expected_delay = np.zeros(3), np.zeros(3)
    for guide, index_integral in zip(guides, [.25e-6, -.5e-6]):
        lp = .5 + guide.strain_phase_factor*.5e-6 + index_integral/guide.effective_index
        lg = .5 + guide.strain_group_factor*.5e-6 + index_integral/guide.group_index
        dw = 2.*np.pi*(np.asarray(frequency) - guide.frequency_ref)
        beta_ref = 2.*np.pi*guide.effective_index*guide.frequency_ref/SPEED_OF_LIGHT
        expected_phase += beta_ref*lp + (guide.group_index/SPEED_OF_LIGHT*dw
                                         + .5*guide.gvd*dw**2)*lg
        expected_delay += (guide.group_index/SPEED_OF_LIGHT + guide.gvd*dw)*lg
    np.testing.assert_allclose(np.asarray(path.one_way_phase(.5, 1.5, frequency)),
                               expected_phase, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.asarray(path.group_delay(.5, 1.5, frequency)),
                               expected_delay, rtol=1e-14, atol=0.)


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_index_phase_reaches_point_rayleigh_and_forward_paths(xp, shape):
    length = SPEED_OF_LIGHT/(4.*GUIDE.effective_index*REFERENCE)
    profile = SpatialProfile([0., length], xp.asarray([.001, .001]))
    sample = RayleighRealization(xp.asarray([.1, .01]), [0., length], [0., length/2., length])
    fiber = replace(make_fiber(profile, length),
                    sections=[Section(0., length, GUIDE, 0., 1e-8)], rayleigh=sample,
                    elements=[PointReflector(length, .2, 0., .8, .7)])
    probe = Probe(xp.full(shape, REFERENCE, dtype=xp.float64))
    response = compute_response(fiber, probe)
    phase = np.pi*.001/GUIDE.effective_index
    expected_r = .1 - .2056*np.exp(-1j*phase)
    expected_t = -.8j*np.exp(-.5j*phase)
    for name, expected in (("reflection", expected_r), ("transmission", expected_t)):
        actual = getattr(response, name)
        assert actual.shape == shape
        assert actual.dtype == xp.complex128
        np.testing.assert_allclose(np.asarray(actual), np.full(shape, expected),
                                   rtol=1e-13, atol=1e-15)


def test_numpy_queries_and_jax_index_values_require_explicit_selection(jax_numpy):
    profile = SpatialProfile([0., 2.], jax_numpy.asarray([1e-6, 2e-6]))
    fiber = make_fiber(profile)
    path = optical_path(fiber)
    frequency = np.array(REFERENCE)
    for method in (path.one_way_phase, path.group_delay, path.phase_factor):
        with pytest.raises(TypeError):
            method(0., 2., frequency)
        assert type(method(0., 2., frequency, xp=jax_numpy)) is type(profile.values)
    with pytest.raises(TypeError):
        compute_response(fiber, Probe(frequency))
    response = compute_response(fiber, Probe(frequency), xp=jax_numpy)
    assert type(response.transmission) is type(profile.values)


def test_index_profile_is_kept_when_the_state_varies_per_sample(xp):
    profile = SpatialProfile([0., 2.], xp.asarray([1e-6, 3e-6]))
    moving = FieldTerm(SpatialProfile([0., 2.], xp.asarray([1e-6, 1e-6])),
                       TimeHistory([0., 1.], xp.asarray([0., 1.])))
    state = FiberState(axial_strain=Field([moving]))
    fiber = make_fiber(profile)
    probe = Probe(xp.asarray([REFERENCE, REFERENCE]), xp.asarray([0., 1.]))
    response = compute_response(fiber, probe, state)
    for index, time in enumerate((0., 1.)):
        frozen = compute_response(fiber, Probe(probe.frequency[index]), state.at_time(time))
        np.testing.assert_allclose(np.asarray(response.transmission[index]),
                                   np.asarray(frozen.transmission), rtol=1e-14, atol=0.)
    assert fiber.index_perturbation is profile


def test_jax_response_derivatives_in_index_values_and_frequency(jax_numpy):
    import jax
    length = SPEED_OF_LIGHT/(4.*GUIDE.effective_index*REFERENCE)
    def objective(parameters):
        profile = SpatialProfile([0., length], parameters[:2]*.01)
        sample = RayleighRealization(jax_numpy.asarray([.1, .01]),
                                     [0., length], [0., length/2., length])
        fiber = replace(make_fiber(profile, length),
                        sections=[Section(0., length, GUIDE, 0., 1e-8)], rayleigh=sample,
                        elements=[PointReflector(length, .2, 0., .8, .7)])
        response = compute_response(fiber, Probe(REFERENCE + parameters[2]*1e9))
        return (jax_numpy.abs(response.reflection)**2
                + .1*jax_numpy.real(response.transmission))
    parameters = jax_numpy.asarray([1., 2., .75], dtype=jax_numpy.float64)
    value = jax.jit(objective)(parameters)
    gradient = jax.jit(jax.grad(objective))(parameters)
    hessian = jax.jit(jax.hessian(objective))(parameters)
    a, b, offset = np.asarray(parameters)
    scale = 2.*np.pi*length/SPEED_OF_LIGHT
    carrier, frequency, mixed = scale*REFERENCE*.005, scale*GUIDE.group_index*1e9, scale*.005e9
    phase = carrier*(a + b) + frequency*offset + mixed*(a + b)*offset
    derivative = np.array([carrier + mixed*offset, carrier + mixed*offset,
                           frequency + mixed*(a + b)])
    second = mixed*np.array([[0., 0., 1.], [0., 0., 1.], [1., 1., 0.]])
    amplitude = .2056
    interference = 2.*.1*amplitude
    first_factor = 2.*interference*np.sin(2.*phase) - .08*np.cos(phase)
    second_factor = 4.*interference*np.cos(2.*phase) + .08*np.sin(phase)
    expected = .01 + amplitude**2 - interference*np.cos(2.*phase) - .08*np.sin(phase)
    expected_hessian = second_factor*np.outer(derivative, derivative) + first_factor*second
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(gradient, first_factor*derivative, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-10, atol=1e-14)
