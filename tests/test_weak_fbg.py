from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Field, FieldTerm, Fiber, FiberState, GuideOptics, PointReflector, Probe,
                        RayleighRealization, Section, SpatialProfile, TimeHistory, WeakFBG,
                        compute_response, optical_path, point_response, rayleigh_response,
                        weak_fbg_reflection)
from pyrayleigh.constants import SPEED_OF_LIGHT

WAVELENGTH, LENGTH, INDEX = 1550e-9, .005, 1.4682
FREQUENCY = SPEED_OF_LIGHT/WAVELENGTH
THERMAL_EXPANSION, THERMO_OPTIC = 5.5e-7, 6.5e-6


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


def make_fiber(elements=(), sample=None, attenuation=.2):
    guide = GuideOptics(FREQUENCY, INDEX, INDEX)
    return Fiber("spool", 2., 293.15, FREQUENCY - 50e9, FREQUENCY + 50e9,
                  [Section(0., 2., guide, attenuation, 0.)], elements, sample)


def uniform_field(value, xp, moving=False):
    profile = SpatialProfile([0., 2.], xp.stack((value, value)))
    history = TimeHistory([0., 2.], xp.asarray([0., 2.])) if moving else None
    return Field([FieldTerm(profile, history)])


@pytest.mark.parametrize("position,error", [(True, TypeError), (np.array(.5), TypeError),
                                             (-1., ValueError), (float("inf"), ValueError)])
def test_grating_position_is_fixed_and_valid(position, error):
    with pytest.raises(error):
        WeakFBG(position, WAVELENGTH, LENGTH, .04)


@pytest.mark.parametrize("name,value,error", [("bragg_wavelength", [WAVELENGTH], ValueError),
                                             ("length", [LENGTH], ValueError),
                                             ("peak_reflectivity", [1j], TypeError)])
def test_grating_parameters_are_real_scalars(name, value, error):
    parameters = dict(bragg_wavelength=WAVELENGTH, length=LENGTH, peak_reflectivity=.04)
    with pytest.raises(error):
        WeakFBG(1., **{**parameters, name: value})


def test_fiber_sorts_mixed_elements_and_keeps_coincident_gratings_after_the_point():
    first, second = WeakFBG(1., WAVELENGTH, LENGTH, .01), WeakFBG(1., WAVELENGTH, LENGTH, .04)
    point = PointReflector(1., .1, 0., .8, .7)
    fiber = make_fiber([first, second, point])
    assert all(actual is expected for actual, expected in zip(fiber.elements, [point, first, second]))
    with pytest.raises(ValueError, match="distinct"):
        make_fiber([first, point, point])
    with pytest.raises(ValueError, match="beyond"):
        make_fiber([WeakFBG(3., WAVELENGTH, LENGTH, .04)])
    drawn = fiber.with_rayleigh(3, cell_length=.5)
    assert drawn.elements[1] is first and drawn.elements[2] is second


@pytest.mark.parametrize("position", [0., 1., 2.])
def test_weak_grating_return_uses_its_path_without_depleting_transmission(xp, position):
    grating = WeakFBG(position, xp.asarray(WAVELENGTH), LENGTH, .04)
    fiber = make_fiber([grating])
    frequency = xp.asarray([FREQUENCY - 10e9, FREQUENCY, FREQUENCY + 10e9])
    response = compute_response(fiber, Probe(frequency))
    path = optical_path(fiber)
    local = weak_fbg_reflection(frequency, WAVELENGTH, LENGTH, .04, INDEX, xp=xp)
    expected = local*path.round_trip(0., position, frequency)
    np.testing.assert_allclose(np.asarray(response.reflection), np.asarray(expected), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(response.transmission),
                               np.asarray(path.transmission(0., 2., frequency)), rtol=1e-13)
    assert response.reflection.dtype == xp.complex128


def test_point_rayleigh_and_coincident_fbg_returns_add_coherently(xp):
    points = [PointReflector(.5, .1, 0., .8, .7), PointReflector(1., .02, 0., .9, .95)]
    sample = RayleighRealization(xp.asarray([.01j, .02]), [.25, 1.5], [0., .5, 2.])
    base = make_fiber(points, sample)
    gratings = [WeakFBG(1., WAVELENGTH, LENGTH, .01), WeakFBG(1., WAVELENGTH, LENGTH, .04)]
    fiber = replace(base, elements=[*gratings, *points])
    frequency = xp.asarray([FREQUENCY, FREQUENCY + 10e9])
    baseline = compute_response(base, Probe(frequency))
    result = compute_response(fiber, Probe(frequency))
    local = sum(grating.reflection(frequency, INDEX, xp=xp) for grating in gratings)
    expected = baseline.reflection + local*(.8*.7*.9*.95)*optical_path(fiber).round_trip(0., 1., frequency)
    np.testing.assert_allclose(np.asarray(result.reflection), np.asarray(expected), rtol=1e-13)
    np.testing.assert_array_equal(np.asarray(result.transmission), np.asarray(baseline.transmission))
    np.testing.assert_array_equal(np.asarray(rayleigh_response(fiber, frequency)),
                                  np.asarray(rayleigh_response(base, frequency)))


def test_thermal_expansion_is_counted_once_in_the_bragg_shift(xp):
    mechanical, temperature = xp.asarray(5e-6), xp.asarray(8.)
    state = FiberState(axial_strain=uniform_field(mechanical + THERMAL_EXPANSION*temperature, xp),
                       temperature_change=uniform_field(temperature, xp))
    grating = WeakFBG(1., WAVELENGTH, LENGTH, .04)
    fiber = make_fiber([grating])
    shifted = WAVELENGTH*(1. + .78*mechanical + (THERMAL_EXPANSION + THERMO_OPTIC)*temperature)
    frequency = SPEED_OF_LIGHT/shifted + xp.asarray([-5e9, 0., 5e9])
    result = point_response(fiber, frequency, state)
    local = weak_fbg_reflection(frequency, shifted, LENGTH, .04, INDEX, xp=xp)
    expected = local*optical_path(fiber, state).round_trip(0., 1., frequency)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-11, atol=1e-13)
    assert grating.bragg_wavelength == WAVELENGTH


@pytest.mark.parametrize("position", [1., 2.])
def test_section_boundary_and_end_face_use_the_right_or_final_section(xp, position):
    left = GuideOptics(FREQUENCY, 1.3, 1.3, strain_phase_factor=.5, temperature_phase_factor=1e-5)
    right = GuideOptics(FREQUENCY, 1.5, 1.5, strain_phase_factor=.9, temperature_phase_factor=2e-5)
    fiber = replace(make_fiber([WeakFBG(position, WAVELENGTH, LENGTH, .04)]),
                    sections=[Section(0., 1., left, 0., 0.), Section(1., 2., right, 0., 0.)])
    state = FiberState(axial_strain=uniform_field(xp.asarray(1e-5), xp),
                       temperature_change=uniform_field(xp.asarray(2.), xp))
    shifted = WAVELENGTH*(1. + .9e-5 + 4e-5)
    frequency = xp.asarray([FREQUENCY, SPEED_OF_LIGHT/shifted])
    expected = weak_fbg_reflection(frequency, shifted, LENGTH, .04, 1.5, xp=xp)
    expected = expected*optical_path(fiber, state).round_trip(0., position, frequency)
    np.testing.assert_allclose(np.asarray(point_response(fiber, frequency, state)),
                               np.asarray(expected), rtol=1e-11, atol=1e-13)


def test_varying_state_freezes_the_resonance_per_probe_sample_without_drift(xp):
    mechanical, temperature = xp.asarray(5e-6), xp.asarray(8.)
    state = FiberState(axial_strain=uniform_field(mechanical + THERMAL_EXPANSION*temperature, xp, True),
                       temperature_change=uniform_field(temperature, xp, True))
    grating = WeakFBG(1., WAVELENGTH, LENGTH, .04)
    fiber = make_fiber([grating])
    time = xp.asarray([[0., 1.], [.5, 0.]])
    shift = time*(.78*mechanical + (THERMAL_EXPANSION + THERMO_OPTIC)*temperature)
    probe = Probe(SPEED_OF_LIGHT/(WAVELENGTH*(1. + shift)), time)
    first = compute_response(fiber, probe, state)
    for row, column in ((0, 0), (0, 1), (1, 0), (1, 1)):
        frozen = state.at_time(time[row, column], xp=xp)
        expected = .2*optical_path(fiber, frozen).round_trip(0., 1., probe.frequency[row, column], xp=xp)
        np.testing.assert_allclose(np.asarray(first.reflection[row, column]), np.asarray(expected), rtol=1e-12)
    second = compute_response(fiber, probe, state)
    np.testing.assert_array_equal(np.asarray(second.reflection), np.asarray(first.reflection))
    assert grating.bragg_wavelength == WAVELENGTH


def test_invalid_shifted_resonance_is_native_nan(xp):
    grating = WeakFBG(0., WAVELENGTH, LENGTH, .04)
    result = grating.reflection(xp.asarray([FREQUENCY]), INDEX, relative_shift=-1.)
    assert np.all(np.isnan(np.asarray(result)))


def test_grating_parameters_participate_in_namespace_selection(jax_numpy):
    grating = WeakFBG(1., WAVELENGTH, LENGTH, jax_numpy.asarray(.04))
    fiber = make_fiber([grating])
    probe = Probe(np.array([FREQUENCY]))
    with pytest.raises(TypeError):
        compute_response(fiber, probe)
    with pytest.raises(TypeError):
        point_response(fiber, probe.frequency)
    result = compute_response(fiber, probe, xp=jax_numpy)
    assert type(result.reflection) is type(grating.peak_reflectivity)


def test_jax_fbg_state_history_and_peak_keep_jit_gradients_and_hessians(jax_numpy):
    import jax
    frequency = FREQUENCY - 5e9
    def power(parameters):
        strain_micro, temperature, time, reflectivity = parameters
        total = strain_micro*1e-6 + THERMAL_EXPANSION*temperature
        state = FiberState(axial_strain=uniform_field(total, jax_numpy, True),
                           temperature_change=uniform_field(temperature, jax_numpy, True))
        fiber = make_fiber([WeakFBG(0., WAVELENGTH, LENGTH, reflectivity)])
        probe = Probe(jax_numpy.asarray(frequency), time)
        response = compute_response(fiber, probe, state)
        return jax_numpy.abs(response.reflection)**2
    def expected(parameters):
        strain_micro, temperature, time, reflectivity = parameters
        shift = time*(.78*strain_micro*1e-6 + (THERMAL_EXPANSION + THERMO_OPTIC)*temperature)
        reflection = weak_fbg_reflection(frequency, WAVELENGTH*(1. + shift), LENGTH,
                                          reflectivity, INDEX, xp=jax_numpy)
        return jax_numpy.abs(reflection)**2
    parameters = jax_numpy.asarray([10., 10., .6, .04])
    for transform in (lambda f: f, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(power))(parameters), jax.jit(transform(expected))(parameters),
                                   rtol=2e-9, atol=1e-11)
