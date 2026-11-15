from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (BendLossTable, Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, RayleighRealization, Section, SpatialProfile,
                        TimeHistory, WeakFBG, multiple_point_response, optical_path,
                        point_response)
from pyrayleigh.constants import SPEED_OF_LIGHT

REFERENCE, INDEX = 193414489032258.06, 1.4682


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


def make_fiber(elements=(), attenuation=0., length=3.):
    guide = GuideOptics(REFERENCE, INDEX, INDEX)
    return Fiber("spool", length, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                  [Section(0., length, guide, attenuation, 0.)], elements)


def cavity(first=.5, last=2.):
    return [PointReflector(first, .2, -.2, .8, .8),
            PointReflector(last, .3, -.3, .9, .9)]


@pytest.mark.parametrize("order", [2, 3])
def test_two_points_use_reverse_reflection_and_both_directional_transmissions(xp, order):
    first = PointReflector(0., .1, xp.asarray(.21*np.exp(.7j)),
                           xp.asarray(.7j), xp.asarray(.6*np.exp(.2j)))
    last = PointReflector(2., xp.asarray(.3*np.exp(-1.1j)), .05, .8, .8)
    fiber = make_fiber([last, first], attenuation=.07)
    frequency = xp.asarray([[REFERENCE - 1e9, REFERENCE], [REFERENCE, REFERENCE + 1e9]])
    result = multiple_point_response(fiber, frequency, max_order=order)
    q = np.asarray(optical_path(fiber).round_trip(0., 2., frequency))
    expected = complex(first.t_left_to_right)*complex(first.t_right_to_left)*(
        complex(first.r_right)*complex(last.r_left)**2*q**2)
    if order == 3:
        expected += complex(first.t_left_to_right)*complex(first.t_right_to_left)*(
            complex(first.r_right)**2*complex(last.r_left)**3*q**3)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-14, atol=1e-17)
    assert result.dtype == xp.complex128 and result.shape == frequency.shape


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_no_cavity_and_disabled_orders_give_native_zero(xp, shape):
    frequency = xp.full(shape, REFERENCE, dtype=xp.float64)
    for elements, order in (([], 3), ([cavity()[0]], 3), (cavity(), 1)):
        result = multiple_point_response(make_fiber(elements), frequency, max_order=order)
        assert result.dtype == xp.complex128 and result.shape == shape
        np.testing.assert_array_equal(np.asarray(result), np.zeros(shape))
    empty = multiple_point_response(make_fiber(cavity()), xp.asarray([], dtype=xp.float64))
    assert empty.shape == (0,)


@pytest.mark.parametrize("start,end,passes", [(.1, .3, 2), (.75, 1.25, 4), (2.25, 2.5, 0)])
@pytest.mark.parametrize("kind", ["distributed", "bend"])
def test_segment_losses_count_actual_passages(xp, start, end, passes, kind):
    bare = make_fiber(cavity())
    guide = bare.sections[0].optics
    table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.7], [.7]]))
    middle = replace(guide, bend_loss=table) if kind == "bend" else guide
    fiber = replace(bare, sections=[Section(0., start, guide, 0., 0.),
                                    Section(start, end, middle, .7 if kind == "distributed" else 0., 0.),
                                    Section(end, 3., guide, 0., 0.)])
    state = FiberState(curvature_x=Field([FieldTerm(SpatialProfile([0., 3.], xp.asarray([5., 5.])))]))
    frequency = xp.asarray([REFERENCE - 1e9, REFERENCE + 1e9])
    actual = multiple_point_response(fiber, frequency, state)
    baseline_fiber = replace(fiber, sections=[replace(section, optics=guide, attenuation=0.)
                                             for section in fiber.sections])
    baseline = multiple_point_response(baseline_fiber, frequency, state)
    np.testing.assert_allclose(np.asarray(actual)/np.asarray(baseline),
                               np.exp(-.5*passes*.7*(end - start)), rtol=2e-13, atol=1e-15)


@pytest.mark.parametrize("position,crossings", [(.25, 1), (1.25, 2), (2.75, 0)])
@pytest.mark.parametrize("opaque", [False, True])
def test_passive_point_transmissions_include_every_crossing(xp, position, crossings, opaque):
    bare = make_fiber(cavity())
    forward, reverse = (0., 0.) if opaque else (.8*np.exp(.2j), .7*np.exp(-.1j))
    extra = PointReflector(position, 0., 0., xp.asarray(forward), xp.asarray(reverse))
    fiber = replace(bare, elements=[*bare.elements, extra])
    frequency = xp.asarray([REFERENCE])
    expected = np.asarray(multiple_point_response(bare, frequency))*(forward*reverse)**crossings
    actual = multiple_point_response(fiber, frequency)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-14, atol=1e-17)


def test_rayleigh_and_weak_gratings_keep_their_separate_returns(xp):
    fiber = make_fiber(cavity(0., 2.))
    sample = RayleighRealization(xp.asarray([.01j]), [1.], [0., 3.])
    grating = WeakFBG(0., 1550e-9, .005, .01)
    mixed = replace(fiber, elements=[grating, *fiber.elements,
                                    WeakFBG(2., 1550e-9, .005, .01)], rayleigh=sample)
    frequency = xp.asarray([REFERENCE])
    before = point_response(mixed, frequency)
    expected = multiple_point_response(fiber, frequency, max_order=3)
    actual = multiple_point_response(mixed, frequency, max_order=3)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(point_response(mixed, frequency)), np.asarray(before))
    assert mixed.rayleigh is sample and grating.bragg_wavelength == 1550e-9


@pytest.mark.parametrize("missing", ["reverse", "last", "through"])
def test_a_blocked_or_zero_cavity_gives_no_extra_return(xp, missing):
    first, last = cavity(0., 2.)
    if missing == "reverse":
        first = replace(first, r_right=xp.asarray(0.))
    elif missing == "last":
        last = replace(last, r_left=xp.asarray(0.))
    else:
        first = replace(first, t_left_to_right=xp.asarray(0.))
    result = multiple_point_response(make_fiber([first, last]), xp.asarray(REFERENCE), max_order=3)
    np.testing.assert_array_equal(np.asarray(result), 0.)


@pytest.mark.parametrize("kind", ["strain", "temperature", "index"])
def test_local_phase_changes_follow_the_actual_cavity_legs(xp, kind):
    scale = 1e-5
    start, end = .75*scale, 1.25*scale
    bare = make_fiber(cavity(.5*scale, 2.*scale), length=3.*scale)
    value = {"strain": 5e-6, "temperature": 10., "index": 2e-5}[kind]
    profile = SpatialProfile([start, end], xp.asarray([value, value]))
    state, fiber = FiberState(), bare
    if kind == "index":
        fiber = replace(bare, index_perturbation=profile)
        change = value
    elif kind == "strain":
        state = FiberState(axial_strain=Field([FieldTerm(profile)]))
        change = INDEX*.78*value
    else:
        state = FiberState(temperature_change=Field([FieldTerm(profile)]))
        change = INDEX*(6.5e-6 + .22*5.5e-7)*value
    frequency = xp.asarray(REFERENCE)
    ratio = multiple_point_response(fiber, frequency, state)/multiple_point_response(bare, frequency)
    phase = 4.*2.*np.pi*REFERENCE/SPEED_OF_LIGHT*change*(end - start)
    np.testing.assert_allclose(np.asarray(ratio), np.exp(-1j*phase), rtol=1e-12, atol=1e-14)


def test_three_weak_points_match_an_independent_recursive_cavity_solution(xp):
    positions = np.array([2., 8., 20.])*1e-6
    reflections = np.array([.005, .007, .009])*np.exp(1j*np.array([.7, -1.1, 2.]))
    powers = np.abs(reflections)**2
    elements = [PointReflector(float(z), xp.asarray(r), xp.asarray(-r.conjugate()),
                               xp.asarray(np.sqrt(1. - power)), xp.asarray(np.sqrt(1. - power)))
                for z, r, power in zip(positions, reflections, powers)]
    attenuation = .1/(30e-6)
    fiber = make_fiber(elements, attenuation, length=30e-6)
    frequency = xp.asarray(np.linspace(REFERENCE - 20e9, REFERENCE + 20e9, 17))
    actual = (point_response(fiber, frequency)
              + multiple_point_response(fiber, frequency, max_order=3))
    beta = 2.*np.pi*INDEX*np.asarray(frequency)/SPEED_OF_LIGHT
    expected = np.zeros(17, complex)
    for index in range(2, -1, -1):
        if index < 2:
            distance = positions[index + 1] - positions[index]
            expected *= np.exp((-attenuation - 2j*beta)*distance)
        r = reflections[index]
        expected = r + (1. - powers[index])*expected/(1. + r.conjugate()*expected)
    expected *= np.exp((-attenuation - 2j*beta)*positions[0])
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0., atol=1e-13)


def test_frequency_validation_and_static_state_requirement():
    with pytest.raises(TypeError, match="real"):
        multiple_point_response(make_fiber(), [REFERENCE + 1j])
    history = TimeHistory([0., 1.], [0., 1.])
    state = FiberState(axial_strain=Field([FieldTerm(SpatialProfile([0., 3.], [1e-6, 1e-6]), history)]))
    with pytest.raises(NotImplementedError, match="static"):
        multiple_point_response(make_fiber(cavity()), REFERENCE, state)
    frozen = state.at_time(.5)
    assert np.isfinite(multiple_point_response(make_fiber(cavity()), REFERENCE, frozen))


def test_reverse_coefficient_participates_in_namespace_selection(jax_numpy):
    first, last = cavity(0., 2.)
    first = replace(first, r_right=jax_numpy.asarray(-.2 + .1j))
    fiber = make_fiber([first, last])
    frequency = np.array([REFERENCE])
    with pytest.raises(TypeError):
        multiple_point_response(fiber, frequency)
    result = multiple_point_response(fiber, frequency, xp=jax_numpy)
    assert type(result) is type(first.r_right) and result.dtype == jax_numpy.complex128


@pytest.mark.parametrize("reverse", [0., -.2])
def test_jax_jit_gradients_and_hessians_keep_reflection_loss_index_and_frequency(jax_numpy, reverse):
    import jax
    length = 30e-6
    def field(parameters):
        reflection, forward, backward, depth, index_change, detuning = parameters
        guide = GuideOptics(REFERENCE, INDEX + index_change*1e-4, 1.5)
        elements = [PointReflector(0., .1, reflection, forward, backward),
                    PointReflector(length, .3, -.3, .8, .8)]
        fiber = replace(make_fiber(elements, depth/length, length),
                        sections=[Section(0., length, guide, depth/length, 0.)])
        value = multiple_point_response(fiber, REFERENCE + detuning*1e9)
        return jax_numpy.real(value)
    def expected(parameters):
        reflection, forward, backward, depth, index_change, detuning = parameters
        phase = 8.*jax_numpy.pi*length*((INDEX + index_change*1e-4)*REFERENCE
                                        + 1.5*detuning*1e9)/SPEED_OF_LIGHT
        return reflection*forward*backward*.09*jax_numpy.exp(-2.*depth)*jax_numpy.cos(phase)
    parameters = jax_numpy.asarray([reverse, .8, .7, .2, .3, .4])
    for transform in (lambda function: function, jax.grad, jax.hessian):
        actual = jax.jit(transform(field))(parameters)
        reference = jax.jit(transform(expected))(parameters)
        np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=1e-13)
