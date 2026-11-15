from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (BendLossTable, Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, Probe, RayleighRealization, Section, SpatialProfile,
                        TimeHistory, WeakFBG, compute_response, multiple_point_response,
                        optical_path, point_response, rayleigh_response)
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


def make_fiber():
    guide = GuideOptics(REFERENCE, INDEX, INDEX)
    return Fiber("spool", 2., 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                  [Section(0., 2., guide, .1, 0.)],
                  [PointReflector(0., .1, -.2, .8, .7),
                   PointReflector(2., .3, -.3, .9, .85)])


def moving_state(xp):
    history = TimeHistory([0., 1.], xp.asarray([.8, 1.2]))
    strain = SpatialProfile([0., 1., 1., 2.], xp.asarray([1e-6, 1e-6, 2e-6, 2e-6]))
    temperature = SpatialProfile([0., 2.], xp.asarray([1., 2.]))
    curvature = SpatialProfile([0., 2.], xp.asarray([5., 5.]))
    return FiberState(axial_strain=Field([FieldTerm(strain, history)]),
                       temperature_change=Field([FieldTerm(temperature, history)]),
                       curvature_x=Field([FieldTerm(curvature, history)]))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_static_response_adds_each_kind_of_return_once(xp, order):
    base = make_fiber()
    sample = RayleighRealization(xp.asarray([.01j, -.02]), [.25, 1.5], [0., 1., 2.])
    grating = WeakFBG(1., 1550e-9, .005, .01)
    fiber = replace(base, elements=[*base.elements, grating], rayleigh=sample)
    probe = Probe(xp.asarray([[REFERENCE - 1e9, REFERENCE, REFERENCE + 1e9]]))
    result = compute_response(fiber, probe, max_order=order)
    expected = (point_response(fiber, probe.frequency) + rayleigh_response(fiber, probe.frequency)
                + multiple_point_response(fiber, probe.frequency, max_order=order))
    np.testing.assert_array_equal(np.asarray(result.reflection), np.asarray(expected))
    direct = compute_response(fiber, probe)
    np.testing.assert_array_equal(np.asarray(result.transmission), np.asarray(direct.transmission))
    if order == 1:
        np.testing.assert_array_equal(np.asarray(result.reflection), np.asarray(direct.reflection))
    assert result.probe is probe and result.shape == probe.shape
    assert result.reflection.dtype == xp.complex128 and type(result.reflection) is type(probe.frequency)
    np.testing.assert_array_equal(np.asarray(sample.coefficients), [.01j, -.02])
    assert grating.bragg_wavelength == 1550e-9


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
def test_moving_response_freezes_all_repeated_passages_at_the_probe_time(xp, order, shape):
    fiber = make_fiber()
    guide = replace(fiber.sections[0].optics,
                    bend_loss=BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.03]])))
    fiber = replace(fiber, sections=[replace(fiber.sections[0], optics=guide)])
    state = moving_state(xp)
    times = np.linspace(.2, .8, int(np.prod(shape))).reshape(shape)
    probe = Probe(xp.asarray(REFERENCE + times*1e9), xp.asarray(times))
    result = compute_response(fiber, probe, state, max_order=order)
    expected_r, expected_t = np.empty(shape, complex), np.empty(shape, complex)
    for index in np.ndindex(shape):
        frequency = probe.frequency[index]
        frozen = state.at_time(probe.time[index], xp=xp)
        expected_r[index] = np.asarray(point_response(fiber, frequency, frozen)
                                       + multiple_point_response(fiber, frequency, frozen, max_order=order))
        expected_t[index] = np.asarray(.8*.9*optical_path(fiber, frozen).transmission(0., 2., frequency))
    np.testing.assert_allclose(np.asarray(result.reflection), expected_r, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(np.asarray(result.transmission), expected_t, rtol=1e-13, atol=1e-16)
    for value in (result.reflection, result.transmission):
        assert value.shape == shape and value.dtype == xp.complex128
        assert type(value) is type(probe.frequency)
    assert result.probe is probe and not state.is_static
    np.testing.assert_array_equal(np.asarray(state.curvature_x.terms[0].profile.values), [5., 5.])
    np.testing.assert_array_equal(np.asarray(state.axial_strain.terms[0].history.values), [.8, 1.2])


@pytest.mark.parametrize("kind", ["empty", "one_point", "grating"])
def test_without_a_point_cavity_higher_orders_keep_the_direct_response(xp, kind):
    fiber = make_fiber()
    elements = {"empty": [], "one_point": fiber.elements[:1],
                "grating": [WeakFBG(1., 1550e-9, .005, .01)]}[kind]
    fiber = replace(fiber, elements=elements)
    probe = Probe(xp.asarray([REFERENCE]))
    direct = compute_response(fiber, probe)
    result = compute_response(fiber, probe, max_order=3)
    np.testing.assert_array_equal(np.asarray(result.reflection), np.asarray(direct.reflection))
    np.testing.assert_array_equal(np.asarray(result.transmission), np.asarray(direct.transmission))


@pytest.mark.parametrize("order,error", [(True, TypeError), (2., TypeError),
                                         (np.array(2), TypeError), (0, ValueError), (4, ValueError)])
@pytest.mark.parametrize("empty_moving", [False, True])
def test_order_is_validated_even_for_an_empty_moving_probe(order, error, empty_moving):
    state = moving_state(numpy_api) if empty_moving else None
    probe = Probe([], []) if empty_moving else Probe(REFERENCE)
    with pytest.raises(error, match="max_order"):
        compute_response(make_fiber(), probe, state, max_order=order)


@pytest.mark.parametrize("empty", [False, True])
def test_moving_higher_orders_still_need_probe_times(empty):
    probe = Probe([]) if empty else Probe(REFERENCE)
    with pytest.raises(ValueError, match="probe.time"):
        compute_response(make_fiber(), probe, moving_state(numpy_api), max_order=2)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("moving", [False, True])
def test_reverse_data_select_the_namespace_only_when_higher_orders_are_requested(jax_numpy, empty, moving):
    fiber = make_fiber()
    first, last = fiber.elements
    fiber = replace(fiber, elements=[replace(first, r_right=jax_numpy.asarray(-.2)), last])
    state = moving_state(numpy_api) if moving else None
    probe = Probe([], []) if empty else Probe([REFERENCE], [.5])
    direct = compute_response(fiber, probe, state)
    assert isinstance(direct.reflection, np.ndarray)
    with pytest.raises(TypeError):
        compute_response(fiber, probe, state, max_order=2)
    result = compute_response(fiber, probe, state, max_order=np.int64(3), xp=jax_numpy)
    assert type(result.reflection) is type(fiber.elements[0].r_right)
    assert result.shape == probe.shape and result.reflection.dtype == jax_numpy.complex128


@pytest.mark.parametrize("reverse", [0., -.2])
def test_jax_moving_histories_times_and_reverse_reflection_keep_gradients_and_hessians(jax_numpy, reverse):
    import jax
    length = SPEED_OF_LIGHT/(4.*INDEX*REFERENCE)
    weights = jax_numpy.asarray([1., 2.])
    def objective(parameters):
        a, b, time, r_right = parameters
        profile = SpatialProfile([0., length], jax_numpy.stack((a, a))*.01)
        history = TimeHistory([0., 2.], jax_numpy.stack((b, 3.*b)))
        state = FiberState(axial_strain=Field([FieldTerm(profile, history)]))
        guide = GuideOptics(REFERENCE, INDEX, INDEX)
        elements = [PointReflector(0., .1, r_right, .8, .7),
                    PointReflector(length, .2, -.2, .8, .7)]
        sample = RayleighRealization(jax_numpy.asarray([.01j]), [0.], [0., length])
        fiber = Fiber("short", length, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                      [Section(0., length, guide, 0., 0.)], elements, sample)
        probe = Probe(jax_numpy.full((2,), REFERENCE, dtype=jax_numpy.float64),
                       jax_numpy.stack((time, 2.*time)))
        response = compute_response(fiber, probe, state, max_order=3)
        return jax_numpy.sum(weights*(jax_numpy.abs(response.reflection)**2
                                      + .1*jax_numpy.real(response.transmission)))
    def expected(parameters):
        a, b, time, r_right = parameters
        shift = jax_numpy.pi*.78*.01*a*b*(1. + jax_numpy.stack((time, 2.*time)))
        q = -jax_numpy.exp(-1j*shift)
        reflection = (.1 + .56*.2*q + .0056j
                      + .56*.2**2*r_right*q**2 + .56*.2**3*r_right**2*q**3)
        transmission = -.64j*jax_numpy.exp(-.5j*shift)
        return jax_numpy.sum(weights*(jax_numpy.abs(reflection)**2 + .1*jax_numpy.real(transmission)))
    parameters = jax_numpy.asarray([1., 2., .4, reverse])
    for transform in (lambda function: function, jax.grad, jax.hessian):
        actual = jax.jit(transform(objective))(parameters)
        reference = jax.jit(transform(expected))(parameters)
        np.testing.assert_allclose(actual, reference, rtol=2e-10, atol=1e-13)
