from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (CrosstalkRealization, Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        MultiCoreFiber, PointReflector, Probe, RayleighRealization, Section,
                        SpatialProfile, TimeHistory, compute_response, optical_path)
from pyrayleigh.constants import SPEED_OF_LIGHT

REFERENCE, INDEX = 193414489032258.06, 1.4682
POSITIONS, EDGES = [0., .5, 1.], [0., .5, 1., 1.25]


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


def make_model(xp=numpy_api, count=2, topology="linear", rayleigh=True):
    guide = GuideOptics(REFERENCE, INDEX, INDEX)
    section = Section(0., 1.25, guide, .2, 0.)
    cores = []
    for core in range(count):
        values = xp.asarray([.01j, .02, -.01j])*(core + 1) if rayleigh else xp.zeros((3,))
        sample = RayleighRealization(values, POSITIONS, EDGES)
        elements = [PointReflector(0., xp.asarray(.05*np.exp(.2j*core)), -.1, .8j, .7j),
                    PointReflector(1., xp.asarray(.1*np.exp(-.3j*core)), -.2, .9, .8)]
        cores.append(Fiber(f"core-{core}", 1.25, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                            [section], elements, sample))
    return MultiCoreFiber(cores).with_crosstalk(42, xp.asarray(-30.), topology=topology, xp=xp)


@pytest.mark.parametrize("count,topology", [(1, "linear"), (3, "linear"), (7, "hex7")])
@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
def test_point_returns_use_their_grid_phases_and_keep_source_data(xp, count, topology, shape):
    model = make_model(xp, count, topology)
    zero = xp.asarray(0., dtype=xp.complex128)
    profile = xp.stack(tuple(xp.stack((core.elements[0].r_left, zero, core.elements[1].r_left))
                              for core in model.cores))
    point_returns = model.crosstalk.mix(profile, model.xt_db_per_km)
    rayleigh = model.crosstalk.mix_rayleigh(tuple(core.rayleigh for core in model.cores), model.xt_db_per_km)
    probe = Probe(xp.full(shape, REFERENCE, dtype=xp.float64))
    phases = np.asarray(model.crosstalk.phases).copy()
    source_returns = [[np.asarray(point.r_left).copy() for point in core.elements] for core in model.cores]
    source_rayleigh = [np.asarray(core.rayleigh.coefficients).copy() for core in model.cores]
    actual = compute_response(model, probe)
    for core, source in enumerate(model.cores):
        points = (replace(source.elements[0], r_left=point_returns[core, 0]),
                  replace(source.elements[1], r_left=point_returns[core, 2]))
        expected = compute_response(replace(source, elements=points, rayleigh=rayleigh[core]), probe)
        direct = compute_response(source, probe)
        np.testing.assert_array_equal(np.asarray(actual.reflection[core, ...]), np.asarray(expected.reflection))
        np.testing.assert_array_equal(np.asarray(actual.transmission[core, ...]), np.asarray(direct.transmission))
        for point, original in zip(source.elements, source_returns[core]):
            np.testing.assert_array_equal(np.asarray(point.r_left), original)
        np.testing.assert_array_equal(np.asarray(source.rayleigh.coefficients), source_rayleigh[core])
    assert actual.shape == (count,) + shape and actual.probe is probe
    assert actual.reflection.dtype == xp.complex128 and type(actual.reflection) is type(probe.frequency)
    np.testing.assert_array_equal(np.asarray(model.crosstalk.phases), phases)


@pytest.mark.parametrize("rayleigh", [False, True])
def test_a_point_and_coincident_rayleigh_see_the_same_coupling_but_different_self_transmission(xp, rayleigh):
    model = make_model(xp, rayleigh=rayleigh)
    values = ([.01, 0., .02j], [.02, 0., .03]) if rayleigh else ([0., 0., 0.], [0., 0., 0.])
    left = [.1 + .02j, .2 - .03j]
    cores = [replace(core, elements=[PointReflector(1., xp.asarray(left[index]), -.1, .8, .7)],
                      rayleigh=RayleighRealization(xp.asarray(values[index]), POSITIONS, EDGES))
             for index, core in enumerate(model.cores)]
    model = replace(model, cores=cores, crosstalk=CrosstalkRealization(2, POSITIONS, xp.zeros((2, 3))))
    probe = Probe(xp.asarray([REFERENCE - 1e9, REFERENCE + 1e9]))
    response = compute_response(model, probe)
    q = np.asarray(optical_path(cores[0]).round_trip(0., 1., probe.frequency))
    for core in range(2):
        other = 1 - core
        point = left[core] + .001*left[other]
        local_rayleigh = values[core][2] + .001*values[other][2]
        expected = values[core][0] + (point + .8*.7*local_rayleigh)*q
        np.testing.assert_allclose(np.asarray(response.reflection[core, ...]), expected, rtol=1e-13, atol=1e-16)


def test_common_time_varying_state_is_applied_after_mixing_point_returns(xp):
    model = make_model(xp)
    history = TimeHistory([0., 1.], xp.asarray([0., 1.]))
    state = FiberState(axial_strain=Field([FieldTerm(
        SpatialProfile([0., 1.25], xp.asarray([1e-6, 2e-6])), history)]))
    probe = Probe(xp.asarray([[REFERENCE, REFERENCE + 1e9]]), xp.asarray([[.2, .8]]))
    actual = compute_response(model, probe, state)
    for column in range(2):
        frozen = state.at_time(probe.time[0, column], xp=xp)
        expected = compute_response(model, Probe(probe.frequency[0, column]), frozen)
        np.testing.assert_array_equal(np.asarray(actual.reflection[:, 0, column]), np.asarray(expected.reflection))
    again = compute_response(model, probe, state)
    np.testing.assert_array_equal(np.asarray(again.reflection), np.asarray(actual.reflection))


@pytest.mark.parametrize("name", ["t_left_to_right", "t_right_to_left"])
def test_different_native_directional_transmissions_give_nan(xp, name):
    model = make_model(xp)
    first, last = model.cores
    points = list(last.elements)
    points[1] = replace(points[1], **{name: xp.asarray(.5j)})
    model = replace(model, cores=[first, replace(last, elements=points)])
    result = compute_response(model, Probe(xp.asarray(REFERENCE)))
    assert np.all(np.isnan(np.asarray(result.reflection)))
    assert np.all(np.isnan(np.asarray(result.transmission)))


@pytest.mark.parametrize("kind", ["different_positions", "different_count", "off_grid", "ambiguous_grid"])
@pytest.mark.parametrize("empty", [False, True])
def test_fixed_point_layout_and_phase_columns_are_checked(kind, empty):
    model = make_model()
    first, last = model.cores
    if kind == "different_positions":
        last = replace(last, elements=[last.elements[0], replace(last.elements[1], position=.5)])
    elif kind == "different_count":
        last = replace(last, elements=last.elements[:1])
    elif kind == "off_grid":
        first = replace(first, elements=[replace(first.elements[1], position=.75)])
        last = replace(last, elements=[replace(last.elements[1], position=.75)])
    else:
        positions, edges = [0., .5, .5], [0., .25, .5, 1.25]
        first = replace(first, elements=[replace(first.elements[1], position=.5)],
                        rayleigh=RayleighRealization([.01, .02, .03], positions, edges))
        last = replace(last, elements=[replace(last.elements[1], position=.5)],
                       rayleigh=RayleighRealization([.02, .03, .04], positions, edges))
        model = replace(model, cores=[first, last], crosstalk=CrosstalkRealization(2, positions, np.zeros((2, 3))))
    model = replace(model, cores=[first, last])
    with pytest.raises(NotImplementedError, match="positions"):
        compute_response(model, Probe([]) if empty else Probe(REFERENCE))


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_multiple_point_paths_remain_explicitly_unsupported_with_crosstalk(order, empty):
    with pytest.raises(NotImplementedError, match="max_order=1"):
        compute_response(make_model(), Probe([]) if empty else Probe(REFERENCE), max_order=order)


def test_point_coefficients_are_included_in_native_namespace_selection(jax_numpy):
    model = make_model()
    first, last = model.cores
    points = list(last.elements)
    points[1] = replace(points[1], r_left=jax_numpy.asarray(.1j))
    model = replace(model, cores=[first, replace(last, elements=points)])
    probe = Probe(REFERENCE)
    with pytest.raises(TypeError):
        compute_response(model, probe)
    response = compute_response(model, probe, xp=jax_numpy)
    assert type(response.reflection) is type(points[1].r_left)


def test_jax_point_returns_coupling_and_common_transmission_keep_gradients_and_hessians(jax_numpy):
    import jax
    length = SPEED_OF_LIGHT/(4.*INDEX*REFERENCE)
    weights = jax_numpy.asarray([[1., 2.], [3., 4.]])
    def objective(parameters):
        level, amplitude, angle, forward, depth, strain, time = parameters
        guide = GuideOptics(REFERENCE, INDEX, INDEX)
        section = Section(0., length, guide, depth/length, 0.)
        cores = []
        for core, reflection in enumerate((jax_numpy.asarray(.1), .2j*amplitude)):
            sample = RayleighRealization(jax_numpy.zeros((2,)), [0., length], [0., length/2., length])
            point = PointReflector(length, reflection, -.1, forward, .7)
            cores.append(Fiber(f"core-{core}", length, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                                [section], [point], sample))
        ones = jax_numpy.ones((2,))
        coupling = CrosstalkRealization(2, [0., length], jax_numpy.stack((angle*ones, 2.*angle*ones)))
        model = MultiCoreFiber(cores, coupling, level)
        profile = SpatialProfile([0., length], .01*strain*ones)
        history = TimeHistory([0., 2.], jax_numpy.asarray([2., 6.]))
        state = FiberState(axial_strain=Field([FieldTerm(profile, history)]))
        probe = Probe(jax_numpy.full((2,), REFERENCE, dtype=jax_numpy.float64),
                       jax_numpy.stack((time, 2.*time)))
        result = compute_response(model, probe, state)
        return jax_numpy.sum(weights*(jax_numpy.abs(result.reflection)**2 + .05*jax_numpy.real(result.reflection)
                                      + .1*jax_numpy.real(result.transmission)))
    def expected(parameters):
        level, amplitude, angle, forward, depth, strain, time = parameters
        phase = jax_numpy.pi*.78*.01*strain*2.*(1. + jax_numpy.stack((time, 2.*time)))
        c = 10.**(level/20.)*np.sqrt(length/1000.)
        q = -jax_numpy.exp(-depth - 1j*phase)
        left = .1 + c*.2j*amplitude*jax_numpy.exp(1j*angle)
        right = .2j*amplitude + c*.1*jax_numpy.exp(2j*angle)
        reflection = jax_numpy.stack((left*q, right*q))
        transmission = -1j*forward*jax_numpy.exp(-.5*depth - .5j*phase)
        return jax_numpy.sum(weights*(jax_numpy.abs(reflection)**2 + .05*jax_numpy.real(reflection)
                                      + .1*jax_numpy.real(transmission)))
    parameters = jax_numpy.asarray([20., 1.2, .4, .8, .2, 1., .4])
    for transform in (lambda function: function, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(objective))(parameters),
                                   jax.jit(transform(expected))(parameters), rtol=2e-10, atol=1e-13)
