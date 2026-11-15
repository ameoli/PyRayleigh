from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Field, FieldTerm, Fiber, FiberState, GuideOptics, MultiCoreFiber,
                        MultiCoreResponse, PointReflector, Probe, RayleighRealization,
                        Section, SpatialProfile, TimeHistory, WeakFBG, compute_response)
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


def make_model(xp=numpy_api, count=2):
    cores = []
    for core in range(count):
        index = xp.asarray(INDEX + .001*core)
        guide = GuideOptics(REFERENCE, index, INDEX + .005*core)
        rayleigh = RayleighRealization(xp.asarray([.01, .02j])*(core + 1), [0., 1.], [0., 1., 1.25])
        elements = [PointReflector(0., .1, -.2, .8, .7),
                    WeakFBG(.75, 1550e-9, .005, .01),
                    PointReflector(1.25, .2, -.1, .9, .85)]
        cores.append(Fiber(f"core-{core}", 1.25, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                            [Section(0., 1.25, guide, .1*(core + 1), 0.)], elements, rayleigh))
    return MultiCoreFiber(cores)


def make_state(value, xp, moving=True):
    profile = SpatialProfile([0., 1.25], xp.asarray([value, value]))
    history = TimeHistory([0., 1.], xp.asarray([0., 1.])) if moving else None
    return FiberState(axial_strain=Field([FieldTerm(profile, history)]))


@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
@pytest.mark.parametrize("order", [1, 3])
def test_core_axis_and_fields_match_the_existing_single_core_solver(xp, shape, order):
    model = make_model(xp)
    model = replace(model, cores=tuple(reversed(model.cores)))
    probe = Probe(xp.full(shape, REFERENCE, dtype=xp.float64))
    response = compute_response(model, probe, max_order=order)
    assert isinstance(response, MultiCoreResponse)
    assert response.probe is probe and response.n_cores == 2
    assert response.shape == (2,) + shape
    for core, fiber in enumerate(model.cores):
        expected = compute_response(fiber, probe, max_order=order)
        for name in ("reflection", "transmission"):
            actual = getattr(response, name)
            assert actual.dtype == xp.complex128 and type(actual) is type(probe.frequency)
            np.testing.assert_array_equal(np.asarray(actual[core, ...]), np.asarray(getattr(expected, name)))


def test_single_core_multicore_model_retains_its_core_axis(xp):
    model = make_model(xp, count=1)
    probe = Probe(xp.asarray([REFERENCE, REFERENCE + 1e9]))
    response = compute_response(model, probe)
    assert response.shape == (1, 2) and response.n_cores == 1
    assert repr(response) == "MultiCoreResponse(shape=(1, 2))"


def test_independent_cores_can_have_different_rayleigh_grids_or_no_sample(xp):
    model = make_model(xp, count=3)
    cores = list(model.cores)
    cores[0] = replace(cores[0], rayleigh=None)
    cores[1] = replace(cores[1], rayleigh=RayleighRealization(
        xp.asarray([.01j, .02, -.03j]), [0., .5, 1.], [0., .5, 1., 1.25]))
    model = replace(model, cores=cores)
    probe = Probe(xp.asarray([REFERENCE, REFERENCE + 1e9]))
    result = compute_response(model, probe)
    for core, fiber in enumerate(model.cores):
        expected = compute_response(fiber, probe)
        np.testing.assert_array_equal(np.asarray(result.reflection[core, ...]), np.asarray(expected.reflection))
        np.testing.assert_array_equal(np.asarray(result.transmission[core, ...]), np.asarray(expected.transmission))


@pytest.mark.parametrize("kind", ["shared", "per_core", "with_default"])
def test_shared_and_per_core_states_follow_the_same_probe_times(xp, kind):
    model = make_model(xp)
    moving = make_state(2e-6, xp)
    if kind == "shared":
        state, states = moving, (moving, moving)
    elif kind == "per_core":
        states = (make_state(-1e-6, xp, moving=False), moving)
        state = (item for item in states)
    else:
        state = states = (None, moving)
    probe = Probe(xp.asarray([[REFERENCE, REFERENCE + 1e9], [REFERENCE - 1e9, REFERENCE]]),
                   xp.asarray([[.1, .4], [.7, .9]]))
    response = compute_response(model, probe, state, max_order=3)
    for core, (fiber, item) in enumerate(zip(model.cores, states)):
        expected = compute_response(fiber, probe, item, max_order=3)
        np.testing.assert_array_equal(np.asarray(response.reflection[core, ...]), np.asarray(expected.reflection))
        np.testing.assert_array_equal(np.asarray(response.transmission[core, ...]), np.asarray(expected.transmission))
    np.testing.assert_array_equal(np.asarray(moving.axial_strain.terms[0].history.values), [0., 1.])


def test_manual_response_owns_native_field_copies_and_keeps_the_probe():
    probe = Probe([REFERENCE, REFERENCE + 1e9])
    reflected, transmitted = np.ones((2, 2), dtype=np.float32), np.full((2, 2), 1j)
    response = MultiCoreResponse(probe, reflection=reflected, transmission=transmitted)
    reflected[:] = 0.
    transmitted[:] = 0.
    assert response.probe is probe and response.reflection.dtype == np.complex128
    np.testing.assert_array_equal(response.reflection, np.ones((2, 2)))
    np.testing.assert_array_equal(response.transmission, np.full((2, 2), 1j))


@pytest.mark.parametrize("r_shape,t_shape", [((), ()), ((2,), (2,)), ((0, 2), (0, 2)),
                                              ((2, 1, 2), (2, 1, 2)), ((2, 3), (2, 3)),
                                              ((2, 2), (1, 2))])
def test_manual_fields_require_a_nonempty_core_axis_and_the_probe_shape(r_shape, t_shape):
    with pytest.raises(ValueError, match="shape"):
        MultiCoreResponse(Probe([REFERENCE, REFERENCE]),
                           reflection=np.zeros(r_shape), transmission=np.zeros(t_shape))


def test_manual_multicore_response_requires_a_probe():
    with pytest.raises(TypeError, match="Probe"):
        MultiCoreResponse(None, reflection=np.zeros((2,)), transmission=np.zeros((2,)))


@pytest.mark.parametrize("empty", [False, True])
def test_active_crosstalk_is_rejected_instead_of_ignored_even_for_empty_probes(empty):
    model = make_model().with_crosstalk(42, -30.)
    probe = Probe([]) if empty else Probe(REFERENCE)
    with pytest.raises(NotImplementedError, match="active crosstalk"):
        compute_response(model, probe)


@pytest.mark.parametrize("state,error", [([], ValueError), ([None], ValueError),
                                         ([None, None, None], ValueError),
                                         ([None, object()], TypeError), (3, TypeError), ("xx", TypeError)])
def test_per_core_state_count_and_types_are_checked(state, error):
    with pytest.raises(error, match="state"):
        compute_response(make_model(), Probe(REFERENCE), state)


@pytest.mark.parametrize("empty", [False, True])
def test_any_moving_core_requires_probe_times_even_for_empty_queries(empty):
    probe = Probe([]) if empty else Probe(REFERENCE)
    with pytest.raises(ValueError, match="probe.time"):
        compute_response(make_model(), probe, (None, make_state(1e-6, numpy_api)))


@pytest.mark.parametrize("order,error", [(True, TypeError), (2., TypeError), (0, ValueError), (4, ValueError)])
def test_order_is_checked_before_evaluating_multicore_fields(order, error):
    with pytest.raises(error, match="max_order"):
        compute_response(make_model(), Probe([]), max_order=order)


def test_mixed_core_backends_need_an_explicit_namespace_and_keep_the_original_probe(jax_numpy):
    model = make_model()
    last = model.cores[-1]
    guide = replace(last.sections[0].optics, effective_index=jax_numpy.asarray(INDEX + .001))
    last = replace(last, sections=[replace(last.sections[0], optics=guide)])
    model = replace(model, cores=[model.cores[0], last])
    probe = Probe([REFERENCE, REFERENCE + 1e9])
    with pytest.raises(TypeError):
        compute_response(model, probe)
    response = compute_response(model, probe, xp=jax_numpy)
    assert response.probe is probe and type(response.reflection) is type(guide.effective_index)
    assert response.reflection.dtype == jax_numpy.complex128


def test_jax_core_states_histories_and_probe_times_keep_gradients_and_hessians(jax_numpy):
    import jax
    length = SPEED_OF_LIGHT/(4.*INDEX*REFERENCE)
    weights = jax_numpy.asarray([[1., 2.], [3., 4.]])
    def objective(parameters):
        amplitude, history_scale, time = parameters
        guide = GuideOptics(REFERENCE, INDEX, INDEX)
        cores, states = [], []
        for core in range(2):
            profile = SpatialProfile([0., length], jax_numpy.ones((2,))*.01*(amplitude + core))
            history = TimeHistory([0., 2.], jax_numpy.stack((history_scale, 3.*history_scale)))
            states.append(FiberState(axial_strain=Field([FieldTerm(profile, history)])))
            sample = RayleighRealization(jax_numpy.asarray([.1]), [0.], [0., length])
            cores.append(Fiber(f"core-{core}", length, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                                [Section(0., length, guide, 0., 0.)],
                                [PointReflector(length, .2*(core + 1), 0., .8, .7)], sample))
        probe = Probe(jax_numpy.full((2,), REFERENCE, dtype=jax_numpy.float64),
                       jax_numpy.stack((time, 2.*time)))
        response = compute_response(MultiCoreFiber(cores), probe, states)
        return jax_numpy.sum(weights*(jax_numpy.abs(response.reflection)**2
                                      + .1*jax_numpy.real(response.transmission)))
    def expected(parameters):
        amplitude, history_scale, time = parameters
        phase = (jax_numpy.pi*.78*.01*jax_numpy.stack((amplitude, amplitude + 1.))[:, None]
                 * history_scale*(1. + jax_numpy.stack((time, 2.*time)))[None, :])
        q = -jax_numpy.exp(-1j*phase)
        reflection = .1 + jax_numpy.asarray([.2, .4])[:, None]*q
        transmission = -.8j*jax_numpy.exp(-.5j*phase)
        return jax_numpy.sum(weights*(jax_numpy.abs(reflection)**2 + .1*jax_numpy.real(transmission)))
    parameters = jax_numpy.asarray([1., 2., .4])
    for transform in (lambda function: function, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(objective))(parameters),
                                   jax.jit(transform(expected))(parameters), rtol=2e-10, atol=1e-13)
