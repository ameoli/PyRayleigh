from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (BendLossTable, CrosstalkRealization, Field, FieldTerm, Fiber,
                        FiberState, GuideOptics, MultiCoreFiber, Probe, RayleighRealization,
                        Section, SpatialProfile, TimeHistory, compute_response)
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


def make_model(xp=numpy_api, count=2, topology="linear"):
    cores = []
    for core in range(count):
        guide = GuideOptics(REFERENCE, xp.asarray(INDEX), xp.asarray(INDEX))
        sample = RayleighRealization(xp.asarray([.01, .02j, -.01])*(core + 1), POSITIONS, EDGES)
        cores.append(Fiber(f"core-{core}", 1.25, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                            [Section(0., 1.25, guide, xp.asarray(.2), 0.)], rayleigh=sample))
    return MultiCoreFiber(cores).with_crosstalk(42, xp.asarray(-30.), topology=topology, xp=xp)


@pytest.mark.parametrize("shape", [(), (2, 2), (0, 2)])
@pytest.mark.parametrize("count,topology", [(1, "linear"), (3, "linear"), (7, "hex7")])
def test_coupled_response_mixes_local_coefficients_before_the_common_path(xp, count, topology, shape):
    model = make_model(xp, count, topology)
    saved = [np.asarray(core.rayleigh.coefficients).copy() for core in model.cores]
    phases = np.asarray(model.crosstalk.phases).copy()
    probe = Probe(xp.full(shape, REFERENCE, dtype=xp.float64))
    actual = compute_response(model, probe, max_order=3)
    mixed = model.crosstalk.mix_rayleigh(tuple(core.rayleigh for core in model.cores), model.xt_db_per_km)
    for core, sample in enumerate(mixed):
        expected = compute_response(replace(model.cores[core], rayleigh=sample), probe, max_order=3)
        np.testing.assert_array_equal(np.asarray(actual.reflection[core, ...]), np.asarray(expected.reflection))
        direct = compute_response(model.cores[core], probe)
        np.testing.assert_array_equal(np.asarray(actual.transmission[core, ...]), np.asarray(direct.transmission))
        np.testing.assert_array_equal(np.asarray(model.cores[core].rayleigh.coefficients), saved[core])
    assert actual.shape == (count,) + shape and actual.probe is probe
    assert actual.reflection.dtype == xp.complex128 and type(actual.reflection) is type(probe.frequency)
    np.testing.assert_array_equal(np.asarray(model.crosstalk.phases), phases)


def test_shared_dynamic_state_bend_loss_and_index_profile_keep_sample_times(xp):
    model = make_model(xp)
    table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.03]]))
    index = SpatialProfile([0., 1.25], xp.asarray([1e-6, 2e-6]))
    cores = [replace(core, sections=[replace(core.sections[0], optics=replace(core.sections[0].optics,
                                                                           bend_loss=table))],
                      index_perturbation=index) for core in model.cores]
    model = replace(model, cores=cores)
    history = TimeHistory([0., 1.], xp.asarray([.8, 1.2]))
    state = FiberState(
        axial_strain=Field([FieldTerm(SpatialProfile([0., 1.25], xp.asarray([2e-6, 2e-6])), history)]),
        temperature_change=Field([FieldTerm(SpatialProfile([0., 1.25], xp.asarray([1., 2.])), history)]),
        curvature_x=Field([FieldTerm(SpatialProfile([0., 1.25], xp.asarray([5., 5.])), history)]))
    probe = Probe(xp.asarray([[REFERENCE, REFERENCE + 1e9], [REFERENCE - 1e9, REFERENCE]]),
                   xp.asarray([[.1, .4], [.7, .9]]))
    response = compute_response(model, probe, (state, state))
    mixed = model.crosstalk.mix_rayleigh(tuple(core.rayleigh for core in cores), model.xt_db_per_km)
    for core, sample in enumerate(mixed):
        expected = compute_response(replace(cores[core], rayleigh=sample), probe, state)
        np.testing.assert_allclose(np.asarray(response.reflection[core, ...]), np.asarray(expected.reflection),
                                   rtol=1e-13, atol=1e-16)
        np.testing.assert_array_equal(np.asarray(response.transmission[core, ...]), np.asarray(expected.transmission))
    again = compute_response(model, probe, state)
    np.testing.assert_array_equal(np.asarray(again.reflection), np.asarray(response.reflection))
    np.testing.assert_array_equal(np.asarray(state.axial_strain.terms[0].history.values), [.8, 1.2])


@pytest.mark.parametrize("name", ["effective_index", "group_index", "gvd", "strain_phase_factor",
                                   "strain_group_factor", "temperature_phase_factor", "temperature_group_factor",
                                   "frequency_ref", "attenuation"])
def test_unequal_native_path_parameters_give_nan_fields(name):
    model = make_model()
    last = model.cores[-1]
    section = last.sections[0]
    if name == "attenuation":
        section = replace(section, attenuation=np.array(.3))
    else:
        old = getattr(section.optics, name)
        change = 1e9 if name == "frequency_ref" else .01
        section = replace(section, optics=replace(section.optics, **{name: np.asarray(old) + change}))
    model = replace(model, cores=[model.cores[0], replace(last, sections=[section])])
    response = compute_response(model, Probe(REFERENCE))
    assert np.all(np.isnan(response.reflection)) and np.all(np.isnan(response.transmission))


@pytest.mark.parametrize("kind", ["index", "bend"])
def test_unequal_native_profile_or_bend_table_values_give_nan(xp, kind):
    model = make_model(xp)
    cores = []
    for core, original in enumerate(model.cores):
        if kind == "index":
            profile = SpatialProfile([0., 1.25], xp.asarray([1e-6, (1. + core)*1e-6]))
            cores.append(replace(original, index_perturbation=profile))
        else:
            table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.03 + .01*core]]))
            guide = replace(original.sections[0].optics, bend_loss=table)
            cores.append(replace(original, sections=[replace(original.sections[0], optics=guide)]))
    result = compute_response(replace(model, cores=cores), Probe(xp.asarray(REFERENCE)))
    assert np.all(np.isnan(np.asarray(result.reflection)))
    assert np.all(np.isnan(np.asarray(result.transmission)))


@pytest.mark.parametrize("kind", ["sections", "index_presence", "index_knots", "bend_presence", "bend_grid"])
@pytest.mark.parametrize("empty", [False, True])
def test_different_fixed_path_layouts_are_explicitly_unsupported(kind, empty):
    model = make_model()
    first, last = model.cores
    section = last.sections[0]
    if kind == "sections":
        last = replace(last, sections=[replace(section, end=.5), replace(section, start=.5)])
    elif kind.startswith("index"):
        first = replace(first, index_perturbation=SpatialProfile([0., 1.25], [0., 0.]))
        if kind == "index_knots":
            last = replace(last, index_perturbation=SpatialProfile([0., .5, 1.25], [0., 0., 0.]))
    else:
        table = BendLossTable([3., 7.], [REFERENCE], [[.01], [.03]])
        first = replace(first, sections=[replace(first.sections[0], optics=replace(first.sections[0].optics,
                                                                                 bend_loss=table))])
        if kind == "bend_grid":
            other = BendLossTable([3., 8.], [REFERENCE], [[.01], [.03]])
            last = replace(last, sections=[replace(section, optics=replace(section.optics, bend_loss=other))])
    probe = Probe([]) if empty else Probe(REFERENCE)
    with pytest.raises(NotImplementedError, match="common"):
        compute_response(replace(model, cores=[first, last]), probe)


@pytest.mark.parametrize("states", [(None, FiberState()), (FiberState(), FiberState())])
def test_coupled_cores_require_one_shared_state(states):
    with pytest.raises(NotImplementedError, match="shared FiberState"):
        compute_response(make_model(), Probe(REFERENCE), states)


@pytest.mark.parametrize("empty", [False, True])
def test_a_shared_moving_state_still_requires_probe_times(empty):
    profile = SpatialProfile([0., 1.25], [1e-6, 1e-6])
    history = TimeHistory([0., 1.], [0., 1.])
    state = FiberState(axial_strain=Field([FieldTerm(profile, history)]))
    with pytest.raises(ValueError, match="probe.time"):
        compute_response(make_model(), Probe([]) if empty else Probe(REFERENCE), state)


def test_coupling_data_and_probe_namespace_require_explicit_conversion(jax_numpy):
    model = make_model(jax_numpy)
    probe = Probe([REFERENCE, REFERENCE + 1e9])
    with pytest.raises(TypeError):
        compute_response(model, probe)
    response = compute_response(model, probe, xp=jax_numpy)
    assert type(response.reflection) is type(model.crosstalk.phases) and response.probe is probe


def test_jit_compares_path_parameters_without_reading_their_values_on_the_host(jax_numpy):
    import jax
    model = make_model(jax_numpy)
    probe = Probe(jax_numpy.asarray(REFERENCE))
    def reflection(delta):
        first, last = model.cores
        guide = replace(last.sections[0].optics, effective_index=INDEX + delta*1e-4)
        last = replace(last, sections=[replace(last.sections[0], optics=guide)])
        return compute_response(replace(model, cores=[first, last]), probe).reflection
    evaluated = jax.jit(reflection)
    assert np.all(np.isfinite(np.asarray(evaluated(jax_numpy.asarray(0.)))))
    assert np.all(np.isnan(np.asarray(evaluated(jax_numpy.asarray(1.)))))


def test_jax_coupling_loss_and_shared_history_keep_gradients_and_hessians(jax_numpy):
    import jax
    length = SPEED_OF_LIGHT/(4.*INDEX*REFERENCE)
    weights = jax_numpy.asarray([[1., 2.], [3., 4.]])
    def power(parameters):
        level, amplitude, angle, depth, strain, time = parameters
        guide = GuideOptics(REFERENCE, INDEX, INDEX)
        section = Section(0., length, guide, depth/length, 0.)
        samples = (RayleighRealization(jax_numpy.asarray([.01, .02j]), [0., length], [0., length/2., length]),
                   RayleighRealization(amplitude*jax_numpy.asarray([-.03j, .04]), [0., length],
                                       [0., length/2., length]))
        cores = [Fiber(f"core-{core}", length, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                        [section], rayleigh=sample) for core, sample in enumerate(samples)]
        ones = jax_numpy.ones((2,))
        coupling = CrosstalkRealization(2, [0., length], jax_numpy.stack((angle*ones, 2.*angle*ones)))
        model = MultiCoreFiber(cores, coupling, level)
        profile = SpatialProfile([0., length], strain*.01*ones)
        history = TimeHistory([0., 2.], jax_numpy.asarray([2., 6.]))
        state = FiberState(axial_strain=Field([FieldTerm(profile, history)]))
        probe = Probe(jax_numpy.full((2,), REFERENCE, dtype=jax_numpy.float64),
                       jax_numpy.stack((time, 2.*time)))
        response = compute_response(model, probe, state)
        return jax_numpy.sum(weights*(jax_numpy.abs(response.reflection)**2
                                      + .1*jax_numpy.real(response.transmission)))
    def expected(parameters):
        level, amplitude, angle, depth, strain, time = parameters
        phase = jax_numpy.pi*.78*.01*strain*2.*(1. + jax_numpy.stack((time, 2.*time)))
        c = 10.**(level/20.)*np.sqrt(length/1000.)
        q = -jax_numpy.exp(-depth - 1j*phase)
        r0 = .01 + (.02j + c*.04*amplitude*jax_numpy.exp(1j*angle))*q
        r1 = -.03j*amplitude + (.04*amplitude + c*.02j*jax_numpy.exp(2j*angle))*q
        reflection = jax_numpy.stack((r0, r1))
        transmission = -1j*jax_numpy.exp(-.5*depth - .5j*phase)
        return jax_numpy.sum(weights*(jax_numpy.abs(reflection)**2 + .1*jax_numpy.real(transmission)))
    parameters = jax_numpy.asarray([20., 1.2, .4, .2, 1., .4])
    for transform in (lambda function: function, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(power))(parameters),
                                   jax.jit(transform(expected))(parameters), rtol=2e-10, atol=1e-15)
