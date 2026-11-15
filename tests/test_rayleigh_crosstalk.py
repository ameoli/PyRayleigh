from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (CrosstalkRealization, Field, FieldTerm, Fiber, FiberState,
                        GuideOptics, Probe, RayleighRealization, Section, SpatialProfile,
                        TimeHistory, compute_response, generate_crosstalk, optical_path)

POSITIONS, EDGES = [0., .5, 1.], [0., .5, 1., 1.25]
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


def make_samples(count, xp=numpy_api):
    return tuple(RayleighRealization(xp.asarray(.01*(np.arange(3) + 1 + 3*core)*(1. + .2j)),
                                     POSITIONS, EDGES) for core in range(count))


@pytest.mark.parametrize("count,topology", [(1, "linear"), (3, "linear"), (7, "hex7")])
def test_mixing_rayleigh_samples_preserves_cells_and_native_coefficients(xp, count, topology):
    sources = make_samples(count, xp)
    coupling = generate_crosstalk(count, POSITIONS, 42, topology=topology, xp=xp)
    original = np.stack([np.asarray(source.coefficients).copy() for source in sources])
    expected = coupling.mix(xp.stack(tuple(source.coefficients for source in sources)), -30.)
    mixed = coupling.mix_rayleigh(iter(sources), xp.asarray(-30.))
    assert isinstance(mixed, tuple) and len(mixed) == count
    for core, sample in enumerate(mixed):
        assert isinstance(sample, RayleighRealization) and sample is not sources[core]
        assert sample.coefficients.dtype == xp.complex128
        assert type(sample.coefficients) is type(sources[core].coefficients)
        np.testing.assert_array_equal(np.asarray(sample.coefficients), np.asarray(expected[core, :]))
        np.testing.assert_array_equal(sample.positions, POSITIONS)
        np.testing.assert_array_equal(sample.edges, EDGES)
        np.testing.assert_array_equal(sample.cell_lengths(), [.5, .5, .25])
        np.testing.assert_array_equal(np.asarray(sources[core].coefficients), original[core])
        assert not sample.positions.flags.writeable and not sample.edges.flags.writeable


def test_repeated_calls_use_original_sources_and_return_owned_numpy_arrays():
    sources = make_samples(2)
    coupling = generate_crosstalk(2, POSITIONS, 42)
    phases = coupling.phases.copy()
    original = [source.coefficients.copy() for source in sources]
    first = coupling.mix_rayleigh(sources, -30.)
    other = coupling.mix_rayleigh(sources, -20.)
    repeated = coupling.mix_rayleigh(sources, -30.)
    for core in range(2):
        np.testing.assert_array_equal(first[core].coefficients, repeated[core].coefficients)
        np.testing.assert_array_equal(sources[core].coefficients, original[core])
    saved = other[1].coefficients.copy()
    first[0].coefficients[:] = 0.
    np.testing.assert_array_equal(other[1].coefficients, saved)
    np.testing.assert_array_equal(sources[0].coefficients, original[0])
    np.testing.assert_array_equal(coupling.phases, phases)


def test_mixed_samples_drive_the_existing_solver_with_a_common_varying_core_path(xp):
    positions, edges = [0., 1.], [0., 1., 1.25]
    sources = (RayleighRealization(xp.asarray([.01, .02j]), positions, edges),
               RayleighRealization(xp.asarray([-.03j, .04]), positions, edges))
    coupling = CrosstalkRealization(2, positions, xp.zeros((2, 2)))
    mixed = coupling.mix_rayleigh(sources, -30.)
    guide = GuideOptics(REFERENCE, INDEX, INDEX)
    fiber = Fiber("common", 1.25, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                  [Section(0., 1.25, guide, .2, 0.)])
    profile = SpatialProfile([0., 1.25], xp.asarray([2e-6, 2e-6]))
    history = TimeHistory([0., 1.], xp.asarray([0., 1.]))
    state = FiberState(axial_strain=Field([FieldTerm(profile, history)]))
    probe = Probe(xp.asarray([REFERENCE - 1e9, REFERENCE + 1e9]), xp.asarray([.2, .8]))
    responses = [compute_response(replace(fiber, rayleigh=sample), probe, state) for sample in mixed]
    local = ((.01, .02j + .00004), (-.03j, .04 + .00002j))
    for core, response in enumerate(responses):
        for index in range(2):
            frozen = state.at_time(probe.time[index], xp=xp)
            q = optical_path(fiber, frozen).round_trip(0., 1., probe.frequency[index], xp=xp)
            expected = local[core][0] + local[core][1]*q
            np.testing.assert_allclose(np.asarray(response.reflection[index]), np.asarray(expected),
                                       rtol=1e-13, atol=1e-16)
    np.testing.assert_array_equal(np.asarray(responses[0].transmission), np.asarray(responses[1].transmission))


@pytest.mark.parametrize("count", [0, 1, 3])
def test_the_number_of_rayleigh_samples_matches_the_stored_cores(count):
    coupling = generate_crosstalk(2, POSITIONS, 42)
    with pytest.raises(ValueError, match="one Rayleigh"):
        coupling.mix_rayleigh(make_samples(count), -30.)


def test_only_rayleigh_realizations_can_be_mixed():
    coupling = generate_crosstalk(2, POSITIONS, 42)
    with pytest.raises(TypeError, match="RayleighRealization"):
        coupling.mix_rayleigh([make_samples(1)[0], np.ones(3)], -30.)


@pytest.mark.parametrize("positions,edges", [([0., .75, 1.], EDGES),
                                             (POSITIONS, [0., .5, 1., 1.5]),
                                             (POSITIONS, [0., .25, 1., 1.25])])
def test_core_samples_must_share_both_positions_and_cell_edges(positions, edges):
    sources = list(make_samples(2))
    sources[1] = RayleighRealization([.01, .02, .03], positions, edges)
    coupling = generate_crosstalk(2, POSITIONS, 42)
    with pytest.raises(ValueError, match="share positions and cell edges"):
        coupling.mix_rayleigh(sources, -30.)


def test_the_crosstalk_positions_must_match_the_rayleigh_grid():
    coupling = generate_crosstalk(2, [0., .25, 1.], 42)
    with pytest.raises(ValueError, match="crosstalk positions"):
        coupling.mix_rayleigh(make_samples(2), -30.)


@pytest.mark.parametrize("native", ["phases", "last_core"])
def test_all_core_coefficients_and_phases_participate_in_namespace_selection(jax_numpy, native):
    sources = list(make_samples(2))
    coupling = generate_crosstalk(2, POSITIONS, 42, xp=jax_numpy if native == "phases" else numpy_api)
    if native == "last_core":
        source = sources[-1]
        sources[-1] = RayleighRealization(jax_numpy.asarray(source.coefficients), source.positions, source.edges)
    with pytest.raises(TypeError):
        coupling.mix_rayleigh(sources, -30.)
    mixed = coupling.mix_rayleigh(sources, -30., xp=jax_numpy)
    for sample in mixed:
        assert type(sample.coefficients) is type(jax_numpy.asarray(0.))
        assert sample.coefficients.dtype == jax_numpy.complex128


def test_jax_rayleigh_coefficients_level_and_phases_keep_jit_gradients_and_hessians(jax_numpy):
    import jax
    def power(parameters):
        level, amplitude, angle = parameters
        ones = jax_numpy.ones((2,))
        sources = (RayleighRealization(.01*ones, [0., 1000.], [0., 1000., 1250.]),
                   RayleighRealization(.01*amplitude*ones, [0., 1000.], [0., 1000., 1250.]))
        phases = jax_numpy.stack((angle*ones, 2.*angle*ones))
        coupling = CrosstalkRealization(2, [0., 1000.], phases)
        mixed = coupling.mix_rayleigh(sources, level)
        values = jax_numpy.stack(tuple(sample.coefficients for sample in mixed))
        return jax_numpy.sum(jax_numpy.abs(values)**2)
    def expected(parameters):
        level, amplitude, angle = parameters
        coupling = 10.**(level/20.)
        return 1e-4*((2. + coupling**2)*(1. + amplitude**2)
                      + 2.*coupling*amplitude*(jax_numpy.cos(angle) + jax_numpy.cos(2.*angle)))
    parameters = jax_numpy.asarray([-30., 1.2, .4])
    for transform in (lambda function: function, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(power))(parameters),
                                   jax.jit(transform(expected))(parameters), rtol=1e-10, atol=1e-14)
