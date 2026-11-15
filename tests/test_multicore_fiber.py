from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (CrosstalkRealization, Fiber, GuideOptics, MultiCoreFiber,
                        RayleighRealization, Section, generate_crosstalk)

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


def make_cores(count=2, xp=numpy_api):
    guide = GuideOptics(REFERENCE, xp.asarray(INDEX), INDEX)
    section = Section(0., 1.25, guide, xp.asarray(.2), 1e-8)
    return [Fiber(f"core-{index}", 1.25, 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                  [section], rayleigh=RayleighRealization(
                      xp.asarray([.01, .02j, .03])*(index + 1), POSITIONS, EDGES))
            for index in range(count)]


def test_core_order_and_shared_frame_are_fixed_without_copying_the_fibers():
    source = make_cores()
    source.reverse()
    first, second = source
    model = MultiCoreFiber(source)
    source.clear()
    assert model.cores[0] is first and model.cores[1] is second
    assert isinstance(model.cores, tuple) and model.n_cores == 2
    assert model.length == 1.25 and model.reference_temperature == 293.15
    assert model.frequency_min == REFERENCE - 50e9 and model.frequency_max == REFERENCE + 50e9
    assert model.crosstalk is None and model.xt_db_per_km is None
    assert repr(model) == "MultiCoreFiber(n_cores=2, length=1.25, crosstalk=False)"


@pytest.mark.parametrize("count,topology", [(1, "linear"), (3, "linear"), (7, "hex7")])
def test_coupled_layout_retains_native_parameters_samples_and_core_objects(xp, count, topology):
    cores = make_cores(count, xp)
    coupling = generate_crosstalk(count, POSITIONS, 42, topology=topology, xp=xp)
    level = xp.asarray(-30.)
    original = [np.asarray(core.rayleigh.coefficients).copy() for core in cores]
    phases = np.asarray(coupling.phases).copy()
    model = MultiCoreFiber(cores, coupling, level)
    assert model.crosstalk is coupling and model.xt_db_per_km is level
    for index, core in enumerate(cores):
        assert model.cores[index] is core
        assert model.cores[index].sections[0].optics is core.sections[0].optics
        assert model.cores[index].rayleigh is core.rayleigh
        np.testing.assert_array_equal(np.asarray(core.rayleigh.coefficients), original[index])
    np.testing.assert_array_equal(np.asarray(model.crosstalk.phases), phases)


def test_with_crosstalk_returns_a_new_layout_and_level_updates_reuse_its_phases(xp):
    model = MultiCoreFiber(make_cores(3, xp))
    level = xp.asarray(-30.)
    coupled = model.with_crosstalk(42, level, xp=xp)
    expected = generate_crosstalk(3, POSITIONS, 42, xp=xp)
    assert coupled is not model and model.crosstalk is None and model.xt_db_per_km is None
    assert coupled.xt_db_per_km is level
    assert all(actual is original for actual, original in zip(coupled.cores, model.cores))
    np.testing.assert_array_equal(np.asarray(coupled.crosstalk.phases), np.asarray(expected.phases))
    changed = replace(coupled, xt_db_per_km=xp.asarray(-20.))
    assert changed.crosstalk is coupled.crosstalk
    assert all(actual is original for actual, original in zip(changed.cores, model.cores))


def test_uncoupled_cores_can_have_independent_rayleigh_grids_or_no_sample():
    cores = make_cores()
    cores[0] = replace(cores[0], rayleigh=None)
    cores[1] = replace(cores[1], rayleigh=RayleighRealization([.01, .02], [0., 1.], [0., 1., 1.25]))
    model = MultiCoreFiber(cores)
    assert model.cores[0].rayleigh is None and model.cores[1].rayleigh.cell_count == 2


@pytest.mark.parametrize("cores,error", [([], ValueError), ([None], TypeError), ([1, 2], TypeError)])
def test_at_least_one_fiber_core_is_required(cores, error):
    with pytest.raises(error, match="core|Fiber"):
        MultiCoreFiber(cores)


@pytest.mark.parametrize("name", ["length", "reference_temperature", "frequency_min", "frequency_max"])
def test_core_frames_must_agree(name):
    first, second = make_cores()
    if name == "length":
        second = replace(second, length=1.5, rayleigh=None,
                          sections=[replace(second.sections[0], end=1.5)])
    else:
        second = replace(second, **{name: getattr(second, name) + 1.})
    with pytest.raises(ValueError, match=name):
        MultiCoreFiber([first, second])


@pytest.mark.parametrize("coupling,level,error", [(None, -30., ValueError),
                                                 (generate_crosstalk(2, POSITIONS, 42), None, ValueError),
                                                 (object(), -30., TypeError),
                                                 (generate_crosstalk(3, POSITIONS, 42), -30., ValueError),
                                                 (generate_crosstalk(2, POSITIONS, 42), [-30.], ValueError),
                                                 (generate_crosstalk(2, POSITIONS, 42), -30j, TypeError)])
def test_crosstalk_and_level_are_supplied_as_a_matching_pair(coupling, level, error):
    with pytest.raises(error):
        MultiCoreFiber(make_cores(), coupling, level)


@pytest.mark.parametrize("missing", [0, 1])
def test_coupled_layout_and_builder_require_rayleigh_in_every_core(missing):
    cores = make_cores()
    cores[missing] = replace(cores[missing], rayleigh=None)
    coupling = generate_crosstalk(2, POSITIONS, 42)
    with pytest.raises(ValueError, match="Rayleigh"):
        MultiCoreFiber(cores, coupling, -30.)
    with pytest.raises(ValueError, match="Rayleigh"):
        MultiCoreFiber(cores).with_crosstalk(42, -30.)


@pytest.mark.parametrize("positions,edges", [([0., .75, 1.], EDGES),
                                             (POSITIONS, [0., .25, 1., 1.25])])
def test_coupled_grids_require_identical_positions_and_cell_edges(positions, edges):
    cores = make_cores()
    cores[1] = replace(cores[1], rayleigh=RayleighRealization([.01, .02, .03], positions, edges))
    with pytest.raises(ValueError, match="positions and cell edges"):
        MultiCoreFiber(cores).with_crosstalk(42, -30.)


def test_the_stored_crosstalk_grid_must_match_the_core_samples():
    coupling = generate_crosstalk(2, [0., .25, 1.], 42)
    with pytest.raises(ValueError, match="positions and cell edges"):
        MultiCoreFiber(make_cores(), coupling, -30.)


def test_jax_layout_keeps_native_core_coefficients_level_and_phase_derivatives(jax_numpy):
    import jax
    def power(parameters):
        level, amplitude, angle = parameters
        guide = GuideOptics(REFERENCE, jax_numpy.asarray(INDEX), INDEX)
        section = Section(0., 1250., guide, 0., 1e-8)
        ones = jax_numpy.ones((2,))
        samples = (RayleighRealization(.01*ones, [0., 1000.], [0., 1000., 1250.]),
                   RayleighRealization(.01*amplitude*ones, [0., 1000.], [0., 1000., 1250.]))
        cores = [Fiber(f"core-{index}", 1250., 293.15, REFERENCE - 50e9, REFERENCE + 50e9,
                        [section], rayleigh=sample) for index, sample in enumerate(samples)]
        coupling = CrosstalkRealization(2, [0., 1000.], jax_numpy.stack((angle*ones, 2.*angle*ones)))
        model = MultiCoreFiber(cores, coupling, level)
        mixed = model.crosstalk.mix_rayleigh(tuple(core.rayleigh for core in model.cores), model.xt_db_per_km)
        return jax_numpy.sum(jax_numpy.abs(jax_numpy.stack(tuple(sample.coefficients for sample in mixed)))**2)
    def expected(parameters):
        level, amplitude, angle = parameters
        coupling = 10.**(level/20.)
        return 1e-4*((2. + coupling**2)*(1. + amplitude**2)
                      + 2.*coupling*amplitude*(jax_numpy.cos(angle) + jax_numpy.cos(2.*angle)))
    parameters = jax_numpy.asarray([-30., 1.2, .4])
    for transform in (lambda function: function, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(power))(parameters),
                                   jax.jit(transform(expected))(parameters), rtol=1e-10, atol=1e-14)
