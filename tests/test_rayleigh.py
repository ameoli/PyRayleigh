from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Fiber, GuideOptics, RayleighRealization, Section,
                        generate_rayleigh, rayleigh_from_normals)


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


def make_realization(**changes):
    values = dict(coefficients=[.01 + .02j, -.03j],
                  positions=[0., 1.], edges=[0., 1., 1.25])
    values.update(changes)
    return RayleighRealization(**values)


def test_native_coefficients_and_partial_last_cell(xp):
    coefficients = xp.asarray([.01 + .02j, -.03j], dtype=xp.complex128)
    sample = make_realization(coefficients=coefficients)
    assert type(sample.coefficients) is type(coefficients)
    assert sample.coefficients.dtype == xp.complex128
    assert sample.cell_count == 2
    assert sample.extent == 1.25
    np.testing.assert_array_equal(np.asarray(sample.coefficients), [.01 + .02j, -.03j])
    np.testing.assert_array_equal(sample.cell_lengths(), [1., .25])
    np.testing.assert_array_equal(sample.positions, [0., 1.])


def test_sample_owns_its_arrays_and_geometry_cannot_be_edited():
    coefficients = np.array([.01, .02])
    positions = np.array([0., 1.])
    edges = np.array([0., 1., 1.25])
    sample = make_realization(coefficients=coefficients, positions=positions, edges=edges)
    coefficients[:] = .5
    positions[:] = .5
    edges[:] = .5
    np.testing.assert_array_equal(sample.coefficients, [.01, .02])
    np.testing.assert_array_equal(sample.positions, [0., 1.])
    np.testing.assert_array_equal(sample.edges, [0., 1., 1.25])
    assert sample.coefficients.dtype == np.complex128
    with pytest.raises(ValueError):
        sample.positions[0] = .5
    with pytest.raises(ValueError):
        sample.edges[1] = .5
    returned_edges = sample.cell_edges()
    returned_edges[1] = .5
    assert sample.edges[1] == 1.


@pytest.mark.parametrize("changes,error", [
    ({"coefficients": []}, ValueError), ({"coefficients": .01}, ValueError),
    ({"coefficients": [[.01, .02]]}, ValueError),
    ({"coefficients": [True, False]}, TypeError),
    ({"positions": [0.]}, ValueError), ({"edges": [0., 1.]}, ValueError),
    ({"edges": [.1, 1., 1.25]}, ValueError),
    ({"edges": [0., 1., 1.]}, ValueError), ({"edges": [0., 2., 1.]}, ValueError),
    ({"positions": [-.1, 1.]}, ValueError), ({"positions": [0., 1.3]}, ValueError),
    ({"positions": [1.1, 1.2]}, ValueError), ({"positions": [0., .9]}, ValueError),
    ({"positions": [float("nan"), 1.]}, ValueError),
    ({"edges": [0., 1., float("inf")]}, ValueError),
    ({"positions": [False, 1.]}, TypeError), ({"positions": [0j, 1.]}, TypeError),
    ({"edges": [[0., 1., 1.25]]}, TypeError),
])
def test_invalid_coefficients_or_cell_geometry(changes, error):
    with pytest.raises(error):
        make_realization(**changes)


def test_cell_edges_are_allowed_as_scatterer_positions():
    sample = make_realization(positions=[1., 1.])
    np.testing.assert_array_equal(sample.positions, [1., 1.])


@pytest.mark.parametrize("name", ["positions", "edges"])
def test_jax_geometry_is_rejected_without_detaching_it(jax_numpy, name):
    geometry = [0., 1.] if name == "positions" else [0., 1., 1.25]
    with pytest.raises(TypeError, match="fixed real"):
        make_realization(**{name: jax_numpy.asarray(geometry)})


def test_jax_local_power_preserves_first_and_second_derivatives(jax_numpy):
    import jax
    disorder = jax_numpy.asarray([1. + 2j, 3. - 1j], dtype=jax_numpy.complex128)
    def local_power(scale):
        sample = make_realization(coefficients=scale*disorder)
        return jax_numpy.sum(jax_numpy.real(jax_numpy.conj(sample.coefficients)
                                           * sample.coefficients))
    scale = jax_numpy.asarray(.1, dtype=jax_numpy.float64)
    value = jax.jit(local_power)(scale)
    first = jax.jit(jax.grad(local_power))(scale)
    second = jax.jit(jax.grad(jax.grad(local_power)))(scale)
    np.testing.assert_allclose(value, .15, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(first, 3., rtol=1e-14, atol=0.)
    np.testing.assert_allclose(second, 30., rtol=1e-14, atol=0.)


def generation_fiber(rate=2e-8, length=1.25):
    optics = GuideOptics(193414489032258.06, 1.4682, 1.4682)
    return Fiber("spool-1", length, 293.15, 1.9e14, 2e14,
                 [Section(0., length, optics, 0., rate)])


@pytest.mark.parametrize("boundary", [.6, 1.])
def test_left_sampled_rates_and_partial_cell_normalization(xp, boundary):
    fiber = generation_fiber()
    optics = fiber.sections[0].optics
    fiber = replace(fiber, sections=[Section(0., boundary, optics, 0., 2e-8),
                                    Section(boundary, 1.25, optics, 0., 8e-8)])
    normals = xp.asarray([[1., 0.], [0., 2.]], dtype=xp.float64)
    sample = rayleigh_from_normals(fiber, normals, cell_length=1.)
    assert sample.coefficients.dtype == xp.complex128
    np.testing.assert_allclose(np.asarray(sample.coefficients), [1e-4, 2e-4j],
                               rtol=1e-14, atol=0.)
    np.testing.assert_array_equal(sample.positions, [0., 1.])
    np.testing.assert_array_equal(sample.cell_lengths(), [1., .25])


def test_seeded_generation_repeats_without_changing_global_random_state():
    fiber = generation_fiber()
    before = np.random.get_state()
    first = generate_rayleigh(fiber, np.int64(7), cell_length=1.)
    second = generate_rayleigh(fiber, 7, cell_length=1.)
    other = generate_rayleigh(fiber, 8, cell_length=1.)
    after = np.random.get_state()
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    assert not np.array_equal(first.coefficients, other.coefficients)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_seeded_generation_uses_the_same_draws_across_backends(xp):
    fiber = generation_fiber()
    expected = generate_rayleigh(fiber, 7, cell_length=1.)
    actual = generate_rayleigh(fiber, 7, cell_length=1., xp=xp)
    assert actual.coefficients.dtype == xp.complex128
    np.testing.assert_allclose(np.asarray(actual.coefficients), expected.coefficients,
                               rtol=1e-14, atol=0.)


@pytest.mark.parametrize("length,cell_length,expected_edges", [
    (1.25, 2., [0., 1.25]), (2., 1., [0., 1., 2.]),
    (.1 + .2, .1, [0., .1, .2, .1 + .2]),
])
def test_generation_clips_the_end_without_empty_cells(length, cell_length, expected_edges):
    sample = generate_rayleigh(generation_fiber(length=length), 7, cell_length=cell_length)
    np.testing.assert_array_equal(sample.edges, expected_edges)
    assert np.all(sample.cell_lengths() > 0.)
    assert sample.extent == length


def test_zero_rate_gives_zero_coefficients(xp):
    sample = generate_rayleigh(generation_fiber(rate=0.), 7, cell_length=1., xp=xp)
    np.testing.assert_array_equal(np.asarray(sample.coefficients), [0j, 0j])


@pytest.mark.parametrize("seed", [-1, .5, True, None, np.bool_(False)])
def test_generation_requires_an_explicit_integer_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        generate_rayleigh(generation_fiber(), seed, cell_length=1.)


@pytest.mark.parametrize("cell_length,error", [
    (0., ValueError), (-1., ValueError), (float("nan"), ValueError),
    (float("inf"), ValueError), (True, TypeError), (np.array(1.), TypeError),
])
def test_generation_requires_a_fixed_positive_cell_length(cell_length, error):
    with pytest.raises(error, match="cell_length"):
        generate_rayleigh(generation_fiber(), 7, cell_length=cell_length)


@pytest.mark.parametrize("normals", [[], [1., 2.], [[1., 2.]], [[1.], [2.]]])
def test_supplied_normals_need_two_columns_and_one_row_per_cell(normals):
    with pytest.raises(ValueError, match="shape"):
        rayleigh_from_normals(generation_fiber(), normals, cell_length=1.)


@pytest.mark.parametrize("normals", [np.ones((2, 2), dtype=bool),
                                      np.ones((2, 2), dtype=complex)])
def test_supplied_normals_must_be_real_numeric_data(normals):
    with pytest.raises(TypeError):
        rayleigh_from_normals(generation_fiber(), normals, cell_length=1.)


def test_generation_infers_jax_from_section_rates(jax_numpy):
    rate = jax_numpy.asarray(2e-8, dtype=jax_numpy.float64)
    sample = generate_rayleigh(generation_fiber(rate), 7, cell_length=1.)
    assert type(sample.coefficients) is type(rate)
    assert sample.coefficients.dtype == jax_numpy.complex128


def test_fixed_draws_preserve_jax_derivatives_in_section_rates(jax_numpy):
    import jax
    normals = np.array([[1., 2.], [3., -1.]])
    def local_field(rates):
        fiber = generation_fiber()
        optics = fiber.sections[0].optics
        fiber = replace(fiber, sections=[Section(0., .6, optics, 0., rates[0]*1e-8),
                                        Section(.6, 1.25, optics, 0., rates[1]*1e-8)])
        sample = rayleigh_from_normals(fiber, normals, cell_length=1., xp=jax_numpy)
        return jax_numpy.real(jax_numpy.sum(sample.coefficients))
    rates = jax_numpy.asarray([2., 8.], dtype=jax_numpy.float64)
    value = jax.jit(local_field)(rates)
    gradient = jax.jit(jax.grad(local_field))(rates)
    hessian = jax.jit(jax.hessian(local_field))(rates)
    np.testing.assert_allclose(value, 4e-4, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(gradient, [2.5e-5, 1.875e-5], rtol=1e-13, atol=0.)
    np.testing.assert_allclose(hessian, np.diag([-6.25e-6, -1.171875e-6]),
                               rtol=1e-13, atol=1e-18)


def test_ensemble_power_and_circularity():
    sample = generate_rayleigh(generation_fiber(length=50000.), 29, cell_length=1.)
    normalized = sample.coefficients / np.sqrt(2e-8)
    np.testing.assert_allclose(np.mean(np.abs(normalized)**2), 1., rtol=.02)
    assert abs(np.mean(normalized)) < .02
    assert abs(np.mean(normalized**2)) < .03
