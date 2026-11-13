import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import RayleighRealization


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
