import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import BendLossTable

REFERENCE = 193414489032258.06


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


def test_native_attenuation_keeps_namespace_shape_and_float64(xp):
    attenuation = xp.asarray([[.01, .04], [.1, .4]], dtype=xp.float32)
    table = BendLossTable([10., 20.], [REFERENCE - 1e9, REFERENCE + 1e9], attenuation)
    assert table.shape == (2, 2)
    assert table.attenuation.dtype == xp.float64
    assert type(table.attenuation) is type(attenuation)
    assert table.curvatures.dtype == table.frequencies.dtype == np.float64
    assert table.covers(REFERENCE - 1e9, REFERENCE + 1e9)
    np.testing.assert_array_equal(np.asarray(table.attenuation), np.asarray(attenuation))


def test_table_owns_data_copies_and_read_only_grids():
    curvatures = np.array([10., 20.])
    frequencies = np.array([REFERENCE])
    attenuation = np.array([[0.], [.1]])
    table = BendLossTable(curvatures, frequencies, attenuation)
    curvatures[:] = 1.
    frequencies[:] = 1.
    attenuation[:] = 0.
    np.testing.assert_array_equal(table.curvatures, [10., 20.])
    np.testing.assert_array_equal(table.frequencies, [REFERENCE])
    np.testing.assert_array_equal(table.attenuation, [[0.], [.1]])
    for grid in (table.curvatures, table.frequencies):
        with pytest.raises(ValueError):
            grid[0] = 1.


def test_one_frequency_means_a_table_independent_of_frequency():
    table = BendLossTable([10., 20.], [REFERENCE], [[0.], [.1]])
    assert table.shape == (2, 1)
    assert table.covers(1., 2.*REFERENCE)
    assert table.covers(REFERENCE, REFERENCE)


@pytest.mark.parametrize("low,high,expected", [
    (-10e9, 10e9, True), (-1e9, 1e9, True), (0., 0., True),
    (-11e9, 0., False), (0., 11e9, False), (20e9, 30e9, False),
])
def test_spectral_table_covers_only_its_sampled_band(low, high, expected):
    table = BendLossTable([10., 20.], [REFERENCE - 10e9, REFERENCE, REFERENCE + 10e9],
                          [[.01, .02, .03], [.1, .2, .3]])
    assert table.covers(np.float64(REFERENCE + low), REFERENCE + high) is expected


@pytest.mark.parametrize("name,grid,error", [
    ("curvatures", [], ValueError), ("curvatures", [10.], ValueError),
    ("curvatures", [0., 10.], ValueError), ("curvatures", [-10., 10.], ValueError),
    ("curvatures", [10., 10.], ValueError), ("curvatures", [20., 10.], ValueError),
    ("curvatures", [10., float("inf")], ValueError),
    ("curvatures", [True, 10.], TypeError), ("curvatures", [[10., 20.]], TypeError),
    ("frequencies", [], ValueError), ("frequencies", [0.], ValueError),
    ("frequencies", [REFERENCE, REFERENCE], ValueError),
    ("frequencies", [float("nan")], ValueError),
    ("frequencies", [REFERENCE + 1j], TypeError),
])
def test_table_grids_are_fixed_positive_and_strictly_increasing(name, grid, error):
    values = dict(curvatures=[10., 20.], frequencies=[REFERENCE], attenuation=[[0.], [.1]])
    values[name] = grid
    with pytest.raises(error):
        BendLossTable(**values)


@pytest.mark.parametrize("name", ["curvatures", "frequencies"])
def test_jax_grids_are_rejected_without_detaching_them(jax_numpy, name):
    values = dict(curvatures=[10., 20.], frequencies=[REFERENCE], attenuation=[[0.], [.1]])
    values[name] = jax_numpy.asarray(values[name])
    with pytest.raises(TypeError, match="fixed real"):
        BendLossTable(**values)


@pytest.mark.parametrize("attenuation,error", [
    (.1, ValueError), ([0., .1], ValueError), ([[.1]], ValueError),
    ([[0., .1], [.2, .3]], ValueError), ([[0j], [.1j]], TypeError),
    ([[False], [True]], TypeError),
])
def test_attenuation_has_one_real_entry_per_grid_pair(attenuation, error):
    with pytest.raises(error):
        BendLossTable([10., 20.], [REFERENCE], attenuation)


@pytest.mark.parametrize("low,high,error", [
    (-1., REFERENCE, ValueError), (REFERENCE, float("inf"), ValueError),
    (float("nan"), REFERENCE, ValueError), (REFERENCE, 1., ValueError),
    (True, REFERENCE, TypeError), (np.array(1.), REFERENCE, TypeError),
])
def test_coverage_uses_fixed_valid_band_limits(low, high, error):
    table = BendLossTable([10., 20.], [REFERENCE], [[0.], [.1]])
    with pytest.raises(error):
        table.covers(low, high)


def test_jax_loss_data_copies_keep_jit_gradients_and_hessians(jax_numpy):
    import jax
    def power(parameters):
        table = BendLossTable([10., 20.], [REFERENCE],
                              jax_numpy.reshape(parameters, (2, 1))*.1)
        depth = table.attenuation[0, 0] + 2.*table.attenuation[1, 0]
        return jax_numpy.exp(-depth)
    parameters = jax_numpy.asarray([2., 3.], dtype=jax_numpy.float64)
    value = jax.jit(power)(parameters)
    gradient = jax.jit(jax.grad(power))(parameters)
    hessian = jax.jit(jax.hessian(power))(parameters)
    expected = np.exp(-.8)
    weights = np.array([.1, .2])
    np.testing.assert_allclose(value, expected, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(gradient, -expected*weights, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(hessian, expected*np.outer(weights, weights),
                               rtol=1e-13, atol=0.)


def test_log_interpolation_in_curvature_and_frequency(xp):
    table = BendLossTable([10., 20., 30.], [REFERENCE - 10e9, REFERENCE + 10e9],
                          xp.asarray([[1., 4.], [9., 36.], [81., 324.]]))
    curvature = xp.asarray([10., 15., 20., 25., 30.])
    frequency = xp.asarray([REFERENCE - 10e9, REFERENCE, REFERENCE + 10e9])
    expected = np.array([1., 3., 9., 27., 81.])[:, None]*np.array([1., 2., 4.])
    actual = table.value(curvature, frequency)
    assert actual.dtype == xp.float64
    assert type(actual) is type(curvature)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.asarray(table.by_frequency(curvature)), expected[:, [0, 2]],
                               rtol=1e-14, atol=0.)


def test_curvature_magnitude_lower_cutoff_and_unsupported_values(xp):
    table = BendLossTable([10., 20.], [REFERENCE], xp.asarray([[1.], [9.]]))
    curvature = xp.asarray([0., 9., 10., -15., 20., 21., -21., float("inf"), float("nan")])
    expected = np.array([0., 0., 1., 3., 9., np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(np.asarray(table.value(curvature, REFERENCE)), expected,
                               rtol=1e-14, atol=0., equal_nan=True)


def test_interpolation_uses_each_nonuniform_grid_interval(xp):
    table = BendLossTable([10., 20., 40.], [REFERENCE - 10e9, REFERENCE, REFERENCE + 30e9],
                          xp.asarray([[1., 4., 16.], [9., 36., 144.], [36., 144., 576.]]))
    curvature = xp.asarray([15., 30.])
    frequency = xp.asarray([REFERENCE - 5e9, REFERENCE + 15e9])
    np.testing.assert_allclose(np.asarray(table.value(curvature, frequency)),
                               [[6., 24.], [36., 144.]], rtol=1e-14, atol=0.)


def test_frequency_queries_clamp_to_the_sampled_band(xp):
    table = BendLossTable([10., 20.], [REFERENCE - 10e9, REFERENCE + 10e9],
                          xp.asarray([[1., 4.], [9., 36.]]))
    frequency = xp.asarray([REFERENCE - 20e9, REFERENCE + 20e9, float("nan")])
    np.testing.assert_allclose(np.asarray(table.value(15., frequency)), [3., 12., np.nan],
                               rtol=1e-14, atol=0., equal_nan=True)


def test_zero_corners_use_linear_interpolation(xp):
    constant = BendLossTable([10., 20.], [REFERENCE], xp.asarray([[0.], [.1]]))
    curvature = xp.asarray([10., 15., 20.])
    frequency = xp.asarray([REFERENCE - 1e12, REFERENCE, REFERENCE + 1e12])
    expected = np.array([0., .05, .1])[:, None]*np.ones((1, 3))
    np.testing.assert_allclose(np.asarray(constant.value(curvature, frequency)), expected,
                               rtol=1e-14, atol=0.)
    spectral = BendLossTable([10., 20.], [REFERENCE - 10e9, REFERENCE + 10e9],
                             xp.asarray([[0., 4.], [0., 36.]]))
    np.testing.assert_allclose(np.asarray(spectral.value(15., REFERENCE)), 6.,
                               rtol=1e-14, atol=0.)


@pytest.mark.parametrize("curvature_shape", [(), (2, 3), (0,)])
@pytest.mark.parametrize("frequency_shape", [(), (2, 1), (0,)])
def test_query_shapes_form_an_outer_grid(xp, curvature_shape, frequency_shape):
    table = BendLossTable([10., 20.], [REFERENCE - 10e9, REFERENCE + 10e9],
                          xp.asarray([[1., 4.], [9., 36.]]))
    curvature = xp.full(curvature_shape, 15., dtype=xp.float64)
    frequency = xp.full(frequency_shape, REFERENCE, dtype=xp.float64)
    result = table.value(curvature, frequency)
    assert result.shape == curvature_shape + frequency_shape
    assert table.by_frequency(curvature).shape == curvature_shape + (2,)
    np.testing.assert_allclose(np.asarray(result), np.full(result.shape, 6.),
                               rtol=1e-14, atol=0.)


@pytest.mark.parametrize("curvature,frequency", [
    (15j, REFERENCE), (15., REFERENCE + 1j), (True, REFERENCE), (15., True),
])
def test_queries_are_real_even_for_a_constant_frequency_table(curvature, frequency):
    table = BendLossTable([10., 20.], [REFERENCE], [[0.], [.1]])
    with pytest.raises(TypeError):
        table.value(curvature, frequency)


def test_numpy_table_and_jax_queries_need_explicit_selection(jax_numpy):
    table = BendLossTable([10., 20.], [REFERENCE - 10e9, REFERENCE + 10e9],
                          [[1., 4.], [9., 36.]])
    curvature = jax_numpy.asarray(15.)
    frequency = jax_numpy.asarray(REFERENCE)
    with pytest.raises(TypeError):
        table.by_frequency(curvature)
    with pytest.raises(TypeError):
        table.value(15., frequency)
    result = table.value(curvature, frequency, xp=jax_numpy)
    assert type(result) is type(curvature)
    np.testing.assert_allclose(np.asarray(result), 6., rtol=1e-14, atol=0.)


def test_jax_log_surface_derivatives_in_data_and_both_queries(jax_numpy):
    import jax
    def rate(parameters):
        x, y, u, v = parameters
        row1 = jax_numpy.stack((x, x + y))
        row2 = jax_numpy.stack((2.*x, 2.*x + y))
        attenuation = jax_numpy.exp(jax_numpy.stack((row1, row2)))
        table = BendLossTable([10., 20.], [REFERENCE - 10e9, REFERENCE + 10e9], attenuation)
        return table.value(10. + 10.*u, REFERENCE - 10e9 + 20e9*v)
    parameters = jax_numpy.asarray([np.log(3.), np.log(4.), .3, .4], dtype=jax_numpy.float64)
    value = jax.jit(rate)(parameters)
    gradient = jax.jit(jax.grad(rate))(parameters)
    hessian = jax.jit(jax.hessian(rate))(parameters)
    x, y, u, v = np.asarray(parameters)
    expected = np.exp(x*(1. + u) + y*v)
    derivative = np.array([1. + u, v, x, y])
    second = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.],
                        [1., 0., 0., 0.], [0., 1., 0., 0.]])
    np.testing.assert_allclose(value, expected, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(gradient, expected*derivative, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(hessian, expected*(np.outer(derivative, derivative) + second),
                               rtol=1e-11, atol=1e-13)


def test_jax_linear_branch_next_to_zero_has_finite_derivatives(jax_numpy):
    import jax
    def rate(parameters):
        right, fraction = parameters
        attenuation = jax_numpy.reshape(jax_numpy.stack((jax_numpy.asarray(0.), right)), (2, 1))
        table = BendLossTable([10., 20.], [REFERENCE], attenuation)
        return table.value(10. + 10.*fraction, REFERENCE)
    parameters = jax_numpy.asarray([.2, .4], dtype=jax_numpy.float64)
    np.testing.assert_allclose(jax.jit(rate)(parameters), .08, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(jax.jit(jax.grad(rate))(parameters), [.4, .2], rtol=1e-13, atol=0.)
    np.testing.assert_allclose(jax.jit(jax.hessian(rate))(parameters), [[0., 1.], [1., 0.]],
                               rtol=1e-13, atol=1e-14)
