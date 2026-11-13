import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import SpatialProfile


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


def test_linear_profile_values_and_integrals_with_negative_knots(xp):
    values = xp.asarray([0., 2., 0.], dtype=xp.float64)
    profile = SpatialProfile([-1., 0., 2.], values)
    position = xp.asarray([-2., -1., -.5, 0., 1., 2., 3.], dtype=xp.float64)
    actual = profile.value(position)
    assert actual.dtype == xp.float64
    assert type(profile.values) is type(values)
    np.testing.assert_array_equal(np.asarray(actual), [0., 0., 1., 2., 1., 0., 0.])
    np.testing.assert_array_equal(np.asarray(profile.antiderivative(position)),
                                  [0., 0., .25, 1., 2.5, 3., 3.])
    np.testing.assert_array_equal(np.asarray(profile.integral(position)),
                                  [-1., -1., -.75, 0., 1.5, 2., 2.])


def test_steps_are_right_continuous_and_add_no_area(xp):
    profile = SpatialProfile([0., 1., 1., 3.], xp.asarray([1., 1., 3., 3.]))
    position = xp.asarray([-1., 0., .5, 1., 2., 3., 4.], dtype=xp.float64)
    np.testing.assert_array_equal(np.asarray(profile.value(position)), [0., 1., 1., 3., 3., 3., 0.])
    np.testing.assert_array_equal(np.asarray(profile.integral(position)), [0., 0., .5, 1., 4., 7., 7.])


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_query_shape_is_preserved(xp, shape):
    profile = SpatialProfile([0., 2.], xp.asarray([1., 3.]))
    position = xp.full(shape, 1., dtype=xp.float64)
    for method, expected in ((profile.value, 2.), (profile.integral, 1.5)):
        actual = method(position)
        assert actual.shape == shape
        assert actual.dtype == xp.float64
        np.testing.assert_array_equal(np.asarray(actual), np.full(shape, expected))


def test_two_equal_knots_have_a_value_but_no_area():
    profile = SpatialProfile([1., 1.], [2., 3.])
    np.testing.assert_array_equal(profile.value([0., 1., 2.]), [0., 3., 0.])
    np.testing.assert_array_equal(profile.integral([0., 1., 2.]), [0., 0., 0.])


def test_profile_copies_inputs_and_keeps_geometry_read_only():
    positions = np.array([0., 2.])
    values = np.array([1., 3.])
    profile = SpatialProfile(positions, values)
    positions[:] = 5.
    values[:] = 5.
    np.testing.assert_array_equal(profile.positions, [0., 2.])
    np.testing.assert_array_equal(profile.values, [1., 3.])
    with pytest.raises(ValueError):
        profile.positions[0] = 1.


@pytest.mark.parametrize("positions,values,error", [
    ([], [], ValueError), ([0.], [1.], ValueError),
    ([1., 0.], [1., 2.], ValueError), ([0., 0., 0.], [1., 2., 3.], ValueError),
    ([0., float("inf")], [1., 1.], ValueError),
    ([0., float("nan")], [1., 1.], ValueError),
    ([0., 1.], [1.], ValueError), ([0., 1.], [[1., 2.]], ValueError),
    ([False, 1.], [1., 2.], TypeError), ([[0., 1.]], [1., 2.], TypeError),
    ([0., 1.], [1j, 2j], TypeError), ([0., 1.], [True, False], TypeError),
])
def test_invalid_profile_data(positions, values, error):
    with pytest.raises(error):
        SpatialProfile(positions, values)


def test_native_jax_geometry_is_rejected_without_detaching_it(jax_numpy):
    with pytest.raises(TypeError, match="fixed real"):
        SpatialProfile(jax_numpy.asarray([0., 1.]), [1., 2.])


def test_numpy_values_and_jax_queries_need_explicit_namespace(jax_numpy):
    profile = SpatialProfile([0., 2.], [1., 3.])
    position = jax_numpy.asarray(1.)
    for method, expected in ((profile.value, 2.), (profile.integral, 1.5)):
        with pytest.raises(TypeError):
            method(position)
        actual = method(position, xp=jax_numpy)
        assert actual.dtype == jax_numpy.float64
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.)


def test_jax_phase_derivatives_in_values_and_query_position(jax_numpy):
    import jax
    def returned_field(parameters):
        profile = SpatialProfile([0., 2.], parameters[:2]*1e-6)
        phase = 5e5*profile.integral(parameters[2])
        return jax_numpy.real(jax_numpy.exp(-1j*phase))
    parameters = jax_numpy.asarray([1., 3., 1.], dtype=jax_numpy.float64)
    value = jax.jit(returned_field)(parameters)
    gradient = jax.jit(jax.grad(returned_field))(parameters)
    hessian = jax.jit(jax.hessian(returned_field))(parameters)
    phase_gradient = np.array([.375, .125, 1.])
    phase_hessian = np.array([[0., 0., .25], [0., 0., .25], [.25, .25, .5]])
    expected_hessian = (-np.cos(.75)*np.outer(phase_gradient, phase_gradient)
                        - np.sin(.75)*phase_hessian)
    np.testing.assert_allclose(value, np.cos(.75), rtol=1e-14, atol=0.)
    np.testing.assert_allclose(gradient, -np.sin(.75)*phase_gradient, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-13, atol=1e-15)
