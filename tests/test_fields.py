import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import Field, FieldTerm, SpatialProfile, TimeHistory


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


def test_time_history_interpolates_and_holds_end_values(xp):
    values = xp.asarray([1., 3., 2.], dtype=xp.float64)
    history = TimeHistory([-1., 0., 2.], values)
    time = xp.asarray([-2., -1., -.5, 0., 1., 2., 3.], dtype=xp.float64)
    actual = history.value(time)
    assert actual.dtype == xp.float64
    assert type(history.values) is type(values)
    np.testing.assert_array_equal(np.asarray(actual), [1., 1., 2., 3., 2.5, 2., 2.])


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_time_history_preserves_query_shape(xp, shape):
    history = TimeHistory([0., 2.], xp.asarray([1., 3.]))
    actual = history.value(xp.full(shape, 1., dtype=xp.float64))
    assert actual.shape == shape
    assert actual.dtype == xp.float64
    np.testing.assert_array_equal(np.asarray(actual), np.full(shape, 2.))


def test_single_time_sample_is_constant(xp):
    history = TimeHistory([1.], xp.asarray([-2.]))
    time = xp.asarray([-10., 1., 10.], dtype=xp.float64)
    np.testing.assert_array_equal(np.asarray(history.value(time)), [-2., -2., -2.])


def test_history_copies_inputs_and_keeps_times_read_only():
    times = np.array([0., 2.])
    values = np.array([1., 3.])
    history = TimeHistory(times, values)
    times[:] = 5.
    values[:] = 5.
    np.testing.assert_array_equal(history.times, [0., 2.])
    np.testing.assert_array_equal(history.values, [1., 3.])
    with pytest.raises(ValueError):
        history.times[0] = 1.


@pytest.mark.parametrize("times,values,error", [
    ([], [], ValueError), ([0., 0.], [1., 2.], ValueError),
    ([1., 0.], [1., 2.], ValueError), ([float("inf")], [1.], ValueError),
    ([float("nan")], [1.], ValueError), ([0., 1.], [1.], ValueError),
    ([0., 1.], [[1., 2.]], ValueError), ([False], [1.], TypeError),
    ([[0., 1.]], [1., 2.], TypeError), ([0.], [1j], TypeError),
    ([0.], [True], TypeError),
])
def test_invalid_time_history_data(times, values, error):
    with pytest.raises(error):
        TimeHistory(times, values)


def test_native_jax_sample_times_are_rejected_without_detaching_them(jax_numpy):
    with pytest.raises(TypeError, match="fixed real"):
        TimeHistory(jax_numpy.asarray([0., 1.]), [1., 2.])


def test_numpy_history_and_jax_time_need_explicit_namespace(jax_numpy):
    history = TimeHistory([0., 2.], [1., 3.])
    time = jax_numpy.asarray(1.)
    with pytest.raises(TypeError):
        history.value(time)
    actual = history.value(time, xp=jax_numpy)
    assert actual.dtype == jax_numpy.float64
    np.testing.assert_allclose(actual, 2., rtol=1e-14, atol=0.)


def test_jax_history_phase_derivatives_in_values_and_query_time(jax_numpy):
    import jax
    def returned_field(parameters):
        history = TimeHistory([-1., 1.], parameters[:2]*1e-6)
        phase = 1e6*history.value(parameters[2])
        return jax_numpy.real(jax_numpy.exp(-1j*phase))
    parameters = jax_numpy.asarray([1., 3., 0.], dtype=jax_numpy.float64)
    value = jax.jit(returned_field)(parameters)
    gradient = jax.jit(jax.grad(returned_field))(parameters)
    hessian = jax.jit(jax.hessian(returned_field))(parameters)
    phase_gradient = np.array([.5, .5, 1.])
    phase_hessian = np.array([[0., 0., -.5], [0., 0., .5], [-.5, .5, 0.]])
    expected_hessian = (-np.cos(2.)*np.outer(phase_gradient, phase_gradient)
                        - np.sin(2.)*phase_hessian)
    np.testing.assert_allclose(value, np.cos(2.), rtol=1e-14, atol=0.)
    np.testing.assert_allclose(gradient, -np.sin(2.)*phase_gradient, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("time", [-2., 2.])
def test_jax_history_is_constant_outside_the_sampled_interval(jax_numpy, time):
    import jax
    history = TimeHistory([-1., 1.], jax_numpy.asarray([1., 3.]))
    query = jax_numpy.asarray(time, dtype=jax_numpy.float64)
    first = jax.jit(jax.grad(history.value))(query)
    second = jax.jit(jax.grad(jax.grad(history.value)))(query)
    np.testing.assert_array_equal(first, 0.)
    np.testing.assert_array_equal(second, 0.)


def test_sum_of_static_and_dynamic_terms_at_paired_samples(xp):
    dynamic = FieldTerm(SpatialProfile([0., 2.], xp.asarray([2., 2.])),
                        TimeHistory([0., 2.], xp.asarray([1., 3.])))
    static = FieldTerm(SpatialProfile([1., 3.], xp.asarray([-1., -1.])))
    field = Field([dynamic, static])
    position = xp.asarray([0., 1., 3.])
    time = xp.asarray([0., 1., 2.])
    np.testing.assert_array_equal(np.asarray(field.value(position, time)), [2., 3., -1.])
    np.testing.assert_array_equal(np.asarray(field.integral(position, time)), [0., 4., 10.])


def test_fields_broadcast_a_position_time_grid(xp):
    dynamic = FieldTerm(SpatialProfile([0., 2.], xp.asarray([2., 2.])),
                        TimeHistory([0., 2.], xp.asarray([1., 3.])))
    static = FieldTerm(SpatialProfile([1., 3.], xp.asarray([-1., -1.])))
    field = Field([dynamic, static])
    position = xp.asarray([[0.], [1.], [3.]])
    time = xp.asarray([0., 1., 2.])
    expected = [[2., 4., 6.], [1., 3., 5.], [-1., -1., -1.]]
    expected_integral = [[0., 0., 0.], [2., 4., 6.], [2., 6., 10.]]
    np.testing.assert_array_equal(np.asarray(field.value(position, time)), expected)
    np.testing.assert_array_equal(np.asarray(field.integral(position, time)), expected_integral)


def test_static_term_broadcasts_over_time_and_defaults_to_time_zero(xp):
    term = FieldTerm(SpatialProfile([0., 2.], xp.asarray([2., 2.])))
    time = xp.zeros((2, 3), dtype=xp.float64)
    for method in (term.value, term.integral):
        actual = method(1., time)
        assert actual.shape == (2, 3)
        assert actual.dtype == xp.float64
        np.testing.assert_array_equal(np.asarray(actual), np.full((2, 3), 2.))
        np.testing.assert_array_equal(np.asarray(method(1.)), 2.)


@pytest.mark.parametrize("shape", [(2, 3), (0,)])
def test_empty_field_is_zero_in_the_query_namespace(xp, shape):
    field = Field()
    assert field.is_zero and field.is_static
    time = xp.zeros(shape, dtype=xp.float64)
    for method in (field.value, field.integral):
        actual = method(1., time)
        assert actual.shape == shape
        assert actual.dtype == xp.float64
        np.testing.assert_array_equal(np.asarray(actual), np.zeros(shape))


def test_field_stores_its_terms_and_flags_depend_only_on_structure():
    profile = SpatialProfile([0., 1.], [0., 0.])
    static = FieldTerm(profile)
    terms = [static]
    field = Field(terms)
    terms.clear()
    assert field.terms == (static,)
    assert field.terms[0].profile is profile
    assert field.is_static and not field.is_zero
    with_history = Field([FieldTerm(profile, TimeHistory([0.], [0.]))])
    assert not with_history.is_static and not with_history.is_zero


def test_field_containers_require_profiles_histories_and_terms():
    profile = SpatialProfile([0., 1.], [1., 1.])
    with pytest.raises(TypeError, match="SpatialProfile"):
        FieldTerm(None)
    with pytest.raises(TypeError, match="TimeHistory"):
        FieldTerm(profile, profile)
    with pytest.raises(TypeError, match="FieldTerm"):
        Field([profile])


def test_mixed_term_namespaces_require_explicit_selection(jax_numpy):
    numpy_profile = SpatialProfile([0., 2.], [1., 3.])
    jax_profile = SpatialProfile([0., 2.], jax_numpy.asarray([1., 3.]))
    history = TimeHistory([0., 2.], jax_numpy.asarray([1., 3.]))
    mixed_term = FieldTerm(numpy_profile, history)
    separate_terms = Field([FieldTerm(numpy_profile), FieldTerm(jax_profile)])
    for obj in (mixed_term, Field([mixed_term]), separate_terms):
        for method, expected in ((obj.value, 4.), (obj.integral, 3.)):
            with pytest.raises(TypeError):
                method(1., 1.)
            actual = method(1., 1., xp=jax_numpy)
            assert actual.dtype == jax_numpy.float64
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.)


def test_field_rejects_incompatible_position_time_shapes(xp):
    term = FieldTerm(SpatialProfile([0., 2.], xp.asarray([1., 1.])))
    position, time = xp.asarray([0., 1.]), xp.asarray([0., 1., 2.])
    for obj in (term, Field([term]), Field()):
        for method in (obj.value, obj.integral):
            with pytest.raises(ValueError):
                method(position, time)


def test_jax_field_phase_derivatives_through_space_time_composition(jax_numpy):
    import jax
    def returned_field(parameters):
        a, b, position, time = parameters
        profile = SpatialProfile([0., 2.], jax_numpy.stack((a, 3.*a))*1e-6)
        history = TimeHistory([0., 2.], jax_numpy.stack((b, 3.*b)))
        baseline = SpatialProfile([0., 2.], jax_numpy.asarray([1e-6, 1e-6]))
        field = Field([FieldTerm(baseline), FieldTerm(profile, history)])
        phase = 1e6*field.integral(position, time)
        return jax_numpy.real(jax_numpy.exp(-1j*phase))
    parameters = jax_numpy.asarray([1., 2., 1., .5], dtype=jax_numpy.float64)
    value = jax.jit(returned_field)(parameters)
    gradient = jax.jit(jax.grad(returned_field))(parameters)
    hessian = jax.jit(jax.hessian(returned_field))(parameters)
    phase_gradient = np.array([4.5, 2.25, 7., 3.])
    phase_hessian = np.array([[0., 2.25, 6., 3.], [2.25, 0., 3., 1.5],
                              [6., 3., 3., 4.], [3., 1.5, 4., 0.]])
    expected_hessian = (-np.cos(5.5)*np.outer(phase_gradient, phase_gradient)
                        - np.sin(5.5)*phase_hessian)
    np.testing.assert_allclose(value, np.cos(5.5), rtol=1e-14, atol=0.)
    np.testing.assert_allclose(gradient, -np.sin(5.5)*phase_gradient, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-13, atol=1e-15)
