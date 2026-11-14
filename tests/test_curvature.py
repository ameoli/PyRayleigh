import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from dataclasses import replace

from pyrayleigh import (BendLossTable, Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, Probe, RayleighRealization, Section, SpatialProfile,
                        TimeHistory, compute_response, optical_path, point_response,
                        rayleigh_response)
from pyrayleigh.propagation.loss import FiberLoss

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


def make_state(xp, moving=False):
    history = TimeHistory([0., 2.], xp.asarray([0., 2.])) if moving else None
    x = Field([FieldTerm(SpatialProfile([0., 2.], xp.asarray([3., 3.])), history)])
    y = Field([FieldTerm(SpatialProfile([0., 2.], xp.asarray([4., 4.])))])
    return FiberState(curvature_x=x, curvature_y=y)


def make_fiber(table=None):
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682, bend_loss=table)
    return Fiber("spool-1", 2., 293.15, REFERENCE - 1e9, REFERENCE + 1e9,
                  [Section(0., 2., guide, .2, 1e-8)], [PointReflector(2., .1, 0., .8, .7)])


def test_curvature_components_broadcast_over_position_and_time(xp):
    state = make_state(xp, moving=True)
    position = xp.asarray([[-1.], [1.], [3.]])
    time = xp.asarray([0., 1., 2.])
    magnitude = state.curvature_magnitude(position, time)
    assert magnitude.dtype == xp.float64
    assert type(magnitude) is type(position)
    np.testing.assert_allclose(np.asarray(magnitude),
                               [[0., 0., 0.], [4., 5., np.sqrt(52.)], [0., 0., 0.]],
                               rtol=1e-14, atol=0.)


def test_snapshot_freezes_curvature_components_too(xp):
    state = make_state(xp, moving=True)
    snapshot = state.at_time(1.)
    assert not state.is_static and snapshot.is_static
    assert snapshot.curvature_x is not state.curvature_x
    for name in ("curvature_x", "curvature_y"):
        field = getattr(snapshot, name)
        assert field.is_static
        assert type(field.terms[0].profile.values) is type(getattr(state, name).terms[0].profile.values)
    np.testing.assert_array_equal(np.asarray(snapshot.curvature_magnitude(1., time=100.)), 5.)


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_reference_curvature_is_zero_in_the_query_namespace(xp, shape):
    position = xp.ones(shape, dtype=xp.float64)
    magnitude = FiberState().curvature_magnitude(position)
    assert magnitude.shape == shape
    assert magnitude.dtype == xp.float64
    np.testing.assert_array_equal(np.asarray(magnitude), np.zeros(shape))


def test_mixed_axis_values_require_explicit_namespace(jax_numpy):
    x = Field([FieldTerm(SpatialProfile([0., 2.], [3., 3.]))])
    y = Field([FieldTerm(SpatialProfile([0., 2.], jax_numpy.asarray([4., 4.])))])
    state = FiberState(curvature_x=x, curvature_y=y)
    with pytest.raises(TypeError):
        state.curvature_magnitude(1.)
    magnitude = state.curvature_magnitude(1., xp=jax_numpy)
    assert type(magnitude) is type(y.terms[0].profile.values)
    np.testing.assert_array_equal(np.asarray(magnitude), 5.)


def test_curvature_without_a_bend_table_keeps_the_reference_response(xp):
    fiber = make_fiber()
    state = make_state(xp, moving=True)
    probe = Probe(xp.asarray([REFERENCE, REFERENCE]), xp.asarray([0., 1.]))
    response = compute_response(fiber, probe, state)
    expected = compute_response(fiber, probe)
    for name in ("reflection", "transmission"):
        np.testing.assert_array_equal(np.asarray(getattr(response, name)),
                                      np.asarray(getattr(expected, name)))


def test_jax_curvature_and_numpy_probe_need_explicit_namespace(jax_numpy):
    state = make_state(jax_numpy, moving=True)
    probe = Probe([REFERENCE], [1.])
    with pytest.raises(TypeError):
        compute_response(make_fiber(), probe, state)
    response = compute_response(make_fiber(), probe, state, xp=jax_numpy)
    native_type = type(state.curvature_x.terms[0].profile.values)
    assert type(response.reflection) is native_type
    assert type(response.transmission) is native_type
    assert response.probe is probe


@pytest.mark.parametrize("moving", [False, True])
def test_bend_loss_enters_static_and_frozen_time_responses(xp, moving):
    table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.09]]))
    fiber = make_fiber(table)
    state = make_state(xp, moving)
    probe = Probe(xp.asarray([REFERENCE, REFERENCE]), xp.asarray([0., 1.]))
    reference = compute_response(fiber, probe)
    response = compute_response(fiber, probe, state)
    rates = np.array([.01*np.sqrt(3.), .03]) if moving else np.array([.03, .03])
    np.testing.assert_allclose(np.asarray(response.reflection),
                               np.asarray(reference.reflection)*np.exp(-2.*rates), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(response.transmission),
                               np.asarray(reference.transmission)*np.exp(-rates), rtol=1e-13)
    path = optical_path(fiber, state.at_time(1.))
    np.testing.assert_allclose(np.asarray(path.attenuation_integral(2., probe.frequency)),
                               [.46, .46], rtol=1e-14)


def test_jax_local_bend_rate_keeps_axis_and_time_derivatives(jax_numpy):
    import jax
    table = BendLossTable([3., 7.], [REFERENCE], jax_numpy.asarray([[.01], [.09]]))
    def rate(parameters):
        a, b, time = parameters
        x = SpatialProfile([0., 2.], jax_numpy.stack((a, 3.*a)))
        history = TimeHistory([0., 1.], jax_numpy.asarray([1., 2.]))
        y = SpatialProfile([0., 2.], jax_numpy.stack((b, b)))
        state = FiberState(curvature_x=Field([FieldTerm(x, history)]),
                           curvature_y=Field([FieldTerm(y)]))
        return table.value(state.curvature_magnitude(1., time), REFERENCE)
    parameters = jax_numpy.asarray([1., 4., .5], dtype=jax_numpy.float64)
    value = jax.jit(rate)(parameters)
    gradient = jax.jit(jax.grad(rate))(parameters)
    hessian = jax.jit(jax.hessian(rate))(parameters)
    dx, dy = np.array([3., 0., 2.]), np.array([0., 1., 0.])
    d2x = np.array([[0., 0., 2.], [0., 0., 0.], [2., 0., 0.]])
    derivative = np.array([1.8, .8, 1.2])
    second = (np.outer(dx, dx) + np.outer(dy, dy) + 3.*d2x
              - np.outer(derivative, derivative))/5.
    slope = np.log(9.)/4.
    expected = .03
    expected_hessian = expected*(slope**2*np.outer(derivative, derivative) + slope*second)
    np.testing.assert_allclose(value, expected, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(gradient, expected*slope*derivative, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-11, atol=1e-14)


def test_jax_straight_fiber_has_finite_zero_bend_rate_derivatives(jax_numpy):
    import jax
    table = BendLossTable([3., 7.], [REFERENCE], jax_numpy.asarray([[.01], [.09]]))
    def rate(parameters):
        x = Field([FieldTerm(SpatialProfile([0., 2.], jax_numpy.stack((parameters[0], parameters[0]))))])
        y = Field([FieldTerm(SpatialProfile([0., 2.], jax_numpy.stack((parameters[1], parameters[1]))))])
        state = FiberState(curvature_x=x, curvature_y=y)
        return table.value(state.curvature_magnitude(1.), REFERENCE)
    parameters = jax_numpy.zeros((2,), dtype=jax_numpy.float64)
    np.testing.assert_array_equal(jax.jit(rate)(parameters), 0.)
    np.testing.assert_array_equal(jax.jit(jax.grad(rate))(parameters), [0., 0.])
    np.testing.assert_array_equal(jax.jit(jax.hessian(rate))(parameters), np.zeros((2, 2)))


@pytest.mark.parametrize("position_shape,frequency_shape", [
    ((), (2, 3)), ((2, 1), (3,)), ((0,), (2,)), ((2,), (0,)),
])
def test_constant_bend_preserves_outer_shapes_and_empty_queries(xp, position_shape, frequency_shape):
    table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.09]]))
    position = xp.full(position_shape, 1., dtype=xp.float64)
    frequency = xp.full(frequency_shape, REFERENCE, dtype=xp.float64)
    result = optical_path(make_fiber(table), make_state(xp)).attenuation_integral(position, frequency)
    assert result.dtype == xp.float64
    assert result.shape == position_shape + frequency_shape
    np.testing.assert_allclose(np.asarray(result), np.full(result.shape, .23), rtol=1e-14)


def test_bend_steps_and_profile_ends_do_not_leak_into_neighbouring_pieces(xp):
    table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.09]]))
    x = Field([FieldTerm(SpatialProfile([0., .5, .5, 2.], xp.asarray([0., 0., 3., 3.]))),
               FieldTerm(SpatialProfile([1., 1.], xp.asarray([100., 100.])))])
    y = Field([FieldTerm(SpatialProfile([.25, 1.5], xp.asarray([4., 4.])))])
    state = FiberState(curvature_x=x, curvature_y=y)
    position = xp.asarray([-1., 0., .25, .5, 1., 1.5, 2., 3.])
    actual = optical_path(make_fiber(table), state).attenuation_integral(position, REFERENCE)
    first = .25*.01*np.sqrt(3.)
    bend = np.array([0., 0., 0., first, first + .015, first + .03,
                     first + .035, first + .035])
    expected = .2*np.clip(np.asarray(position), 0., 2.) + bend
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-13, atol=0.)


def test_each_section_uses_its_own_bend_table_or_none(xp):
    table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.09]]))
    fiber = make_fiber(table)
    bent = fiber.sections[0].optics
    plain = replace(bent, bend_loss=None)
    stronger = replace(bent, bend_loss=replace(table, attenuation=3.*table.attenuation))
    fiber = replace(fiber, sections=[Section(0., .5, bent, .2, 0.),
                                     Section(.5, 1., plain, .4, 0.),
                                     Section(1., 2., stronger, .6, 0.)])
    result = optical_path(fiber, make_state(xp)).attenuation_integral(
        xp.asarray([.5, 1., 2.]), REFERENCE)
    np.testing.assert_allclose(np.asarray(result), [.115, .315, 1.005], rtol=1e-14)


def test_linear_curvature_crosses_both_sides_of_the_table_and_frequency_is_integrated(xp):
    levels = np.array([3., 4., 7.])
    rates = .01*np.exp(.5*(levels - 3.))
    table = BendLossTable(levels, [REFERENCE - 1e9, REFERENCE + 1e9],
                          xp.asarray(np.column_stack((rates, 4.*rates)).tolist()))
    state = FiberState(curvature_x=Field([
        FieldTerm(SpatialProfile([0., 2.], xp.asarray([-6., 6.])))]))
    position = xp.asarray([[-1., 0., .5], [1., 1.5, 2.]])
    frequency = xp.asarray([REFERENCE - 1e9, REFERENCE, REFERENCE + 1e9])
    result = optical_path(make_fiber(table), state).attenuation_integral(position, frequency)
    area = .01*np.expm1(1.5)/3.
    bend = np.array([[0., 0., area], [area, area, 2.*area]])
    expected = (.2*np.clip(np.asarray(position), 0., 2.))[..., None] + bend[..., None]*[1., 2., 4.]
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-13, atol=0.)


def test_linear_vector_curvature_matches_independent_quadrature(xp):
    from scipy.integrate import quad
    table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.09]]))
    state = FiberState(
        curvature_x=Field([FieldTerm(SpatialProfile([0., 2.], xp.asarray([3., 5.])))]),
        curvature_y=Field([FieldTerm(SpatialProfile([0., 2.], xp.asarray([4., 2.])))]))
    result = optical_path(make_fiber(table), state).attenuation_integral(2., REFERENCE)
    bend, _ = quad(lambda s: .01*np.exp(np.log(9.)*(np.hypot(3. + s, 4. - s) - 3.)/4.),
                    0., 2., epsabs=1e-14, epsrel=1e-12)
    np.testing.assert_allclose(np.asarray(result), .4 + bend, rtol=1e-12)


def test_off_grid_frequency_interpolation_precedes_spatial_integration(xp):
    table = BendLossTable([3., 7.], [REFERENCE - 1e9, REFERENCE + 1e9],
                          xp.asarray([[.01, .04], [.01*np.exp(1.), .04*np.exp(3.)]]))
    state = FiberState(curvature_x=Field([
        FieldTerm(SpatialProfile([0., 2.], xp.asarray([3., 7.])))]))
    fractions = np.array([.25, .5, .75])
    frequency = xp.asarray((REFERENCE - 1e9 + fractions*2e9).tolist())
    result = optical_path(make_fiber(table), state).attenuation_integral(2., frequency)
    slope = .25 + .5*fractions
    expected = .4 + .01*4.**fractions*np.expm1(4.*slope)/(2.*slope)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-13)


def test_coherent_point_and_rayleigh_returns_include_bend_loss_at_each_position(xp):
    table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.09]]))
    sample = RayleighRealization(xp.asarray([.01j, .02]), [.5, 1.5], [0., 1., 2.])
    fiber = replace(make_fiber(table), rayleigh=sample)
    state = make_state(xp)
    frequency = xp.asarray([REFERENCE, REFERENCE + 1e8])
    reference = optical_path(fiber)
    point = .1*np.asarray(reference.round_trip(0., 2., frequency))*np.exp(-.06)
    rayleigh = sum(coefficient*np.asarray(reference.round_trip(0., position, frequency))
                    *np.exp(-.03*position)
                    for position, coefficient in zip((.5, 1.5), (.01j, .02)))
    np.testing.assert_allclose(np.asarray(point_response(fiber, frequency, state)), point, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(rayleigh_response(fiber, frequency, state)), rayleigh,
                               rtol=1e-13)
    result = compute_response(fiber, Probe(frequency), state)
    np.testing.assert_allclose(np.asarray(result.reflection), point + rayleigh, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(result.transmission),
                               .8*np.asarray(reference.transmission(0., 2., frequency))*np.exp(-.03),
                               rtol=1e-13)


def test_unsupported_curvature_is_nan_only_after_its_nonzero_material_span(xp):
    table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.09]]))
    state = FiberState(curvature_x=Field([
        FieldTerm(SpatialProfile([1., 2.], xp.asarray([8., 8.])))]))
    result = optical_path(make_fiber(table), state).attenuation_integral(
        xp.asarray([0., .5, 1., 1.5, 2.]), REFERENCE)
    np.testing.assert_allclose(np.asarray(result)[:3], [0., .1, .2], rtol=1e-14)
    assert np.all(np.isnan(np.asarray(result)[3:]))


def test_bend_loss_requires_a_static_state():
    with pytest.raises(TypeError, match="FiberState"):
        FiberLoss(make_fiber(), None)
    with pytest.raises(NotImplementedError, match="static FiberState"):
        FiberLoss(make_fiber(), make_state(numpy_api, moving=True))


def test_jax_bend_tables_select_the_namespace_for_all_response_entry_points(jax_numpy):
    table = BendLossTable([3., 7.], [REFERENCE], jax_numpy.asarray([[.01], [.09]]))
    fiber = make_fiber(table)
    state = FiberState(curvature_x=Field([FieldTerm(SpatialProfile([0., 2.], [5., 5.]))]))
    frequency = np.asarray([REFERENCE])
    path = optical_path(fiber, state)
    assert path.phase_factor(0., 2., frequency).dtype == np.complex128
    for method, args in ((path.attenuation_integral, (2., frequency)),
                          (path.transmission, (0., 2., frequency)),
                          (point_response, (fiber, frequency, state)),
                          (rayleigh_response, (fiber, frequency, state)),
                          (compute_response, (fiber, Probe(frequency), state))):
        with pytest.raises(TypeError):
            method(*args)
        result = method(*args, xp=jax_numpy)
        if hasattr(result, "reflection"):
            result = result.reflection
        assert type(result) is type(table.attenuation)


def test_jax_integrated_onset_keeps_the_moving_crossing_derivatives(jax_numpy):
    import jax
    table = BendLossTable([3., 7.], [REFERENCE], jax_numpy.asarray([[.01], [.01*np.exp(2.)]]))
    def cumulative(initial):
        profile = SpatialProfile([0., 2.], jax_numpy.stack((initial, initial + 4.)))
        state = FiberState(curvature_x=Field([FieldTerm(profile)]))
        return optical_path(make_fiber(table), state).attenuation_integral(2., REFERENCE)
    initial = jax_numpy.asarray(2., dtype=jax_numpy.float64)
    np.testing.assert_allclose(jax.jit(cumulative)(initial), .4 + .01*np.expm1(1.5), rtol=1e-13)
    np.testing.assert_allclose(jax.jit(jax.grad(cumulative))(initial), .005*np.exp(1.5), rtol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.hessian(cumulative))(initial), .0025*np.exp(1.5), rtol=1e-11)


def test_jax_bend_position_derivative_is_the_local_total_rate(jax_numpy):
    import jax
    table = BendLossTable([3., 7.], [REFERENCE], jax_numpy.asarray([[.01], [.01*np.exp(2.)]]))
    state = FiberState(curvature_x=Field([
        FieldTerm(SpatialProfile([0., 2.], jax_numpy.asarray([2., 6.])))]))
    def cumulative(position):
        return optical_path(make_fiber(table), state).attenuation_integral(position, REFERENCE)
    position = jax_numpy.asarray(1.7, dtype=jax_numpy.float64)
    np.testing.assert_allclose(jax.jit(jax.grad(cumulative))(position), .2 + .01*np.exp(1.2), rtol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.hessian(cumulative))(position), .01*np.exp(1.2), rtol=1e-11)


def test_jax_response_keeps_curvature_and_table_gradients_and_hessian(jax_numpy):
    import jax
    def power(parameters):
        x, y, low, high = parameters
        table = BendLossTable([3., 7.], [REFERENCE], jax_numpy.reshape(
            jax_numpy.stack((low, high)), (2, 1)))
        state = FiberState(
            curvature_x=Field([FieldTerm(SpatialProfile([0., 2.], jax_numpy.stack((x, x))))]),
            curvature_y=Field([FieldTerm(SpatialProfile([0., 2.], jax_numpy.stack((y, y))))]))
        result = compute_response(make_fiber(table), Probe(jax_numpy.asarray(REFERENCE)), state)
        return jax_numpy.abs(result.transmission)**2
    def expected(parameters):
        x, y, low, high = parameters
        fraction = (jax_numpy.sqrt(x*x + y*y) - 3.)/4.
        rate = jax_numpy.exp((1. - fraction)*jax_numpy.log(low) + fraction*jax_numpy.log(high))
        return .64*jax_numpy.exp(-.4 - 2.*rate)
    parameters = jax_numpy.asarray([3., 4., .01, .09])
    for transform in (lambda f: f, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(power))(parameters),
                                   transform(expected)(parameters), rtol=1e-11, atol=1e-13)


def test_jax_straight_path_keeps_finite_zero_bend_derivatives(jax_numpy):
    import jax
    table = BendLossTable([3., 7.], [REFERENCE], jax_numpy.asarray([[.01], [.09]]))
    def power(parameters):
        x, y = parameters
        state = FiberState(
            curvature_x=Field([FieldTerm(SpatialProfile([0., 2.], jax_numpy.stack((x, x))))]),
            curvature_y=Field([FieldTerm(SpatialProfile([0., 2.], jax_numpy.stack((y, y))))]))
        field = optical_path(make_fiber(table), state).transmission(0., 2., REFERENCE)
        return jax_numpy.abs(field)**2
    parameters = jax_numpy.zeros((2,), dtype=jax_numpy.float64)
    np.testing.assert_allclose(jax.jit(power)(parameters), np.exp(-.4), rtol=1e-14)
    np.testing.assert_array_equal(jax.jit(jax.grad(power))(parameters), [0., 0.])
    np.testing.assert_array_equal(jax.jit(jax.hessian(power))(parameters), np.zeros((2, 2)))


def test_jax_constant_profile_keeps_independent_endpoint_derivatives(jax_numpy):
    import jax
    table = BendLossTable([3., 7.], [REFERENCE], jax_numpy.asarray([[.01], [.09]]))
    def cumulative(values):
        state = FiberState(curvature_x=Field([FieldTerm(SpatialProfile([0., 2.], values))]))
        return optical_path(make_fiber(table), state).attenuation_integral(2., REFERENCE)
    values = jax_numpy.asarray([5., 5.])
    slope, rate = np.log(9.)/4., .03
    np.testing.assert_allclose(jax.jit(jax.grad(cumulative))(values),
                               [rate*slope, rate*slope], rtol=1e-12)
    expected = rate*slope**2*np.array([[2./3., 1./3.], [1./3., 2./3.]])
    np.testing.assert_allclose(jax.jit(jax.hessian(cumulative))(values), expected, rtol=1e-11)
