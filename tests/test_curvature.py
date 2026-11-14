import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (BendLossTable, Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, Probe, Section, SpatialProfile, TimeHistory,
                        compute_response, optical_path)

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
def test_bend_path_integration_is_explicitly_pending(xp, moving):
    table = BendLossTable([3., 7.], [REFERENCE], xp.asarray([[.01], [.09]]))
    fiber = make_fiber(table)
    state = make_state(xp, moving)
    probe = Probe(xp.asarray([REFERENCE]), xp.asarray([1.]))
    with pytest.raises(NotImplementedError, match="bend-loss integration"):
        optical_path(fiber, state.at_time(1.))
    with pytest.raises(NotImplementedError, match="bend-loss integration"):
        compute_response(fiber, probe, state)
    assert optical_path(fiber).state.is_static


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
