import math

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Field, FieldTerm, Fiber, FiberState, GuideOptics, Probe, Section,
                        SpatialProfile, compute_response)
from pyrayleigh.modes import (loss_db_from_attenuation, per_turn_table, tabulated_bend_loss_db,
                              tabulated_bend_loss_table)

REFERENCE = 193414489032258.06
RADII = [.02, .005, .01]
LOSSES = [.03, 30., 3.]


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


def test_log_radius_interpolation_keeps_sorted_samples_and_broadcast_turns(xp):
    radius = xp.asarray([.005, .0075, .01, .015, .02])
    turns = xp.asarray([[0.], [.25], [2.]])
    result = tabulated_bend_loss_db(radius, turns, RADII, xp.asarray(LOSSES))
    expected = np.array([30., np.sqrt(90.), 3., .3, .03])*[[0.], [.25], [2.]]
    assert result.dtype == xp.float64
    assert type(result) is type(radius)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-14)
    empty = tabulated_bend_loss_db(xp.zeros((0,)), 1., RADII, xp.asarray(LOSSES))
    assert empty.shape == (0,)


def test_direct_model_clamps_to_end_samples_outside_the_radius_range(xp):
    result = tabulated_bend_loss_db(xp.asarray([.001, .004, .021, .1]), .5,
                                    RADII, xp.asarray(LOSSES))
    np.testing.assert_allclose(np.asarray(result), [15., 15., .015, .015], rtol=1e-14)


def test_per_turn_table_keeps_native_loss_order_and_physical_turn_lengths(xp):
    losses = xp.asarray([0., 30., 3.], dtype=xp.float32)
    table = per_turn_table(iter(RADII), losses, frequency=REFERENCE)
    assert table.shape == (3, 1)
    assert table.attenuation.dtype == xp.float64
    assert type(table.attenuation) is type(losses)
    np.testing.assert_array_equal(table.curvatures, [50., 100., 200.])
    lengths = xp.asarray((2.*math.pi*np.array([.02, .01, .005])).tolist())
    recovered = loss_db_from_attenuation(table.attenuation[:, 0], lengths)
    np.testing.assert_allclose(np.asarray(recovered), [0., 3., 30.], rtol=1e-14)


def test_resampling_retains_original_points_and_uses_the_radius_log_law(xp):
    table = tabulated_bend_loss_table(RADII, xp.asarray(LOSSES), frequency=REFERENCE)
    assert table.shape == (65, 1)
    assert table.attenuation.dtype == xp.float64
    for radius in RADII:
        assert 1./radius in table.curvatures
    radii = 1./table.curvatures
    recovered = np.asarray(table.attenuation[:, 0])*2.*math.pi*radii*10./math.log(10.)
    expected = np.exp(np.interp(radii, [.005, .01, .02], np.log([30., 3., .03])))
    np.testing.assert_allclose(recovered, expected, rtol=1e-13)


def test_more_subdivisions_reduce_the_curvature_interpolation_error(xp):
    radii = [.005, .02]
    losses = xp.asarray([30., .03])
    query = np.linspace(.005, .02, 101)
    expected = np.exp(np.interp(query, radii, np.log([30., .03])))
    expected = expected*math.log(10.)/(20.*math.pi*query)
    errors = []
    for count in (2, 32):
        table = tabulated_bend_loss_table(radii, losses, points_per_interval=count)
        result = table.value(xp.asarray((1./query).tolist()), REFERENCE)
        errors.append(np.max(np.abs(np.asarray(result)/expected - 1.)))
    assert errors[1] < errors[0]/20.


def test_resampled_table_keeps_bend_loss_table_curvature_limits(xp):
    table = tabulated_bend_loss_table(RADII, xp.asarray(LOSSES), points_per_interval=2)
    result = table.value(xp.asarray([0., 40., 50., 200., 201.]), REFERENCE)
    expected = [0., 0., .03*math.log(10.)/(20.*math.pi*.02),
                 30.*math.log(10.)/(20.*math.pi*.005), np.nan]
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-14, equal_nan=True)


@pytest.mark.parametrize("function,args", [
    (per_turn_table, ([.005, .005], [1., 2.])),
    (tabulated_bend_loss_db, (.01, 1., [0., .01], [1., 2.])),
    (tabulated_bend_loss_table, ([.005], [1.])),
    (per_turn_table, (RADII, [1., 2.])),
    (tabulated_bend_loss_db, (.01, 1., RADII, [[1., 2., 3.]])),
    (tabulated_bend_loss_table, (RADII, 1.)),
])
def test_table_geometry_and_loss_shapes_are_checked(function, args):
    with pytest.raises(ValueError):
        function(*args)


@pytest.mark.parametrize("count,error", [(0, ValueError), (-1, ValueError),
                                         (True, TypeError), (2., TypeError)])
def test_subdivision_count_is_a_fixed_positive_integer(count, error):
    with pytest.raises(error):
        tabulated_bend_loss_table(RADII, LOSSES, points_per_interval=count)


def test_numpy_integer_subdivision_count_is_supported():
    table = tabulated_bend_loss_table(RADII, LOSSES, points_per_interval=np.int64(2))
    assert table.shape == (5, 1)


@pytest.mark.parametrize("function,args", [
    (per_turn_table, (RADII, [1j, 2j, 3j])),
    (tabulated_bend_loss_db, (1j, 1., RADII, LOSSES)),
    (tabulated_bend_loss_table, (RADII, [True, False, True])),
])
def test_losses_and_queries_are_real(function, args):
    with pytest.raises(TypeError):
        function(*args)


def test_native_queries_do_not_detach_fixed_metadata(jax_numpy):
    with pytest.raises(TypeError, match="fixed real"):
        tabulated_bend_loss_db(.01, 1., jax_numpy.asarray(RADII), jax_numpy.asarray(LOSSES))
    with pytest.raises(TypeError, match="fixed integer"):
        tabulated_bend_loss_table(RADII, LOSSES, points_per_interval=jax_numpy.asarray(2))


def test_mixed_namespaces_need_explicit_selection(jax_numpy):
    radius = np.asarray([.0075])
    losses = jax_numpy.asarray(LOSSES)
    with pytest.raises(TypeError):
        tabulated_bend_loss_db(radius, 1., RADII, losses)
    result = tabulated_bend_loss_db(radius, 1., RADII, losses, xp=jax_numpy)
    assert type(result) is type(losses)


def test_jax_log_loss_keeps_radius_turn_and_sample_derivatives(jax_numpy):
    import jax
    def loss(parameters):
        radius, turns, low, high = parameters
        return tabulated_bend_loss_db(radius, turns, [.005, .015],
                                       jax_numpy.stack((low, high)))
    parameters = jax_numpy.asarray([.0075, .75, 30., .3])
    fraction, width, turns, low, high = .25, .01, .75, 30., .3
    expected = turns*low**(1. - fraction)*high**fraction
    slope = np.array([math.log(high/low)/width, 1./turns, (1. - fraction)/low, fraction/high])
    second = np.diag([0., -1./turns**2, -(1. - fraction)/low**2, -fraction/high**2])
    second[0, 2] = second[2, 0] = -1./(width*low)
    second[0, 3] = second[3, 0] = 1./(width*high)
    np.testing.assert_allclose(jax.jit(loss)(parameters), expected, rtol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(parameters), expected*slope, rtol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.hessian(loss))(parameters),
                               expected*(np.outer(slope, slope) + second), rtol=1e-11, atol=1e-12)
    def clamped(radius):
        return tabulated_bend_loss_db(radius, 1., [.005, .015], jax_numpy.asarray([30., .3]))
    radius = jax_numpy.asarray(.02)
    np.testing.assert_allclose(jax.jit(clamped)(radius), .3, rtol=1e-14)
    np.testing.assert_array_equal(jax.jit(jax.grad(clamped))(radius), 0.)
    np.testing.assert_array_equal(jax.jit(jax.hessian(clamped))(radius), 0.)


def test_jax_resampled_losses_keep_gradients_through_the_optical_response(jax_numpy):
    import jax
    radius, turns = .01, .75
    length = 2.*math.pi*radius*turns
    profile = SpatialProfile([0., length], jax_numpy.asarray([1./radius, 1./radius]))
    state = FiberState(curvature_x=Field([FieldTerm(profile)]))
    probe = Probe(jax_numpy.asarray(REFERENCE))
    def power(losses):
        table = tabulated_bend_loss_table([.005, .015], losses, points_per_interval=2)
        guide = GuideOptics(REFERENCE, 1.4682, 1.4682, bend_loss=table)
        fiber = Fiber("spool", length, 293.15, REFERENCE - 1e9, REFERENCE + 1e9,
                      [Section(0., length, guide, 0., 0.)])
        response = compute_response(fiber, probe, state)
        return jax_numpy.abs(response.transmission)**2
    low, high = 30., .3
    rate = math.sqrt(low*high)
    factor = turns*math.log(10.)/10.
    expected = math.exp(-factor*rate)
    slope = -factor*rate/2.*np.array([1./low, 1./high])
    second = -factor*rate/4.*np.array([[-1./low**2, 1./(low*high)],
                                      [1./(low*high), -1./high**2]])
    parameters = jax_numpy.asarray([low, high])
    np.testing.assert_allclose(jax.jit(power)(parameters), expected, rtol=1e-13)
    np.testing.assert_allclose(jax.jit(jax.grad(power))(parameters), expected*slope, rtol=1e-11)
    np.testing.assert_allclose(jax.jit(jax.hessian(power))(parameters),
                               expected*(np.outer(slope, slope) + second), rtol=1e-10, atol=1e-12)
