import math

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (BendLossTable, Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, Probe, Section, SpatialProfile, compute_response,
                        optical_path)
from pyrayleigh.modes import (attenuation_from_loss_db, bend_length, exponential_bend_loss_db,
                              exponential_bend_loss_table, loss_db_from_attenuation)
from pyrayleigh.modes.bends import DEFAULT_TABLE_RADII, NOMINAL_FREQUENCY

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


def make_fiber(table, length=2.):
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682, bend_loss=table)
    return Fiber("spool-1", length, 293.15, REFERENCE - 1e9, REFERENCE + 1e9,
                  [Section(0., length, guide, 0., 0.)],
                  [PointReflector(length, .1, 0., .8, .7)])


def curvature_state(start, end, radius, xp):
    profile = SpatialProfile([start, end], xp.asarray([1./radius, 1./radius]))
    return FiberState(curvature_x=Field([FieldTerm(profile)]))


def test_exponential_loss_broadcasts_radius_and_fractional_turns(xp):
    radius = xp.asarray([[5e-3], [10e-3]], dtype=xp.float32)
    turns = xp.asarray([0., .5, 2.], dtype=xp.float64)
    result = exponential_bend_loss_db(radius, turns)
    expected = 100.*np.exp(-np.asarray(radius, dtype=np.float64)/5e-3)*[0., .5, 2.]
    assert result.dtype == xp.float64
    assert type(result) is type(radius)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-14)
    length = bend_length(radius, turns)
    np.testing.assert_allclose(np.asarray(length), 2.*np.pi*np.asarray(radius, dtype=np.float64)*[0., .5, 2.],
                               rtol=1e-14)
    empty = exponential_bend_loss_db(xp.zeros((0, 1)), turns)
    assert empty.shape == (0, 3)


def test_default_and_custom_pyofdr_parameters(xp):
    np.testing.assert_allclose(np.asarray(exponential_bend_loss_db(5e-3, 1., xp=xp)),
                               100./math.e, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(exponential_bend_loss_db(10e-3, 1., 50., 10e-3, xp=xp)),
                               50./math.e, rtol=1e-14)
    assert float(exponential_bend_loss_db(100e-3, 10., xp=xp)) < 1e-5


def test_total_loss_and_alpha_conversions_keep_broadcast_shapes(xp):
    losses = xp.asarray([[0.], [10.]], dtype=xp.float64)
    lengths = xp.asarray([.5, 2.], dtype=xp.float64)
    alpha = attenuation_from_loss_db(losses, lengths)
    assert alpha.dtype == xp.float64
    assert type(alpha) is type(lengths)
    np.testing.assert_allclose(np.asarray(alpha), [[0., 0.], [2.*math.log(10.), .5*math.log(10.)]],
                               rtol=1e-14)
    recovered = loss_db_from_attenuation(alpha, lengths)
    np.testing.assert_allclose(np.asarray(recovered), [[0., 0.], [10., 10.]], rtol=1e-14)
    np.testing.assert_allclose(np.exp(-np.asarray(alpha)*np.asarray(lengths)),
                               [[1., 1.], [.1, .1]], rtol=1e-14)


def test_table_sorts_fixed_radii_and_converts_one_physical_turn(xp):
    radii = [10e-3, 5e-3, 20e-3]
    amplitude = xp.asarray(40., dtype=xp.float64)
    critical = xp.asarray(10e-3, dtype=xp.float64)
    table = exponential_bend_loss_table(iter(radii), amplitude, critical, frequency=REFERENCE)
    assert isinstance(table, BendLossTable)
    assert table.shape == (3, 1)
    assert table.attenuation.dtype == xp.float64
    assert type(table.attenuation) is type(amplitude)
    np.testing.assert_array_equal(table.curvatures, [50., 100., 200.])
    ordered = np.array([20e-3, 10e-3, 5e-3])
    total = loss_db_from_attenuation(table.attenuation[:, 0], 2.*math.pi*xp.asarray(ordered.tolist()))
    np.testing.assert_allclose(np.asarray(total), 40.*np.exp(-ordered/10e-3), rtol=1e-14)
    queried = table.value(xp.asarray([100.]), xp.asarray([REFERENCE - 1e9, REFERENCE + 1e9]))
    np.testing.assert_allclose(np.asarray(queried), np.repeat(np.asarray(table.attenuation[1:2, :]), 2, axis=1),
                               rtol=1e-14)


def test_table_has_the_documented_default_grid():
    table = exponential_bend_loss_table()
    assert table.shape == (200, 1)
    assert not DEFAULT_TABLE_RADII.flags.writeable
    np.testing.assert_array_equal(table.curvatures, 1./DEFAULT_TABLE_RADII)
    np.testing.assert_array_equal(table.frequencies, [NOMINAL_FREQUENCY])


@pytest.mark.parametrize("amplitude,critical", [(0., 5e-3), (100., 1e-8)])
def test_zero_and_underflowed_losses_stay_zero(xp, amplitude, critical):
    table = exponential_bend_loss_table([5e-3, 10e-3], amplitude, critical, xp=xp)
    np.testing.assert_array_equal(np.asarray(table.attenuation), [[0.], [0.]])
    assert table.attenuation.dtype == xp.float64


def test_table_uses_existing_curvature_interpolation_between_sampled_radii(xp):
    table = exponential_bend_loss_table([5e-3, 20e-3], xp=xp)
    result = table.value(125., REFERENCE)
    expected = np.sqrt(np.prod(np.asarray(table.attenuation)))
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-14)


def test_sampled_arc_has_pyofdr_total_loss_in_transmission_and_reflection(xp):
    radius, turns = 15e-3, .75
    length = 2.*math.pi*radius*turns
    table = exponential_bend_loss_table([10e-3, radius, 20e-3], xp=xp)
    fiber = make_fiber(table, length)
    state = curvature_state(0., length, radius, xp)
    result = compute_response(fiber, Probe(xp.asarray([REFERENCE, REFERENCE + 1e9])), state)
    total_db = turns*100.*math.exp(-radius/5e-3)
    np.testing.assert_allclose(np.abs(np.asarray(result.transmission))**2,
                               .64*10.**(-total_db/10.), rtol=1e-13)
    np.testing.assert_allclose(np.abs(np.asarray(result.reflection)),
                               .1*10.**(-total_db/10.), rtol=1e-13)


def test_pyofdr_loss_can_be_spread_over_an_arbitrary_material_span(xp):
    radius, turns, start, end = 15e-3, .75, .4, 1.1
    total = exponential_bend_loss_db(radius, turns, xp=xp)
    alpha = attenuation_from_loss_db(total, end - start)
    table = BendLossTable([1./radius, 2./radius], [REFERENCE], xp.reshape(xp.stack((alpha, alpha)), (2, 1)))
    state = curvature_state(start, end, radius, xp)
    path = optical_path(make_fiber(table), state)
    position = xp.asarray([-1., 0., .3, start, .6, end, 2., 3.])
    depth = path.attenuation_integral(position, REFERENCE)
    loss = loss_db_from_attenuation(depth, 1.)
    expected = turns*100.*math.exp(-radius/5e-3)*np.clip(np.asarray(position) - start, 0., end - start)/(end - start)
    np.testing.assert_allclose(np.asarray(loss), expected, rtol=1e-13, atol=0.)


@pytest.mark.parametrize("radii,error", [
    ([], ValueError), ([.01], ValueError), ([0., .01], ValueError),
    ([-.01, .01], ValueError), ([.01, .01], ValueError),
    ([.01, float("inf")], ValueError), ([.01, float("nan")], ValueError),
    ([True, .01], TypeError), ([.01j, .02], TypeError), ([[.01, .02]], TypeError),
])
def test_table_requires_fixed_positive_distinct_radii(radii, error):
    with pytest.raises(error):
        exponential_bend_loss_table(radii)


@pytest.mark.parametrize("name,value,error", [
    ("loss_per_turn_db", [100.], ValueError), ("critical_radius", [5e-3], ValueError),
    ("loss_per_turn_db", 100j, TypeError), ("critical_radius", True, TypeError),
    ("frequency", 0., ValueError), ("frequency", float("inf"), ValueError),
])
def test_table_parameter_shapes_dtypes_and_fixed_frequency(name, value, error):
    with pytest.raises(error):
        exponential_bend_loss_table([5e-3, 10e-3], **{name: value})


@pytest.mark.parametrize("function,args", [
    (bend_length, (1j, 1.)), (attenuation_from_loss_db, (1j, 1.)),
    (loss_db_from_attenuation, (1j, 1.)), (exponential_bend_loss_db, (1j, 1.)),
])
def test_bend_kernels_require_real_values(function, args):
    with pytest.raises(TypeError):
        function(*args)


def test_jax_radii_and_frequency_cannot_be_detached_into_table_metadata(jax_numpy):
    with pytest.raises(TypeError, match="fixed real"):
        exponential_bend_loss_table(jax_numpy.asarray([5e-3, 10e-3]))
    with pytest.raises(TypeError, match="fixed real"):
        exponential_bend_loss_table([5e-3, 10e-3], frequency=jax_numpy.asarray(REFERENCE))


def test_mixed_model_parameters_need_an_explicit_namespace(jax_numpy):
    amplitude, critical = np.asarray(100.), jax_numpy.asarray(5e-3)
    for function, args in ((exponential_bend_loss_db, (.01, 1., amplitude, critical)),
                           (exponential_bend_loss_table, ([.005, .01], amplitude, critical))):
        with pytest.raises(TypeError):
            function(*args)
        result = function(*args, xp=jax_numpy)
        if isinstance(result, BendLossTable):
            result = result.attenuation
        assert type(result) is type(critical)


def test_jax_total_loss_keeps_radius_turns_and_model_derivatives(jax_numpy):
    import jax
    def loss(parameters):
        return exponential_bend_loss_db(*parameters)
    values = np.array([7e-3, 2.5, 100., 5e-3])
    radius, turns, amplitude, critical = values
    expected = turns*amplitude*math.exp(-radius/critical)
    slope = np.array([-1./critical, 1./turns, 1./amplitude, radius/critical**2])
    second = np.diag([0., -1./turns**2, -1./amplitude**2, -2.*radius/critical**3])
    second[0, 3] = second[3, 0] = 1./critical**2
    parameters = jax_numpy.asarray(values.tolist())
    np.testing.assert_allclose(jax.jit(loss)(parameters), expected, rtol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(parameters), expected*slope, rtol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.hessian(loss))(parameters),
                               expected*(np.outer(slope, slope) + second), rtol=1e-11, atol=1e-12)


def test_jax_table_model_gradients_reach_the_optical_response(jax_numpy):
    import jax
    radius, turns = 15e-3, .75
    length = 2.*math.pi*radius*turns
    state = curvature_state(0., length, radius, jax_numpy)
    probe = Probe(jax_numpy.asarray(REFERENCE))
    def power(parameters):
        table = exponential_bend_loss_table([10e-3, radius, 20e-3], *parameters)
        response = compute_response(make_fiber(table, length), probe, state)
        return jax_numpy.abs(response.transmission)**2
    amplitude, critical = 4., 10e-3
    factor = turns*math.log(10.)/10.
    exponential = math.exp(-radius/critical)
    expected = .64*math.exp(-factor*amplitude*exponential)
    slope = -factor*exponential*np.array([1., amplitude*radius/critical**2])
    second = -factor*exponential*np.array([
        [0., radius/critical**2],
        [radius/critical**2, amplitude*(radius**2/critical**4 - 2.*radius/critical**3)]])
    parameters = jax_numpy.asarray([amplitude, critical])
    np.testing.assert_allclose(jax.jit(power)(parameters), expected, rtol=1e-13)
    np.testing.assert_allclose(jax.jit(jax.grad(power))(parameters), expected*slope, rtol=1e-11)
    np.testing.assert_allclose(jax.jit(jax.hessian(power))(parameters),
                               expected*(np.outer(slope, slope) + second), rtol=1e-10, atol=1e-12)
