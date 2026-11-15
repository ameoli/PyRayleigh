import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import crosstalk_from_phases, crosstalk_pairs


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


def test_linear_pairs_keep_the_directed_pyofdr_phase_order():
    assert crosstalk_pairs(1) == ()
    assert crosstalk_pairs(np.int64(2)) == ((1, 0), (0, 1))
    assert crosstalk_pairs(3) == ((1, 0), (0, 1), (2, 1), (1, 2))


def test_hex7_pairs_include_the_centre_and_wrapped_ring_neighbors():
    pairs = crosstalk_pairs(7, "hex7")
    assert pairs[:6] == tuple((source, 0) for source in range(1, 7))
    assert pairs[6:9] == ((0, 1), (2, 1), (6, 1))
    assert pairs[-3:] == ((0, 6), (5, 6), (1, 6))
    assert len(pairs) == len(set(pairs)) == 24
    assert all(source != target and (target, source) in pairs for source, target in pairs)
    assert [sum(target == core for _, target in pairs) for core in range(7)] == [6, 3, 3, 3, 3, 3, 3]


@pytest.mark.parametrize("count,topology,error", [(True, "linear", TypeError),
                                                  (np.bool_(True), "linear", TypeError),
                                                  (2., "linear", TypeError),
                                                  (np.array(2), "linear", TypeError),
                                                  (0, "linear", ValueError), (-1, "linear", ValueError),
                                                  (3, "hex7", ValueError), (2, "ring", ValueError),
                                                  (2, None, TypeError)])
def test_topology_is_fixed_and_validated(count, topology, error):
    with pytest.raises(error):
        crosstalk_pairs(count, topology)


def test_linear_mixing_keeps_complex_phase_and_power_per_km_units(xp):
    profile = xp.asarray([[1. + 1j, 2., 3j, -1.], [2., 1j, -1., .5],
                           [-1j, 3., 2., 1j]], dtype=xp.complex64)
    positions = xp.asarray([0., 250., 1000., 1500.])
    phases = xp.asarray([[0., .2, .4, .6], [.3, .5, .7, .9],
                          [.1, .2, .3, .4], [-.2, -.4, -.6, -.8]])
    actual = crosstalk_from_phases(profile, positions, xp.asarray(-30.), phases)
    original, angles = np.asarray(profile, dtype=complex), np.asarray(phases)
    amplitude = np.sqrt(1e-3*np.asarray(positions)/1000.)
    expected = np.stack((original[0] + original[1]*amplitude*np.exp(1j*angles[0]),
                         original[1] + original[0]*amplitude*np.exp(1j*angles[1])
                                     + original[2]*amplitude*np.exp(1j*angles[2]),
                         original[2] + original[1]*amplitude*np.exp(1j*angles[3])))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(actual)[:, 0], original[:, 0])
    assert actual.dtype == xp.complex128 and type(actual) is type(profile)
    np.testing.assert_array_equal(np.asarray(profile), original)
    np.testing.assert_array_equal(np.asarray(phases), angles)


def test_first_order_mixing_uses_original_sources_without_a_second_hop(xp):
    profile = xp.asarray([[1., 1.], [0., 0.], [0., 0.]])
    result = crosstalk_from_phases(profile, xp.asarray([0., 1000.]), -30., xp.zeros((4, 2)))
    np.testing.assert_array_equal(np.asarray(result)[0], [1., 1.])
    np.testing.assert_allclose(np.asarray(result)[1], [0., np.sqrt(.001)], rtol=1e-14)
    np.testing.assert_array_equal(np.asarray(result)[2], [0., 0.])


def test_hex7_mixes_six_center_neighbors_and_three_per_outer_core(xp):
    profile = xp.ones((7, 2))
    result = crosstalk_from_phases(profile, xp.asarray([0., 1000.]), -30.,
                                   xp.zeros((24, 2)), topology="hex7")
    expected = np.ones((7, 2))
    expected[:, 1] += np.array([6., 3., 3., 3., 3., 3., 3.])*np.sqrt(.001)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("phase", [0., np.pi])
def test_neighbor_fields_can_cancel_or_add_coherently(xp, phase):
    profile = xp.asarray([[1.], [0.], [-1.]])
    phases = xp.asarray([[0.], [0.], [phase], [0.]])
    actual = crosstalk_from_phases(profile, xp.asarray([1000.]), -30., phases)
    expected = np.sqrt(.001)*(1. - np.exp(1j*phase))
    np.testing.assert_allclose(np.asarray(actual)[1, 0], expected, rtol=1e-14, atol=1e-17)


@pytest.mark.parametrize("count,columns,topology", [(1, 2, "linear"), (3, 0, "linear"), (7, 0, "hex7")])
def test_single_core_and_empty_profiles_are_native_copies(xp, count, columns, topology):
    profile = xp.ones((count, columns), dtype=xp.float32)
    phases = xp.zeros((len(crosstalk_pairs(count, topology)), columns))
    result = crosstalk_from_phases(profile, xp.ones((columns,)), -30., phases, topology=topology)
    assert result.shape == (count, columns) and result.dtype == xp.complex128
    assert type(result) is type(profile)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(profile))


def test_reusing_phases_is_deterministic_and_numpy_output_does_not_alias_the_input():
    profile = np.array([[1j, 1.], [2., -.5j]])
    positions, phases = np.array([0., 1000.]), np.array([[.1, .2], [.3, .4]])
    first = crosstalk_from_phases(profile, positions, -30., phases)
    second = crosstalk_from_phases(profile, positions, -30., phases)
    np.testing.assert_array_equal(first, second)
    first[:] = 0.
    np.testing.assert_array_equal(profile, [[1j, 1.], [2., -.5j]])
    np.testing.assert_array_equal(positions, [0., 1000.])
    np.testing.assert_array_equal(phases, [[.1, .2], [.3, .4]])


@pytest.mark.parametrize("name,value,error", [("profile", [1., 2.], ValueError),
                                              ("profile", np.ones((0, 2)), ValueError),
                                              ("positions", [0.], ValueError),
                                              ("positions", [[0., 1.]], ValueError),
                                              ("xt_db_per_km", [-30.], ValueError),
                                              ("phases", [[0., 0.]], ValueError),
                                              ("positions", [0j, 1j], TypeError),
                                              ("xt_db_per_km", -30j, TypeError),
                                              ("phases", np.zeros((2, 2), complex), TypeError)])
def test_profile_and_coupling_shapes_and_real_data_are_checked(name, value, error):
    parameters = dict(profile=np.ones((2, 2)), positions=[0., 1000.], xt_db_per_km=-30.,
                       phases=np.zeros((2, 2)))
    with pytest.raises(error):
        crosstalk_from_phases(**{**parameters, name: value})


def test_mixed_numpy_and_jax_data_require_explicit_selection(jax_numpy):
    profile = jax_numpy.ones((2, 2))
    positions, phases = np.array([0., 1000.]), np.zeros((2, 2))
    with pytest.raises(TypeError):
        crosstalk_from_phases(profile, positions, -30., phases)
    result = crosstalk_from_phases(profile, positions, -30., phases, xp=jax_numpy)
    assert type(result) is type(profile) and result.dtype == jax_numpy.complex128


@pytest.mark.parametrize("name", ["profile", "positions", "xt_db_per_km", "phases"])
def test_each_native_parameter_can_select_jax(jax_numpy, name):
    parameters = dict(profile=[[1., 1.], [2., 2.]], positions=[0., 1000.], xt_db_per_km=-30.,
                       phases=[[0., 0.], [0., 0.]])
    parameters[name] = jax_numpy.asarray(parameters[name])
    result = crosstalk_from_phases(**parameters)
    assert type(result) is type(parameters[name]) and result.dtype == jax_numpy.complex128


def test_jax_level_profiles_and_phases_keep_gradients_and_hessians_at_the_input_face(jax_numpy):
    import jax
    def power(parameters):
        level, amplitude, angle = parameters
        ones = jax_numpy.ones((2,))
        profile = jax_numpy.stack((ones, amplitude*ones))
        phases = jax_numpy.stack((angle*ones, 2.*angle*ones))
        mixed = crosstalk_from_phases(profile, jax_numpy.asarray([0., 1000.]), level, phases)
        return jax_numpy.sum(jax_numpy.abs(mixed)**2)
    def expected(parameters):
        level, amplitude, angle = parameters
        coupling = 10.**(level/20.)
        return ((2. + coupling**2)*(1. + amplitude**2)
                + 2.*coupling*amplitude*(jax_numpy.cos(angle) + jax_numpy.cos(2.*angle)))
    parameters = jax_numpy.asarray([-30., 1.2, .4])
    for transform in (lambda function: function, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(power))(parameters),
                                   jax.jit(transform(expected))(parameters), rtol=1e-11, atol=1e-13)


def test_jax_position_derivatives_remain_native_away_from_zero(jax_numpy):
    import jax
    def field(position):
        mixed = crosstalk_from_phases(jax_numpy.asarray([[0.], [1.]]),
                                      jax_numpy.reshape(position, (1,)), -30., jax_numpy.zeros((2, 1)))
        return jax_numpy.real(mixed[0, 0])
    position = jax_numpy.asarray(1000.)
    amplitude = np.sqrt(.001)
    np.testing.assert_allclose(jax.jit(field)(position), amplitude, rtol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.grad(field))(position), amplitude/2000., rtol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.hessian(field))(position), -amplitude/(4.*1000.**2), rtol=1e-14)
