import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (CrosstalkRealization, crosstalk_from_phases, crosstalk_pairs,
                        generate_crosstalk)


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


@pytest.mark.parametrize("count,topology", [(1, "linear"), (3, "linear"), (7, "hex7")])
def test_seeded_phases_follow_the_pyofdr_pair_stream_in_every_backend(xp, count, topology):
    positions = [0., .25, 10., 1000.]
    sample = generate_crosstalk(count, positions, np.int64(42), topology=topology, xp=xp)
    rng = np.random.Generator(np.random.PCG64(42))
    expected = np.empty((len(crosstalk_pairs(count, topology)), len(positions)))
    for row in range(expected.shape[0]):
        expected[row, :] = rng.uniform(0., 2.*np.pi, len(positions))
    np.testing.assert_array_equal(np.asarray(sample.phases), expected)
    assert sample.phases.dtype == xp.float64 and type(sample.phases) is type(xp.asarray(0.))
    assert sample.n_cores == count and sample.position_count == 4
    assert sample.topology == topology and sample.pairs == crosstalk_pairs(count, topology)
    assert not sample.positions.flags.writeable


def test_same_seed_repeats_and_another_seed_changes_the_sample_without_global_rng_changes():
    before = np.random.get_state()
    first = generate_crosstalk(3, [0., 1., 10.], 42)
    second = generate_crosstalk(3, [0., 1., 10.], 42)
    other = generate_crosstalk(3, [0., 1., 10.], 43)
    after = np.random.get_state()
    np.testing.assert_array_equal(first.phases, second.phases)
    assert not np.array_equal(first.phases, other.phases)
    for initial, final in zip(before, after):
        np.testing.assert_array_equal(initial, final)


def test_manual_sample_owns_its_phases_and_read_only_fixed_positions():
    positions = np.array([0., 1000.])
    phases = np.array([[.1, .2], [.3, .4]], dtype=np.float32)
    sample = CrosstalkRealization(2, positions, phases)
    positions[:] = 10.
    phases[:] = 0.
    np.testing.assert_array_equal(sample.positions, [0., 1000.])
    np.testing.assert_array_equal(sample.phases, np.array([[.1, .2], [.3, .4]], dtype=np.float32))
    assert sample.phases.dtype == np.float64
    with pytest.raises(ValueError):
        sample.positions[0] = 1.
    assert repr(sample) == "CrosstalkRealization(n_cores=2, n_positions=2, topology='linear')"


@pytest.mark.parametrize("count,topology", [(3, "linear"), (7, "hex7")])
def test_mixing_reuses_the_sample_when_profiles_or_coupling_level_change(xp, count, topology):
    positions = np.array([0., 250., 1000.])
    sample = generate_crosstalk(count, positions, 42, topology=topology, xp=xp)
    phases = np.asarray(sample.phases).copy()
    profile = xp.asarray(np.arange(count*3).reshape(count, 3) + .1j)
    first = sample.mix(profile, -30.)
    expected = crosstalk_from_phases(profile, xp.asarray(positions), -30., sample.phases, topology=topology)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(expected))
    for scale, level in ((2., -20.), (.5, -40.)):
        actual = sample.mix(scale*profile, xp.asarray(level))
        reference = crosstalk_from_phases(scale*profile, xp.asarray(positions), level, sample.phases,
                                          topology=topology)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
    np.testing.assert_array_equal(np.asarray(sample.mix(profile, -30.)), np.asarray(first))
    np.testing.assert_array_equal(np.asarray(sample.phases), phases)
    assert first.dtype == xp.complex128 and type(first) is type(profile)


@pytest.mark.parametrize("count,positions,topology", [(1, [0., 1.], "linear"),
                                                      (1, [], "linear"), (3, [], "linear"), (7, [], "hex7")])
def test_empty_and_single_core_samples_keep_the_native_profile_shape(xp, count, positions, topology):
    sample = generate_crosstalk(count, positions, 42, topology=topology, xp=xp)
    profile = xp.ones((count, len(positions)), dtype=xp.float32)
    result = sample.mix(profile, -30.)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(profile))
    assert result.shape == profile.shape and result.dtype == xp.complex128
    assert type(result) is type(profile)


@pytest.mark.parametrize("seed", [True, np.bool_(True), -1, .5, "42", None, 1j])
def test_seed_is_a_nonnegative_integer(seed):
    with pytest.raises(ValueError, match="seed"):
        generate_crosstalk(2, [0., 1.], seed)


@pytest.mark.parametrize("positions,error", [([-1., 0.], ValueError), ([float("inf")], ValueError),
                                             ([float("nan")], ValueError), ([True], TypeError),
                                             ([[0., 1.]], TypeError)])
def test_positions_are_fixed_finite_and_nonnegative(positions, error):
    with pytest.raises(error, match="positions"):
        generate_crosstalk(2, positions, 42)


@pytest.mark.parametrize("phases,error", [([0., 0.], ValueError), (np.zeros((3, 2)), ValueError),
                                         (np.zeros((2, 1)), ValueError), (np.zeros((2, 2), complex), TypeError)])
def test_manual_phases_have_one_real_row_per_pair_and_one_column_per_position(phases, error):
    with pytest.raises(error):
        CrosstalkRealization(2, [0., 1.], phases)


@pytest.mark.parametrize("profile", [np.ones((3, 2)), np.ones((2, 1)), np.ones((2, 2, 1)), [1., 2.]])
def test_mix_requires_the_stored_core_and_position_counts(profile):
    sample = generate_crosstalk(2, [0., 1.], 42)
    with pytest.raises(ValueError, match="profile"):
        sample.mix(profile, -30.)


def test_numpy_phases_and_jax_profiles_require_explicit_namespace_selection(jax_numpy):
    sample = generate_crosstalk(2, [0., 1000.], 42)
    profile = jax_numpy.ones((2, 2))
    with pytest.raises(TypeError):
        sample.mix(profile, -30.)
    result = sample.mix(profile, -30., xp=jax_numpy)
    assert type(result) is type(profile) and result.dtype == jax_numpy.complex128


def test_jax_phases_select_jax_without_treating_fixed_numpy_positions_as_native_data(jax_numpy):
    positions = np.array([0., 1000.])
    sample = generate_crosstalk(2, positions, 42, xp=jax_numpy)
    result = sample.mix([[1., 1.], [2., 2.]], -30.)
    assert type(result) is type(sample.phases) and result.dtype == jax_numpy.complex128
    with pytest.raises(TypeError, match="positions"):
        CrosstalkRealization(2, jax_numpy.asarray(positions), sample.phases)


def test_native_manual_phases_keep_jit_gradients_and_hessians(jax_numpy):
    import jax
    def power(parameters):
        level, amplitude, angle = parameters
        ones = jax_numpy.ones((2,))
        phases = jax_numpy.stack((angle*ones, 2.*angle*ones))
        sample = CrosstalkRealization(2, [0., 1000.], phases)
        profile = jax_numpy.stack((ones, amplitude*ones))
        return jax_numpy.sum(jax_numpy.abs(sample.mix(profile, level))**2)
    def expected(parameters):
        level, amplitude, angle = parameters
        coupling = 10.**(level/20.)
        return ((2. + coupling**2)*(1. + amplitude**2)
                + 2.*coupling*amplitude*(jax_numpy.cos(angle) + jax_numpy.cos(2.*angle)))
    parameters = jax_numpy.asarray([-30., 1.2, .4])
    for transform in (lambda function: function, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(power))(parameters),
                                   jax.jit(transform(expected))(parameters), rtol=1e-11, atol=1e-13)


def test_generated_sample_keeps_level_and_profile_derivatives_without_redrawing(jax_numpy):
    import jax
    sample = generate_crosstalk(2, [0., 1000.], 42, xp=jax_numpy)
    angles = np.asarray(sample.phases).copy()
    def power(parameters):
        level, amplitude = parameters
        ones = jax_numpy.ones((2,))
        profile = jax_numpy.stack((ones, amplitude*ones))
        return jax_numpy.sum(jax_numpy.abs(sample.mix(profile, level))**2)
    def expected(parameters):
        level, amplitude = parameters
        coupling = 10.**(level/20.)
        return ((2. + coupling**2)*(1. + amplitude**2)
                + 2.*coupling*amplitude*(np.cos(angles[0, 1]) + np.cos(angles[1, 1])))
    parameters = jax_numpy.asarray([-30., 1.2])
    for transform in (lambda function: function, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(power))(parameters),
                                   jax.jit(transform(expected))(parameters), rtol=1e-11, atol=1e-13)
    np.testing.assert_array_equal(np.asarray(sample.phases), angles)
