from dataclasses import replace

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Fiber, GuideOptics, PointReflector, Probe, RayleighRealization,
                        Section, compute_response,
                        ornstein_uhlenbeck_index_from_normals as index_from_normals,
                        ornstein_uhlenbeck_index_profile as index_profile)
from pyrayleigh.constants import SPEED_OF_LIGHT


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


@pytest.mark.parametrize("length,count", [(.025, 2), (.2, 3), (.225, 4)])
def test_fixed_grid_and_decay_of_the_initial_sample(xp, length, count):
    normals = xp.asarray([1.] + [0.]*(count - 1), dtype=xp.float64)
    profile = index_from_normals(np.float64(length), .03, .5, np.float64(.1), normals)
    assert len(profile.positions) == count
    assert profile.positions[0] == 0.
    assert profile.positions[-1] == length
    assert profile.values.dtype == xp.float64
    assert type(profile.values) is type(normals)
    expected = .03*np.exp(-profile.positions/.5)
    np.testing.assert_allclose(np.asarray(profile.values), expected, rtol=1e-14, atol=0.)


def test_stationary_covariance_and_linear_interpolation_variance(xp):
    profiles = [index_from_normals(.225, .03, .5, .1, xp.asarray(row))
                for row in np.eye(4)]
    positions = profiles[0].positions
    transfer = np.stack([np.asarray(profile.values) for profile in profiles], axis=1)
    covariance = transfer @ transfer.T
    expected = .03**2*np.exp(-np.abs(positions[:, None] - positions[None, :])/.5)
    np.testing.assert_allclose(covariance, expected, rtol=1e-14, atol=0.)
    midpoints = xp.asarray((positions[:-1] + positions[1:])/2.)
    interpolated = np.stack([np.asarray(profile.value(midpoints)) for profile in profiles])
    variance = np.sum(interpolated**2, axis=0)
    expected_midpoint = .03**2*(1. + np.exp(-np.diff(positions)/.5))/2.
    np.testing.assert_allclose(variance, expected_midpoint, rtol=1e-14, atol=0.)


@pytest.mark.parametrize("sigma", [0., .03])
def test_seeded_profile_matches_fixed_pcg64_draws_in_each_backend(xp, sigma):
    normals = np.random.Generator(np.random.PCG64(27)).standard_normal(4)
    profile = index_profile(.225, xp.asarray(sigma), xp.asarray(.5), .1, np.int64(27))
    expected = index_from_normals(.225, sigma, .5, .1, xp.asarray(normals))
    assert type(profile.values) is type(expected.values)
    np.testing.assert_array_equal(profile.positions, expected.positions)
    np.testing.assert_array_equal(np.asarray(profile.values), np.asarray(expected.values))


def test_profile_generation_leaves_the_global_rng_unchanged():
    before = np.random.get_state()
    first = index_profile(.225, .03, .5, .1, 27)
    second = index_profile(.225, .03, .5, .1, 27)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(first.values, second.values)


def test_innovation_survives_spacing_much_smaller_than_correlation_length(xp):
    profile = index_from_normals(1e-12, 1., 1e6, 1e-12, xp.asarray([0., 1.]))
    np.testing.assert_allclose(np.asarray(profile.values)[1], np.sqrt(2e-18),
                               rtol=1e-14, atol=0.)


@pytest.mark.parametrize("length,spacing,error", [
    (0., .1, ValueError), (-1., .1, ValueError), (float("nan"), .1, ValueError),
    (1., 0., ValueError), (1., float("inf"), ValueError),
    (True, .1, TypeError), (np.array(.225), .1, TypeError),
])
def test_positions_require_fixed_finite_positive_geometry(length, spacing, error):
    with pytest.raises(error):
        index_from_normals(length, .03, .5, spacing, [1., 1., 1., 1.])


@pytest.mark.parametrize("sigma,correlation,normals,error", [
    ([.03], .5, [1., 1., 1., 1.], ValueError),
    (.03, [.5], [1., 1., 1., 1.], ValueError),
    (.03j, .5, [1., 1., 1., 1.], TypeError),
    (.03, .5, [], ValueError), (.03, .5, [1., 1., 1.], ValueError),
    (.03, .5, [[1., 1., 1., 1.]], ValueError),
    (.03, .5, [1j, 1j, 1j, 1j], TypeError),
    (.03, .5, [True, True, True, True], TypeError),
])
def test_native_parameters_and_normals_have_the_required_dtype_and_shape(
        sigma, correlation, normals, error):
    with pytest.raises(error):
        index_from_normals(.225, sigma, correlation, .1, normals)


@pytest.mark.parametrize("seed", [-1, .5, "27", True, np.bool_(False)])
def test_seed_is_a_fixed_nonnegative_integer(seed):
    with pytest.raises(ValueError, match="seed"):
        index_profile(.225, .03, .5, .1, seed)


def test_numpy_normals_and_jax_parameters_need_explicit_selection(jax_numpy):
    normals = np.ones(4)
    sigma = jax_numpy.asarray(.03)
    with pytest.raises(TypeError):
        index_from_normals(.225, sigma, .5, .1, normals)
    profile = index_from_normals(.225, sigma, .5, .1, normals, xp=jax_numpy)
    assert type(profile.values) is type(sigma)
    assert profile.values.dtype == jax_numpy.float64


def test_jax_response_keeps_sigma_correlation_and_normal_derivatives(jax_numpy):
    import jax
    reference = SPEED_OF_LIGHT/1550e-9
    guide = GuideOptics(reference, 1.4682, 1.4682)
    length = SPEED_OF_LIGHT/(4.*guide.effective_index*reference)
    def objective(parameters):
        sigma, correlation, eta = parameters
        normals = jax_numpy.stack((jax_numpy.asarray(1.), eta))
        profile = index_from_normals(length, sigma*.01, correlation*10.*length,
                                      length, normals)
        sample = RayleighRealization(jax_numpy.asarray([.1]), [0.], [0., length])
        fiber = Fiber("short", length, 293.15, reference - 50e9, reference + 50e9,
                      [Section(0., length, guide, 0., 1e-8)],
                      [PointReflector(length, .2, 0., .8, .7)], sample)
        fiber = replace(fiber, index_perturbation=profile)
        response = compute_response(fiber, Probe(jax_numpy.asarray(reference)))
        return (jax_numpy.abs(response.reflection)**2
                + .1*jax_numpy.real(response.transmission))
    parameters = jax_numpy.asarray([1., 1., .5], dtype=jax_numpy.float64)
    value = jax.jit(objective)(parameters)
    gradient = jax.jit(jax.grad(objective))(parameters)
    hessian = jax.jit(jax.hessian(objective))(parameters)
    sigma, correlation, eta = np.asarray(parameters)
    a = np.exp(-.1/correlation)
    da = a*.1/correlation**2
    dda = a*(.01/correlation**4 - .2/correlation**3)
    v = np.sqrt(1. - a*a)
    dv = -a*da/v
    ddv = -(da*da + a*dda)/v - (a*da)**2/v**3
    scale = .005*np.pi/guide.effective_index
    f = 1. + a + eta*v
    df, ddf = da + eta*dv, dda + eta*ddv
    phase = scale*sigma*f
    derivative = scale*np.array([f, sigma*df, sigma*v])
    second = scale*np.array([[0., df, v], [df, sigma*ddf, sigma*dv],
                              [v, sigma*dv, 0.]])
    first_factor = .04*np.sin(phase) - .04*np.cos(phase/2.)
    second_factor = .04*np.cos(phase) + .02*np.sin(phase/2.)
    expected = .05 - .04*np.cos(phase) - .08*np.sin(phase/2.)
    expected_hessian = second_factor*np.outer(derivative, derivative) + first_factor*second
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(gradient, first_factor*derivative, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-10, atol=1e-14)
