import math

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (BendLossTable, Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, Probe, Section, SpatialProfile, compute_response)
from pyrayleigh.constants import SPEED_OF_LIGHT
from pyrayleigh.modes import marcuse_bend_attenuation, marcuse_bend_loss_table
from pyrayleigh.modes.bends import DEFAULT_TABLE_RADII

A, N1, N2 = 4.1e-6, 1.444745306273739, 1.44
FREQUENCY = SPEED_OF_LIGHT/1550e-9
RADII = [.01, .005, .02]


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


def fiber_and_state(table, xp):
    radius, turns = .01, .75
    length = 2.*math.pi*radius*turns
    guide = GuideOptics(FREQUENCY, 1.4682, 1.4682, bend_loss=table)
    fiber = Fiber("spool", length, 293.15, FREQUENCY - 20e9, FREQUENCY + 20e9,
                  [Section(0., length, guide, 0., 0.)], [PointReflector(length, .1, 0., .8, .7)])
    profile = SpatialProfile([0., length], xp.asarray([100., 100.]))
    return fiber, FiberState(curvature_x=Field([FieldTerm(profile)]))


def test_table_sorts_radii_and_samples_each_curvature_frequency_pair(xp):
    frequencies = [SPEED_OF_LIGHT/1625e-9, FREQUENCY, SPEED_OF_LIGHT/1310e-9]
    core = xp.asarray(A, dtype=xp.float64)
    table = marcuse_bend_loss_table(frequencies, core, N1, N2, iter(RADII))
    assert isinstance(table, BendLossTable)
    assert table.shape == (3, 3)
    assert table.attenuation.dtype == xp.float64
    assert type(table.attenuation) is type(core)
    np.testing.assert_array_equal(table.curvatures, [50., 100., 200.])
    np.testing.assert_array_equal(table.frequencies, frequencies)
    for row, radius in enumerate([.02, .01, .005]):
        expected = marcuse_bend_attenuation(xp.asarray(frequencies), core, N1, N2, radius)
        np.testing.assert_allclose(np.asarray(table.attenuation[row, :]), np.asarray(expected), rtol=1e-13)


def test_default_radius_grid_is_preserved():
    table = marcuse_bend_loss_table([FREQUENCY], A, N1, N2)
    assert table.shape == (200, 1)
    np.testing.assert_array_equal(table.curvatures, 1./DEFAULT_TABLE_RADII)


def test_table_metadata_owns_copies_without_modifying_input_order():
    frequencies, radii = np.array([FREQUENCY]), np.array(RADII)
    table = marcuse_bend_loss_table(frequencies, A, N1, N2, radii)
    np.testing.assert_array_equal(radii, RADII)
    frequencies[:] = 1.
    radii[:] = 1.
    np.testing.assert_array_equal(table.frequencies, [FREQUENCY])
    np.testing.assert_array_equal(table.curvatures, [50., 100., 200.])
    assert not table.frequencies.flags.writeable and not table.curvatures.flags.writeable


def test_one_frequency_freezes_the_centre_loss_through_the_optical_response(xp):
    table = marcuse_bend_loss_table([FREQUENCY], A, N1, N2, RADII, xp=xp)
    fiber, state = fiber_and_state(table, xp)
    probe = Probe(xp.asarray([FREQUENCY - 20e9, FREQUENCY, FREQUENCY + 20e9]))
    result = compute_response(fiber, probe, state)
    total = .75*3.356204768598211
    np.testing.assert_allclose(np.abs(np.asarray(result.transmission))**2,
                               np.full((3,), .64*10.**(-total/10.)), rtol=2e-10)
    np.testing.assert_allclose(np.abs(np.asarray(result.reflection)),
                               np.full((3,), .1*10.**(-total/10.)), rtol=2e-10)


def test_spectral_table_interpolates_and_must_cover_the_fiber_band(xp):
    frequencies = [FREQUENCY - 20e9, FREQUENCY + 20e9]
    table = marcuse_bend_loss_table(frequencies, A, N1, N2, RADII, xp=xp)
    expected = np.sqrt(np.prod(np.asarray(table.attenuation[1, :])))
    np.testing.assert_allclose(np.asarray(table.value(100., FREQUENCY, xp=xp)), expected, rtol=1e-13)
    fiber_and_state(table, xp)
    narrow = marcuse_bend_loss_table([FREQUENCY - 10e9, FREQUENCY + 10e9], A, N1, N2, RADII, xp=xp)
    with pytest.raises(ValueError, match="cover"):
        fiber_and_state(narrow, xp)


def test_underflowed_and_unsupported_samples_keep_the_rate_function_convention(xp):
    table = marcuse_bend_loss_table([FREQUENCY], A, N1, N2, [1.2, 1.3], xp=xp)
    np.testing.assert_array_equal(np.asarray(table.attenuation), [[0.], [0.]])
    unsupported = marcuse_bend_loss_table([FREQUENCY, 1e15], A, N1, N2, RADII, xp=xp)
    assert np.all(np.isfinite(np.asarray(unsupported.attenuation[:, 0])))
    assert np.all(np.isnan(np.asarray(unsupported.attenuation[:, 1])))


@pytest.mark.parametrize("grid,error", [
    ([], ValueError), ([0.], ValueError), ([FREQUENCY, FREQUENCY], ValueError),
    ([FREQUENCY, FREQUENCY/2.], ValueError), ([float("nan")], ValueError),
    ([True], TypeError), ([[FREQUENCY]], TypeError),
])
def test_frequency_grid_is_fixed_positive_and_increasing(grid, error):
    with pytest.raises(error):
        marcuse_bend_loss_table(grid, A, N1, N2, RADII)


@pytest.mark.parametrize("grid,error", [
    ([], ValueError), ([.01], ValueError), ([.01, .01], ValueError),
    ([-.01, .02], ValueError), ([True, .02], TypeError),
])
def test_radius_grid_is_fixed_positive_and_distinct(grid, error):
    with pytest.raises(error):
        marcuse_bend_loss_table([FREQUENCY], A, N1, N2, grid)


@pytest.mark.parametrize("name", ["core_radius", "core_index", "cladding_index", "radius_factor"])
def test_guide_parameters_are_scalar(name):
    parameters = dict(core_radius=A, core_index=N1, cladding_index=N2, radius_factor=1.28)
    parameters[name] = [parameters[name]]
    with pytest.raises(ValueError, match="scalars"):
        marcuse_bend_loss_table([FREQUENCY], bend_radii=RADII, **parameters)


def test_guide_parameters_are_real():
    with pytest.raises(TypeError):
        marcuse_bend_loss_table([FREQUENCY], A, N1 + 1j, N2, RADII)


def test_jax_grids_are_rejected_without_detaching_metadata(jax_numpy):
    with pytest.raises(TypeError, match="fixed real"):
        marcuse_bend_loss_table(jax_numpy.asarray([FREQUENCY]), A, N1, N2, RADII)
    with pytest.raises(TypeError, match="fixed real"):
        marcuse_bend_loss_table([FREQUENCY], A, N1, N2, jax_numpy.asarray(RADII))


def test_native_parameters_select_the_namespace_and_mixed_data_need_explicit_selection(jax_numpy):
    core, index = np.asarray(A), jax_numpy.asarray(N1)
    with pytest.raises(TypeError):
        marcuse_bend_loss_table([FREQUENCY], core, index, N2, RADII)
    table = marcuse_bend_loss_table([FREQUENCY], core, index, N2, RADII, xp=jax_numpy)
    assert type(table.attenuation) is type(index)


def test_jax_factory_parameters_keep_jit_gradients_and_hessians_through_the_response(jax_numpy):
    import jax
    def power(parameters):
        a_scale, n1, n2, correction = parameters
        table = marcuse_bend_loss_table([FREQUENCY], A*a_scale, n1, n2, [.005, .01], correction)
        fiber, state = fiber_and_state(table, jax_numpy)
        response = compute_response(fiber, Probe(jax_numpy.asarray(FREQUENCY)), state)
        return jax_numpy.abs(response.transmission)**2
    def expected(parameters):
        a_scale, n1, n2, correction = parameters
        rate = marcuse_bend_attenuation(FREQUENCY, A*a_scale, n1, n2, .01, correction)
        return .64*jax_numpy.exp(-2.*math.pi*.01*.75*rate)
    parameters = jax_numpy.asarray([1., N1, N2, 1.28])
    for transform in (lambda f: f, jax.grad, jax.hessian):
        np.testing.assert_allclose(jax.jit(transform(power))(parameters), jax.jit(transform(expected))(parameters),
                                   rtol=2e-8, atol=1e-10)
