import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import BendLossTable, Fiber, GuideOptics, Section, units
from pyrayleigh.constants import SPEED_OF_LIGHT

REFERENCE = 193414489032258.06
GVD = -2.1682619391414893e-26


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


def test_reference_phase_and_group_delay(xp):
    values = [xp.asarray(value, dtype=xp.float64) for value in (REFERENCE, 1.45, 1.47, GVD)]
    guide = GuideOptics(*values)
    np.testing.assert_allclose(np.asarray(guide.reference_propagation_constant),
                               5877818.513168, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(np.asarray(guide.group_delay_per_length(values[0])),
                               4.903392199412835e-9, rtol=1e-14, atol=0.)
    np.testing.assert_array_equal(np.asarray(guide.propagation_constant_offset(values[0])), 0.)


def test_zero_dispersion_constant_index_matches_pyofdr_convention(xp):
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682)
    frequency = xp.asarray([[REFERENCE - 1e10, REFERENCE, REFERENCE + 1e10]], dtype=xp.float64)
    beta = guide.propagation_constant(frequency)
    expected = 2.*np.pi*1.4682*np.asarray(frequency)/SPEED_OF_LIGHT
    assert beta.shape == (1, 3)
    assert beta.dtype == xp.float64
    np.testing.assert_allclose(np.asarray(beta), expected, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(guide.group_delay_per_length(frequency)),
                               np.full((1, 3), 1.4682/SPEED_OF_LIGHT), rtol=1e-14, atol=0.)


def test_small_frequency_offset_keeps_precision(xp):
    guide = GuideOptics(REFERENCE, 1.45, 1.47, GVD)
    frequency = xp.asarray(REFERENCE + 1., dtype=xp.float64)
    actual = guide.propagation_constant_offset(frequency)
    np.testing.assert_allclose(np.asarray(actual), 3.080892182268972e-8, rtol=1e-13, atol=0.)


def test_second_order_dispersion_offset(xp):
    guide = GuideOptics(REFERENCE, 1.45, 1.47, GVD)
    frequency = xp.asarray(REFERENCE + 1e10, dtype=xp.float64)
    actual = guide.propagation_constant_offset(frequency)
    np.testing.assert_allclose(np.asarray(actual), 308.0891754271221, rtol=1e-13, atol=0.)


def test_jax_frequency_derivatives_and_group_delay(jax_numpy):
    import jax
    guide = GuideOptics(REFERENCE, 1.45, 1.47, GVD)
    def beta_offset(frequency_thz):
        return guide.propagation_constant_offset(frequency_thz*1e12)
    frequency = jax_numpy.asarray(REFERENCE/1e12 + .01, dtype=jax_numpy.float64)
    first = jax.jit(jax.grad(beta_offset))(frequency)
    second = jax.jit(jax.grad(jax.grad(beta_offset)))(frequency)
    expected_first = 2.*np.pi*1e12*guide.group_delay_per_length(frequency*1e12)
    np.testing.assert_allclose(first, expected_first, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(second, 4.*np.pi**2*GVD*1e24, rtol=1e-13, atol=0.)


def test_jax_guide_parameters_remain_differentiable(jax_numpy):
    import jax
    def beta(parameters):
        guide = GuideOptics(REFERENCE, parameters[0], parameters[1], parameters[2]*1e-27)
        return guide.propagation_constant(REFERENCE + 1e10)
    parameters = jax_numpy.asarray([1.45, 1.47, GVD*1e27], dtype=jax_numpy.float64)
    value = jax.jit(beta)(parameters)
    gradient = jax.jit(jax.grad(beta))(parameters)
    delta_omega = 2.*np.pi*1e10
    expected = [2.*np.pi*REFERENCE/SPEED_OF_LIGHT,
                delta_omega/SPEED_OF_LIGHT, .5*delta_omega**2*1e-27]
    np.testing.assert_allclose(gradient, expected, rtol=1e-13, atol=0.)
    reference = GuideOptics(REFERENCE, 1.45, 1.47, GVD).propagation_constant(REFERENCE + 1e10)
    np.testing.assert_allclose(value, reference, rtol=1e-14)


def test_numpy_guide_constants_need_an_explicit_jax_namespace(jax_numpy):
    import jax
    reference = units.wavelength_to_frequency(1550e-9)
    guide = GuideOptics(reference, 1.4682, 1.4682)
    frequency = jax_numpy.asarray(REFERENCE/1e12 + .01, dtype=jax_numpy.float64)
    with pytest.raises(TypeError):
        guide.propagation_constant(frequency*1e12)
    def beta(frequency_thz):
        return guide.propagation_constant(frequency_thz*1e12, xp=jax_numpy)
    derivative = jax.jit(jax.grad(beta))(frequency)
    np.testing.assert_allclose(derivative, 2.*np.pi*1e12*1.4682/SPEED_OF_LIGHT,
                               rtol=1e-13, atol=0.)


def test_guide_parameters_are_scalars_and_real():
    with pytest.raises(ValueError, match="scalars"):
        GuideOptics(REFERENCE, np.array([1.45, 1.46]), 1.47).propagation_constant(REFERENCE)
    with pytest.raises(TypeError, match="real values"):
        GuideOptics(REFERENCE, 1.45 + .1j, 1.47).propagation_constant(REFERENCE)


def test_default_sensitivities_use_the_total_strain_convention():
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682)
    np.testing.assert_allclose([guide.strain_phase_factor, guide.strain_group_factor],
                               [.78, .78], rtol=1e-14, atol=0.)
    np.testing.assert_allclose([guide.temperature_phase_factor, guide.temperature_group_factor],
                               [6.621e-6, 6.621e-6], rtol=1e-14, atol=0.)
    assert guide.thermal_expansion == 5.5e-7
    assert guide.relative_phase_change() == 0.
    assert guide.relative_group_delay_change() == 0.


def test_sensitivity_inputs_broadcast_in_their_native_namespace(xp):
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682)
    strain = xp.asarray([[0.], [2e-6]], dtype=xp.float64)
    temperature = xp.asarray([0., 1., 2.], dtype=xp.float64)
    expected = [[0., 6.621e-6, 1.3242e-5], [1.56e-6, 8.181e-6, 1.4802e-5]]
    for method in (guide.relative_phase_change, guide.relative_group_delay_change):
        actual = method(strain, temperature)
        assert actual.shape == (2, 3)
        assert actual.dtype == xp.float64
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-14, atol=0.)


def test_free_and_fixed_total_strain_heating(xp):
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682)
    temperature = xp.asarray([0., 1., 2.], dtype=xp.float64)
    strain = guide.thermal_expansion*temperature
    for method in (guide.relative_phase_change, guide.relative_group_delay_change):
        free = method(strain, temperature)
        fixed = method(0., temperature)
        np.testing.assert_allclose(np.asarray(free), [0., 7.05e-6, 1.41e-5], rtol=1e-14, atol=0.)
        np.testing.assert_allclose(np.asarray(fixed), [0., 6.621e-6, 1.3242e-5], rtol=1e-14, atol=0.)


def test_phase_and_group_factors_are_independent_native_parameters(xp):
    phase = xp.asarray(.75, dtype=xp.float64)
    group = xp.asarray(.85, dtype=xp.float64)
    expansion = xp.asarray(5.5e-7, dtype=xp.float64)
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682,
                        strain_phase_factor=phase, strain_group_factor=group,
                        temperature_phase_factor=6e-6, temperature_group_factor=7e-6,
                        thermal_expansion=expansion)
    assert guide.strain_phase_factor is phase
    assert guide.strain_group_factor is group
    assert guide.thermal_expansion is expansion
    np.testing.assert_allclose(np.asarray(guide.relative_phase_change(1e-6, 2.)),
                               1.275e-5, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.asarray(guide.relative_group_delay_change(1e-6, 2.)),
                               1.485e-5, rtol=1e-14, atol=0.)


@pytest.mark.parametrize("name", ["strain_phase_factor", "strain_group_factor",
                                  "temperature_phase_factor", "temperature_group_factor",
                                  "thermal_expansion"])
def test_sensitivity_parameters_are_real_scalars(name):
    with pytest.raises(ValueError, match="scalars"):
        GuideOptics(REFERENCE, 1.4682, 1.4682, **{name: np.array([1.])})
    with pytest.raises(TypeError):
        GuideOptics(REFERENCE, 1.4682, 1.4682, **{name: 1j})


def test_numpy_sensitivities_need_explicit_jax_selection(jax_numpy):
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682,
                        strain_phase_factor=np.array(.78), strain_group_factor=np.array(.78))
    strain = jax_numpy.asarray(1e-6, dtype=jax_numpy.float64)
    for method in (guide.relative_phase_change, guide.relative_group_delay_change):
        with pytest.raises(TypeError):
            method(strain)
        actual = method(strain, xp=jax_numpy)
        assert actual.dtype == jax_numpy.float64
        np.testing.assert_allclose(actual, .78e-6, rtol=1e-14, atol=0.)


@pytest.mark.parametrize("method_name", ["relative_phase_change", "relative_group_delay_change"])
def test_jax_sensitivity_derivatives_in_physical_coefficients(jax_numpy, method_name):
    import jax
    phase_scale = .02*2.*np.pi*1.4682/1550e-9
    mechanical_strain, temperature = 10e-6, 1.
    def returned_field(parameters):
        photoelastic, thermo_optic, expansion = parameters
        thermo_optic, expansion = thermo_optic*1e-6, expansion*1e-6
        strain_factor = 1. - photoelastic
        temperature_factor = thermo_optic + photoelastic*expansion
        guide = GuideOptics(REFERENCE, 1.4682, 1.4682,
                            strain_phase_factor=strain_factor, strain_group_factor=strain_factor,
                            temperature_phase_factor=temperature_factor,
                            temperature_group_factor=temperature_factor,
                            thermal_expansion=expansion)
        total_strain = mechanical_strain + guide.thermal_expansion*temperature
        change = getattr(guide, method_name)(total_strain, temperature)
        return jax_numpy.real(jax_numpy.exp(-1j*phase_scale*change))
    parameters = jax_numpy.asarray([.22, 6.5, .55], dtype=jax_numpy.float64)
    value = jax.jit(returned_field)(parameters)
    gradient = jax.jit(jax.grad(returned_field))(parameters)
    hessian = jax.jit(jax.hessian(returned_field))(parameters)
    phase = phase_scale*(.78*mechanical_strain + 7.05e-6*temperature)
    phase_gradient = phase_scale*np.array([-mechanical_strain, 1e-6*temperature, 1e-6*temperature])
    np.testing.assert_allclose(value, np.cos(phase), rtol=1e-13, atol=0.)
    np.testing.assert_allclose(gradient, -np.sin(phase)*phase_gradient, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(hessian, -np.cos(phase)*np.outer(phase_gradient, phase_gradient),
                               rtol=1e-12, atol=1e-15)


def test_bend_table_is_retained_without_changing_the_reference_guide(xp):
    table = BendLossTable([10., 20.], [REFERENCE], xp.asarray([[.01], [.1]]))
    guide = GuideOptics(REFERENCE, 1.45, 1.47, GVD, bend_loss=table)
    plain = GuideOptics(REFERENCE, 1.45, 1.47, GVD)
    assert guide.bend_loss is table
    assert plain.bend_loss is None
    frequency = xp.asarray([REFERENCE - 1e9, REFERENCE, REFERENCE + 1e9])
    for name in ("propagation_constant", "propagation_constant_offset", "group_delay_per_length"):
        np.testing.assert_array_equal(np.asarray(getattr(guide, name)(frequency)),
                                      np.asarray(getattr(plain, name)(frequency)))


@pytest.mark.parametrize("table", [0., {}, np.array([[.01], [.1]])])
def test_bend_loss_requires_a_table_or_none(table):
    with pytest.raises(TypeError, match="BendLossTable"):
        GuideOptics(REFERENCE, 1.45, 1.47, bend_loss=table)


def test_jax_table_parameters_keep_derivatives_through_guide_and_fiber(jax_numpy):
    import jax
    def power(parameters):
        left, right, fraction = parameters
        values = jax_numpy.reshape(jax_numpy.exp(jax_numpy.stack((left, right))), (2, 1))
        table = BendLossTable([10., 20.], [REFERENCE], values)
        guide = GuideOptics(REFERENCE, 1.45, 1.47, bend_loss=table)
        part = Section(5e-13, 2. - 5e-13, guide, 0., 1e-8)
        fiber = Fiber("test", 2., 293.15, REFERENCE - 1e9, REFERENCE + 1e9, [part])
        stored = fiber.sections[0].optics.bend_loss
        rate = stored.value(10. + 10.*fraction, REFERENCE)
        return jax_numpy.exp(-fiber.length*rate)
    parameters = jax_numpy.asarray([np.log(.01), np.log(.04), .5], dtype=jax_numpy.float64)
    value = jax.jit(power)(parameters)
    gradient = jax.jit(jax.grad(power))(parameters)
    hessian = jax.jit(jax.hessian(power))(parameters)
    left, right, fraction = np.asarray(parameters)
    depth = 2.*np.exp((1. - fraction)*left + fraction*right)
    expected = np.exp(-depth)
    derivative = np.array([1. - fraction, fraction, right - left])
    second = np.array([[0., 0., -1.], [0., 0., 1.], [-1., 1., 0.]])
    expected_hessian = expected*((depth**2 - depth)*np.outer(derivative, derivative)
                                 - depth*second)
    np.testing.assert_allclose(value, expected, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(gradient, -depth*expected*derivative, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-11, atol=1e-14)
