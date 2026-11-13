import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Fiber, Field, FieldTerm, FiberState, GuideOptics, OpticalPath,
                        Section, SpatialProfile, TimeHistory, optical_path, units)
from pyrayleigh.constants import SPEED_OF_LIGHT

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


def make_fiber(length=2., sections=None):
    if sections is None:
        sections = [Section(0., length, GuideOptics(REFERENCE, 1.4682, 1.4682), 0., 1e-8)]
    return Fiber("spool-1", length, 293.15, REFERENCE - 50e9, REFERENCE + 50e9, sections)


def piecewise_fiber_and_state(xp):
    first = GuideOptics(REFERENCE, 1.45, 1.47, -2.2e-26,
                        strain_phase_factor=.8, strain_group_factor=.7,
                        temperature_phase_factor=6e-6, temperature_group_factor=5e-6)
    second = GuideOptics(REFERENCE + 5e9, 1.46, 1.48, -1.8e-26,
                         strain_phase_factor=.6, strain_group_factor=.9,
                         temperature_phase_factor=4e-6, temperature_group_factor=7e-6)
    fiber = make_fiber(sections=[Section(0., 1., first, 0., 1e-8),
                                 Section(1., 2., second, 0., 1e-8)])
    strain = SpatialProfile([0., 2.], xp.asarray([0., 2e-6]))
    temperature = SpatialProfile([0., 2.], xp.asarray([1., 3.]))
    state = FiberState(axial_strain=Field([FieldTerm(strain)]),
                       temperature_change=Field([FieldTerm(temperature)]))
    return fiber, state


def test_reference_path_matches_the_uniform_guide(xp):
    path = optical_path(make_fiber())
    frequency = xp.asarray([[REFERENCE - 1e9, REFERENCE, REFERENCE + 1e9]])
    phase = path.one_way_phase(.25, 1.75, frequency)
    delay = path.group_delay(.25, 1.75, frequency)
    assert phase.dtype == delay.dtype == xp.float64
    assert phase.shape == delay.shape == (1, 3)
    expected = 2.*np.pi*1.4682*np.asarray(frequency)*1.5/SPEED_OF_LIGHT
    np.testing.assert_allclose(np.asarray(phase), expected, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.asarray(delay), np.full((1, 3), 1.4682*1.5/SPEED_OF_LIGHT),
                               rtol=1e-14, atol=0.)


def test_numpy_scalar_geometry_preserves_the_numerical_namespace(xp):
    guide = GuideOptics(REFERENCE, 1.4682, 1.4682)
    section = Section(np.float64(0.), np.float64(2.), guide, 0., 1e-8)
    path = optical_path(make_fiber(length=np.float64(2.), sections=[section]))
    frequency = xp.asarray(REFERENCE, dtype=xp.float64)
    start, end = np.float64(.25), np.float64(1.75)
    phase = path.one_way_phase(start, end, frequency)
    delay = path.group_delay(start, end, frequency)
    factor = path.phase_factor(start, end, frequency)
    assert phase.dtype == delay.dtype == xp.float64
    assert factor.dtype == xp.complex128
    np.testing.assert_allclose(np.asarray(phase), 2.*np.pi*1.4682*REFERENCE*1.5/SPEED_OF_LIGHT,
                               rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.asarray(delay), 1.4682*1.5/SPEED_OF_LIGHT, rtol=1e-14, atol=0.)


def test_section_weights_and_partial_field_integrals(xp):
    fiber, state = piecewise_fiber_and_state(xp)
    path = optical_path(fiber, state)
    frequency = xp.asarray([REFERENCE - 1e10, REFERENCE, REFERENCE + 1e10])
    phase_lengths = [.50000555, .500004875]
    group_lengths = [.5000046375, .5000084375]
    expected_phase, expected_delay = np.zeros(3), np.zeros(3)
    for section, lp, lg in zip(fiber.sections, phase_lengths, group_lengths):
        guide = section.optics
        dw = 2.*np.pi*(np.asarray(frequency) - guide.frequency_ref)
        beta_ref = 2.*np.pi*guide.effective_index*guide.frequency_ref/SPEED_OF_LIGHT
        expected_phase += beta_ref*lp + (guide.group_index/SPEED_OF_LIGHT*dw + .5*guide.gvd*dw**2)*lg
        expected_delay += (guide.group_index/SPEED_OF_LIGHT + guide.gvd*dw)*lg
    np.testing.assert_allclose(np.asarray(path.one_way_phase(.5, 1.5, frequency)), expected_phase,
                               rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.asarray(path.group_delay(.5, 1.5, frequency)), expected_delay,
                               rtol=1e-14, atol=0.)
    factor = path.phase_factor(.5, 1.5, frequency)
    assert factor.dtype == xp.complex128
    np.testing.assert_allclose(np.abs(np.asarray(factor)), np.ones(3), rtol=1e-14, atol=0.)
    np.testing.assert_allclose(np.asarray(path.phase_factor(.5, 1.5, frequency, round_trip=True)),
                               np.asarray(factor)**2, rtol=1e-14, atol=0.)


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_zero_length_path_preserves_query_shape(xp, shape):
    path = optical_path(make_fiber())
    frequency = xp.full(shape, REFERENCE, dtype=xp.float64)
    for method in (path.one_way_phase, path.group_delay):
        actual = method(1., 1., frequency)
        assert actual.shape == shape
        np.testing.assert_array_equal(np.asarray(actual), np.zeros(shape))
    factor = path.phase_factor(1., 1., frequency)
    assert factor.shape == shape
    np.testing.assert_array_equal(np.asarray(factor), np.ones(shape, dtype=complex))


def test_split_carrier_retains_a_one_hertz_frequency_change(xp):
    path = optical_path(make_fiber(length=1000.))
    frequency = xp.asarray([REFERENCE, REFERENCE + 1.], dtype=xp.float64)
    factor = path.phase_factor(0., 1000., frequency)
    ratio = factor[1]/factor[0]
    expected = np.exp(-1j*2.*np.pi*1.4682*1000./SPEED_OF_LIGHT)
    np.testing.assert_allclose(np.asarray(ratio), expected, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("start,end,error", [
    (-1., 1., ValueError), (1., 0., ValueError), (0., 2.1, ValueError),
    (0., float("nan"), ValueError), (True, 1., TypeError), (np.array(0.), 1., TypeError),
])
def test_endpoints_are_fixed_ordered_and_inside_the_fiber(start, end, error):
    with pytest.raises(error):
        optical_path(make_fiber()).one_way_phase(start, end, REFERENCE)


def test_path_requires_a_fiber_and_a_static_state():
    with pytest.raises(TypeError, match="Fiber"):
        optical_path(None)
    with pytest.raises(TypeError, match="FiberState"):
        OpticalPath(make_fiber(), Field())
    dynamic = Field([FieldTerm(SpatialProfile([0., 2.], [1., 1.]), TimeHistory([0.], [1.]))])
    with pytest.raises(NotImplementedError, match="static"):
        optical_path(make_fiber(), FiberState(axial_strain=dynamic))


def test_numpy_guide_and_jax_frequency_need_explicit_selection(jax_numpy):
    guide = GuideOptics(units.wavelength_to_frequency(1550e-9), 1.4682, 1.4682)
    path = optical_path(make_fiber(sections=[Section(0., 2., guide, 0., 1e-8)]))
    frequency = jax_numpy.asarray(REFERENCE, dtype=jax_numpy.float64)
    for method in (path.one_way_phase, path.group_delay, path.phase_factor):
        with pytest.raises(TypeError):
            method(0., 2., frequency)
        actual = method(0., 2., frequency, xp=jax_numpy)
        assert type(actual) is type(frequency)


def test_jax_frequency_derivatives_equal_group_delay(jax_numpy):
    import jax
    path = optical_path(*piecewise_fiber_and_state(numpy_api))
    def phase(frequency_thz):
        return path.one_way_phase(.5, 1.5, frequency_thz*1e12, xp=jax_numpy)
    frequency = jax_numpy.asarray(REFERENCE/1e12 + .01, dtype=jax_numpy.float64)
    first = jax.jit(jax.grad(phase))(frequency)
    second = jax.jit(jax.grad(jax.grad(phase)))(frequency)
    delay = path.group_delay(.5, 1.5, frequency*1e12, xp=jax_numpy)
    np.testing.assert_allclose(first, 2.*np.pi*1e12*delay, rtol=1e-13, atol=0.)
    expected_second = 4.*np.pi**2*1e24*(-2.2e-26*.5000046375 - 1.8e-26*.5000084375)
    np.testing.assert_allclose(second, expected_second, rtol=1e-13, atol=0.)


def test_jax_phase_and_complex_factor_keep_guide_and_state_derivatives(jax_numpy):
    import jax
    def build_path(parameters):
        guide = GuideOptics(REFERENCE, 1.4682, 1.4682,
                            temperature_phase_factor=parameters[1]*1e-6)
        fiber = make_fiber(sections=[Section(0., 2., guide, 0., 1e-8)])
        profile = SpatialProfile([0., 2.], jax_numpy.stack((parameters[0], parameters[0])))
        return optical_path(fiber, FiberState(temperature_change=Field([FieldTerm(profile)])))
    def phase(parameters):
        return build_path(parameters).one_way_phase(0., 1., REFERENCE)
    def factor(parameters):
        return build_path(parameters).phase_factor(0., 1., REFERENCE)
    def real_field(parameters):
        return jax_numpy.real(factor(parameters))
    parameters = jax_numpy.asarray([2., 6.], dtype=jax_numpy.float64)
    beta_ref = 2.*np.pi*1.4682*REFERENCE/SPEED_OF_LIGHT
    phase_gradient = beta_ref*1e-6*np.array([6., 2.])
    phase_hessian = beta_ref*1e-6*np.array([[0., 1.], [1., 0.]])
    np.testing.assert_allclose(jax.jit(phase)(parameters), beta_ref*(1. + 12e-6), rtol=1e-14)
    np.testing.assert_allclose(jax.jit(jax.grad(phase))(parameters), phase_gradient, rtol=1e-13)
    np.testing.assert_allclose(jax.jit(jax.hessian(phase))(parameters), phase_hessian,
                               rtol=1e-13, atol=1e-15)
    field = np.asarray(jax.jit(factor)(parameters))
    np.testing.assert_allclose(jax.jit(jax.grad(real_field))(parameters), field.imag*phase_gradient,
                               rtol=1e-12, atol=1e-13)
    expected_hessian = -field.real*np.outer(phase_gradient, phase_gradient) + field.imag*phase_hessian
    np.testing.assert_allclose(jax.jit(jax.hessian(real_field))(parameters), expected_hessian,
                               rtol=1e-12, atol=1e-13)
