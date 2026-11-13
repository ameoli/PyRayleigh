import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import GuideOptics, Probe

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


@pytest.mark.parametrize("frequency,time,shape", [
    (REFERENCE, None, ()),
    ([REFERENCE, REFERENCE + 1.], [0., .01], (2,)),
    ([[REFERENCE, REFERENCE + 1.]], None, (1, 2)),
    ([], [], (0,)),
])
def test_python_samples(frequency, time, shape):
    probe = Probe(frequency, time)
    assert probe.shape == shape
    assert probe.frequency.dtype == np.float64
    np.testing.assert_array_equal(probe.frequency, np.asarray(frequency))
    if time is None:
        assert probe.time is None
    else:
        assert probe.time.dtype == np.float64
        np.testing.assert_array_equal(probe.time, np.asarray(time))


@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
def test_native_arrays_keep_namespace_shape_and_precision(xp, shape):
    frequency = xp.full(shape, REFERENCE, dtype=xp.float64)
    time = xp.zeros(shape, dtype=xp.float64)
    probe = Probe(frequency, time)
    assert type(probe.frequency) is type(frequency)
    assert type(probe.time) is type(time)
    assert probe.shape == shape
    assert probe.frequency.dtype == xp.float64
    assert probe.time.dtype == xp.float64
    np.testing.assert_array_equal(np.asarray(probe.frequency), np.full(shape, REFERENCE))
    np.testing.assert_array_equal(np.asarray(probe.time), np.zeros(shape))


def test_input_arrays_can_change_without_changing_the_probe():
    frequency = np.array([REFERENCE, REFERENCE + 1.])
    time = np.array([0., .01])
    probe = Probe(frequency, time)
    frequency[:] = 1.
    time[:] = 1.
    np.testing.assert_array_equal(probe.frequency, [REFERENCE, REFERENCE + 1.])
    np.testing.assert_array_equal(probe.time, [0., .01])


@pytest.mark.parametrize("frequency,time", [
    ([REFERENCE, REFERENCE], 0.),
    (REFERENCE, [0.]),
    ([[REFERENCE, REFERENCE]], [0., .01]),
    ([REFERENCE, REFERENCE], []),
])
def test_time_requires_the_exact_frequency_shape(frequency, time):
    with pytest.raises(ValueError, match="same shape"):
        Probe(frequency, time)


@pytest.mark.parametrize("frequency,time", [
    ([REFERENCE + 1j], None), ([REFERENCE], [1j]), (True, None),
])
def test_samples_are_real_numeric_data(frequency, time):
    with pytest.raises(TypeError):
        Probe(frequency, time)


def test_mixed_numpy_and_jax_inputs_are_rejected(jax_numpy):
    with pytest.raises(TypeError):
        Probe(np.array([REFERENCE]), jax_numpy.asarray([0.]))


def test_native_time_selects_jax_for_python_frequency(jax_numpy):
    time = jax_numpy.asarray(0., dtype=jax_numpy.float64)
    probe = Probe(REFERENCE, time)
    assert type(probe.frequency) is type(time)
    assert probe.frequency.dtype == jax_numpy.float64


def test_jax_frequency_without_time_keeps_guide_derivatives(jax_numpy):
    import jax
    guide = GuideOptics(REFERENCE, 1.45, 1.47, GVD)
    def beta_offset(frequency_thz):
        probe = Probe(frequency_thz*1e12)
        return guide.propagation_constant_offset(probe.frequency)
    frequency = jax_numpy.asarray(REFERENCE/1e12 + .01, dtype=jax_numpy.float64)
    first = jax.jit(jax.grad(beta_offset))(frequency)
    second = jax.jit(jax.grad(jax.grad(beta_offset)))(frequency)
    expected_first = 2.*np.pi*1e12*guide.group_delay_per_length(frequency*1e12)
    np.testing.assert_allclose(first, expected_first, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(second, 4.*np.pi**2*GVD*1e24, rtol=1e-13, atol=0.)


def test_jax_frequency_and_time_copies_preserve_derivatives(jax_numpy):
    import jax
    def cw_field_real(parameters):
        probe = Probe(parameters[0]*1e12, parameters[1]*1e-12)
        return jax_numpy.cos(2.*jax_numpy.pi*probe.frequency*probe.time)
    parameters = jax_numpy.asarray([193.4, .005], dtype=jax_numpy.float64)
    value = jax.jit(cw_field_real)(parameters)
    gradient = jax.jit(jax.grad(cw_field_real))(parameters)
    hessian = jax.jit(jax.hessian(cw_field_real))(parameters)
    frequency, time = np.asarray(parameters)
    k = 2.*np.pi
    phase = k*frequency*time
    expected_gradient = -np.sin(phase)*np.array([k*time, k*frequency])
    mixed = -k*np.sin(phase) - k**2*frequency*time*np.cos(phase)
    expected_hessian = np.array([[-(k*time)**2*np.cos(phase), mixed],
                                 [mixed, -(k*frequency)**2*np.cos(phase)]])
    np.testing.assert_allclose(value, np.cos(phase), rtol=1e-13, atol=0.)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=0.)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-12, atol=0.)
