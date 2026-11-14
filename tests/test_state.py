import numpy as np
import pytest

from pyrayleigh import Field, FieldTerm, FiberState, SpatialProfile, TimeHistory


def test_default_state_is_the_zero_reference_state():
    state = FiberState()
    assert state.is_static
    for field in (state.axial_strain, state.temperature_change, state.curvature_x, state.curvature_y):
        assert field.is_zero
        np.testing.assert_array_equal(field.value([0., 1.]), [0., 0.])
        np.testing.assert_array_equal(field.integral([0., 1.]), [0., 0.])


def test_state_keeps_supplied_fields_and_their_independent_values():
    strain = Field([FieldTerm(SpatialProfile([0., 2.], [1e-6, 1e-6]))])
    temperature = Field([FieldTerm(SpatialProfile([0., 2.], [0., 10.]),
                                   TimeHistory([0., 1.], [0., 1.]))])
    state = FiberState(axial_strain=strain, temperature_change=temperature)
    assert state.axial_strain is strain
    assert state.temperature_change is temperature
    np.testing.assert_allclose(state.axial_strain.value(1., .5), 1e-6, rtol=1e-14, atol=0.)
    np.testing.assert_array_equal(state.temperature_change.value(1., .5), 2.5)
    np.testing.assert_array_equal(state.temperature_change.integral(1., .5), 1.25)


@pytest.mark.parametrize("name", ["axial_strain", "temperature_change", "curvature_x", "curvature_y"])
def test_state_static_flag_depends_on_history_presence(name):
    dynamic = Field([FieldTerm(SpatialProfile([0., 2.], [0., 0.]),
                               TimeHistory([0.], [0.]))])
    state = FiberState(**{name: dynamic})
    assert not state.is_static


@pytest.mark.parametrize("name,value", [
    ("axial_strain", None), ("axial_strain", 1e-6),
    ("temperature_change", 293.15),
    ("temperature_change", FieldTerm(SpatialProfile([0., 1.], [1., 1.]))),
    ("curvature_x", None), ("curvature_y", 1.),
    ("curvature_x", SpatialProfile([0., 1.], [1., 1.])),
])
def test_state_requires_explicit_fields(name, value):
    with pytest.raises(TypeError, match=name):
        FiberState(**{name: value})


def test_jax_state_fields_keep_independent_parameter_derivatives():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        def returned_field(parameters):
            strain = SpatialProfile([0., 2.], jnp.stack((parameters[0], parameters[0]))*1e-6)
            temperature = SpatialProfile([0., 2.], jnp.stack((parameters[1], parameters[1])))
            state = FiberState(axial_strain=Field([FieldTerm(strain)]),
                               temperature_change=Field([FieldTerm(temperature)]))
            phase = (1e6*state.axial_strain.integral(1.)
                     + .5*state.temperature_change.integral(1.))
            return jnp.real(jnp.exp(-1j*phase))

        parameters = jnp.asarray([1., 2.], dtype=jnp.float64)
        value = jax.jit(returned_field)(parameters)
        gradient = jax.jit(jax.grad(returned_field))(parameters)
        hessian = jax.jit(jax.hessian(returned_field))(parameters)
        phase_gradient = np.array([1., .5])
        np.testing.assert_allclose(value, np.cos(2.), rtol=1e-14, atol=0.)
        np.testing.assert_allclose(gradient, -np.sin(2.)*phase_gradient, rtol=1e-13, atol=0.)
        np.testing.assert_allclose(hessian, -np.cos(2.)*np.outer(phase_gradient, phase_gradient),
                                   rtol=1e-13, atol=0.)
    finally:
        jax.config.update("jax_enable_x64", previous)
