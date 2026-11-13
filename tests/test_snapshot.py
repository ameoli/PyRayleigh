import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import (Field, FieldTerm, Fiber, FiberState, GuideOptics,
                        PointReflector, Probe, Section, SpatialProfile,
                        TimeHistory, compute_response)


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


@pytest.mark.parametrize("time", [-1., .5, 3.])
def test_snapshot_preserves_steps_values_and_exact_spatial_integrals(xp, time):
    dynamic = FieldTerm(SpatialProfile([0., 1., 1., 3.], xp.asarray([1., 1., 2., 2.])),
                        TimeHistory([0., 2.], xp.asarray([1., 3.])))
    static = FieldTerm(SpatialProfile([-1., 4.], xp.asarray([-.5, -.5])))
    field = Field([dynamic, static])
    snapshot = field.at_time(xp.asarray(time))
    assert snapshot.is_static
    assert not field.is_static
    assert len(snapshot.terms) == len(field.terms)
    for original, frozen in zip(field.terms, snapshot.terms):
        assert frozen.history is None
        assert type(frozen.profile.values) is type(original.profile.values)
        np.testing.assert_array_equal(frozen.profile.positions, original.profile.positions)
    position = xp.asarray([-2., -1., 0., .5, 1., 2., 3., 4., 5.])
    for method in ("value", "integral"):
        expected = np.asarray(getattr(field, method)(position, time))
        actual = getattr(snapshot, method)(position, time=100.)
        assert actual.dtype == xp.float64
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-14, atol=0.)


def test_snapshot_of_both_state_fields_drives_the_static_response(xp):
    strain = Field([FieldTerm(SpatialProfile([0., 2.], xp.asarray([2e-6, 2e-6])),
                              TimeHistory([0., 2.], xp.asarray([1., 3.])))])
    temperature = Field([FieldTerm(SpatialProfile([0., 4.], xp.asarray([0., 4.])),
                                   TimeHistory([0., 2.], xp.asarray([3., 1.])))])
    state = FiberState(axial_strain=strain, temperature_change=temperature)
    snapshot = state.at_time(xp.asarray(1.))
    assert snapshot.is_static and not state.is_static
    position = xp.asarray([0., 1., 2.])
    np.testing.assert_allclose(np.asarray(snapshot.axial_strain.value(position)),
                               [4e-6, 4e-6, 4e-6], rtol=1e-14, atol=0.)
    np.testing.assert_array_equal(np.asarray(snapshot.temperature_change.value(position)),
                                  [0., 2., 4.])
    expected_state = FiberState(
        axial_strain=Field([FieldTerm(SpatialProfile([0., 2.], xp.asarray([4e-6, 4e-6])))]),
        temperature_change=Field([FieldTerm(SpatialProfile([0., 4.], xp.asarray([0., 8.])))]))
    reference = 193414489032258.06
    guide = GuideOptics(reference, 1.4682, 1.4682)
    fiber = Fiber("spool-1", 2., 293.15, reference - 50e9, reference + 50e9,
                  [Section(0., 2., guide, .1, 1e-8)],
                  [PointReflector(1.25, .1, 0., .8, .7)])
    probe = Probe(xp.asarray([reference, reference + 1e9]), xp.asarray([0., 2.]))
    response = compute_response(fiber, probe, snapshot)
    expected = compute_response(fiber, probe, expected_state)
    for name in ("reflection", "transmission"):
        np.testing.assert_allclose(np.asarray(getattr(response, name)),
                                   np.asarray(getattr(expected, name)), rtol=1e-13, atol=0.)


def test_static_and_empty_states_can_be_frozen_without_changing_the_source():
    original = Field([FieldTerm(SpatialProfile([0., 2.], [1., 2.]))])
    state = FiberState(axial_strain=original)
    snapshot = state.at_time(.5)
    original.terms[0].profile.values[:] = 0.
    np.testing.assert_array_equal(snapshot.axial_strain.value([0., 2.]), [1., 2.])
    assert snapshot.temperature_change.is_zero
    empty = FiberState().at_time(.5)
    assert empty.is_static
    assert empty.axial_strain.is_zero and empty.temperature_change.is_zero


@pytest.mark.parametrize("time,error", [
    ([], ValueError), ([.5], ValueError), (np.array([[.5]]), ValueError),
    (True, TypeError), (.5j, TypeError),
])
def test_snapshot_requires_one_real_scalar_time_even_for_empty_fields(time, error):
    for model in (Field(), FiberState()):
        with pytest.raises(error):
            model.at_time(time)


def test_mixed_values_and_fields_require_one_explicit_namespace(jax_numpy):
    numpy_profile = SpatialProfile([0., 2.], [1., 1.])
    jax_profile = SpatialProfile([0., 2.], jax_numpy.asarray([1., 1.]))
    history = TimeHistory([0., 2.], jax_numpy.asarray([1., 3.]))
    mixed_field = Field([FieldTerm(numpy_profile, history), FieldTerm(numpy_profile)])
    mixed_state = FiberState(axial_strain=Field([FieldTerm(numpy_profile)]),
                             temperature_change=Field([FieldTerm(jax_profile, history)]))
    for model in (mixed_field, mixed_state):
        with pytest.raises(TypeError):
            model.at_time(.5)
        snapshot = model.at_time(.5, xp=jax_numpy)
        fields = ((snapshot,) if isinstance(snapshot, Field) else
                  (snapshot.axial_strain, snapshot.temperature_change))
        for field in fields:
            for term in field.terms:
                assert type(term.profile.values) is type(jax_profile.values)
    np.testing.assert_allclose(np.asarray(mixed_field.at_time(.5, xp=jax_numpy).value(1.)),
                               2.5, rtol=1e-14, atol=0.)


def test_jax_snapshot_keeps_values_and_time_derivatives(jax_numpy):
    import jax
    def returned_field(parameters):
        a, b, temperature, time = parameters
        strain = SpatialProfile([0., 2.], jax_numpy.stack((a, a))*1e-6)
        history = TimeHistory([0., 2.], jax_numpy.stack((b, 3.*b)))
        thermal = SpatialProfile([0., 2.], jax_numpy.stack((temperature, temperature)))
        state = FiberState(axial_strain=Field([FieldTerm(strain, history)]),
                           temperature_change=Field([FieldTerm(thermal)]))
        snapshot = state.at_time(time)
        phase = (1e6*snapshot.axial_strain.integral(1.)
                 + .5*snapshot.temperature_change.integral(1.))
        return jax_numpy.cos(phase)
    parameters = jax_numpy.asarray([1., 2., 1., .5], dtype=jax_numpy.float64)
    value = jax.jit(returned_field)(parameters)
    gradient = jax.jit(jax.grad(returned_field))(parameters)
    hessian = jax.jit(jax.hessian(returned_field))(parameters)
    phase_gradient = np.array([3., 1.5, .5, 2.])
    phase_hessian = np.array([[0., 1.5, 0., 2.], [1.5, 0., 0., 1.],
                              [0., 0., 0., 0.], [2., 1., 0., 0.]])
    expected_hessian = (-np.cos(3.5)*np.outer(phase_gradient, phase_gradient)
                        - np.sin(3.5)*phase_hessian)
    np.testing.assert_allclose(value, np.cos(3.5), rtol=1e-14, atol=0.)
    np.testing.assert_allclose(gradient, -np.sin(3.5)*phase_gradient, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-13, atol=1e-15)
