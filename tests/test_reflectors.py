import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh import PointReflector, lossless_reflector


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


def test_port_order_and_complex_dtype(xp):
    point = PointReflector(2., .2, -.2, .8, .7)
    matrix = point.scattering_matrix(xp=xp)
    assert matrix.shape == (2, 2)
    assert matrix.dtype == xp.complex128
    np.testing.assert_array_equal(np.asarray(matrix), [[.2, .7], [.8, -.2]])


def test_lossless_coefficients(xp):
    point = lossless_reflector(3., .04, .3, xp=xp)
    r_left = .2*np.exp(.3j)
    transmission = .9797958971132712
    expected = [[r_left, transmission], [transmission, -np.conj(r_left)]]
    np.testing.assert_allclose(np.asarray(point.scattering_matrix()), expected, rtol=1e-14, atol=0.)
    assert point.position == 3.


@pytest.mark.parametrize("reflectance", [0., .04, 1.])
def test_reciprocity_and_energy_conservation(xp, reflectance):
    matrix = lossless_reflector(0., reflectance, .3, xp=xp).scattering_matrix()
    gram = xp.conj(xp.permute_dims(matrix, (1, 0))) @ matrix
    np.testing.assert_allclose(np.asarray(gram), np.eye(2), rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(matrix), np.asarray(xp.permute_dims(matrix, (1, 0))))


@pytest.mark.parametrize("position,error", [
    (-1., ValueError), (float("nan"), ValueError), (float("inf"), ValueError),
    (True, TypeError), (np.array(0.), TypeError),
])
def test_position_is_fixed_finite_and_nonnegative(position, error):
    with pytest.raises(error):
        lossless_reflector(position, .04)


def test_coefficients_and_factory_parameters_are_scalars():
    with pytest.raises(ValueError, match="scalars"):
        PointReflector(0., np.array([.2]), -.2, .8, .8)
    with pytest.raises(TypeError):
        PointReflector(0., True, -.2, .8, .8)
    with pytest.raises(ValueError, match="scalars"):
        lossless_reflector(0., [.04], .3)
    with pytest.raises(ValueError, match="scalars"):
        lossless_reflector(0., .04, [.3])


def test_jax_matrix_preserves_reflectance_and_phase_derivatives(jax_numpy):
    import jax
    def recombined_field(parameters):
        point = lossless_reflector(2., parameters[0], parameters[1])
        matrix = point.scattering_matrix()
        return jax_numpy.real(matrix[0, 0] + matrix[1, 0])
    parameters = jax_numpy.asarray([.04, .3], dtype=jax_numpy.float64)
    value = jax.jit(recombined_field)(parameters)
    gradient = jax.jit(jax.grad(recombined_field))(parameters)
    hessian = jax.jit(jax.hessian(recombined_field))(parameters)
    reflectance, phase = np.asarray(parameters)
    r, t = np.sqrt(reflectance), np.sqrt(1. - reflectance)
    expected_gradient = [np.cos(phase)/(2.*r) - 1./(2.*t), -r*np.sin(phase)]
    mixed = -np.sin(phase)/(2.*r)
    expected_hessian = [[-np.cos(phase)/(4.*r**3) - 1./(4.*t**3), mixed],
                        [mixed, -r*np.cos(phase)]]
    np.testing.assert_allclose(value, r*np.cos(phase) + t, rtol=1e-14, atol=0.)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-13, atol=0.)
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-13, atol=0.)


def test_mixed_coefficient_namespaces_need_explicit_selection(jax_numpy):
    r_left = jax_numpy.asarray(.2 + .1j, dtype=jax_numpy.complex128)
    transmission = np.array(.9)
    point = PointReflector(0., r_left, -jax_numpy.conj(r_left), transmission, transmission)
    with pytest.raises(TypeError):
        point.scattering_matrix()
    matrix = point.scattering_matrix(xp=jax_numpy)
    assert matrix.dtype == jax_numpy.complex128
    np.testing.assert_allclose(np.asarray(matrix), [[.2 + .1j, .9], [.9, -.2 + .1j]])
