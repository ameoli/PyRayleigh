from pathlib import Path
import subprocess
import sys

import array_api_compat.numpy as numpy_api
import array_api_strict as strict
import numpy as np
import pytest

from pyrayleigh.backends import array_namespace, asarray


@pytest.fixture
def jax_numpy():
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    yield jnp
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("xp", [numpy_api, strict])
@pytest.mark.parametrize("values", [[1, 2], [[1 + 2j, 3 - 4j]], 1e-10, []])
def test_physical_array_values_and_precision(xp, values):
    array = asarray(values, xp=xp)
    expected_dtype = np.complex128 if np.iscomplexobj(values) else np.float64
    expected = np.asarray(values, dtype=expected_dtype)
    np.testing.assert_array_equal(np.asarray(array), expected)
    assert array.shape == expected.shape
    assert xp.isdtype(array.dtype, "complex floating" if np.iscomplexobj(values) else "real floating")
    assert array.dtype == (xp.complex128 if np.iscomplexobj(values) else xp.float64)


def interference_intensity(phase):
    xp = array_namespace(phase)
    phase = asarray(phase, xp=xp)
    i = xp.asarray(1j, dtype=xp.complex128)
    phasors = xp.exp(i * xp.astype(phase, xp.complex128))
    field = phasors[0] + 2. * phasors[1]
    return xp.real(field * xp.conj(field))


@pytest.mark.parametrize("xp", [numpy_api, strict])
def test_two_reflector_interference(xp):
    phase = asarray([.2, .7], xp=xp)
    expected = 5. + 4. * np.cos(.2 - .7)
    np.testing.assert_allclose(np.asarray(interference_intensity(phase)), expected, rtol=1e-14)


def test_jax_precision_jit_and_interference_derivatives(jax_numpy):
    import jax
    phase = asarray([.2, .7], xp=jax_numpy)
    assert phase.dtype == jax_numpy.float64
    value = jax.jit(interference_intensity)(phase)
    gradient = jax.jit(jax.grad(interference_intensity))(phase)
    hessian = jax.jit(jax.hessian(interference_intensity))(phase)
    difference = .2 - .7
    np.testing.assert_allclose(value, 5. + 4. * np.cos(difference), rtol=1e-14)
    np.testing.assert_allclose(gradient, [-4.*np.sin(difference), 4.*np.sin(difference)], rtol=1e-14)
    expected_hessian = 4.*np.cos(difference)*np.array([[-1., 1.], [1., -1.]])
    np.testing.assert_allclose(hessian, expected_hessian, rtol=1e-14)


def test_jax_complex_conversion_preserves_derivatives(jax_numpy):
    import jax
    def loss(x):
        z = asarray(x + 2j*x, xp=jax_numpy)
        values = jax_numpy.stack((z, 2*z))
        return jax_numpy.sum(jax_numpy.real(values * jax_numpy.conj(values)))
    x = asarray([1., 2.], xp=jax_numpy)
    np.testing.assert_allclose(jax.grad(loss)(x), 50.*np.asarray(x))
    np.testing.assert_allclose(jax.hessian(loss)(x), 50.*np.eye(2))
    assert asarray([1 + 2j], xp=jax_numpy).dtype == jax_numpy.complex128


def test_jax_x64_is_required_without_changing_the_setting():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    previous = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", False)
        values = jnp.asarray([1.], dtype=jnp.float32)
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            array_namespace(values)
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            asarray([1.], xp=jnp)
        assert not jax.config.x64_enabled
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_mixed_numpy_and_jax_arrays_are_rejected(jax_numpy):
    with pytest.raises(TypeError):
        array_namespace(np.array([1.]), jax_numpy.asarray([1.]))


def test_masks_are_not_physical_numeric_arrays():
    with pytest.raises(TypeError, match="physical arrays"):
        asarray([True, False], xp=numpy_api)


def test_numpy_use_does_not_require_jax():
    code = """
import builtins
import numpy as np
original_import = builtins.__import__
def without_jax(name, *args, **kwargs):
    if name == 'jax' or name.startswith('jax.'):
        raise ModuleNotFoundError('JAX unavailable', name='jax')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_jax
from pyrayleigh.backends import array_namespace, asarray
xp = array_namespace(np.array([1., 2.]))
values = asarray([1., 2.], xp=xp)
assert values.dtype == np.float64
assert xp.sum(values) == 3.
"""
    source = Path(__file__).resolve().parents[1] / "src"
    code = f"import sys; sys.path.insert(0, {str(source)!r})\n" + code
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
