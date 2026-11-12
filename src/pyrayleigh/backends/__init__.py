import array_api_compat as apc


def _require_x64(xp):
    if apc.is_jax_namespace(xp):
        import jax
        if not jax.config.x64_enabled:
            raise RuntimeError('Enable jax_enable_x64 before using PyRayleigh with JAX.')


def array_namespace(*arrays):
    """Array API namespace of native arrays, with the required JAX precision."""
    xp = apc.array_namespace(*arrays)
    _require_x64(xp)
    return xp


def asarray(values, *, xp):
    """Physical numerical data as float64 or complex128 in the given namespace."""
    _require_x64(xp)
    array = xp.asarray(values)
    if not xp.isdtype(array.dtype, "numeric"):
        raise TypeError("physical arrays must contain real or complex numbers")
    dtype = xp.complex128 if xp.isdtype(array.dtype, "complex floating") else xp.float64
    return xp.astype(array, dtype, copy=False)
