# PyRayleigh

Linear optical response of optical fibers under a prescribed state.

The initial scope is the fiber physics currently covered by PyOFDR.
Laser, interferometer, detector and acquisition models belong in the
interrogator packages, such as PyOFDR and PyOTDR.

The planned structure separates the built fiber, its state and the response:

```
src/pyrayleigh/
  fiber/          Fiber, Section and optical properties
  state/          FiberState, spatial fields and time histories
  scattering/     Rayleigh realization, point reflectors and FBGs
  propagation/    phase, loss and response solvers
  backends/       Array API helpers for NumPy and JAX
  probe.py        requested frequencies and times
  response.py     Response and compute_response(fiber, probe, state)
```

All lengths are in metres, frequencies in Hz, strain is dimensionless and
temperature changes are in K.

The numerical kernels will use the Array API Standard, with NumPy as the
reference and JAX for differentiation. Physical numerical arrays use
float64 and complex128. Random realizations will be drawn once and reused;
parameter-dependent responses will be recomputed. Discrete layouts will
remain fixed when taking derivatives.

The current code provides the package scaffold and numerical helpers.
Fiber models will be added incrementally.

Install the development dependencies, including the optional JAX backend:

```sh
python -m pip install -e ".[jax,dev]"
python -m pytest -q
```

Enable JAX x64 before creating arrays or compiling functions:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

The helpers check this setting without changing it. NumPy use does not
require importing or installing JAX.

`pyrayleigh.backends.array_namespace(*arrays)` selects the numerical namespace
from native arrays. Mixed NumPy/JAX inputs are rejected. For Python values,
`asarray(values, xp=namespace)` explicitly selects the destination namespace
and converts real/integer data to float64 and complex data to complex128.
This helper accepts physical numerical data; masks use `xp.asarray` directly.

The tests use Array API Strict to check the common subset, and JAX to check
JIT compilation and first/second derivatives through complex arithmetic.
