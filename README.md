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

The current code provides numerical helpers, SI conversions, guide optics,
uniform sections and the built fiber layout. Scattering models and
propagation solvers will be added incrementally.

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

`pyrayleigh.units` converts power ratios, dB, attenuation, wavelength,
frequency and chromatic dispersion. Attenuation is stored as the power
coefficient alpha [1/m], with `P(z) = P(0) exp(-alpha*z)`; a one-way field
uses half that exponent. The dispersion parameter D is in s/m^2, and beta2
is in s^2/m. Wavelengths and optical frequencies are positive.

```python
from pyrayleigh import units

alpha = units.attenuation_from_db_per_km(.2)
beta2 = units.dispersion_to_gvd_at_wavelength(17 * units.PS_PER_NM_KM, 1550e-9)
```

The conversions infer the namespace from native arrays; Python values use
NumPy unless `xp` is supplied. They keep the same x64 requirement for JAX.

`GuideOptics` stores the reference frequency [Hz], effective index, group
index and beta2 [s^2/m]. With `dw = 2*pi*(frequency-frequency_ref)` it uses
`beta = beta_ref + group_index*dw/c + beta2*dw**2/2`. The carrier and its
offset can be evaluated separately, retaining small frequency shifts.

```python
from pyrayleigh import GuideOptics

nu_ref = units.wavelength_to_frequency(1550e-9)
guide = GuideOptics(nu_ref, effective_index=1.4682, group_index=1.4682, gvd=beta2)
beta = guide.propagation_constant(nu_ref)
delay_per_metre = guide.group_delay_per_length(nu_ref)
```

For differentiation with respect to guide parameters, construct the guide
inside the transformed function with its native JAX inputs. Passing the
guide object itself as a JAX argument is not yet supported. Each method
accepts `xp` for explicit namespace selection; automatic selection considers
both frequency and guide parameter arrays.

If guide constants came from NumPy and frequencies are JAX arrays, select
JAX explicitly (or prepare the constants as JAX arrays too):

```python
import jax.numpy as jnp

frequency = jnp.asarray(nu_ref + 1e9)
beta_jax = guide.propagation_constant(frequency, xp=jnp)
```

`Section` describes a fixed material interval with a guide, a constant power
attenuation coefficient and a guided Rayleigh coefficient. Both coefficients
are explicit, in 1/m. Their physical domain is finite and nonnegative; only
their real scalar dtype/shape is checked here, preserving native JAX values.
The geometry is validated as fixed real numbers with `0 <= start < end`.

```python
from pyrayleigh import Section

section = Section(0., 10., guide, attenuation=alpha,
                  rayleigh_coefficient=units.db_to_linear(-82.))
length = section.length
alpha_at_reference = section.attenuation_at(nu_ref)
```

`attenuation_at` returns the same coefficient with the frequency input's
shape, with optional `xp` selection. Different sections do not imply a
splice or reflection at their boundary; those will be explicit elements.
For JAX coefficient derivatives, construct the section inside the transformed
function with fixed geometry. Passing a section object as a JAX argument is
not yet supported.

`Fiber` collects ordered sections covering the material interval `[0, length]`.
Its identifier, length, reference temperature and declared optical band are
fixed metadata, independent of the interrogator and of the prescribed state.
Sections are stored as a tuple; gaps, overlaps and incomplete coverage are
rejected. Boundary differences up to `1e-12 * length` are snapped on copies
of the affected sections, retaining their native optical parameters.

```python
from pyrayleigh import Fiber

fiber = Fiber("spool-1", length=10., reference_temperature=293.15,
              frequency_min=nu_ref - 50e9, frequency_max=nu_ref + 50e9,
              sections=[section])
```

Reference temperature is in K and the band uses absolute optical frequencies
in Hz. Like section geometry, these metadata are fixed finite real numbers;
all are positive and `frequency_min < frequency_max`. Construct the fiber
inside the transformed function when fitting its sections' parameters;
passing a fiber object as a JAX argument is not yet supported.
