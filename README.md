# PyRayleigh

Linear optical response of optical fibers under a prescribed state.

The initial scope is the fiber physics currently covered by PyOFDR.
Laser, interferometer, detector and acquisition models belong in the
interrogator packages, such as PyOFDR and PyOTDR.

The current library is configured through Python model objects. The planned
PyOFDR integration will read its YAML in PyOFDR and construct these objects
from the fiber parameters. A dedicated format for saving built fibers can
be added later, storing the actual disorder as well as the model parameters.

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
uniform sections, the built fiber layout, probe samples, point reflectors,
Rayleigh sample storage and generation, spatial profiles, time histories
and their composition into fields and a prescribed fiber state. Guide
sensitivities describe the local coupling to strain and temperature.
Static optical paths integrate phase, group delay and distributed loss through
the sections. Point-reflector returns can be summed coherently.
Further scattering models and propagation solvers will be added incrementally.

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

`Probe(frequency, time=None)` describes where an interrogator requests the
fiber response. Frequencies are absolute optical frequencies in Hz; optional
times are in seconds and must have exactly the same shape. Scalars, matrices
and empty arrays are accepted. A probe stores its own float64 copies in the
input namespace, so changing the input arrays does not change the samples.

```python
from pyrayleigh import Probe

probe = Probe([nu_ref - 1e9, nu_ref, nu_ref + 1e9], time=[0., .005, .01])
sample_shape = probe.shape
```

The physical domain is positive finite frequencies and finite times; this
container checks real numeric dtype and matching shapes. Mixed NumPy/JAX
arrays are rejected, so prepare both inputs in the intended namespace.
JAX derivatives through either frequency or time are supported when the
probe is constructed inside the transformed function. Passing the probe
object itself as a JAX argument is not yet supported.

`PointReflector` holds four field coefficients at a fixed material position
[m]. Its local scattering matrix uses
`[b_left, b_right] = S @ [a_left, a_right]` with
`S = [[r_left, t_right_to_left], [t_left_to_right, r_right]]`.
Coefficients are dimensionless and power normalized; propagation to the
element is separate. The coefficients must be finite and passive. Only
scalar numeric metadata is checked here, preserving native JAX parameters.

```python
from pyrayleigh import lossless_reflector

point = lossless_reflector(position=5., reflectance=.04, phase=.3)
scattering_matrix = point.scattering_matrix()
```

The lossless factory uses `r_left = sqrt(R)*exp(i*phase)`,
`r_right = -conj(r_left)` and real `t = sqrt(1-R)` in both directions.
It requires `0 <= R <= 1`; derivatives use the smooth interior `0 < R < 1`.
Both the factory and `scattering_matrix` accept explicit `xp` selection.
Construct the element inside the transformed function when fitting R or
phase. No random phase is drawn during evaluation.

`Fiber` accepts an optional `elements` sequence (empty by default). Point
reflectors are stored as a tuple sorted by material position, retaining
the supplied objects and their native coefficients. Positions must be
distinct and inside `[0, length]`; both ends and section boundaries are
allowed. Positions are fixed geometry, with no rounding to a spatial grid.

```python
from dataclasses import replace

fiber_with_points = replace(fiber, elements=[point])
first_point = fiber_with_points.elements[0]
```

Changing section properties does not create a point reflector. This step
stores the elements in the layout; propagation and the full fiber response
will be implemented separately. For JAX element parameters, construct the
fiber inside the transformed function as for section parameters.

`RayleighRealization` stores one local point coefficient per material cell.
The complex coefficients describe reflection from the left, with propagation
phase and attenuation applied separately. They must be finite with magnitude
at most one; only numeric dtype and shape are checked here. They are copied
as complex128 in their input namespace, retaining JAX derivatives.

```python
from pyrayleigh import RayleighRealization

rayleigh = RayleighRealization(
    coefficients=[1e-4 + 2e-4j, -3e-4j],
    positions=[0., 1.],
    edges=[0., 1., 1.25],
)
cell_lengths = rayleigh.cell_lengths()  # [1., .25] m
```

Positions and cell edges are fixed geometry in metres, supplied as Python
or NumPy values. Their read-only NumPy copies are checked: edges start at
zero and strictly increase; each coefficient has one position inside its
own cell, including either edge. Explicit edges retain a partial final
cell, and positions at the left edges can represent the PyOFDR lattice.
`cell_count` and `extent` describe the layout; `cell_edges()` and
`cell_lengths()` return new arrays of its geometry.

For JAX coefficient derivatives, construct the realization inside the
transformed function with a fixed layout. Geometry is not differentiated;
native JAX geometry arrays are rejected rather than converted to NumPy.

`generate_rayleigh` draws a sample from the sections' Rayleigh coefficients:

```python
from pyrayleigh import generate_rayleigh

rayleigh = generate_rayleigh(fiber, seed=1234, cell_length=1.)
```

The cell length [m] is explicit fixed geometry. To reproduce the PyOFDR
grid, choose it once as `c/(2*n*bandwidth)` from the reference fiber and
the optical bandwidth. Positions are the left cell edges, and the last
cell ends at the fiber length. Round-off at that end does not create an
empty cell. R is sampled at each left edge, including
for cells crossing a section boundary, rather than averaged over the cell.
The local coefficient is `sqrt(R*ds/2)*(X + i*Y)`: independent standard
normal draws give ensemble power `E[abs(r)**2] = R*ds`.

A private NumPy PCG64 generator draws all real parts followed by all
imaginary parts, as in PyOFDR. It leaves the global random state unchanged
and uses the same draws across backends. The seed identifies this stream;
per-core seed derivation belongs to the caller. Store the actual sample:
NumPy does not guarantee identical normal draws across versions.

For fitting, draw the normals once and reuse `rayleigh_from_normals` while
changing the sections' R values. It performs no random draws:

```python
import numpy as np
from pyrayleigh import rayleigh_from_normals

rng = np.random.default_rng(1234)
normals = np.stack((rng.standard_normal(rayleigh.cell_count),
                    rng.standard_normal(rayleigh.cell_count)), axis=1)
rayleigh_jax = rayleigh_from_normals(fiber, normals, cell_length=1., xp=jnp)
```

The normals have shape `(cell_count, 2)`, with real and imaginary columns.
Both functions accept explicit `xp` selection. `rayleigh_from_normals`
otherwise infers it from the normals and section rates; select JAX explicitly
when reusing NumPy draws with JAX rates. Construct the fiber and sample
inside the transformed function, with fixed draws and cell length. Rates
must be finite and nonnegative, with `R*ds << 1` for weak scattering;
derivatives in R use `R > 0`.

`Fiber.rayleigh` holds an optional `RayleighRealization`, with `None` as the
default. Its cell edges must cover exactly `[0, fiber.length]`. The supplied
sample is kept as is, including its native coefficient arrays. Construction
does not draw, resample or snap its geometry; cells can cross section
boundaries and can have different lengths.

```python
fiber_with_rayleigh = replace(fiber, rayleigh=rayleigh)
fiber_generated = fiber.with_rayleigh(seed=1234, cell_length=1.)
```

`with_rayleigh` returns a copy containing a new sample, using the same seed,
cell length and optional `xp` arguments as `generate_rayleigh`. The original
fiber keeps its sections, elements and stored sample unchanged.

Stored coefficients are not automatically rescaled when section rates change.
For fitting, rebuild the sample with fixed normals inside the transformed
function, then attach it with `replace(fiber, rayleigh=sample)`. Both steps
preserve JAX derivatives. Propagation will use this stored sample; the full
fiber response will be implemented separately.

`SpatialProfile(positions, values)` describes a static scalar field along
the material coordinate, with linear interpolation between knots and zero
outside them. Positions [m] are fixed Python or NumPy values: finite,
nondecreasing, with at least two entries. A knot can occur twice to describe
a step; its value there is the value on the right. At the last knot the
profile takes its last supplied value. Knots may lie on either side of zero.

```python
from pyrayleigh import SpatialProfile

strain = SpatialProfile([0., 1., 1., 3.], [0., 0., 10e-6, 10e-6])
local_strain = strain.value([.5, 1., 2.])
integrated_strain = strain.integral([.5, 1., 2.])
```

`value` evaluates the field, `antiderivative` integrates from `-inf`, and
`integral` integrates from zero. The integrals are exact for the linear
pieces; repeated knots add no area. Values carry the field's unit, such as
dimensionless strain, temperature change [K] or index change. Integrals have
that unit times metres. This representation will supply the strain,
temperature and index perturbations used by the phase model.

Values are finite real inputs, checked for dtype and matching shape and
copied as float64 in their native namespace. Geometry is copied as read-only
NumPy data. All three methods accept finite queries of any shape and an
optional `xp`; automatic namespace selection considers both values and
queries. Select JAX explicitly when the stored values are NumPy arrays.
Construct the profile inside the transformed function when fitting its
values. JAX derivatives in values and query position are supported; position
derivatives are intended away from the knots.

`TimeHistory(times, values)` describes prescribed samples in laboratory
time [s]. Times are fixed finite Python or NumPy values, strictly increasing,
with at least one sample. Values interpolate linearly between samples;
outside that interval the first or last value is held constant. One sample
therefore describes a constant history.

```python
from pyrayleigh import TimeHistory

history = TimeHistory([0., 1., 3.], [1., 3., 2.])
time_factors = history.value([-1., .5, 2., 4.])  # [1., 2., 2.5, 2.]
```

History values can be dimensionless multipliers of a spatial profile, or
carry the field's unit when the spatial profile is a unit shape. Times are
copied as read-only NumPy data. Values must be finite real inputs; only
dtype and matching shape are checked, and float64 copies keep their native
namespace. `value(time, xp=None)` preserves the query shape and selects the
namespace from both stored values and queries, as for `SpatialProfile`.

For JAX fitting, construct the history inside the transformed function.
Values and query time are differentiable, with time derivatives intended
away from the sample times. These are sampled, linearly interpolated
histories.

`FieldTerm(profile, history=None)` represents `profile(s)*history(t)`.
Without a history it is static. `Field(terms=())` sums any number of terms
of the same physical quantity; an empty field is zero.

```python
from pyrayleigh import Field, FieldTerm

baseline = SpatialProfile([0., 3.], [2e-6, 2e-6])
strain_field = Field([FieldTerm(baseline), FieldTerm(strain, history)])
strain_samples = strain_field.value([.5, 2.], time=[0., 1.])
strain_integrals = strain_field.integral([.5, 2.], time=[0., 1.])
```

Both containers provide `value(position, time=0., xp=None)` and
`integral(position, time=0., xp=None)`. The latter integrates in space from
zero at fixed laboratory time, multiplying each exact profile integral by
its history value. The field's unit is the product of profile and history
units; all summed terms must describe the same quantity.

Positions and times follow array broadcasting rules. Matching shapes give
paired samples; shapes `(N, 1)` and `(M,)` give an `(N, M)` position-time
grid. Static terms and empty fields follow the same shape rules.
Automatic namespace selection considers both queries and all profile and
history values. Use `xp` explicitly for mixed NumPy/JAX data.

`Field.is_static` means all terms have `history=None`. `Field.is_zero` means
the term tuple is empty; zero-valued terms are retained. These flags inspect
structure, preserving native trainable values. For JAX fitting, construct
the terms and field inside the transformed function. Values, spatial
integrals and their derivatives remain native through the composition.

`FiberState` collects the prescribed fields separately from the built fiber.
This step includes `axial_strain` [1] and `temperature_change` [K], both
explicit `Field` objects. Defaults are empty zero fields. Inputs are keyword
only and retained as supplied, including their native trainable values.

```python
from pyrayleigh import FiberState

state = FiberState(axial_strain=strain_field)
state_strain = state.axial_strain.value([.5, 2.], time=[0., 1.])
```

As in the private model, axial strain means the **total** strain of the
fiber axis, including thermal expansion. A free heated fiber has
`axial_strain = alpha_L * temperature_change`. Temperature change means
`T - fiber.reference_temperature`.

To reproduce PyOFDR's current additive strain and temperature model, the
planned integration must form `epsilon_total = epsilon_mechanical + alpha_L*dT`
with matching optical coefficients, so thermal expansion is counted once.
`FiberState` stores the supplied fields; the optical coupling and the
mechanical transfer from host to fiber will be implemented separately.

`is_static` is true when neither field contains a time history, based on
structure rather than numerical values. Evaluate each field through its
`value` and `integral` methods, with the same broadcasting and optional `xp`
selection. For JAX fitting, construct the state inside the transformed
function. The state remains separate from `Probe`, whose times query its
fields, and from `Fiber`, whose geometry and stored disorder remain fixed.

`GuideOptics` stores separate phase and group sensitivities, using the
private model's names and defaults:

| Parameter | Default | Unit |
| --- | --- | --- |
| `strain_phase_factor` | `0.78` | 1 |
| `strain_group_factor` | `0.78` | 1 |
| `temperature_phase_factor` | `6.621e-6` | 1/K |
| `temperature_group_factor` | `6.621e-6` | 1/K |
| `thermal_expansion` | `5.5e-7` | 1/K |

The strain factors default to `1-p_e`, with `p_e = .22`. The temperature
factors are at fixed **total** strain: `xi + p_e*alpha_L`, with
`xi = 6.5e-6` [1/K]. Free heating supplies `epsilon_total = alpha_L*dT`,
recovering the relative change `(xi + alpha_L)*dT`. A heated fiber with
zero total axial strain instead represents fixed axial length.

```python
phase_change = guide.relative_phase_change(axial_strain=10e-6, temperature_change=1.)
group_delay_change = guide.relative_group_delay_change(10e-6, 1.)
```

Both methods return a first-order relative change, without adding one.
Phase sensitivity refers to `frequency_ref`; group sensitivity is the
relative change of group delay per material length. Inputs broadcast,
with optional `xp` selection from the state values and the relevant factors.
Guide parameters are real scalar inputs, checked at construction and kept
native. Finite values and the documented physical domains are required;
numeric range checks are left out to preserve JAX differentiation.

The factors are independent inputs. When changing physical `p_e`, `xi` or
`alpha_L`, supply the corresponding phase and group factors together with
`thermal_expansion`. For total strain `epsilon_mechanical + alpha_L*dT`,
the phase law then recovers PyOFDR's
`(1-p_e)*epsilon_mechanical + (xi+alpha_L)*dT`.
Construct the guide inside the transformed function when fitting these
parameters. Reference propagation and group-delay methods describe the
unperturbed guide.

`optical_path(fiber, state=None)` builds an `OpticalPath` in a static state.
Omitting the state selects the reference fiber. Endpoints are fixed finite
real scalars in material coordinates [m], with `0 <= start <= end <= fiber.length`.

```python
from pyrayleigh import optical_path

path = optical_path(fiber)
phase = path.one_way_phase(0., fiber.length, probe.frequency)
field_phase = path.phase_factor(0., fiber.length, probe.frequency)
delay = path.group_delay(0., fiber.length, probe.frequency)
```

Within each traversed section, let `d` be its traversed length and `dW` the
exact field integrals over that interval. The phase and group lengths are
`Lp = d + Kp_strain*dW_strain + Kp_temperature*dW_temperature` and
`Lg = d + Kg_strain*dW_strain + Kg_temperature*dW_temperature`. The one-way
phase is the sum of `beta_ref*Lp + (beta(nu)-beta_ref)*Lg`; the group delay
is the sum of `beta'(nu)*Lg`. Sensitivities belong to each section, so a
profile crossing a boundary gets the weights of both traversed sections.

`phase_factor` returns `exp(-i*phase)`, or `exp(-2i*phase)` with
`round_trip=True`. Carrier and frequency-offset terms are exponentiated
separately, preserving small frequency changes on long paths.
`one_way_phase` returns their sum as a diagnostic absolute phase.
Frequencies are finite positive Hz within the declared optical band;
only real dtype is checked here, preserving native JAX queries.

All three methods accept `xp`; automatic selection considers frequency,
guide parameters and state values. Select JAX explicitly when mixing NumPy
constants and JAX arrays. Construct the path inside the transformed function
when fitting its guide or state. Endpoints remain fixed geometry.
This step handles static states; a state with a time history raises
`NotImplementedError`. Scattering amplitudes and the complete fiber response
will be added separately.

`OpticalPath.attenuation_integral(position, frequency)` returns cumulative
distributed power loss `A = int_0^position alpha ds` [1]. Each constant
section rate is integrated over its exact overlap with the path. Positions
before the input give zero, and positions past the output give the full
fiber loss. The output shape is `position.shape + frequency.shape`; the
current section rates are frequency independent.

```python
loss = path.attenuation_integral([0., fiber.length], probe.frequency)
transmission = path.transmission(0., fiber.length, probe.frequency)
round_trip = path.round_trip(0., fiber.length, probe.frequency)
```

For an interval, `dA = A(end) - A(start)`. One-way field transmission is
`exp(-dA/2)*exp(-i*phi)`; there-and-back transmission is
`exp(-dA)*exp(-2i*phi)`. Their power factors are `exp(-dA)` and
`exp(-2dA)`. Both methods use fixed endpoints as for `phase_factor` and
accept optional `xp` selection. They include the attenuation parameters
when selecting the namespace, preserving JAX derivatives in the rates.

The loss helper is `path.loss` (`propagation.loss.FiberLoss`). It uses the
full `Section.attenuation`, as in PyOFDR, including when a Rayleigh sample
is attached. Stored Rayleigh coefficients leave this distributed loss
unchanged. Loss uses material lengths; the prescribed state changes the
phase and delay. These factors describe distributed propagation. The
scattering solver will combine discrete element coefficients separately;
bend-loss models will follow.

`point_response(fiber, frequency, state=None, xp=None)` gives the complex
reflection at the left port from the fiber's discrete `PointReflector`
elements, with one reflection per contribution:

`H(nu) = sum_j r_left,j * prod_(k<j)(t_forward,k*t_reverse,k) * P_roundtrip(0,s_j,nu)`.

```python
from pyrayleigh import point_response

reflected = point_response(fiber_with_points, probe.frequency)
```

`P_roundtrip` is the distributed field factor of `OpticalPath.round_trip`.
Elements are already sorted by `Fiber`; each return crosses the preceding
elements in both directions, while its own reflection uses `r_left`
directly. Transmission products retain their complex phases, and a zero
transmission blocks later returns without division. Points at either fiber
end are supported. An empty element list gives zero reflection.

The result has the frequency query's shape and complex128 dtype. Automatic
namespace selection includes the frequency, guide and state values, section
attenuation and the participating point coefficients. Use explicit `xp` for
mixed NumPy/JAX arrays. For fitting, construct the fiber and elements inside
the transformed function; coefficients and losses retain JAX derivatives.
The state is static, as for `OpticalPath`. This step handles direct point
returns; stored Rayleigh contributions and multiple-reflection paths will
be added separately. Evaluation draws no random numbers.
