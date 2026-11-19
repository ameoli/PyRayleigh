# Optical conventions and loss budget

The NumPy reference is available through `compute_response(..., method="exact")`.
The existing `compute_response(..., max_order=...)` calls keep their numerical
behavior, using `method="truncated"` by default.

## Scope

The first exact implementation covers one scalar core, a static `FiberState`,
stored Rayleigh samples and `PointReflector` elements. It composes all orders
of scattering in that discretized model. It does not make the Rayleigh grid,
the prescribed material model or the time evolution exact.

PyRayleigh returns complex optical fields. Laser noise, interferometers,
ADC samples, FFT distance axes and strain estimation belong to the interrogator.
The fiber realization is supplied once and reused when its state changes.

```python
from pyrayleigh import compute_response

result = compute_response(fiber, probe, state, method="exact",
                          loss_convention="background")
```

`probe` is a `Probe` and `state` can be omitted for the reference state.
The result is the existing `Response`, with the probe's frequency shape.
This reference rejects non-NumPy parameters rather than converting them.
`xp=None`, `numpy` and `array_api_compat.numpy` are accepted for this method;
other namespaces are not. The standalone `exact_response` function remains
available from `pyrayleigh.propagation.exact`.

## Fields and ports

Use the temporal phasor convention `exp(+i*2*pi*nu*t)`. Frequencies are absolute
Hz, material positions are in m, and optical phases are in rad. Numerical
parameters use float64 and fields use complex128.

Port amplitudes are power normalized. At either port, `a` is incoming and `b`
is outgoing. With the left port first:

```text
[b_L]   [r_L   t_RL] [a_L]
[b_R] = [t_LR  r_R ] [a_R]
```

With unit illumination from the left and a matched right port, `a_L=1` and
`a_R=0`. `Response.reflection` is `r_L` of the complete fiber and
`Response.transmission` is `t_LR`. Their squared magnitudes are power fractions.
Reference planes are the input at material position 0 and the output at L.

Use all four supplied `PointReflector` coefficients. In particular, do not
reconstruct `r_R` from `r_L`, or replace directional transmissions with an
assumed common value. The relation `r_R=-conj(r_L)` applies to the existing
lossless factory with real positive transmission, not to an arbitrary two-port.

For a lossless two-port, `S.conj().T @ S = I`. A passive two-port satisfies
`S.conj().T @ S <= I` in the positive-semidefinite sense. Check the largest
singular value of the full matrix, not only power for illumination from the
left. Reciprocity gives `t_LR=t_RL` when the physical component is reciprocal;
it does not require equal reflection coefficients.

## Propagation

For a material interval, use `optical_path(fiber, state).transmission(...)`:

$$
p(s_0,s_1,\nu)=\exp(-A/2)\exp(-i\Phi),\qquad
A=\int_{s_0}^{s_1}\alpha_{\rm background}(s,\nu)\,ds.
$$

`A` is a dimensionless power-loss exponent. One-way power is multiplied by
`exp(-A)` and the reciprocal round-trip field by `exp(-A)`. For uniform loss,
`alpha = attenuation_dB_per_km * log(10)/10000` in 1/m.

The propagation two-port has zero reflection and `t_LR=t_RL=p`. Its phase,
dispersion, strain and temperature response come from `OpticalPath`; the new
solver must not duplicate those equations. Keep carrier and detuning terms
separate as in the current implementation. Group delay uses `dPhi/domega`.

## Rayleigh cells

Keep the stored positions, cell edges and coefficients. The existing generator
places samples at left cell edges, with the last cell clipped to the fiber.
Changing the probe or state must not redraw or reposition them.

The guided backward-scattering rate `rho` has units 1/m. The current sampling
law is `r_j=sqrt(rho_j*ds_j/2)*(X_j+i*Y_j)`, so `E[abs(r_j)**2]=rho_j*ds_j`.
For the exact method, a cell uses the reciprocal lossless matrix

```text
[r_j                    sqrt(1-abs(r_j)**2)]
[sqrt(1-abs(r_j)**2)     -conj(r_j)         ]
```

This explicitly removes guided forward power through its transmission.
It represents the backward guided mode, not all radiation and absorption.
Finite coefficients with `abs(r_j)**2 <= 1` are required. Reject nonphysical
samples before the square root; do not clip or renormalize the Gaussian draws.
The current generator and legacy solver retain their existing behavior.

## Loss budget

Three quantities have different meanings:

| Quantity | Meaning | Unit |
|---|---|---|
| `alpha_total_measured` | Attenuation inferred from a stated measurement/calibration | 1/m |
| `rho_guided_backscatter` | Rate used to generate the explicit backward guided scatterers | 1/m |
| `alpha_background` | Residual distributed loss not represented by those scatterers | 1/m |

The exact solver uses `alpha_background` for distributed propagation.
It includes absorption and unmodelled radiation or bend loss. It excludes the
guided backscattering represented explicitly by the Rayleigh matrices.
Discrete connector loss is already in the point matrix and must not be added
again to the distributed loss. Likewise, do not encode the same bend loss in
both a section coefficient and an active bend-loss table.

Today `Section.attenuation` is the power coefficient used by `OpticalPath`.
The API does not record whether it came from total measured loss or a residual
background. Its stored value and legacy interpretation remain unchanged.

An exact calculation requires `loss_convention="background"` to declare that
the supplied section coefficients represent background loss. Omitting it or
using another value is an error. This adds no argument or default to `Section`
and does not perform a loss calibration for the caller. The dispatcher rejects
loss options on the truncated method rather than silently ignoring them.

Do not automatically subtract `rho` from an attenuation coefficient. A mapping
such as `alpha_background ~= alpha_total_measured - rho_guided_backscatter`
needs a justified weak-scattering calibration with compatible definitions and
a nonnegative result. It is not a correction based on the power of one random
realization. Record any chosen mapping explicitly in the experiment setup.

For an ideal lossless experiment, `alpha_background=0` and nonzero Rayleigh
scattering are valid: the cell matrices redistribute power between the ports.
Refining the Rayleigh grid does not recalibrate the background loss.

## Geometry and state

Order events by material position. At a coincident point and Rayleigh sample,
compose the `PointReflector` first and the Rayleigh sample second. This agrees
with the current direct Rayleigh response crossing points at or before its
position. Coincident Rayleigh samples retain their stored order.
A section boundary introduces no extra reflection; propagation
integrates the sections on each side. A zero-length interval is transparent.

`FiberState.axial_strain` is total axial strain, including thermal expansion;
`temperature_change` is a change from the fiber's reference temperature in K.
`GuideOptics` defines the thermal sensitivity at fixed total strain. A caller
starting from mechanical strain must convert it once, without adding thermal
expansion again inside the solver.

Evaluate propagation on the actual material legs. For the point-only path
`0 -> b -> a -> b -> 0`, a strain region before a is crossed twice, between
a and b four times, and beyond b zero times. The apparent position `2*b-a`
is not a propagation endpoint. Do not remove a physical contribution because
its apparent distance exceeds the material fiber length.

The first exact method rejects time-varying states, multicore/crosstalk and
`WeakFBG` inputs explicitly. A caller can first supply a frozen static state;
this is not retarded propagation. Distributed Bragg gratings need a separate
two-port model and geometry contract. Existing methods keep their support.

## Comparisons and numerical policy

The existing `max_order=1/2/3` calls keep their meaning: direct Rayleigh and
localized returns, with optional repeated point reflections. Their transmitted
field is the direct one-way field; Rayleigh cells currently add no depletion.

The exact selector is separate from `max_order`. An explicitly supplied path
order, including `max_order=1`, is an error for `method="exact"`. With the
truncated method an omitted order still means 1; explicit orders must remain
integers 1, 2 or 3. `Response` keeps its current fields, shapes and units.
Unknown methods are rejected. There is no `method="born"` alias for the
truncated model. Use `direct_response_consistent` for the single-reflection
comparison with the exact model's local transmissions.

To isolate multipath, compare exact and a direct-consistent calculation using
the same local matrices, transmissions, background loss and realization.
The latter retains one reflection event per path. Comparing exact with the
legacy method also includes their different treatment of Rayleigh depletion.

### Direct response with the same local transmissions

```python
from pyrayleigh import direct_response_consistent

direct = direct_response_consistent(fiber, probe, state,
                                     loss_convention="background")
residual = result.reflection - direct.reflection
```

Here `result` is the exact `Response` computed above. The direct function
also returns a `Response`: reflection includes exactly one reflection event,
and transmission includes no reflections. It has the same static, scalar,
NumPy-only scope and requires the same background-loss declaration.

For ordered local scatterers, with reciprocal distributed factor `P(0,s)`,

```text
H_direct = sum_j r_L,j * P(0,s_j)^2 * product_(k<j)(t_LR,k * t_RL,k)
T_direct = P(0,L) * product_j t_LR,j
```

Each return crosses all earlier blocks in both directions, using their own
directional transmissions. It does not cross its own transmission. The
point/cell tie order is the same as for exact. Stored Rayleigh coefficients
are reused, with `sqrt(1-abs(r_j)**2)` at every crossing, without new draws.

Both calculations use the same internal block builder and `OpticalPath`.
The direct calculation visits the nodes once; it does not enumerate paths
or introduce a denominator cutoff. Wave memory is O(M), plus O(N) node data.
It is run only when requested, without extra fields in an exact-only call.

This is a path truncation for a diagnostic comparison. With strong coherent
returns it can give `abs(H_direct)**2 + abs(T_direct)**2 > 1`: the omitted
paths also contribute interference. Do not normalize or clip that result, or
interpret it as a separate complete passive fiber. Exact passivity is checked
on the full response. This decomposition depends on the chosen local blocks.

Subtract complex fields: `H_residual = H_exact - H_direct`. Power includes the
interference term `2*real(H_direct*conj(H_residual))`; it is not additive across
those two fields.

### Optional diagnostic result

```python
from pyrayleigh import compute_response_diagnostics

diagnostic = compute_response_diagnostics(fiber, probe, state,
                                         loss_convention="background")
H_total = diagnostic.total.reflection
H_direct = diagnostic.direct.reflection
H_residual = diagnostic.residual.reflection
power = abs(H_direct)**2 + abs(H_residual)**2 + diagnostic.reflection_interference
```

`DiagnosticResponse` holds three `Response` objects with the original probe.
`residual` is formed as `total - direct` for reflection and transmission.
Reflected residual paths have at least three reflection events; transmitted
residual paths have at least two. The same scalar static NumPy scope applies.
There is no new selector or default in `compute_response`.

The signed `reflection_interference` and `transmission_interference` properties
give the power cross term for each port. They are calculated on request, not
stored as extra arrays. In particular, `abs(H_residual)**2` is not the change
in total reflected power: that change also includes the interference term.

This diagnostic runs exact and direct separately on the same fiber, state,
probe and stored disorder. It retains six complex field arrays, versus two
for a plain response; both block traversals are performed only when the
diagnostic function is called. There is no shared-pass optimization yet.
The exact denominator cutoff is accepted and its failures are propagated.

Metadata records fiber ID, cell count, model scope, backend, precision, boundary
condition, background-loss declaration and denominator cutoff. It is a summary,
not a saved fiber or state, and contains no inferred random seed. Keep the
actual realization and state with the experiment for a reproducible comparison.

Do not replace a small cascade denominator with an epsilon. Near resonances
require diagnostics and a conditioning check; exactly degenerate cases need
explicit treatment or a reported failure. Test opaque elements to ensure
downstream components cannot contribute through a blocked port.

The reference composes `TwoPort` blocks with the scalar `star` product. It
checks finite inputs, nonnegative background losses and rates, full point
passivity, and the Rayleigh coefficient domain before composing the fiber.
`denominator_tolerance` defaults to `64*eps(float64)`. A denominator at or below
the cutoff raises `NumericalResonanceError`, with sample index, frequency,
material position and the failed denominator. It is a refusal cutoff, not an
accuracy bound for high-Q cavities. No epsilon is added to a denominator.
The same optional keyword can be passed through `compute_response` with
`method="exact"`; its value is checked by the cascade.

If the complete prefix has zero left-to-right transmission at every sample,
the solver returns its left reflection and zero transmission. A disconnected
closed cavity downstream then needs no evaluation. Otherwise a singular
composition is reported. An opaque component is not a source of NaNs.

Frequency samples are evaluated together and nodes are visited in spatial
order. Wave arrays take O(M) memory for M probe samples; node metadata takes
O(N). Profile integration still uses the existing `OpticalPath` routines.
This is a reference implementation, without a performance optimization pass.

Start with a NumPy reference. Keep the Array API design and add JAX parity
after the reference is verified. Physical checks must not convert JAX tracers
to NumPy or Python booleans; later traced execution needs separate validation
or numerical status reporting.

## First validation example

`examples/fabry_perot.ipynb` compares with an independent analytic two-mirror
solution. For the uniform reference it calculates propagation directly from
lengths, index and loss, without calling the cascade or `OpticalPath`.
It compares complex reflection and transmission, energy residuals, weak and
strong reflectors, and local-strain crossing counts. Parameters, frequency
grid, precision, backend and tolerances are recorded. This is a scientific example;
matching legacy ADC traces alone does not validate the exact optical model.

The notebook also illuminates a uniform point-only fiber from both ends.
Reversing the geometry swaps the local port coefficients, without complex
conjugation. Binary-exact positions avoid an extra geometric rounding error.
All four fields are compared with the analytic cavity matrix. Passivity is
checked on its largest singular value; lossless cases also require unitarity.
The examples include distributed background loss and an opaque mirror.

At `R=0.9999`, a fixed 12 GHz band is sampled at 1501, 6001, 24001 and 96001
frequencies. The 8 MHz grid misses the narrow peak; the 0.125 MHz grid gives
peak power 0.99973 and FWHM about 3.07944 MHz, versus 3.07767 MHz analytically.
Fields at shared frequencies agree. Refining the frequency grid resolves the
spectrum; it does not improve the spatial Rayleigh model or regularize a
small denominator. The analytic reference still uses float64, and its stated
round-off allowances are not a general high-Q accuracy guarantee.

The final notebook section requests a `DiagnosticResponse` for a point-only
cavity over an 80 GHz band. It checks direct and residual fields against the
closed ghost series for both ports, and reconstructs power with the signed
interference term. The plotted component powers are not independent budgets.

## Validation covered so far

Local checks below are kept outside the repository test suite. The notebook
is the runnable example shipped with the source. This table records coverage,
not completion of every validation or convergence task in the roadmap.

| Check | Coverage | Where |
|---|---|---|
| T1 | Transparent fiber, single point/cell, opaque prefixes and end mirrors | Local NumPy checks |
| T2 | Independent two-mirror complex fields, weak/strong reflectors and loss | Fabry-Perot notebook |
| T3 | Static local strain and 2/4/0 crossings of the first repeated point path | Fabry-Perot notebook |
| T4 | Full-matrix passivity, unitarity and reciprocity; mixed point/Rayleigh matrices against interface equations | Notebook and local NumPy checks |
| T5 | Weak-limit scaling of a fixed Rayleigh sample; fixed-transmission single-reflection limit | Local NumPy checks |
| T6 | Frequency-grid resolution, near-resonance diagnostics and point paths with apparent position beyond L | Notebook and local NumPy checks |

The T5 comparison with the truncated response includes its missing Rayleigh
depletion. Use the direct-consistent response to isolate multiple scattering
with the same local transmissions. Local checks compare its path sum with
independent single-bounce formulas and the two-mirror residual with the
analytic ghost series, including apparent positions beyond L.
Coupled spatial refinement, Rayleigh statistics and exact JAX
parity remain separate validation work. Reversal checks on mixed networks
avoid coincident point/cell nodes: reversing those would also reverse their
local composition order, which the current fixed tie convention does not allow.

The reference-guide `rayleigh_resolution_report` estimates a proposed cell
length against an optical band. Its advisory ratios and phase bounds do not
establish convergence. See [Rayleigh sampling](rayleigh_sampling.md) for units,
scope and the distinction between proposed and stored grids.
