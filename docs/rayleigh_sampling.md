# Rayleigh cell length and optical band

`rayleigh_resolution_report` estimates how a proposed uniform lattice compares
with a nominal optical resolution. Frequencies are absolute Hz; lengths are m.

```python
from pyrayleigh import rayleigh_resolution_report

report = rayleigh_resolution_report(fiber, (nu_min, nu_max), cell_length=.002)
print(report["resolution_m"])
print(report["cell_to_resolution"])
print(report["notes"])
```

For bandwidth B and group index n_g, the nominal scale is

```text
delta_z = c / (2*n_g*B)
```

This is a scale for the frequency-to-distance response, not a maximum allowed
physical cell width. The report returns trial lengths delta_z/2, /4 and /8
for a convergence study. A width greater than delta_z/4 adds an advisory note;
it does not raise a warning, reject the model or change the grid.

For B=10 GHz and n_g=1.47, delta_z is about 10.197 mm. A 2 mm cell has
ratio 0.1961; on a 0.1 m fiber the proposed lattice contains 50 cells.

## What is evaluated

The report uses each section's reference `GuideOptics`. Its GVD gives a group
index that varies linearly with frequency, so endpoint values bound it over
the band. Each section has minimum/maximum nominal resolution; the top-level
`resolution_m` is the smallest value across the sections and frequencies.
Each section row applies that guide to one nominal cell width; a cell crossing
a section boundary is bounded by the worst guide, not assigned to a new section.
The proposed maximum cell width is clipped to the fiber length when one cell
covers the whole fiber. `cell_count` follows the uniform generator's endpoint
rule, without allocating the lattice; the final cell would be clipped at L.
Sub-cell round-off in the material grid is not a resolution estimate.

| Report field | Meaning |
|---|---|
| `frequency_band_hz`, `bandwidth_hz` | Requested absolute band and its width |
| `cell_length_m`, `max_cell_length_m` | Requested lattice spacing and nominal maximum cell width |
| `cell_count` | Number of proposed cells, including the clipped final cell |
| `resolution_m`, `cell_to_resolution` | Smallest nominal resolution and cell width divided by it |
| `trial_cell_lengths_m` | Three candidate widths for a later convergence study |
| `sections` | Section bounds, group-index range, resolution range and phase estimates |
| `notes` | Advisory information, without automatic solver changes |

Each section's `phase_span_bound_rad` bounds the change of round-trip phase
across the full optical band over one nominal cell:

```text
phase_span_bound = 4*pi*B*max(n_g)*cell_width/c
```

It is a frequency-span estimate. `carrier_phase_rad` separately gives twice
the reference propagation constant times the cell width. The large carrier
phase is not a discretization error, and is not subtracted from two large
absolute phases when estimating the band variation.

## Scope and interpretation

This helper is a NumPy float64 reference-guide report for one scalar core. It does
not evaluate imposed strain, temperature or a static index profile; the latter
is flagged in `notes` if present. It does not inspect existing Rayleigh samples
or their coefficients. A proposed spacing can differ from a stored realization;
the report does not claim to describe that stored lattice.

Neither the stored disorder nor the published generator is changed. Existing
samples remain at their supplied positions, and the generator still uses left
cell edges. No random numbers are drawn and no coefficients are clipped.

A small cell/resolution ratio does not certify exact-solver convergence or
passivity. Independent grids need statistical comparisons; changing N with the
same seed does not preserve one complex realization. A comparison of nearly
the same fiber needs coupled refinement with an explicit phase/position rule.
The report does not implement that refinement or validate the Gaussian law.

## Independent-grid statistical example

Run [rayleigh_statistics.ipynb](../examples/rayleigh_statistics.ipynb) for a
first weak-scattering reference study. The fiber is uniform, 0.0625 m long,
with n_eff=n_g=1.47, zero GVD, no point elements, zero background loss and
rho*L=1e-4. The optical band is 10 GHz around 1550 nm, sampled at 65 frequencies.
Each of four left-edge grids has its own 512-realization ensemble. Seeds are
`120000 + 10000*grid_index + realization_index`; different grids are not coupled.

Within one realization, exact, direct-consistent and legacy fields use the
same stored sample. In this point-free case, `max_order=1` is the undepleted
single-scattering field, with analytic Gaussian statistics. The notebook
checks coefficient normalization, reflected power at one frequency, its fading
CDF, complex frequency covariance and exact lossless energy conservation.
Frequency bins within a realization are not treated as independent samples.

The independent covariance references are a finite left-edge sum and the
uniform continuum integral. The deterministic difference between these two
references is distinguished from the measured ensemble error. In the reference
run (core revision `4d9f7b7`, NumPy 2.5.4, complex128):

| Cells | Cell/resolution | Max discrete-continuum covariance gap | Measured-discrete covariance RMS |
|---|---|---|---|
| 8 | 0.7662 | 0.18845 | 0.03470 |
| 16 | 0.3831 | 0.06648 | 0.04399 |
| 32 | 0.1915 | 0.03160 | 0.03715 |
| 64 | 0.09577 | 0.01567 | 0.05312 |

Covariance errors are normalized by rho*L. The nominal Monte Carlo standard
error is about 1/sqrt(512)=0.0442 per frequency offset. Thus the measured error
does not have to fall monotonically with cell count; more realizations are
needed to resolve a grid bias below that uncertainty. Mean power and a fading
CDF can look correct even on the coarse grid, so they are not convergence
certificates by themselves.

The exact/direct-consistent relative field differences are about 7e-5 to 8e-5
in this weak case. Exact/legacy differences also include missing legacy
depletion. This example does not establish convergence for strong backscatter,
cavities, heterogeneous or perturbed fibers, and it does not implement coupled
refinement. It adds a runnable example, with local audit scripts kept outside
the repository test suite.

## Conditional coefficient split

`split_rayleigh_coefficients` is a deterministic primitive for splitting each
parent coefficient into two equal-width children. It accepts supplied normals
and returns an (N, 2) array in the selected Array API backend, including JAX.
It does not create a new spatial grid or draw random numbers.

```python
from pyrayleigh import split_rayleigh_coefficients

children = split_rayleigh_coefficients(
    parent_coefficients, normals, rate, parent_cell_lengths,
    right_phase=reference_phase_difference,
)
```

`normals` contains one independent standard-normal real/imaginary pair per
parent and must be independent of the parent draws. `rate` is the parent's
guided-backscatter rate in 1/m, and lengths are in m. Scalars or one value per
parent are accepted for rate, length and phase. Let V=rate*length, and define

```text
eta = sqrt(V/8) * (X + i*Y)
r_left  = r_parent/2 + eta
r_right = (r_parent/2 - eta) * exp(+i*theta)
```

For fixed model parameters and circular complex Gaussian parents with
variance V, the children have variance V/2 and are independent in the
unconditional ensemble. Conditional on a fixed parent,
their phase-aligned fluctuations are anticorrelated. Do not estimate V from
the power of one parent draw. Use the known prior rate and cell width.

### Optical phase convention

The invariant is `r_left + exp(-i*theta)*r_right = r_parent`. For a parent at
the left cell edge, the left child stays there and the right child is at the
midpoint. Theta is the extra round-trip optical phase to that midpoint at a
chosen reference frequency and reference state. It is an angle in radians,
not a complex propagation factor. The positive phase in the returned right
coefficient compensates the negative phase of propagation.

The default theta=0 preserves the raw coefficient sum. That alone generally
does not preserve a complex optical return when scatterer positions change.
A supplied optical phase preserves only the undepleted single-scattering
return at the chosen reference when the interval is lossless and has no
intervening point element. Loss or an element between the children also changes
return amplitudes, which this phase-only compensation does not preserve.
It does not force equal spectra at other
frequencies or equal exact responses with depletion and multiple scattering.
Choose the reference frame once: do not rephase the stored disorder when
applying a new strain or temperature state.

### Current limits

This primitive checks numeric dtype and shape, leaving physical values native
for differentiation, as the existing coefficient constructor does. Parents
must obey the declared prior; at zero variance their physical value is zero.
Gaussian children can exceed |r|=1, so the exact solver's domain check remains
necessary. No clipping, rejection sampling or coefficient normalization is
performed by the split.

The equal split assumes a constant rate within each parent cell. If a parent
crosses a change in rate, using new child section rates is a different model
unless its variances are made consistent. Geometry construction, a rule for
section boundaries and an optical convergence study of coupled grids remain
separate work. Existing samples and the published generator are unchanged.
