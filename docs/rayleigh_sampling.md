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
