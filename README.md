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
  backends/       NumPy and native Torch operations
  probe.py        requested frequencies and times
  response.py     Response and compute_response(fiber, probe, state)
```

All lengths are in metres, frequencies in Hz, strain is dimensionless and
temperature changes are in K.

The NumPy and optional Torch backends will use float64 and complex128.
The Torch calculations will keep the graph in Torch. Random realizations
will be drawn once and reused; parameter-dependent responses will be
recomputed. Discrete layouts will remain fixed when taking derivatives.

This first commit contains only the package scaffold.
