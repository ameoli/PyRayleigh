# PyRayleigh
![tests](https://github.com/ameoli/PyRayleigh/actions/workflows/tests.yml/badge.svg)
![release](https://img.shields.io/github/v/release/ameoli/PyRayleigh?include_prereleases)
![license](https://img.shields.io/github/license/ameoli/PyRayleigh)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![status](https://img.shields.io/badge/status-alpha-orange)

PyRayleigh computes the complex reflection and transmission of an optical fiber.
The goal is to move the fiber physics from PyOFDR into a library that can also be used by PyOTDR.
It currently includes Rayleigh scattering, point reflectors, distributed loss, index changes, strain and temperature.
The current solver uses one reflection per path.

The calculations use the Array API with NumPy and optional JAX for derivatives.
Numerical arrays use float64 and complex128. Lengths are in metres, frequencies in Hz and temperature changes in K. Random fiber profiles are generated once and reused.

## Structure

```text
src/pyrayleigh/
  fiber/          fiber layout, sections, optical properties and index profiles
  state/          strain, temperature, spatial profiles and time histories
  scattering/     Rayleigh samples and point reflectors
  propagation/    phase, delay, loss and reflected fields
  backends/       Array API helpers
  probe.py        frequencies and laboratory times to evaluate
  response.py     reflection and transmission
  units.py        unit conversions
  constants.py    physical constants
tests/            tests for NumPy, JAX and Array API Strict
```

`Fiber` describes the built fiber. `FiberState` contains its strain and temperature fields. `Probe` contains the requested frequencies and times.
`compute_response` puts these together and returns a `Response`.

The models are built with Python objects. PyOFDR will keep reading its configuration file and build these objects. Laser, interferometer and detector models stay in the interrogator package.

## Install

Python 3.10 or newer is required. From this checkout:

```sh
python -m pip install -e .
```

For JAX and the test dependencies:

```sh
python -m pip install -e ".[jax,dev]"
python -m pytest -q
```

Enable JAX x64 before creating arrays or using JIT:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Small example

```python
import numpy as np
from pyrayleigh import Fiber, GuideOptics, Probe, Section, compute_response

nu = 193.4e12
guide = GuideOptics(nu, effective_index=1.4682, group_index=1.4682)
section = Section(0., 1., guide, attenuation=4.6e-5, rayleigh_coefficient=1e-8)
fiber = Fiber("fiber-1", 1., 293.15, nu - 5e9, nu + 5e9, [section])
fiber = fiber.with_rayleigh(seed=1, cell_length=.01)
probe = Probe(np.linspace(nu - 5e9, nu + 5e9, 101))

response = compute_response(fiber, probe)
reflected_power = abs(response.reflection)**2
transmitted_power = abs(response.transmission)**2
```

`reflection` and `transmission` are complex field amplitudes with the same shape as the probe. Their squared magnitudes give power fractions for unit input power.

## Citing

If you use PyRayleigh in published work, please cite it.

```bibtex
@software{pyrayleigh,
  author = {tbd},
  title  = {tbd},
  year   = {2026},
  url    = {https://github.com/ameoli/PyRayleigh},
}
```

## References

- Froggatt and Moore, *High-spatial-resolution distributed strain measurement in optical fiber with Rayleigh scatter*, Appl Opt 1998.
- Soller et al., *High resolution OFDR for characterization of components and assemblies*, Opt Express 2005.
- Hartog, *An Introduction to Distributed Optical Fibre Sensors*, CRC 2017.

## License
MIT -- see `LICENSE`.


## Issues and contributions

Bugs and feature requests: [GitHub issues](https://github.com/ameoli/PyRayleigh/issues).
Pull requests welcome.
