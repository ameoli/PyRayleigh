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

## Development

| Step | Added | Commit |
| --- | --- | --- |
| 1 | Package layout | [34813f8](https://github.com/ameoli/PyRayleigh/commit/34813f8) |
| 2 | Array API helpers, NumPy and JAX | [8ded2c7](https://github.com/ameoli/PyRayleigh/commit/8ded2c7) |
| 3 | SI unit conversions | [50d2773](https://github.com/ameoli/PyRayleigh/commit/50d2773) |
| 4 | Guide indices and dispersion | [ebe557d](https://github.com/ameoli/PyRayleigh/commit/ebe557d) |
| 5 | Uniform fiber sections | [17c89bc](https://github.com/ameoli/PyRayleigh/commit/17c89bc) |
| 6 | Fiber layout | [b81bc8e](https://github.com/ameoli/PyRayleigh/commit/b81bc8e) |
| 7 | Probe frequencies and times | [e4e5c18](https://github.com/ameoli/PyRayleigh/commit/e4e5c18) |
| 8 | Point reflector coefficients | [b955b91](https://github.com/ameoli/PyRayleigh/commit/b955b91) |
| 9 | Point elements in the fiber | [c33b014](https://github.com/ameoli/PyRayleigh/commit/c33b014) |
| 10 | Stored Rayleigh samples | [7d8b8c2](https://github.com/ameoli/PyRayleigh/commit/7d8b8c2) |
| 11 | Rayleigh generation and fixed normal samples | [2ab2b66](https://github.com/ameoli/PyRayleigh/commit/2ab2b66) |
| 12 | Rayleigh samples attached to the fiber | [22e5983](https://github.com/ameoli/PyRayleigh/commit/22e5983) |
| 13 | Spatial profiles | [2acd66e](https://github.com/ameoli/PyRayleigh/commit/2acd66e) |
| 14 | Time histories | [4632c3c](https://github.com/ameoli/PyRayleigh/commit/4632c3c) |
| 15 | Fields made from profiles and histories | [7089158](https://github.com/ameoli/PyRayleigh/commit/7089158) |
| 16 | Fiber strain and temperature state | [00b9886](https://github.com/ameoli/PyRayleigh/commit/00b9886) |
| 17 | Optical response to strain and temperature | [1997502](https://github.com/ameoli/PyRayleigh/commit/1997502) |
| 18 | Optical phase and group delay | [d82cf1a](https://github.com/ameoli/PyRayleigh/commit/d82cf1a) |
| 19 | Distributed loss and path transmission | [1eb14d4](https://github.com/ameoli/PyRayleigh/commit/1eb14d4) |
| 20 | Direct point reflector returns | [1142b71](https://github.com/ameoli/PyRayleigh/commit/1142b71) |
| 21 | Stored Rayleigh returns | [3567632](https://github.com/ameoli/PyRayleigh/commit/3567632) |
| 22 | Combined reflection and transmission | [11ab859](https://github.com/ameoli/PyRayleigh/commit/11ab859) |
| 23 | Freeze fields and state at one time | [d0e6e67](https://github.com/ameoli/PyRayleigh/commit/d0e6e67) |
| 24 | State evaluated at each probe time | [b272c82](https://github.com/ameoli/PyRayleigh/commit/b272c82) |
| 25 | Static index perturbations | [52ce3a3](https://github.com/ameoli/PyRayleigh/commit/52ce3a3) |
| 26 | OU index fluctuations | [84aebcc](https://github.com/ameoli/PyRayleigh/commit/84aebcc) |
| 27 | Constant index segments and simpler README | [a960b51](https://github.com/ameoli/PyRayleigh/commit/a960b51) |
| 28 | Sum of stored index profiles | This commit |
