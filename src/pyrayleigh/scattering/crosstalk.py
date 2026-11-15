"""Phase-scrambled first-order coupling between core profiles."""

from dataclasses import dataclass, field
from numbers import Integral
import operator

import numpy as np

from ..backends import complex_arrays, real_arrays
from .rayleigh import RayleighRealization, _fixed_positions


def crosstalk_pairs(n_cores, topology="linear"):
    """Fixed (source, target) pairs, in PyOFDR's target-major phase order."""
    if isinstance(n_cores, bool) or not isinstance(n_cores, Integral):
        raise TypeError("n_cores must be a fixed integer")
    if n_cores < 1:
        raise ValueError("n_cores must be positive")
    if not isinstance(topology, str):
        raise TypeError("topology must be a string")
    if topology == "linear":
        return tuple((source, target) for target in range(n_cores)
                     for source in (target - 1, target + 1) if 0 <= source < n_cores)
    if topology == "hex7":
        if n_cores != 7:
            raise ValueError("hex7 requires seven cores")
        neighbors = ((1, 2, 3, 4, 5, 6), (0, 2, 6), (0, 1, 3), (0, 2, 4),
                     (0, 3, 5), (0, 4, 6), (0, 5, 1))
        return tuple((source, target) for target, sources in enumerate(neighbors) for source in sources)
    raise ValueError("topology must be linear or hex7")


def crosstalk_from_phases(profile, positions, xt_db_per_km, phases, *, topology="linear", xp=None):
    """Mix a (n_cores, n_positions) complex profile using fixed phase draws.

    positions [m] are finite and nonnegative, one per profile column.
    xt_db_per_km is a finite scalar: 10**(xt_db_per_km/10) is the coupled
    power per km and per directed pair. phases [rad] has shape
    (len(crosstalk_pairs(n_cores, topology)), n_positions), with finite
    real values. Uniform draws in [0, 2*pi) reproduce PyOFDR's model.
    Only dtype and shape are checked, keeping numerical data native.

    Each source adds sqrt(10**(xt_db_per_km/10)*z/1000)*exp(i*phase)
    times its original field to the target. The source field is not depleted;
    weak coupling is assumed. The input is kept and a new array is returned.
    Reuse the same phases across evaluations. Position derivatives require
    z > 0; the amplitude is not differentiable in z at the input face.
    """
    xp, _ = complex_arrays(profile, positions, xt_db_per_km, phases, xp=xp)
    _, (profile,) = complex_arrays(profile, xp=xp)
    _, (positions, level, phases) = real_arrays(positions, xt_db_per_km, phases, xp=xp)
    if profile.ndim != 2:
        raise ValueError("profile must have shape (n_cores, n_positions)")
    n_cores, count = profile.shape
    pairs = crosstalk_pairs(n_cores, topology)
    if positions.shape != (count,):
        raise ValueError("positions must have one entry per profile column")
    if level.ndim != 0:
        raise ValueError("xt_db_per_km must be a scalar")
    if phases.shape != (len(pairs), count):
        raise ValueError("phases must have shape (n_pairs, n_positions)")
    amplitude = 10.**(level/20.)*xp.sqrt(positions/1000.)
    i = xp.asarray(1j, dtype=xp.complex128)
    coupling = xp.astype(amplitude, xp.complex128)[None, :]*xp.exp(i*xp.astype(phases, xp.complex128))
    rows = [profile[core, :] for core in range(n_cores)]
    for index, (source, target) in enumerate(pairs):
        rows[target] = rows[target] + profile[source, :]*coupling[index, :]
    return xp.stack(rows, axis=0)


@dataclass(frozen=True, eq=False)
class CrosstalkRealization:
    """Stored phases [rad] for a fixed core topology and position grid [m].

    positions are fixed nonnegative Python or NumPy values, copied as a
    read-only array. Native phases have shape (n_pairs, n_positions), in
    crosstalk_pairs order. They are copied as float64; finite real values
    are required, with only dtype and shape checked. Mixing reuses them.
    """

    n_cores: int
    positions: object = field(repr=False)
    phases: object = field(repr=False)
    topology: str = "linear"

    def __post_init__(self):
        pairs = crosstalk_pairs(self.n_cores, self.topology)
        positions = _fixed_positions(self.positions, "positions")
        if np.any(positions < 0.):
            raise ValueError("positions must be nonnegative")
        xp, (phases,) = real_arrays(self.phases)
        if phases.shape != (len(pairs), positions.shape[0]):
            raise ValueError("phases must have shape (n_pairs, n_positions)")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "phases", xp.asarray(phases, copy=True))

    @property
    def pairs(self):
        return crosstalk_pairs(self.n_cores, self.topology)

    @property
    def position_count(self):
        return self.positions.shape[0]

    def mix(self, profile, xt_db_per_km, *, xp=None):
        """Apply this sample to matching core profiles at any coupling level."""
        xp, _ = complex_arrays(profile, xt_db_per_km, self.phases, xp=xp)
        _, (profile,) = complex_arrays(profile, xp=xp)
        if profile.shape != (self.n_cores, self.position_count):
            raise ValueError("profile must match the stored cores and positions")
        positions = xp.asarray(self.positions.tolist(), dtype=xp.float64)
        return crosstalk_from_phases(profile, positions, xt_db_per_km, self.phases,
                                     topology=self.topology, xp=xp)

    def mix_rayleigh(self, samples, xt_db_per_km, *, xp=None):
        """Return new Rayleigh samples after mixing their local coefficients.

        There is one sample per core, on identical positions and cell edges.
        Positions must match the first columns of this crosstalk sample;
        extra columns are kept for point and multiple returns. This operation
        mixes the supplied reference coefficients; it evaluates no propagation or state.
        With a common core path, the results can use the single-core solver.
        Core-dependent propagation and state are outside this operation.
        Stored source samples and crosstalk phases are retained.
        """
        samples = tuple(samples)
        if len(samples) != self.n_cores:
            raise ValueError("samples must contain one Rayleigh realization per core")
        if any(not isinstance(sample, RayleighRealization) for sample in samples):
            raise TypeError("samples must contain only RayleighRealization objects")
        reference = samples[0]
        count = reference.cell_count
        if not np.array_equal(reference.positions, self.positions[:count]):
            raise ValueError("crosstalk positions must start with the Rayleigh positions")
        for sample in samples:
            if (not np.array_equal(sample.positions, reference.positions)
                    or not np.array_equal(sample.edges, reference.edges)):
                raise ValueError("Rayleigh samples must share positions and cell edges")
        xp, arrays = complex_arrays(*(sample.coefficients for sample in samples),
                                    self.phases, xt_db_per_km, xp=xp)
        profile = xp.stack(arrays[:self.n_cores], axis=0)
        _, (phases,) = real_arrays(self.phases, xp=xp)
        positions = xp.asarray(reference.positions.tolist(), dtype=xp.float64)
        mixed = crosstalk_from_phases(profile, positions, xt_db_per_km, phases[:, :count],
                                      topology=self.topology, xp=xp)
        return tuple(RayleighRealization(mixed[core, :], sample.positions, sample.edges)
                     for core, sample in enumerate(samples))

    def __repr__(self):
        return (f"CrosstalkRealization(n_cores={self.n_cores}, "
                f"n_positions={self.position_count}, topology={self.topology!r})")


def generate_crosstalk(n_cores, positions, seed, *, topology="linear", extra_positions=(), xp=None):
    """Draw U(0, 2*pi) phases once using a private NumPy PCG64 stream.

    Base draws follow PyOFDR's directed-pair order and are identical in every
    backend. seed is the stream seed; PyOFDR keeps its component seed
    derivation. The returned sample is independent of the coupling level.
    Extra fixed positions are appended, ignoring repeats and positions already
    in the base grid, allowing round-off. The base columns keep their original seeded phases.
    Extra draws are grouped by position, keeping earlier columns when more
    positions are appended with the same seed.
    """
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a nonnegative integer")
    try:
        seed = operator.index(seed)
    except TypeError:
        raise ValueError("seed must be a nonnegative integer") from None
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    pairs = crosstalk_pairs(n_cores, topology)
    positions = _fixed_positions(positions, "positions")
    extra = _fixed_positions(extra_positions, "extra_positions")
    if np.any(extra < 0.):
        raise ValueError("extra_positions must be nonnegative")
    additional = []
    for position in extra:
        known = np.concatenate((positions, additional))
        if not np.any(np.isclose(known, position, rtol=8.*np.finfo(float).eps, atol=0.)):
            additional.append(position)
    rng = np.random.Generator(np.random.PCG64(seed))
    phases = rng.uniform(0., 2.*np.pi, (len(pairs), len(positions)))
    extra_phases = rng.uniform(0., 2.*np.pi, (len(additional), len(pairs))).T
    positions = np.concatenate((positions, additional))
    phases = np.concatenate((phases, extra_phases), axis=1)
    _, (phases,) = real_arrays(phases, xp=xp)
    return CrosstalkRealization(n_cores, positions, phases, topology)
