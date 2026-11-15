"""A fixed collection of core fibers and optional stored crosstalk."""

from dataclasses import dataclass, field, replace
import math

import numpy as np

from ..backends import real_arrays
from ..scattering import CrosstalkRealization, generate_crosstalk
from .fiber import Fiber


@dataclass(frozen=True, eq=False)
class MultiCoreFiber:
    """Core indices follow the supplied order; the original fibers are kept.

    Cores share length, reference temperature and declared frequency band.
    Their native optical parameters and stored disorder are retained.
    Optional crosstalk starts with an identical Rayleigh grid in every core.
    Further phase columns can serve points and ghosts, also beyond the fiber.
    Its level is a finite native scalar in dB/km. Only dtype
    and shape of the level are checked. This class stores the layout;
    it applies no mixing, propagation or state on construction.
    """

    cores: tuple = field(repr=False)
    crosstalk: CrosstalkRealization | None = None
    xt_db_per_km: object = field(default=None, repr=False)

    def __post_init__(self):
        cores = tuple(self.cores)
        if not cores:
            raise ValueError("a multicore fiber needs at least one core")
        if any(not isinstance(core, Fiber) for core in cores):
            raise TypeError("cores must contain only Fiber objects")
        for name in ("length", "reference_temperature", "frequency_min", "frequency_max"):
            if any(getattr(core, name) != getattr(cores[0], name) for core in cores[1:]):
                raise ValueError(f"cores must share {name}")
        object.__setattr__(self, "cores", cores)
        if (self.crosstalk is None) != (self.xt_db_per_km is None):
            raise ValueError("crosstalk and xt_db_per_km must be supplied together")
        if self.crosstalk is None:
            return
        if not isinstance(self.crosstalk, CrosstalkRealization):
            raise TypeError("crosstalk must be a CrosstalkRealization")
        if self.crosstalk.n_cores != self.n_cores:
            raise ValueError("crosstalk must match the number of cores")
        _, (level,) = real_arrays(self.xt_db_per_km)
        if level.ndim != 0:
            raise ValueError("xt_db_per_km must be a scalar")
        if any(core.rayleigh is None for core in cores):
            raise ValueError("crosstalk requires a stored Rayleigh sample in every core")
        edges = cores[0].rayleigh.edges
        for core in cores:
            sample = core.rayleigh
            if (not np.array_equal(sample.positions, self.crosstalk.positions[:sample.cell_count])
                    or not np.array_equal(sample.edges, edges)):
                raise ValueError("coupled cores must share Rayleigh positions and cell edges")

    @property
    def n_cores(self):
        return len(self.cores)

    @property
    def length(self):
        return self.cores[0].length

    @property
    def reference_temperature(self):
        return self.cores[0].reference_temperature

    @property
    def frequency_min(self):
        return self.cores[0].frequency_min

    @property
    def frequency_max(self):
        return self.cores[0].frequency_max

    def with_crosstalk(self, seed, xt_db_per_km, *, topology="linear", extra_positions=(), max_order=None, xp=None):
        """Draw stored phases, optionally appending fixed point or ghost positions.

        With max_order=1, 2 or 3, include the first core's point paths before
        extra_positions. Coefficients are not inspected. Without max_order,
        only the Rayleigh grid and explicit extra_positions are used.
        """
        sample = self.cores[0].rayleigh
        if sample is None:
            raise ValueError("crosstalk requires stored Rayleigh samples")
        if max_order is not None:
            from ..propagation.paths import point_reflection_paths
            reference = self.cores[0]
            routes = point_reflection_paths(reference, max_order=max_order)
            automatic = [math.fsum(reference.elements[index].position if event % 2 == 0
                                    else -reference.elements[index].position
                                    for event, index in enumerate(route)) for route in routes]
            extra_positions = (*automatic, *extra_positions)
        coupling = generate_crosstalk(self.n_cores, sample.positions, seed, topology=topology,
                                     extra_positions=extra_positions, xp=xp)
        return replace(self, crosstalk=coupling, xt_db_per_km=xt_db_per_km)

    def __repr__(self):
        return (f"MultiCoreFiber(n_cores={self.n_cores}, length={self.length}, "
                f"crosstalk={self.crosstalk is not None})")
