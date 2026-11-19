"""Optional exact/direct comparison for a fixed scalar fiber."""

from dataclasses import dataclass, field

import array_api_compat as apc
import numpy as np

from .propagation.cascade import _TOLERANCE
from .propagation.direct import direct_response_consistent
from .propagation.exact import exact_response
from .response import Response


@dataclass(frozen=True, eq=False)
class DiagnosticResponse:
    """Optical fields, with residual = total - direct in both ports.

    The residual belongs to the same local block model, not to the legacy
    approximation. Power also needs its interference with the direct field.
    """

    total: Response
    direct: Response
    metadata: dict = field(default_factory=dict)
    residual: Response = field(init=False)

    def __post_init__(self):
        if not isinstance(self.total, Response) or not isinstance(self.direct, Response):
            raise TypeError("total and direct must be Response objects")
        if self.total.probe is not self.direct.probe:
            raise ValueError("total and direct must share the same Probe")
        xp = apc.array_namespace(self.total.reflection, self.total.transmission,
            self.direct.reflection, self.direct.transmission, self.total.probe.frequency)
        if not apc.is_numpy_namespace(xp):
            raise NotImplementedError("DiagnosticResponse currently supports NumPy only")
        object.__setattr__(self, "metadata", dict(self.metadata))
        residual = Response(self.total.probe,
            reflection=self.total.reflection-self.direct.reflection,
            transmission=self.total.transmission-self.direct.transmission)
        object.__setattr__(self, "residual", residual)

    @property
    def probe(self):
        return self.total.probe

    @property
    def shape(self):
        return self.total.shape

    @property
    def reflection_interference(self):
        """Signed power term 2*real(direct_r*conj(residual_r))."""
        return 2*np.real(self.direct.reflection*np.conj(self.residual.reflection))

    @property
    def transmission_interference(self):
        """Signed power term 2*real(direct_t*conj(residual_t))."""
        return 2*np.real(self.direct.transmission*np.conj(self.residual.transmission))


def compute_response_diagnostics(fiber, probe, state=None, *, loss_convention,
                                 denominator_tolerance=_TOLERANCE):
    """Compute exact, direct-consistent and residual fields on the same input.

    This runs the two scalar static NumPy calculations separately. Normal
    compute_response calls do not allocate these extra fields. Metadata is a
    short model summary, without copying the realization or inventing a seed.
    """
    total = exact_response(fiber, probe, state, loss_convention=loss_convention,
        denominator_tolerance=denominator_tolerance)
    direct = direct_response_consistent(fiber, probe, state, loss_convention=loss_convention)
    metadata = {"fiber_id": fiber.fiber_id, "loss_convention": loss_convention,
        "model": "scalar_static", "backend": "numpy", "dtype": "complex128",
        "boundary": "matched_right", "denominator_tolerance": float(denominator_tolerance),
        "rayleigh_cells": 0 if fiber.rayleigh is None else fiber.rayleigh.cell_count}
    return DiagnosticResponse(total, direct, metadata)
