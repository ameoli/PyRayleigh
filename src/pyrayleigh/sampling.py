"""Nominal spatial sampling estimates for a proposed Rayleigh lattice."""

import math

import array_api_compat as apc
import numpy as np

from .backends import real_arrays
from .constants import SPEED_OF_LIGHT
from .fiber import Fiber, MultiCoreFiber


def rayleigh_resolution_report(fiber, frequency_band, *, cell_length):
    """Compare a proposed uniform cell length [m] with an optical band [Hz].

    This NumPy report uses reference guides, including their GVD. It does not
    evaluate an imposed state or static index profile. Stored scatterers are
    neither inspected nor regenerated. The estimates are not a convergence test.
    """
    if isinstance(fiber, MultiCoreFiber):
        raise NotImplementedError("request a resolution report for one core at a time")
    if not isinstance(fiber, Fiber):
        raise TypeError("fiber must be a Fiber")
    parameters = [frequency_band, cell_length]
    for section in fiber.sections:
        guide = section.optics
        parameters.extend((guide.frequency_ref, guide.effective_index, guide.group_index, guide.gvd))
    xp, values = real_arrays(*parameters)
    if not apc.is_numpy_namespace(xp):
        raise NotImplementedError("rayleigh_resolution_report currently supports NumPy only")
    band, width = values[:2]
    if band.shape != (2,) or width.ndim != 0:
        raise ValueError("frequency_band needs two endpoints and cell_length must be scalar")
    if any(value.ndim != 0 for value in values[2:]):
        raise ValueError("guide parameters must be scalar")
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("resolution report inputs must be finite")
    low, high = float(band[0]), float(band[1])
    width = float(width)
    if not 0. < low < high or low < fiber.frequency_min or high > fiber.frequency_max:
        raise ValueError("frequency_band must increase inside the positive fiber band")
    if width <= 0.:
        raise ValueError("cell_length must be positive")

    bandwidth = high-low
    max_width = min(width, fiber.length)
    sections = []
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        count = max(1, math.ceil(np.float64(fiber.length)/width))
        if count > 1 and (count-1)*width >= fiber.length:
            count -= 1
        for section in fiber.sections:
            guide = section.optics
            if guide.frequency_ref <= 0. or guide.effective_index <= 0. or guide.group_index <= 0.:
                raise ValueError("guide reference frequency and indices must be positive")
            group = SPEED_OF_LIGHT*guide.group_delay_per_length(band, xp=xp)
            if not np.all(np.isfinite(group)) or np.any(group <= 0.):
                raise ValueError("positive finite group delay is required throughout the band")
            group_min, group_max = float(np.min(group)), float(np.max(group))
            resolution_min = SPEED_OF_LIGHT/(2*bandwidth*group_max)
            resolution_max = SPEED_OF_LIGHT/(2*bandwidth*group_min)
            phase_span = 4*np.pi*bandwidth*group_max*max_width/SPEED_OF_LIGHT
            carrier_phase = 2*guide.reference_propagation_constant*max_width
            if not np.all(np.isfinite((resolution_min, resolution_max, phase_span, carrier_phase))):
                raise ValueError("resolution estimates must be finite")
            sections.append({"start_m": float(section.start), "end_m": float(section.end),
                "group_index_min": group_min, "group_index_max": group_max,
                "resolution_min_m": resolution_min, "resolution_max_m": resolution_max,
                "cell_to_resolution": max_width/resolution_min,
                "phase_span_bound_rad": phase_span, "carrier_phase_rad": float(carrier_phase)})

    resolution = min(section["resolution_min_m"] for section in sections)
    notes = ["Nominal reference-guide estimate; compare several spatial grids."]
    if max_width/resolution > .25:
        notes.append("The cell width exceeds one quarter of the smallest nominal resolution.")
    if fiber.index_perturbation is not None:
        notes.append("The static index profile is not included in this estimate.")
    return {"scope": "reference_guides", "frequency_band_hz": (low, high),
        "bandwidth_hz": bandwidth, "cell_length_m": width, "max_cell_length_m": max_width,
        "cell_count": count, "resolution_m": resolution, "cell_to_resolution": max_width/resolution,
        "trial_cell_lengths_m": (resolution/2, resolution/4, resolution/8),
        "sections": sections, "notes": notes}
