"""Linear optical response of optical fibers."""

__version__ = "0.1.0.dev0"

from .fiber import (BendLossTable, Fiber, GuideOptics, MultiCoreFiber, Section, add_profiles,
                    index_segments_profile, ornstein_uhlenbeck_index_from_normals,
                    ornstein_uhlenbeck_index_profile)
from .probe import Probe
from .state import Field, FieldTerm, FiberState, SpatialProfile, TimeHistory, cox_strain
from .propagation import (OpticalPath, multiple_point_response, optical_path, point_reflection_paths,
                           point_response, rayleigh_response, reciprocal_transmission_factors,
                           reflection_paths_from_positions)
from .scattering import (CrosstalkRealization, PointReflector, RayleighRealization, WeakFBG,
                         crosstalk_from_phases, crosstalk_pairs, generate_crosstalk, generate_rayleigh,
                         lossless_reflector, rayleigh_coefficients_from_normals,
                         rayleigh_from_normals, weak_fbg_reflection)
from .response import MultiCoreResponse, Response, compute_response
