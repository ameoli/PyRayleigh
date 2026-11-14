"""Linear optical response of optical fibers."""

__version__ = "0.1.0.dev0"

from .fiber import (Fiber, GuideOptics, Section, ornstein_uhlenbeck_index_from_normals,
                    ornstein_uhlenbeck_index_profile)
from .probe import Probe
from .state import Field, FieldTerm, FiberState, SpatialProfile, TimeHistory
from .propagation import OpticalPath, optical_path, point_response, rayleigh_response
from .scattering import (PointReflector, RayleighRealization, generate_rayleigh,
                         lossless_reflector, rayleigh_from_normals)
from .response import Response, compute_response
