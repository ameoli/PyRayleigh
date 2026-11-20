"""Native two-port fields and NumPy composition for the scalar reference solver."""

from dataclasses import dataclass, field
from numbers import Real

import array_api_compat as apc
import numpy as np

from ..backends import complex_arrays


_TOLERANCE = 64*np.finfo(np.float64).eps


def _check_tolerance(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0.):
        raise ValueError("tolerance must be a finite nonnegative real number")


class NumericalResonanceError(FloatingPointError):
    """A cascade denominator is below the requested numerical cutoff."""

    def __init__(self, index, denominator, tolerance, *, frequency=None, position=None):
        self.index = index
        self.denominator = complex(denominator)
        self.magnitude = abs(self.denominator)
        self.tolerance = float(tolerance)
        self.frequency = frequency
        self.position = position
        message = (f"star denominator at index {index} has magnitude "
                   f"{self.magnitude:.6g} <= {self.tolerance:.6g}")
        if frequency is not None:
            message += f" at {frequency:.12g} Hz"
        if position is not None:
            message += f", material position {position:.12g} m"
        super().__init__(message)


@dataclass(frozen=True, eq=False)
class TwoPort:
    """Copied native complex128 fields with a common broadcast shape.

    The matrix is [[r_left, t_right_to_left], [t_left_to_right, r_right]].
    NumPy fields must be finite and are read-only. Other namespaces keep
    only dtype/shape checks, allowing traced parameters. Numerical passivity
    checks and star composition currently require NumPy.
    """

    r_left: object = field(repr=False)
    r_right: object = field(repr=False)
    t_left_to_right: object = field(repr=False)
    t_right_to_left: object = field(repr=False)

    def __post_init__(self):
        names = ("r_left", "r_right", "t_left_to_right", "t_right_to_left")
        xp, arrays = complex_arrays(*(getattr(self, name) for name in names))
        is_numpy = apc.is_numpy_namespace(xp)
        arrays = xp.broadcast_arrays(*arrays)
        for name, value in zip(names, arrays):
            if is_numpy and not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            value = xp.asarray(value, dtype=xp.complex128, copy=True)
            if is_numpy:
                value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def shape(self):
        return self.r_left.shape

    def scattering_matrix(self):
        """Matrix axes are the final two axes: shape + (2, 2)."""
        xp, _ = complex_arrays(self.r_left)
        return xp.stack((xp.stack((self.r_left, self.t_right_to_left), axis=-1),
                         xp.stack((self.t_left_to_right, self.r_right), axis=-1)), axis=-2)

    def validate_passivity(self, tolerance=_TOLERANCE):
        """Reject matrices whose largest singular value exceeds 1+tolerance."""
        _check_tolerance(tolerance)
        if not apc.is_numpy_namespace(apc.array_namespace(self.r_left)):
            raise NotImplementedError("validate_passivity currently supports NumPy only")
        largest = np.linalg.svd(self.scattering_matrix(), compute_uv=False)[..., 0]
        invalid = ~np.isfinite(largest) | (largest > 1.+tolerance)
        if np.any(invalid):
            index = tuple(int(i) for i in np.argwhere(invalid)[0])
            raise ValueError(f"non-passive TwoPort at index {index}: sigma_max={largest[index]:.6g}")

    def __repr__(self):
        return f"TwoPort(shape={self.shape})"


def star(left, right, *, denominator_tolerance=_TOLERANCE):
    """Connect left's right port to right's left port.

    Validate physical components before composing them. The tolerance is an
    absolute refusal cutoff, not a regularization or an accuracy guarantee.
    Singular closed cavities also raise; no degenerate-case limit is taken.
    """
    if not isinstance(left, TwoPort) or not isinstance(right, TwoPort):
        raise TypeError("star requires two TwoPort objects")
    _check_tolerance(denominator_tolerance)
    if not apc.is_numpy_namespace(apc.array_namespace(left.r_left, right.r_left)):
        raise NotImplementedError("star currently supports NumPy only")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        denominator = np.asarray(1.-left.r_right*right.r_left)
        invalid = np.abs(denominator) <= denominator_tolerance
        if np.any(invalid):
            index = tuple(int(i) for i in np.argwhere(invalid)[0])
            raise NumericalResonanceError(index, denominator[index], denominator_tolerance)
        return TwoPort(
            r_left=left.r_left + left.t_right_to_left*right.r_left*left.t_left_to_right/denominator,
            r_right=right.r_right + right.t_left_to_right*left.r_right*right.t_right_to_left/denominator,
            t_left_to_right=right.t_left_to_right*left.t_left_to_right/denominator,
            t_right_to_left=left.t_right_to_left*right.t_right_to_left/denominator,
        )
