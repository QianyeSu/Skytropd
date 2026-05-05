from __future__ import annotations

import numpy as np

from . import _zero_crossing_backend as _backend


def fortran_zero_crossing(
    F_flat: np.ndarray, lat: np.ndarray, lat_uncertainty: float
) -> np.ndarray:
    """Return zero crossings from the compiled backend."""

    return _backend.zero_crossing(
        np.ascontiguousarray(F_flat, dtype=np.float64),
        np.ascontiguousarray(lat, dtype=np.float64),
        float(lat_uncertainty),
    )


def fortran_descending_threshold_crossing(
    profile_flat: np.ndarray,
    lat: np.ndarray,
    start_idx: np.ndarray,
    threshold: np.ndarray,
) -> np.ndarray:
    """Return descending threshold crossings from the compiled backend."""

    return _backend.descending_threshold_crossing(
        np.ascontiguousarray(profile_flat, dtype=np.float64),
        np.ascontiguousarray(lat, dtype=np.float64),
        np.ascontiguousarray(start_idx, dtype=np.int32),
        np.ascontiguousarray(threshold, dtype=np.float64),
    )
