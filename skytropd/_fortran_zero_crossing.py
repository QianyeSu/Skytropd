from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    from . import _zero_crossing_backend as _backend

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on local build environment
    _backend = None
    _IMPORT_ERROR = str(exc)


def fortran_zero_crossing(
    F_flat: np.ndarray, lat: np.ndarray, lat_uncertainty: float
) -> Optional[np.ndarray]:
    """Return zero crossings from the compiled backend when available."""

    if _backend is None:
        return None
    return _backend.zero_crossing(
        np.ascontiguousarray(F_flat, dtype=np.float64),
        np.ascontiguousarray(lat, dtype=np.float64),
        float(lat_uncertainty),
    )


def fortran_zero_crossing_status() -> Tuple[bool, Optional[str]]:
    """Report whether the compiled backend can be imported."""

    return _backend is not None, _IMPORT_ERROR
