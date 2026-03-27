import numpy as np
import pytest

from skytropd._fortran_zero_crossing import (
    fortran_zero_crossing,
    fortran_zero_crossing_status,
)
from skytropd.functions import TropD_Calculate_ZeroCrossing, _zero_crossing_python


def _build_test_field():
    lat = np.arange(-89.5, 90.5, 1.0)
    x = lat[(lat > 0) & (lat < 60)]
    profiles = np.stack(
        [
            np.cos(np.radians(2.0 * x)) - 0.2,
            np.cos(np.radians(2.0 * x)) - 0.35,
            np.where(x < 22.0, 0.6, -0.4),
            np.where((x > 28.0) & (x < 31.0), np.nan, np.cos(np.radians(1.7 * x)) - 0.3),
        ],
        axis=0,
    )
    return profiles, x


@pytest.mark.skipif(
    not fortran_zero_crossing_status()[0],
    reason=fortran_zero_crossing_status()[1] or "Fortran backend unavailable",
)
def test_fortran_backend_matches_python_core():
    field, lat = _build_test_field()

    expected = _zero_crossing_python(field, lat, lat_uncertainty=0.0)
    actual = fortran_zero_crossing(field, lat, lat_uncertainty=0.0)

    assert actual is not None
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.skipif(
    not fortran_zero_crossing_status()[0],
    reason=fortran_zero_crossing_status()[1] or "Fortran backend unavailable",
)
def test_public_zero_crossing_matches_python_core_with_fortran_backend():
    field, lat = _build_test_field()

    expected = _zero_crossing_python(np.atleast_2d(field), lat, lat_uncertainty=5.0)
    actual = TropD_Calculate_ZeroCrossing(field, lat, lat_uncertainty=5.0)

    assert np.allclose(actual, expected, equal_nan=True)
