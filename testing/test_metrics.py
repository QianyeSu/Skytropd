import numpy as np
import pytest

from skytropd.functions import TropD_Calculate_StreamFunction
from skytropd.metrics import TropD_Metric_PSI


def _build_symmetric_meridional_wind():
    lats = np.arange(-87.5, 90.0, 5.0)
    levs = np.linspace(1000.0, 100.0, 37)
    lat_shape = np.sin(3.0 * np.pi * np.abs(lats)[:, None] / np.max(np.abs(lats)))
    vertical_shape = np.cos(
        np.pi * (levs[None, :] - levs[-1]) / (levs[0] - levs[-1])
    )
    V = np.sign(lats)[:, None] * 20.0 * lat_shape * vertical_shape
    return V, lats, levs


@pytest.mark.parametrize(
    "method",
    ["Psi_500", "Psi_500_10Perc", "Psi_300_700", "Psi_500_Int", "Psi_Int"],
)
def test_precomputed_psi_matches_v_metric(method):
    V, lats, levs = _build_symmetric_meridional_wind()
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)

    phi_from_v = TropD_Metric_PSI(V, lats, levs, method=method)
    phi_from_psi = TropD_Metric_PSI(Psi, lats, levs, method=method, field_type="PSI")

    assert len(phi_from_v) == len(phi_from_psi)
    for phi_v, phi_psi in zip(phi_from_v, phi_from_psi):
        assert np.allclose(phi_v, phi_psi, equal_nan=True)


def test_precomputed_psi_shape_mismatch():
    V, lats, levs = _build_symmetric_meridional_wind()
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)

    with pytest.raises(ValueError):
        TropD_Metric_PSI(Psi[..., :-1], lats, levs, field_type="PSI")


def test_precomputed_psi_invalid_field_type():
    V, lats, levs = _build_symmetric_meridional_wind()

    with pytest.raises(ValueError):
        TropD_Metric_PSI(V, lats, levs, field_type="bad")

