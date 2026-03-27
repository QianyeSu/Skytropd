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


def _build_threshold_only_psi():
    lats = np.array([-60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    levs = np.array([300.0, 500.0, 700.0])
    profile_map = {
        0.0: 6.0,
        10.0: 8.0,
        20.0: 10.0,
        30.0: 2.0,
        40.0: 0.5,
        50.0: 0.2,
        60.0: 0.1,
    }
    profile = np.array([profile_map[abs(lat)] for lat in lats], dtype=float)
    psi_lat = np.where(lats < 0.0, -profile, profile)
    psi = np.repeat(psi_lat[:, None], levs.size, axis=1)
    return psi, lats, levs


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


@pytest.mark.parametrize("method", ["Psi_500", "Psi_300_700", "Psi_500_Int", "Psi_Int"])
def test_psi_threshold_fallback_replaces_nan_zero_crossing(method):
    Psi, lats, levs = _build_threshold_only_psi()

    phi_without_fallback = TropD_Metric_PSI(
        Psi, lats, levs, method=method, field_type="PSI", threshold=None
    )
    phi_with_fallback = TropD_Metric_PSI(
        Psi, lats, levs, method=method, field_type="PSI", threshold=0.1
    )

    assert len(phi_without_fallback) == len(phi_with_fallback) == 2
    for phi_nan, phi_fallback in zip(phi_without_fallback, phi_with_fallback):
        assert np.all(np.isnan(phi_nan))
        assert np.all(np.isfinite(phi_fallback))


def test_psi_500_threshold_fallback_matches_explicit_threshold_metric():
    Psi, lats, levs = _build_threshold_only_psi()

    phi_with_fallback = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500", field_type="PSI", threshold=0.1
    )
    phi_threshold = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_10Perc", field_type="PSI", threshold=0.1
    )

    for phi_fallback, phi_explicit in zip(phi_with_fallback, phi_threshold):
        assert np.allclose(phi_fallback, phi_explicit, equal_nan=True)


def test_psi_threshold_rejects_negative_values():
    Psi, lats, levs = _build_threshold_only_psi()

    with pytest.raises(ValueError):
        TropD_Metric_PSI(Psi, lats, levs, field_type="PSI", threshold=-0.1)



def test_metric_code_parser_handles_prefixed_names():
    from skytropd.metrics import _metric_code_from_name

    assert _metric_code_from_name("TropD_Metric_TPB") == "TPB"
    assert _metric_code_from_name("TropD_Metric_PSI_precomputed") == "PSI"
