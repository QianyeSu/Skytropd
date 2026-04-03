import numpy as np
import pytest
from scipy.interpolate import interp1d

from skytropd.functions import (
    TropD_Calculate_MaxLat,
    TropD_Calculate_StreamFunction,
    TropD_Calculate_ZeroCrossing,
)
from skytropd.metrics import TropD_Metric_PE, TropD_Metric_PSI


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


def _reference_metric_pe(pe, lat, lat_uncertainty=0.0):
    pe = np.atleast_2d(np.asarray(pe))
    lat = np.asarray(lat)

    if lat[-1] < lat[0]:
        pe = pe[..., ::-1]
        lat = lat[::-1]

    dpedy = np.diff(pe, axis=-1)
    lat_mid = (lat[:-1] + lat[1:]) / 2.0
    pe_grad = interp1d(
        lat_mid,
        dpedy,
        axis=-1,
        bounds_error=False,
        fill_value=(dpedy[..., 0], dpedy[..., -1]),
    )(lat)

    eq_boundary = 5.0
    subpolar_boundary = 50.0
    polar_boundary = 60.0
    mask = (lat > eq_boundary) & (lat < polar_boundary)
    lat_masked = lat[mask]
    subpolar_mask = (lat > eq_boundary) & (lat < subpolar_boundary)
    emax_lat = TropD_Calculate_MaxLat(-pe[..., subpolar_mask], lat[subpolar_mask], n=30)

    lat_after_emax = lat_masked > emax_lat[..., None]
    pe_after_emax = np.where(lat_after_emax, pe[..., mask], np.nan)
    zc1 = TropD_Calculate_ZeroCrossing(
        pe_after_emax, lat_masked, lat_uncertainty=lat_uncertainty
    )

    increases = np.zeros_like(zc1, dtype=bool)
    pe_grad_flat = pe_grad.reshape(-1, lat.size)
    zc1_flat = zc1.reshape(-1)
    for i, pe_grad_i in enumerate(pe_grad_flat):
        grad_interp = interp1d(lat, pe_grad_i, axis=-1)
        increases.reshape(-1)[i] = grad_interp(zc1_flat[i]) > 0

    lat_after_zc = lat_masked > zc1[..., None]
    pe_after_zc = np.where(lat_after_zc, pe[..., mask], np.nan)
    zc2 = TropD_Calculate_ZeroCrossing(
        pe_after_zc, lat_masked, lat_uncertainty=lat_uncertainty
    )

    return np.where(increases, zc1, zc2)


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


def test_metric_pe_matches_reference_for_multidimensional_input():
    lat = np.linspace(0.0, 95.0, 96)
    base = np.tanh((lat - 30.0) / 8.0) - 0.3 * np.exp(-((lat - 20.0) / 6.0) ** 2)
    wave1 = 0.25 * np.sin(np.deg2rad(12.0 * lat))
    wave2 = 0.18 * np.cos(np.deg2rad(17.0 * lat))

    pe = np.stack(
        [
            base,
            base + wave1,
            base - wave1,
            base + wave2,
            -1.0 - 0.01 * lat,
            base + wave1 - wave2,
        ],
        axis=0,
    ).reshape(2, 3, lat.size)

    expected = _reference_metric_pe(pe, lat)
    actual = TropD_Metric_PE.__wrapped__(pe, lat)

    assert np.any(np.isfinite(expected))
    assert np.any(np.isnan(expected))
    assert np.allclose(actual, expected, equal_nan=True)



def test_metric_code_parser_handles_prefixed_names():
    from skytropd.metrics import _metric_code_from_name

    assert _metric_code_from_name("TropD_Metric_TPB") == "TPB"
    assert _metric_code_from_name("TropD_Metric_PSI_precomputed") == "PSI"
