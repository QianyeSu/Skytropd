from functools import lru_cache
from pathlib import Path
import re
import numpy as np
import pytest
from scipy.interpolate import interp1d
import xarray as xr

import skytropd.metrics as metrics_mod
from skytropd.functions import (
    KAPPA,
    find_nearest,
    TropD_Calculate_MaxLat,
    TropD_Calculate_StreamFunction,
    TropD_Calculate_TropopauseHeight,
    TropD_Calculate_ZeroCrossing,
)
from skytropd.metrics import (
    TropD_Metric_EDJ,
    TropD_Metric_PE,
    TropD_Metric_PSI,
    TropD_Metric_STJ,
    TropD_Metric_TPB,
)

_LAYER_PERCENT_METHOD = re.compile(
    r"^Psi_\d+(?:\.\d+)?_\d+(?:\.\d+)?_\d+(?:\.\d+)?Perc(?:_(?:center2d|profile))?$",
    flags=re.IGNORECASE,
)


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


def _reference_quadratic_peak_fit(field, lat_mask, n_fit=1):
    field = np.asarray(field)
    lat_mask = np.asarray(lat_mask)
    phi = np.zeros(field.shape[:-1], dtype=float)
    umax = np.zeros(field.shape[:-1], dtype=float)
    flat = field.reshape(-1, field.shape[-1])

    for i, profile in enumerate(flat):
        peak_idx = np.nanargmax(profile)
        out_idx = np.unravel_index(i, phi.shape)

        if peak_idx == 0 or peak_idx == profile.size - 1:
            phi[out_idx] = lat_mask[peak_idx]
            continue

        if n_fit > peak_idx or n_fit > profile.size - peak_idx + 1:
            fit_half_width = min(peak_idx, profile.size - peak_idx + 1)
        else:
            fit_half_width = n_fit

        coeffs = np.polynomial.polynomial.polyfit(
            lat_mask[peak_idx - fit_half_width : peak_idx + fit_half_width + 1],
            profile[peak_idx - fit_half_width : peak_idx + fit_half_width + 1],
            deg=2,
        )
        phi[out_idx] = -coeffs[1] / (2.0 * coeffs[2])
        umax[out_idx] = (
            4.0 * coeffs[2] * coeffs[0] - coeffs[1] * coeffs[1]
        ) / 4.0 / coeffs[2]

    return phi, umax


def _reference_metric_tpb_single_hemisphere(
    T, lat, lev, method="max_gradient", Z=None, Cutoff=1.5e4, **maxlat_kwargs
):
    T = np.asarray(T)
    lat = np.asarray(lat)
    lev = np.asarray(lev)
    if Z is not None:
        Z = np.asarray(Z)

    if T.shape[-2:] != (lat.size, lev.size):
        raise ValueError
    if method not in ["max_gradient", "max_potemp", "cutoff"]:
        raise ValueError

    if lat[-1] < lat[0]:
        lat = lat[::-1]
        T = T[..., ::-1, :]
        if Z is not None:
            Z = Z[..., ::-1, :]

    polar_boundary = 60.0
    eq_boundary = 0.0
    mask = (lat > eq_boundary) & (lat < polar_boundary)

    if "max_" in method:
        if method == "max_potemp":
            maxlat_kwargs["n"] = maxlat_kwargs.get("n", 30)
            PT = T / (lev / 1000.0) ** KAPPA
            Pt, PTt = TropD_Calculate_TropopauseHeight(T, lev, Z=PT)
            F = PTt - np.nanmin(PT, axis=-1)
        else:
            Pt = TropD_Calculate_TropopauseHeight(T, lev, Z=None)
            F = np.diff(Pt, axis=-1) / (lat[1] - lat[0])
            lat = (lat[1:] + lat[:-1]) / 2.0
            F *= np.sign(lat)
            mask = (lat > eq_boundary) & (lat < polar_boundary)
        F = np.where(np.isfinite(F), F, 0.0)
        return TropD_Calculate_MaxLat(F[..., mask], lat[mask], **maxlat_kwargs)

    if Z is None:
        raise ValueError
    Pt, Ht = TropD_Calculate_TropopauseHeight(T, lev, Z)
    return TropD_Calculate_ZeroCrossing(Ht[..., mask] - Cutoff, lat[mask])


def _reference_metric_tpb_oldstyle(
    T, lat, lev, method="max_gradient", Z=None, Cutoff=1.5e4, **maxlat_kwargs
):
    T = np.asarray(T)
    lat = np.asarray(lat)
    if Z is not None:
        Z = np.asarray(Z)

    phi_list = []
    if (lat < -20.0).any():
        shmask = lat < 0.5
        phi_sh = _reference_metric_tpb_single_hemisphere(
            T[..., shmask, :],
            -lat[shmask],
            lev,
            method=method,
            Z=None if Z is None else Z[..., shmask, :],
            Cutoff=Cutoff,
            **dict(maxlat_kwargs),
        )
        phi_list.append(-phi_sh)
    if (lat > 20.0).any():
        nhmask = lat > -0.5
        phi_nh = _reference_metric_tpb_single_hemisphere(
            T[..., nhmask, :],
            lat[nhmask],
            lev,
            method=method,
            Z=None if Z is None else Z[..., nhmask, :],
            Cutoff=Cutoff,
            **dict(maxlat_kwargs),
        )
        phi_list.append(phi_nh)

    return tuple(phi_list)


@lru_cache(maxsize=1)
def _load_validation_tpb_sample():
    data_dir = Path(__file__).resolve().parents[1] / "ValidationData"
    with xr.open_dataset(data_dir / "ta.nc") as tds, xr.open_dataset(
        data_dir / "zg.nc"
    ) as zds:
        temperature = (
            tds["ta"]
            .isel(time=slice(0, 3))
            .transpose("time", "lat", "lev")
            .load()
            .values
        )
        geopotential = (
            zds["zg"]
            .isel(time=slice(0, 3))
            .transpose("time", "lat", "lev")
            .load()
            .values
        )
        lat = tds["lat"].values
        lev = tds["lev"].values

    return temperature, geopotential, lat, lev


def _build_fit_test_wind():
    lats = np.arange(0.0, 91.0, 2.5)
    levs = np.array([1000.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0])
    time = np.arange(3)
    member = np.arange(2)
    U = np.empty((time.size, member.size, lats.size, levs.size), dtype=float)

    for it, t in enumerate(time):
        for im, m in enumerate(member):
            edj_center = 42.0 + 2.0 * m - 1.0 * t
            stj_center = 28.0 + 1.5 * t + 1.0 * m
            low_jet = 26.0 * np.exp(-((lats - edj_center) / 7.0) ** 2)
            upper_jet = 36.0 * np.exp(-((lats - stj_center) / 6.0) ** 2)
            background = 4.0 + 0.1 * lats

            for k, lev in enumerate(levs):
                if lev >= 700.0:
                    U[it, im, :, k] = background + low_jet + 0.2 * upper_jet
                else:
                    U[it, im, :, k] = background + upper_jet + 0.15 * low_jet

    return U, lats, levs


def _reference_psi_layer_percent_center2d(
    Psi, lat, lev, top_hpa=500.0, bottom_hpa=800.0, percent=5.0
):
    Psi = np.asarray(Psi, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lev = np.asarray(lev, dtype=float)

    layer_mask = (lev >= min(top_hpa, bottom_hpa)) & (lev <= max(top_hpa, bottom_hpa))
    if not np.any(layer_mask):
        raise ValueError("requested layer does not intersect available pressure levels")

    profile = Psi[..., layer_mask].mean(axis=-1)
    nh_mask = lat > 0.0
    if not np.any(nh_mask):
        raise ValueError("No NH latitudes found.")

    out = np.full(Psi.shape[:-2], np.nan, dtype=float)
    psi_flat = Psi.reshape(-1, lat.size, lev.size)
    out_flat = out.reshape(-1)
    nh_lat_indices = np.where(nh_mask)[0]
    profile_flat = profile.reshape(-1, lat.size)

    for i in range(psi_flat.shape[0]):
        nh_field = psi_flat[i, nh_mask, :]
        if not np.isfinite(nh_field).any():
            continue
        nh_flat_index = int(np.nanargmax(nh_field))
        center_lat_subindex, center_level_index = np.unravel_index(
            nh_flat_index, nh_field.shape
        )
        center_lat_index = nh_lat_indices[center_lat_subindex]
        threshold = (float(percent) / 100.0) * float(
            nh_field[center_lat_subindex, center_level_index]
        )

        lat_search = lat[center_lat_index:]
        profile_search = profile_flat[i, center_lat_index:]
        for j in range(lat_search.size - 1):
            val0 = float(profile_search[j])
            val1 = float(profile_search[j + 1])
            if not (np.isfinite(val0) and np.isfinite(val1)):
                continue
            if val0 >= threshold and val1 <= threshold:
                if np.isclose(val0, val1):
                    out_flat[i] = 0.5 * (lat_search[j] + lat_search[j + 1])
                else:
                    out_flat[i] = lat_search[j] + (
                        (threshold - val0) * (lat_search[j + 1] - lat_search[j]) / (val1 - val0)
                    )
                break

    return out


@pytest.mark.parametrize(
    "method",
    [
        "Psi_500",
        "Psi_500_10Perc",
        "Psi_300_700",
        "Psi_500_800",
        "Psi_500_800_5Perc",
        "Psi_500_800_5Perc_center2d",
        "Psi_500_800_5Perc_profile",
        "Psi_500_800_10Perc",
        "Psi_500_800_10Perc_center2d",
        "Psi_500_800_10Perc_profile",
        "Psi_500_Int",
        "Psi_Int",
    ],
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


def test_psi_custom_layer_method_matches_manual_layer_average():
    V, lats, levs = _build_symmetric_meridional_wind()
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)
    layer_mask = (levs >= 500.0) & (levs <= 800.0)
    actual = TropD_Metric_PSI(Psi, lats, levs, method="Psi_500_800", field_type="PSI")
    reverse = TropD_Metric_PSI(Psi, lats, levs, method="Psi_800_500", field_type="PSI")
    manual = TropD_Metric_PSI(
        Psi[..., layer_mask],
        lats,
        levs[layer_mask],
        method="Psi_Int",
        field_type="PSI",
    )

    assert len(actual) == len(reverse) == len(manual) == 2
    for phi_actual, phi_reverse in zip(actual, reverse):
        assert np.allclose(phi_actual, phi_reverse, equal_nan=True)
    for phi_actual, phi_manual in zip(actual, manual):
        assert np.allclose(phi_actual, phi_manual, equal_nan=True)


def test_psi_300_700_alias_preserves_existing_behavior():
    V, lats, levs = _build_symmetric_meridional_wind()
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)

    expected = TropD_Metric_PSI(Psi, lats, levs, method="Psi_300_700", field_type="PSI")
    actual = TropD_Metric_PSI(Psi, lats, levs, method="Psi_700_300", field_type="PSI")

    assert len(actual) == len(expected) == 2
    for actual_phi, expected_phi in zip(actual, expected):
        assert np.allclose(actual_phi, expected_phi, equal_nan=True)


def test_psi_5perc_default_matches_center2d_variant():
    V, lats, levs = _build_symmetric_meridional_wind()
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)

    default_phi = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_800_5Perc", field_type="PSI"
    )
    explicit_phi = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_800_5Perc_center2d", field_type="PSI"
    )

    assert len(default_phi) == len(explicit_phi) == 2
    for phi_default, phi_explicit in zip(default_phi, explicit_phi):
        assert np.allclose(phi_default, phi_explicit, equal_nan=True)


def test_psi_5perc_center2d_matches_reference_script_logic():
    V, lats, levs = _build_symmetric_meridional_wind()
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)

    expected_nh = _reference_psi_layer_percent_center2d(
        Psi, lats, levs, 500.0, 800.0, percent=5.0
    )
    actual = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_800_5Perc_center2d", field_type="PSI"
    )

    assert len(actual) == 2
    assert np.allclose(actual[1], expected_nh, equal_nan=True)


def test_psi_10perc_default_matches_center2d_variant():
    V, lats, levs = _build_symmetric_meridional_wind()
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)

    default_phi = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_800_10Perc", field_type="PSI"
    )
    explicit_phi = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_800_10Perc_center2d", field_type="PSI"
    )

    assert len(default_phi) == len(explicit_phi) == 2
    for phi_default, phi_explicit in zip(default_phi, explicit_phi):
        assert np.allclose(phi_default, phi_explicit, equal_nan=True)


def test_psi_10perc_center2d_matches_generalized_reference_script_logic():
    V, lats, levs = _build_symmetric_meridional_wind()
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)

    expected_nh = _reference_psi_layer_percent_center2d(
        Psi, lats, levs, 500.0, 800.0, percent=10.0
    )
    actual = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_800_10Perc_center2d", field_type="PSI"
    )

    assert len(actual) == 2
    assert np.allclose(actual[1], expected_nh, equal_nan=True)


def test_psi_layer_percent_falls_back_to_layer_zero_crossing_poleward_of_50deg():
    sample_path = Path(
        r"M:\CESM2LE\model\model_1011_001\model_1011_001_Hadley_Circulation"
        r"\model_1011_001_Hadley_Remove_BarotropicV_1850_2100.nc"
    )
    if not sample_path.exists():
        pytest.skip(f"sample dataset not available: {sample_path}")

    with xr.open_dataset(sample_path) as ds:
        psi = ds["Hadley_MSF"].transpose("time", "lat", "level").load()

    lats = psi["lat"].values
    levs = psi["level"].values
    phi_layer = TropD_Metric_PSI(
        psi.values, lats, levs, method="Psi_500_800", field_type="PSI"
    )[1]
    phi_percent = TropD_Metric_PSI(
        psi.values, lats, levs, method="Psi_500_800_5Perc", field_type="PSI"
    )[1]

    assert np.sum(phi_percent > 60.0) == 0
    assert np.nanmax(phi_percent) <= 60.0
    assert np.any(np.isfinite(phi_layer))
    assert np.any(np.isclose(phi_percent, phi_layer, equal_nan=False))


def test_psi_5perc_uses_compiled_descending_threshold_backend_when_available(monkeypatch):
    V, lats, levs = _build_symmetric_meridional_wind()
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)

    calls = {"count": 0}
    original = metrics_mod.fortran_descending_threshold_crossing

    def wrapped(profile_flat, lat, start_idx, threshold):
        calls["count"] += 1
        return original(profile_flat, lat, start_idx, threshold)

    monkeypatch.setattr(metrics_mod, "fortran_descending_threshold_crossing", wrapped)
    result = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_800_5Perc_center2d", field_type="PSI"
    )

    assert calls["count"] == 2
    assert len(result) == 2


def test_streamfunction_accepts_trailing_lev_lat_order():
    V, lats, levs = _build_symmetric_meridional_wind()
    V = np.stack([V, 1.1 * V], axis=0)

    expected = TropD_Calculate_StreamFunction(V, lats, levs)
    actual = TropD_Calculate_StreamFunction(np.swapaxes(V, -2, -1), lats, levs)

    assert actual.shape == np.swapaxes(expected, -2, -1).shape
    assert np.allclose(actual, np.swapaxes(expected, -2, -1), equal_nan=True)


def test_psi_layer_percent_accepts_trailing_lev_lat_order_with_leading_dims():
    V, lats, levs = _build_symmetric_meridional_wind()
    V = np.stack([V, 1.05 * V, 0.95 * V], axis=0)
    V = np.stack([V, 1.1 * V], axis=0)
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)

    expected_from_v = TropD_Metric_PSI(V, lats, levs, method="Psi_500_800_10Perc")
    expected_from_psi = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_800_10Perc", field_type="PSI"
    )

    V_lev_lat = np.swapaxes(V, -2, -1)
    Psi_lev_lat = np.swapaxes(Psi, -2, -1)
    actual_from_v = TropD_Metric_PSI(
        V_lev_lat, lats, levs, method="Psi_500_800_10Perc"
    )
    actual_from_psi = TropD_Metric_PSI(
        Psi_lev_lat, lats, levs, method="Psi_500_800_10Perc", field_type="PSI"
    )

    for actual, expected in zip(actual_from_v, expected_from_v):
        assert np.allclose(actual, expected, equal_nan=True)
    for actual, expected in zip(actual_from_psi, expected_from_psi):
        assert np.allclose(actual, expected, equal_nan=True)


def test_psi_5perc_profile_variant_is_distinct_method():
    V, lats, levs = _build_symmetric_meridional_wind()
    Psi = TropD_Calculate_StreamFunction(V, lats, levs)

    center2d = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_800_5Perc_center2d", field_type="PSI"
    )
    profile = TropD_Metric_PSI(
        Psi, lats, levs, method="Psi_500_800_5Perc_profile", field_type="PSI"
    )

    assert len(center2d) == len(profile) == 2
    for phi_center2d, phi_profile in zip(center2d, profile):
        assert np.ndim(phi_center2d) == np.ndim(phi_profile)


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


def test_metric_edj_fit_matches_reference():
    U, lats, levs = _build_fit_test_wind()
    mask = (lats > 15.0) & (lats < 70.0)
    u850 = U[..., find_nearest(levs, 850.0)]

    expected_phi, expected_umax = _reference_quadratic_peak_fit(
        u850[..., mask], lats[mask], n_fit=1
    )
    actual_phi, actual_umax = TropD_Metric_EDJ.__wrapped__(
        U, lats, levs, method="fit", n_fit=1
    )

    assert np.allclose(actual_phi, expected_phi, equal_nan=True)
    assert np.allclose(actual_umax, expected_umax, equal_nan=True)


def test_metric_stj_fit_matches_reference():
    U, lats, levs = _build_fit_test_wind()
    layer_400_to_100 = (levs >= 100.0) & (levs <= 400.0)
    lev_int = levs[layer_400_to_100]
    u_int = np.trapezoid(U[..., layer_400_to_100], lev_int, axis=-1) / (
        lev_int[-1] - lev_int[0]
    )
    adjusted_u = u_int - U[..., find_nearest(levs, 850.0)]
    mask = (lats > 10.0) & (lats < 60.0)

    expected_phi, expected_umax = _reference_quadratic_peak_fit(
        adjusted_u[..., mask], lats[mask], n_fit=1
    )
    actual_phi, actual_umax = TropD_Metric_STJ.__wrapped__(
        U, lats, levs, method="fit", n_fit=1
    )

    assert np.allclose(actual_phi, expected_phi, equal_nan=True)
    assert np.allclose(actual_umax, expected_umax, equal_nan=True)


@pytest.mark.parametrize("method", ["max_gradient", "max_potemp", "cutoff"])
def test_metric_tpb_matches_oldstyle_reference(method):
    temperature, geopotential, lat, lev = _load_validation_tpb_sample()
    kwargs = {"method": method}
    if method == "cutoff":
        kwargs["Z"] = geopotential

    expected = _reference_metric_tpb_oldstyle(temperature, lat, lev, **kwargs)
    actual = TropD_Metric_TPB(temperature, lat, lev, **kwargs)

    assert len(actual) == len(expected) == 2
    for actual_phi, expected_phi in zip(actual, expected):
        assert np.allclose(actual_phi, expected_phi, equal_nan=True)


def test_metric_tpb_single_hemisphere_return_matches_oldstyle_reference():
    temperature, geopotential, lat, lev = _load_validation_tpb_sample()
    nhmask = lat > -0.5

    expected = _reference_metric_tpb_oldstyle(
        temperature[..., nhmask, :],
        lat[nhmask],
        lev,
        method="cutoff",
        Z=geopotential[..., nhmask, :],
    )
    actual = TropD_Metric_TPB(
        temperature[..., nhmask, :],
        lat[nhmask],
        lev,
        method="cutoff",
        Z=geopotential[..., nhmask, :],
    )

    assert len(actual) == len(expected) == 1
    assert np.allclose(actual[0], expected[0], equal_nan=True)



def test_metric_code_parser_handles_prefixed_names():
    from skytropd.metrics import _metric_code_from_name

    assert _metric_code_from_name("TropD_Metric_TPB") == "TPB"
    assert _metric_code_from_name("TropD_Metric_PSI_precomputed") == "PSI"
