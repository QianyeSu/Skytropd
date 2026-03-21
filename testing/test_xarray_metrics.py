from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

import skytropd as pyt


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
def test_xr_psi_precomputed_matches_wind_input(method):
    V, lats, levs = _build_symmetric_meridional_wind()
    V = np.stack([V, 1.1 * V], axis=0)
    times = np.arange(V.shape[0])
    ds_v = xr.Dataset(
        {"v": (("time", "lat", "lev"), V)},
        coords={"time": times, "lat": lats, "lev": levs},
    )
    psi = pyt.TropD_Calculate_StreamFunction(V, lats, levs)
    ds_psi = xr.Dataset(
        {"psi": (("time", "lat", "lev"), psi)},
        coords={"time": times, "lat": lats, "lev": levs},
    )

    phi_from_v = ds_v.pyt_metrics.xr_psi(method=method)
    phi_from_psi = ds_psi.pyt_metrics.xr_psi(method=method, field_type="PSI")

    assert np.allclose(phi_from_v.values, phi_from_psi.values, equal_nan=True)
    assert list(phi_from_psi.coords["hemsph"].values) == ["SH", "NH"]


def test_xr_edj_uses_pressure_axis_when_present():
    lats = np.arange(-87.5, 90.0, 5.0)
    levs = np.array([500.0, 850.0, 1000.0])
    profile0 = np.exp(-((lats + 35.0) / 10.0) ** 2) + 1.2 * np.exp(
        -((lats - 45.0) / 8.0) ** 2
    )
    profile1 = np.exp(-((lats + 40.0) / 10.0) ** 2) + 1.1 * np.exp(
        -((lats - 50.0) / 8.0) ** 2
    )
    u850 = np.stack([profile0, profile1], axis=0)
    U = np.stack([0.6 * u850, u850, 0.8 * u850], axis=-1)
    ds = xr.Dataset(
        {"u": (("time", "lat", "lev"), U)},
        coords={"time": np.arange(U.shape[0]), "lat": lats, "lev": levs},
    )

    phi_from_ds = ds.pyt_metrics.xr_edj(method="max").squeeze("method")
    phi_expected = pyt.TropD_Metric_EDJ(U, lats, levs, method="max")

    assert np.allclose(phi_from_ds.sel(hemsph="SH").values, phi_expected[0])
    assert np.allclose(phi_from_ds.sel(hemsph="NH").values, phi_expected[1])


def test_xr_tpb_cutoff_matches_numpy_metric():
    data_dir = Path(__file__).resolve().parents[1] / "ValidationData"
    with xr.open_dataset(data_dir / "ta.nc") as tds, xr.open_dataset(
        data_dir / "zg.nc"
    ) as zds:
        temperature = (
            tds["ta"]
            .isel(time=slice(0, 3))
            .transpose("time", "lat", "lev")
            .rename("T")
            .load()
        )
        geopotential = (
            zds["zg"]
            .isel(time=slice(0, 3))
            .transpose("time", "lat", "lev")
            .rename("Z")
            .load()
        )

    ds = xr.merge([temperature.to_dataset(), geopotential.to_dataset()])
    phi_from_ds = ds.pyt_metrics.xr_tpb(method="cutoff").squeeze("method")
    phi_expected = pyt.TropD_Metric_TPB(
        temperature.values,
        temperature["lat"].values,
        temperature["lev"].values,
        method="cutoff",
        Z=geopotential.values,
    )

    assert np.allclose(phi_from_ds.sel(hemsph="SH").values, phi_expected[0])
    assert np.allclose(phi_from_ds.sel(hemsph="NH").values, phi_expected[1])