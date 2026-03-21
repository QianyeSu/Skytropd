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

