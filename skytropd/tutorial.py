from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError:  # pragma: no cover - exercised only when xarray is absent.
    xr = None

from scipy.io import netcdf_file


def _validation_file(filename: str) -> Path:
    package_root = Path(__file__).resolve().parent
    for base in (package_root, package_root.parent):
        candidate = base / "ValidationData" / filename
        if candidate.exists():
            return candidate
    return package_root / "ValidationData" / filename


def buildV():
    filename = _validation_file("va.nc")
    # 1) PSI -- Streamfunction zero crossing
    # read meridional velocity V(time,lat,lev), latitude and level
    if xr is not None:
        with xr.open_dataset(filename) as dataset:
            V = dataset["va"].transpose("time", "lat", "lev").isel(time=0).load().values
            lat = dataset["lat"].load().values
            lev = dataset["lev"].load().values
    else:
        with netcdf_file(filename, "r", mmap=False) as dataset:
            V = np.array(dataset.variables["va"][:], copy=True)
            # Change axes of V to be [time, lat, lev]
            V = np.transpose(V, (2, 1, 0))
            V = V[0, :, :]
            lat = np.array(dataset.variables["lat"][:], copy=True)
            lev = np.array(dataset.variables["lev"][:], copy=True)

    return lat, lev, V


lat, lev, V = buildV()
