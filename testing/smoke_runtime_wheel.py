from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

import skytropd as pyt
from skytropd._fortran_zero_crossing import fortran_zero_crossing_status


def _load_validation_v():
    repo_root = Path(__file__).resolve().parents[1]
    filename = repo_root / "ValidationData" / "va.nc"
    if not filename.exists():
        raise FileNotFoundError(f"missing validation data file: {filename}")

    with netcdf_file(filename, "r", mmap=False) as dataset:
        v = np.array(dataset.variables["va"][:], copy=True)
        v = np.transpose(v, (2, 1, 0))
        v = v[0, :, :]
        lat = np.array(dataset.variables["lat"][:], copy=True)
        lev = np.array(dataset.variables["lev"][:], copy=True)

    return v, lat, lev


def main() -> None:
    backend_ok, backend_error = fortran_zero_crossing_status()
    assert backend_ok, backend_error

    package_dir = Path(pyt.__file__).resolve().parent
    assert not (package_dir / "ValidationData").exists()
    assert not (package_dir / "ValidationMetrics").exists()

    v, lat, lev = _load_validation_v()
    psi = pyt.TropD_Calculate_StreamFunction(v, lat, lev)

    phi_from_v = pyt.TropD_Metric_PSI(v, lat, lev)
    phi_from_psi = pyt.TropD_Metric_PSI(psi, lat, lev, field_type="PSI")

    assert len(phi_from_v) == len(phi_from_psi) == 2
    for phi_v, phi_psi in zip(phi_from_v, phi_from_psi):
        assert np.all(np.isfinite(phi_v))
        assert np.all(np.isfinite(phi_psi))
        assert np.allclose(phi_v, phi_psi, equal_nan=True)


if __name__ == "__main__":
    main()
