from pathlib import Path

import numpy as np

import skytropd as pyt
import skytropd.tutorial as tutorial
from skytropd._fortran_zero_crossing import fortran_zero_crossing_status


def main() -> None:
    package_dir = Path(pyt.__file__).resolve().parent

    assert (package_dir / "ValidationData" / "va.nc").exists()
    assert (package_dir / "ValidationMetrics" / "EDJ.nc").exists()
    assert tutorial.V.shape == (tutorial.lat.size, tutorial.lev.size)

    backend_ok, backend_error = fortran_zero_crossing_status()
    assert backend_ok, backend_error

    lat = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    field = np.sin(2.0 * np.radians(lat))
    phi = pyt.TropD_Calculate_MaxLat(field, lat)
    assert np.isfinite(phi)

    zc_field = np.array([2.0, 1.0, -1.0, -2.0, -3.0])
    zc = pyt.TropD_Calculate_ZeroCrossing(zc_field, lat)
    assert np.isfinite(zc)

    print(package_dir)
    print(phi)
    print(zc)


if __name__ == "__main__":
    main()
