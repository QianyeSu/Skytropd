from pathlib import Path

import numpy as np

import skytropd as pyt
import skytropd.tutorial as tutorial


def main() -> None:
    package_dir = Path(pyt.__file__).resolve().parent

    assert (package_dir / "ValidationData" / "va.nc").exists()
    assert (package_dir / "ValidationMetrics" / "EDJ.nc").exists()
    assert tutorial.V.shape == (tutorial.lat.size, tutorial.lev.size)

    lat = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    field = np.sin(2.0 * np.radians(lat))
    phi = pyt.TropD_Calculate_MaxLat(field, lat)
    assert np.isfinite(phi)

    print(package_dir)
    print(phi)


if __name__ == "__main__":
    main()
