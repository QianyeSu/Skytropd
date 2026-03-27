import numpy as np

import skytropd as pyt
from skytropd._fortran_zero_crossing import fortran_zero_crossing_status


def main() -> None:
    backend_ok, backend_error = fortran_zero_crossing_status()
    assert backend_ok, backend_error

    lat = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    field = np.sin(2.0 * np.radians(lat))
    phi = pyt.TropD_Calculate_MaxLat(field, lat)
    assert np.isfinite(phi)

    zc_field = np.array([2.0, 1.0, -1.0, -2.0, -3.0])
    zc = pyt.TropD_Calculate_ZeroCrossing(zc_field, lat)
    assert np.isfinite(zc)

    lev = np.array([1000.0, 850.0, 700.0, 500.0, 300.0])
    psi = np.array(
        [
            [1.5, 1.2, 1.0, 0.6, 0.2],
            [1.0, 0.8, 0.5, 0.1, -0.2],
            [0.4, 0.2, -0.1, -0.4, -0.7],
            [-0.3, -0.5, -0.8, -1.0, -1.3],
            [-0.8, -1.0, -1.2, -1.5, -1.7],
        ]
    )
    phi_sh, phi_nh = pyt.TropD_Metric_PSI(psi, lat, lev, field_type="PSI")
    assert np.isnan(phi_sh)
    assert np.isfinite(phi_nh)


if __name__ == "__main__":
    main()
