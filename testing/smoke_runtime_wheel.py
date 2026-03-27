import numpy as np

import skytropd as pyt
from skytropd._fortran_zero_crossing import fortran_zero_crossing_status


def main() -> None:
    backend_ok, backend_error = fortran_zero_crossing_status()
    assert backend_ok, backend_error

    lat = np.arange(-50.0, 51.0, 10.0)
    field = np.sin(2.0 * np.radians(np.abs(lat)))
    phi = pyt.TropD_Calculate_MaxLat(field, lat)
    assert np.isfinite(phi)

    zc_field = np.array([-3.0, -2.0, -1.0, 1.0, 2.0])
    zc = pyt.TropD_Calculate_ZeroCrossing(zc_field, lat)
    assert np.isfinite(zc)

    lev = np.array([1000.0, 850.0, 700.0, 500.0, 300.0])
    lat_shape = np.sin(3.0 * np.pi * np.abs(lat)[:, None] / np.max(np.abs(lat)))
    vertical_shape = np.cos(
        np.pi * (lev[None, :] - lev[-1]) / (lev[0] - lev[-1])
    )
    V = np.sign(lat)[:, None] * 20.0 * lat_shape * vertical_shape
    psi = pyt.TropD_Calculate_StreamFunction(V, lat, lev)
    phi_sh, phi_nh = pyt.TropD_Metric_PSI(psi, lat, lev, field_type="PSI")
    assert np.all(np.isfinite(phi_sh))
    assert np.all(np.isfinite(phi_nh))


if __name__ == "__main__":
    main()
