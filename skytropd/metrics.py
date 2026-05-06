# Written by Ori Adam Mar.21.2017
# Edited by Alison Ming Jul.4.2017
# rewrite for readability/vectorization - sjs 1.27.22
from typing import Optional, Tuple, Callable, Union
import warnings
from functools import wraps
from inspect import signature
import re
import numpy as np
from numpy.polynomial import polynomial
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from .functions import (
    find_nearest,
    TropD_Calculate_MaxLat,
    TropD_Calculate_StreamFunction,
    TropD_Calculate_TropopauseHeight,
    TropD_Calculate_ZeroCrossing,
)
from ._fortran_zero_crossing import fortran_descending_threshold_crossing

# kappa = R_dry / Cp_dry
KAPPA = 287.04 / 1005.7

try:
    trapezoid = np.trapezoid
except AttributeError:
    trapezoid = np.trapz


def _metric_code_from_name(func_name: str) -> str:
    """Infer the base metric code from a metric wrapper function name."""

    known_codes = ("EDJ", "OLR", "PE", "PSI", "PSL", "STJ", "TPB", "UAS", "GWL", "1sigma")
    prefix = "TropD_Metric_"
    metric_name = func_name[len(prefix) :] if func_name.startswith(prefix) else func_name
    for code in known_codes:
        if (
            metric_name == code
            or metric_name.startswith(code + "_")
            or metric_name.endswith("_" + code)
            or ("_" + code + "_") in metric_name
        ):
            return code
    return metric_name.split("_")[-1]


def hemisphere_handler(
    metric_func: Callable[..., Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]
) -> Callable:
    """
    Wrapper for metrics to allow one or two hemispheres of data to be passed

    Parameters
    ----------
    metric_func : Callable
        the skytropd metric function to wrap

    Returns
    -------
    Callable
        wrapped function with same name, docstring, and call signature as the original
        function, but which operates over each hemisphere independently
    """

    @wraps(metric_func)
    def wrapped_metric_func(
        arr: np.ndarray, lat: np.ndarray, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        wrapper for skytropd metric functions which allows one or two hemispheres of data
        to be passed

        Parameters
        ----------
        arr : np.ndarray
            first argument to a skytropd metric function is always the field used for the
            metric (zonal wind, OLR, meridional wind, etc.)
        lat : np.ndarray
            second argument to a metric function is always the latitude array
        *args
            Other arguments passed to the metric function, typically will only be the
            pressure level array if any
        **kwargs
            Keyword arguments passed to the metric function, all have a `method` keyword,
            some have others

        Returns
        -------
        PhiSH : np.ndarray, optional
            metric latitude in SH (if SH latitudes are provided)
        PhiNH : np.ndarray, optional
            metric latitude in NH (if NH latitudes are provided)
        UmaxSH : np.ndarray, optional
            maximum of SH zonal wind (if using `TropD_Metric_EDJ` or `TropD_Metric_STJ`
            and `method="fit"`)
        UmaxNH : np.ndarray, optional
            maximum of NH zonal wind (if using `TropD_Metric_EDJ` or `TropD_Metric_STJ`
            and `method="fit"`)
        """

        kwargs.pop("hem", None)
        # get the metric identifier from the function name
        metric_code = _metric_code_from_name(metric_func.__name__)
        # we need to know which dimension is the latitude dimension to split it
        # appropriately. In functions which accept vertically-resolved input data,
        # the vertical level is usually last and latitude is second to last.
        # TropD_Metric_PSI also accepts the trailing order (..., lev, lat).
        has_extra_dim = False
        # these all require vertically-resolved data
        if metric_code in ["STJ", "TPB", "PSI"]:
            has_extra_dim = True
        # these take it optionally, so we need to figure out if it was provided, either as
        # an arg (it will be third positional and probably the only arg in args) or kwarg
        elif metric_code in ["UAS", "EDJ"]:
            has_extra_dim = (len(args) > 0) or (kwargs.get("lev", None) is not None)
        # Shah metrics may also have an extra last dimension if zonal_mean_tracer=False
        # (but it is lon, not lev)
        elif metric_code in ["GWL", "1sigma"]:
            has_extra_dim = not kwargs.get(
                "zonal_mean_tracer",
                signature(metric_func).parameters["zonal_mean_tracer"].default,
            )
        lat_axis = -2 if has_extra_dim else -1
        if metric_code == "PSI":
            lev = args[0] if len(args) > 0 else kwargs.get("lev", None)
            if lev is not None:
                lev = np.asarray(lev)
                if arr.shape[-2:] == (lev.size, lat.size):
                    lat_axis = -1
        # now the TropD_Metric_TPB accepts a Z array kwarg that also needs to be split
        # based on latitude, so we need to get that if it is there
        Z = None
        if metric_code == "TPB":
            Z = kwargs.pop("Z", None)
        # now we just split by hemisphere and apply as if the data was for the NH
        Phi_list = []
        if (lat < -20.0).any():
            # let's make sure we include the equator point just in case
            SHmask = lat < 0.5
            SHarr_mask = [slice(None)] * arr.ndim
            SHarr_mask[lat_axis] = SHmask
            if Z is not None:
                kwargs["Z"] = Z[tuple(SHarr_mask)]
            Phi_list.append(
                metric_func(
                    # for the TropD_Metric_PSI, it takes meridional wind. In order to
                    # make meridional wind in the SH like the NH, we have to flip the sign
                    arr[tuple(SHarr_mask)] * (-1.0 if metric_code == "PSI" else 1.0),
                    -lat[SHmask],
                    *args,
                    **kwargs,
                )
            )
            if isinstance(Phi_list[0], tuple):
                Phi_list_temp = list(Phi_list[0])
                #Phi_list[0][0] *= -1.0
                Phi_list_temp[0] *= -1.0
                Phi_list[0] = tuple(Phi_list_temp)
            else:
                Phi_list[0] *= -1.0
        if (lat > 20.0).any():
            NHmask = lat > -0.5
            NHarr_mask = [slice(None)] * arr.ndim
            NHarr_mask[lat_axis] = NHmask
            if Z is not None:
                kwargs["Z"] = Z[tuple(NHarr_mask)]
            Phi_list.append(
                metric_func(arr[tuple(NHarr_mask)], lat[NHmask], *args, **kwargs)
            )
        # one final snag before we return. TropD_Metric_EDJ and TropD_Metric_STJ
        # potentially return extra arrays if using `method="fit"`, so we need to
        # flatten our return list and potentially swap some variables if global data was
        # provided so that metric latitudes are always returned first
        if metric_code in ["EDJ", "STJ"]:
            method_used = kwargs.get(
                "method", signature(metric_func).parameters["method"].default
            )
            if method_used == "fit":
                Phi_list = [item for pair in Phi_list for item in pair]
                if len(Phi_list) == 4:
                    Phi_list[1], Phi_list[2] = Phi_list[2], Phi_list[1]
        return tuple(Phi_list)  # type: ignore

    return wrapped_metric_func


def _coerce_trailing_lat_lev_axes(
    field: np.ndarray, lat: np.ndarray, lev: np.ndarray, field_name: str
) -> np.ndarray:
    """Return ``field`` with trailing axes ordered as ``(..., lat, lev)``."""

    field = np.asarray(field)
    lat = np.asarray(lat)
    lev = np.asarray(lev)

    if field.shape[-2:] == (lat.size, lev.size):
        return field
    if field.shape[-2:] == (lev.size, lat.size):
        return np.swapaxes(field, -2, -1)
    raise ValueError(
        f"final dimensions on {field_name} {field.shape[-2:]} and grid coordinates "
        f"must match either (lat, lev)=({lat.size},{lev.size}) or "
        f"(lev, lat)=({lev.size},{lat.size})"
    )


def _interp_last_axis_common_grid(
    x: np.ndarray, xp: np.ndarray, fp: np.ndarray
) -> np.ndarray:
    """Vectorized linear interpolation for profiles that share a common x-grid."""

    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp)

    if xp.ndim != 1:
        raise ValueError("xp must be one-dimensional")
    if fp.shape[-1] != xp.size:
        raise ValueError(
            f"fp last axis of size {fp.shape[-1]} is not aligned with xp size {xp.size}"
        )
    if x.shape != fp.shape[:-1]:
        raise ValueError(
            f"x shape {x.shape} must match fp leading shape {fp.shape[:-1]}"
        )

    x_eval = np.where(np.isfinite(x), x, xp[0])
    idx = np.searchsorted(xp, x_eval, side="right") - 1
    idx = np.clip(idx, 0, xp.size - 2)

    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = np.take_along_axis(fp, idx[..., None], axis=-1)[..., 0]
    y1 = np.take_along_axis(fp, (idx + 1)[..., None], axis=-1)[..., 0]

    frac = np.divide(
        x_eval - x0,
        x1 - x0,
        out=np.zeros_like(x0, dtype=np.result_type(fp.dtype, float)),
        where=x1 != x0,
    )
    out = y0 + frac * (y1 - y0)
    return np.where(np.isfinite(x), out, np.nan)


def _interp_midpoint_field_to_grid(midpoint_field: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Interpolate midpoint values back to the original 1-D grid along the last axis."""

    midpoint_field = np.asarray(midpoint_field)
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if midpoint_field.shape[-1] != x.size - 1:
        raise ValueError(
            "midpoint_field last axis must be one smaller than the target x-grid"
        )

    out = np.empty(midpoint_field.shape[:-1] + (x.size,), dtype=midpoint_field.dtype)
    out[..., 0] = midpoint_field[..., 0]
    out[..., -1] = midpoint_field[..., -1]
    if x.size > 2:
        dx_prev = x[1:-1] - x[:-2]
        dx_next = x[2:] - x[1:-1]
        out[..., 1:-1] = (
            midpoint_field[..., :-1] * dx_next + midpoint_field[..., 1:] * dx_prev
        ) / (dx_prev + dx_next)
    return out


def _maxlat_last_axis_no_nan(F: np.ndarray, lat: np.ndarray, n: int) -> np.ndarray:
    """Fast path for MaxLat when latitude is the last axis and there are no NaNs."""

    F = np.asarray(F)
    lat = np.asarray(lat, dtype=float)
    if F.shape[-1] != lat.size:
        raise ValueError(
            f"last axis of F with size {F.shape[-1]} is not aligned with lat size {lat.size}"
        )

    Fmin = F.min(axis=-1, keepdims=True)
    Frange = F.max(axis=-1, keepdims=True) - Fmin
    scaled = np.divide(
        F - Fmin,
        Frange,
        out=np.full_like(F, np.nan, dtype=np.result_type(F.dtype, float)),
        where=Frange != 0,
    )
    weights = scaled**n
    return trapezoid(weights * lat, lat, axis=-1) / trapezoid(weights, lat, axis=-1)


def _quadratic_peak_fit_last_axis(
    field: np.ndarray, lat: np.ndarray, n_fit: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Return quadratic-fit peak latitude and strength for data on the last axis."""

    field = np.asarray(field)
    lat = np.asarray(lat, dtype=float)
    if field.shape[-1] != lat.size:
        raise ValueError(
            f"last axis of field with size {field.shape[-1]} is not aligned with lat size {lat.size}"
        )

    field_flat = field.reshape(-1, lat.size)
    out_shape = field.shape[:-1]
    phi = np.zeros(field_flat.shape[0], dtype=np.result_type(field.dtype, float))
    umax = np.zeros(field_flat.shape[0], dtype=np.result_type(field.dtype, float))
    imax = np.nanargmax(field_flat, axis=-1)

    boundary = (imax == 0) | (imax == lat.size - 1)
    phi[boundary] = lat[imax[boundary]]

    interior_rows = np.where(~boundary)[0]
    if interior_rows.size == 0:
        return phi.reshape(out_shape), umax.reshape(out_shape)

    if int(n_fit) == 1:
        interior_imax = imax[interior_rows]
        for peak_idx in np.unique(interior_imax):
            rows = interior_rows[interior_imax == peak_idx]
            coeffs = polynomial.polyfit(
                lat[peak_idx - 1 : peak_idx + 2],
                field_flat[rows, peak_idx - 1 : peak_idx + 2].T,
                deg=2,
            )
            phi[rows] = -coeffs[1] / (2.0 * coeffs[2])
            umax[rows] = (
                4.0 * coeffs[2] * coeffs[0] - coeffs[1] * coeffs[1]
            ) / 4.0 / coeffs[2]
        return phi.reshape(out_shape), umax.reshape(out_shape)

    for row in interior_rows:
        peak_idx = imax[row]
        profile = field_flat[row]

        if n_fit > peak_idx or n_fit > profile.size - peak_idx + 1:
            fit_half_width = min(peak_idx, profile.size - peak_idx + 1)
        else:
            fit_half_width = n_fit

        coeffs = polynomial.polyfit(
            lat[peak_idx - fit_half_width : peak_idx + fit_half_width + 1],
            profile[peak_idx - fit_half_width : peak_idx + fit_half_width + 1],
            deg=2,
        )
        phi[row] = -coeffs[1] / (2.0 * coeffs[2])
        umax[row] = (
            4.0 * coeffs[2] * coeffs[0] - coeffs[1] * coeffs[1]
        ) / 4.0 / coeffs[2]

    return phi.reshape(out_shape), umax.reshape(out_shape)


def _psi_metric_latitude(
    Psi: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "Psi_500",
    lat_uncertainty: float = 0.0,
    threshold: Optional[float] = 0.1,
) -> np.ndarray:
    """Return HC edge latitude from a precomputed mass streamfunction."""

    Psi = _coerce_trailing_lat_lev_axes(Psi, lat, lev, "Psi")
    lat = np.asarray(lat)
    lev = np.asarray(lev)
    if threshold is not None:
        threshold = float(threshold)
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative or None")

    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
        Psi = Psi[..., ::-1, :]
        lat = lat[::-1]
    cos_lat = np.cos(lat * np.pi / 180.0)[:, None]

    def _psi_zero_crossing_latitude(
        P_field: np.ndarray, allow_threshold_fallback: bool
    ) -> np.ndarray:
        subpolar_boundary = 30.0
        polar_boundary = 60.0
        mask = (lat > 0) & (lat < polar_boundary)
        subpolar_mask = (lat > 0) & (lat < subpolar_boundary)
        lat_masked = lat[mask]

        # 1. Find latitude of maximal tropical Psi
        Pmax_lat_masked = TropD_Calculate_MaxLat(
            P_field[..., subpolar_mask], lat[subpolar_mask]
        )

        # 2. Find latitude of minimal subtropical Psi poleward of tropical maximum
        lat_after_Pmax = lat_masked >= Pmax_lat_masked[..., None]
        Plat_after_Pmax = np.where(lat_after_Pmax, P_field[..., mask], np.nan)
        Pmin_lat_masked = TropD_Calculate_MaxLat(-Plat_after_Pmax, lat_masked)

        # 3. Find the zero crossing between the above latitudes
        lat_in_between = (lat_masked <= Pmin_lat_masked[..., None]) & lat_after_Pmax
        Plat_in_between = np.where(lat_in_between, P_field[..., mask], np.nan)

        Phi_zero = TropD_Calculate_ZeroCrossing(
            Plat_in_between, lat_masked, lat_uncertainty=lat_uncertainty
        )
        if not allow_threshold_fallback or threshold is None:
            return Phi_zero

        Pmax = P_field[..., subpolar_mask].max(axis=-1)[..., None]
        Phi_threshold = TropD_Calculate_ZeroCrossing(
            Plat_in_between - threshold * Pmax,
            lat_masked,
            lat_uncertainty=lat_uncertainty,
        )
        return np.where(np.isnan(Phi_zero), Phi_threshold, Phi_zero)

    def _psi_direct_threshold_latitude(P_field: np.ndarray, threshold_value: float) -> np.ndarray:
        subpolar_boundary = 30.0
        polar_boundary = 60.0
        mask = (lat > 0) & (lat < polar_boundary)
        subpolar_mask = (lat > 0) & (lat < subpolar_boundary)
        lat_masked = lat[mask]

        Pmax_lat_masked = TropD_Calculate_MaxLat(
            P_field[..., subpolar_mask], lat[subpolar_mask]
        )
        lat_after_Pmax = lat_masked >= Pmax_lat_masked[..., None]
        Plat_after_Pmax = np.where(lat_after_Pmax, P_field[..., mask], np.nan)
        Pmin_lat_masked = TropD_Calculate_MaxLat(-Plat_after_Pmax, lat_masked)
        lat_in_between = (lat_masked <= Pmin_lat_masked[..., None]) & lat_after_Pmax
        Plat_in_between = np.where(lat_in_between, P_field[..., mask], np.nan)
        Pmax = P_field[..., subpolar_mask].max(axis=-1)[..., None]

        return TropD_Calculate_ZeroCrossing(
            Plat_in_between - threshold_value * Pmax,
            lat_masked,
            lat_uncertainty=lat_uncertainty,
        )

    psi_method = method.strip()
    layer_match = re.fullmatch(
        r"Psi_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)",
        psi_method,
        flags=re.IGNORECASE,
    )
    layer_percent_match = re.fullmatch(
        r"Psi_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)Perc(?:_(center2d|profile))?",
        psi_method,
        flags=re.IGNORECASE,
    )

    if (psi_method == "Psi_500") or (psi_method == "Psi_500_10Perc"):
        # Use Psi at the level nearest to 500 hPa
        P = Psi[..., find_nearest(lev, 500.0)]
    elif psi_method == "Psi_500_Int":
        # Use integrated Psi from p=0 to level nearest to 500 hPa
        PPsi = cumtrapz(Psi * cos_lat, 100.0 * lev, axis=-1, initial=0.0)
        P = PPsi[..., find_nearest(lev, 500.0)]
    elif psi_method == "Psi_Int":
        # Use vertical mean of Psi
        P = trapezoid(Psi * cos_lat, 100.0 * lev, axis=-1)
    elif layer_match is not None:
        lower = float(layer_match.group(1))
        upper = float(layer_match.group(2))
        lower_bound = min(lower, upper)
        upper_bound = max(lower, upper)
        # Use Psi averaged between the requested pressure levels
        layer_mask = (lev >= lower_bound) & (lev <= upper_bound)
        if not np.any(layer_mask):
            raise ValueError(
                "no pressure levels found within the requested PSI layer "
                f"{lower:g}-{upper:g} hPa"
            )
        P = trapezoid(
            Psi[..., layer_mask] * cos_lat, lev[layer_mask] * 100.0, axis=-1
        )
    elif layer_percent_match is not None:
        lower = float(layer_percent_match.group(1))
        upper = float(layer_percent_match.group(2))
        threshold_fraction = float(layer_percent_match.group(3)) / 100.0
        threshold_mode = (layer_percent_match.group(4) or "center2d").lower()
        lower_bound = min(lower, upper)
        upper_bound = max(lower, upper)
        layer_mask = (lev >= lower_bound) & (lev <= upper_bound)
        if not np.any(layer_mask):
            raise ValueError(
                "no pressure levels found within the requested PSI layer "
                f"{lower:g}-{upper:g} hPa"
            )

        # Hill et al. (2025) interpret this edge using a layer-mean profile
        # and a threshold tied to the Hadley-cell center amplitude.
        profile = Psi[..., layer_mask].mean(axis=-1)
        nh_mask = lat > 0.0
        if not np.any(nh_mask):
            raise ValueError(
                "requires at least one positive latitude for PSI layer Perc methods"
            )

        out_shape = Psi.shape[:-2]
        phi = np.full(out_shape, np.nan, dtype=np.result_type(Psi.dtype, float))
        psi_flat = Psi.reshape(-1, lat.size, lev.size)
        profile_flat = profile.reshape(-1, lat.size)
        phi_flat = phi.reshape(-1)
        nh_lat_indices = np.where(nh_mask)[0]
        start_idx = np.full(psi_flat.shape[0], -1, dtype=np.int32)
        threshold_values = np.full(psi_flat.shape[0], np.nan, dtype=float)

        for i in range(psi_flat.shape[0]):
            nh_field = psi_flat[i, nh_mask, :]
            if not np.isfinite(nh_field).any():
                continue
            center_flat_index = int(np.nanargmax(nh_field))
            center_lat_subindex, center_level_index = np.unravel_index(
                center_flat_index, nh_field.shape
            )
            center_lat_index = nh_lat_indices[center_lat_subindex]
            center_value_2d = float(nh_field[center_lat_subindex, center_level_index])
            center_value_profile = float(profile_flat[i, center_lat_index])

            if threshold_mode == "center2d":
                threshold_values[i] = threshold_fraction * center_value_2d
            elif threshold_mode == "profile":
                threshold_values[i] = threshold_fraction * center_value_profile
            else:
                raise ValueError("unrecognized method " + method)
            start_idx[i] = center_lat_index

        phi_flat[:] = fortran_descending_threshold_crossing(
            profile_flat, lat, start_idx, threshold_values
        )
        fallback_mask = (~np.isfinite(phi)) | (phi > 60.0)
        if not np.any(fallback_mask):
            return phi

        # Repository safeguard: if the layer-threshold edge remains poleward of
        # 60N, fall back only for those cases to the same layer's zero-crossing edge.
        P_zero = trapezoid(
            Psi[..., layer_mask] * cos_lat, lev[layer_mask] * 100.0, axis=-1
        )
        phi[fallback_mask] = _psi_zero_crossing_latitude(
            P_zero[fallback_mask], allow_threshold_fallback=False
        )
        return phi
    else:
        raise ValueError("unrecognized method " + method)

    if psi_method == "Psi_500_10Perc":
        threshold_value = 0.1 if threshold is None else threshold
        Phi = _psi_direct_threshold_latitude(P, threshold_value)
    else:
        Phi = _psi_zero_crossing_latitude(P, allow_threshold_fallback=True)

    return Phi


@hemisphere_handler
def TropD_Metric_EDJ(
    U: np.ndarray,
    lat: np.ndarray,
    lev: Optional[np.ndarray] = None,
    method: str = "peak",
    n_fit: int = 1,
    **maxlat_kwargs,
) -> np.ndarray:
    """
    TropD Eddy Driven Jet (EDJ) Metric

    Latitude of maximum of the zonal wind at the level closest to 850 hPa

    Parameters
    ----------
    U : numpy.ndarray (dim1, ..., lat[, lev])
        N-dimensional array of zonal-mean zonal wind data. Also accepts surface/
        850hPa wind
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray, optional (lev,)
        vertical level array in hPa, used to find wind closest to 850hPa. if not
        provided, last axis is assumed to be lat
    method : {"peak", "max", "fit"}, optional
        Method for determining latitude of maximum zonal wind, by default "peak":

        * "peak": Latitude of the maximum of the zonal wind at the level closest
                  to 850hPa (smoothing parameter ``n=30``)
        * "max": Latitude of the maximum of the zonal wind at the level closest to
                 850hPa (smoothing parameter ``n=6``)
        * "fit": Latitude of the maximum of the zonal wind at the level closest to
                 850hPa using a quadratic polynomial fit of data from grid points
                 surrounding the grid point of the maximum

    n_fit : int, optional
        used when ``method="fit"``, determines the number of points around the max to use
        while fitting, by default 1
    **kwargs : optional
        additional keyword arguments for :py:func:`TropD_Calculate_MaxLat` (not used
        for ``method="fit"``)

        n : int, optional
            Rank of moment used to calculate the location of max, e.g.,
            ``n=1,2,4,6,8,...``, by default 6 if ``method="max"``, 30 if ``method="peak"``

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-2[, dimN-1])
        N-2(N-1) dimensional latitudes of the EDJ
    Umax : numpy.ndarray (dim1, ..., dimN-2), conditional
        N-2 dimensional EDJ strength, returned if ``method="fit"``

    """

    U = np.asarray(U)
    lat = np.asarray(lat)
    get_lev = lev is not None
    U_grid_shape = U.shape[-2:] if get_lev else U.shape[-1]
    input_grid_shape = (lat.size, lev.size) if get_lev else lat.size  # type: ignore
    if U_grid_shape != input_grid_shape:
        raise ValueError(
            f"last axes of U w/ shape {U_grid_shape},"
            " not aligned with input grid of shape " + str(input_grid_shape)
        )
    if method not in ["max", "peak", "fit"]:
        raise ValueError("unrecognized method " + method)
    n_fit = int(n_fit)

    eq_boundary = 15.0
    polar_boundary = 70.0
    mask = (lat > eq_boundary) & (lat < polar_boundary)

    if get_lev:
        u850 = U[..., find_nearest(lev, 850.0)]  # type: ignore
    else:
        u850 = U

    if method != "fit":  # max or peak
        # update default n to 30 for peak
        if method == "peak":
            maxlat_kwargs["n"] = maxlat_kwargs.get("n", 30)
        else:  # max
            maxlat_kwargs["n"] = maxlat_kwargs.get("n", 6)
        # lat should already be last axis
        maxlat_kwargs.pop("axis", None)

        Phi = TropD_Calculate_MaxLat(u850[..., mask], lat[mask], **maxlat_kwargs)

        return Phi

    else:  # method == 'fit':
        lat_mask = lat[mask]
        Phi, Umax = _quadratic_peak_fit_last_axis(u850[..., mask], lat_mask, n_fit=n_fit)

        return Phi, Umax  # type: ignore


@hemisphere_handler
def TropD_Metric_OLR(
    olr: np.ndarray,
    lat: np.ndarray,
    method: str = "250W",
    Cutoff: Optional[float] = None,
    **maxlat_kwargs,
) -> np.ndarray:
    """
    TropD Outgoing Longwave Radiation (OLR) Metric

    Latitude of maximum OLR or first latitude poleward of maximum where OLR reaches
    crosses a predefined cutoff

    Parameters
    ----------
    olr : numpy.ndarray (dim1, ..., lat)
        zonal mean TOA olr (positive)
    lat : numpy.ndarray (lat,)
        latitude array
    method : {"250W", "20W", "cutoff", "10Perc", "max", "peak"}, optional
        Method for determining the OLR maximum/threshold, by default "250W":

        * "250W": the 1st latitude poleward of the tropical OLR max in each hemisphere
                    where OLR crosses :math:`250W/m^2`
        * "20W": the 1st latitude poleward of the tropical OLR max in each hemisphere
                where OLR crosses [the tropical OLR max minus :math:`20W/m^2`]
        * "cutoff": the 1st latitude poleward of the tropical OLR max in each
                    hemisphere where OLR crosses a specified cutoff value
        * "10Perc": the 1st latitude poleward of the tropical OLR max in each
                    hemisphere where OLR is 10% smaller than the tropical OLR max
        * "max": the latitude of the tropical olr max in each hemisphere with
                    smoothing parameter ``n=6``
        * "peak": the latitude of maximum of tropical olr in each hemisphere with
                smoothing parameter ``n=30``

    Cutoff : float, optional
        if ``method="cutoff"``, specifies the OLR cutoff value in :math:`W/m^2`

    **kwargs : optional
        additional keyword arguments for :py:func:`TropD_Calculate_MaxLat`

        n : int, optional
            Rank of moment used to calculate the location of max, e.g.,
            ``n=1,2,4,6,8,...``, by default 6 if ``method="max"``, 30 if ``method="peak"``

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional latitudes of the near-equator OLR max/threshold crossing
    """

    olr = np.asarray(olr)
    lat = np.asarray(lat)
    if olr.shape[-1] != lat.size:
        raise ValueError(
            f"last axis of olr had shape {olr.shape[-1]},"
            f" not aligned with lat size {lat.size}"
        )
    if "n" in maxlat_kwargs and method != "max":
        warnings.warn("smoothing parameter n only utilized for method == max")
        maxlat_kwargs.pop("n")
    if Cutoff is not None and method != "cutoff":
        warnings.warn("cutoff parameter only utilized for method == cutoff")

    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
        olr = olr[..., ::-1]
        lat = lat[::-1]

    eq_boundary = 5.0
    subpolar_boundary = 40.0
    polar_boundary = 60.0

    subpolar_mask = (lat > eq_boundary) & (lat < subpolar_boundary)

    if method == "peak":
        maxlat_kwargs["n"] = 30
    elif method == "max":
        maxlat_kwargs["n"] = 6
    # lat should already be last axis
    maxlat_kwargs.pop("axis", None)

    olr_max_lat = TropD_Calculate_MaxLat(
        olr[..., subpolar_mask], lat[subpolar_mask], **maxlat_kwargs
    )

    if method in ["cutoff", "250W", "20W", "10Perc"]:
        # get tropical OLR max for methods 20W and 10Perc
        olr_max = olr[..., subpolar_mask].max(axis=-1)[..., None]

        # set cutoff dependent on method
        if method == "250W":
            Cutoff = 250.0
        elif method == "20W":
            Cutoff = olr_max - 20.0
        elif method == "10Perc":
            Cutoff = 0.9 * olr_max

        # identify regions poleward of the OLR max in both hemispheres
        mask = (lat > eq_boundary) & (lat < polar_boundary)
        max_mask = lat[mask] > olr_max_lat[..., None]

        # OLR in each hemisphere, only valid poleward of max
        olr = np.where(max_mask, olr[..., mask], np.nan)

        # get latitude where OLR falls below cutoff poleward of tropical max
        Phi = TropD_Calculate_ZeroCrossing(olr - Cutoff, lat[mask])
    # these methods don't need to find a threshold after
    # the OLR max, just need the max
    elif method in ["max", "peak"]:
        Phi = olr_max_lat
    else:
        raise ValueError("unrecognized method " + method)

    return Phi


@hemisphere_handler
def TropD_Metric_PE(
    pe: np.ndarray,
    lat: np.ndarray,
    method: str = "zero_crossing",
    lat_uncertainty: float = 0.0,
) -> np.ndarray:
    """
    TropD Precipitation Minus Evaporation (PE) Metric

    Find the first zero crossing of zonal-mean precipitation minus evaporation poleward
    of the subtropical minimum

    Parameters
    ----------
    pe : numpy.ndarray (dim1,  ..., lat,)
        zonal-mean precipitation minus evaporation
    lat : numpy.ndarray (lat,)
        latitude array
    method : {"zero_crossing"}, optional
        Method to compute the zero crossing for precipitation minus evaporation, by
        default "zero_crossing".

        * "zero_crossing": the first latitude poleward of the subtropical P-E min
                           where P-E changes from negative to positive.

    lat_uncertainty : float, optional
        The minimal distance allowed between adjacent zero crossings in degrees,
        by default 0.0

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional latitudes of the 1st subtropical P-E zero crossing
    """

    pe = np.atleast_2d(pe)
    lat = np.asarray(lat)
    if pe.shape[-1] != lat.size:
        raise ValueError(
            f"last axis of P-E had shape {pe.shape[-1]},"
            f" not aligned with lat size {lat.size}"
        )
    if method != "zero_crossing":
        raise ValueError("unrecognized method " + method)

    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
        pe = pe[..., ::-1]
        lat = lat[::-1]

    # The gradient of PE is used to determine whether PE
    # becomes positive at the zero crossing
    dpedy = np.diff(pe, axis=-1)
    pe_grad = _interp_midpoint_field_to_grid(dpedy, lat)

    # define latitudes of boundaries certain regions
    eq_boundary = 5.0
    subpolar_boundary = 50.0
    polar_boundary = 60.0

    # split into hemispheres
    mask = (lat > eq_boundary) & (lat < polar_boundary)
    lat_masked = lat[mask]

    # find E-P maximum (P-E min) latitude in subtropics
    # first define the subpolar region to search, excluding poles due to low P-E
    subpolar_mask = (lat > eq_boundary) & (lat < subpolar_boundary)
    Emax_lat_masked = _maxlat_last_axis_no_nan(-pe[..., subpolar_mask], lat[subpolar_mask], n=30)

    # find zero crossings poleward of E-P max
    # first define regions poleward of E-P max in each hemisphere
    lat_after_Emax = lat_masked > Emax_lat_masked[..., None]
    pelat_after_Emax = np.where(lat_after_Emax, pe[..., mask], np.nan)
    ZC1_lat_masked = TropD_Calculate_ZeroCrossing(
        pelat_after_Emax, lat_masked, lat_uncertainty=lat_uncertainty
    )

    # we've got the zero crossing poleward of E-P max, but it might not be the
    # right one. Now we need to go through and check the P-E gradient
    # to make sure P-E is increasing poleward.
    # if it is, use that latitude, else, use the next zero crossing

    # first check if the (northward) gradient value at the
    # zero crossing is increasing poleward
    grad_at_ZC1 = _interp_last_axis_common_grid(ZC1_lat_masked, lat, pe_grad)
    pelat_increases_at_ZC = np.isfinite(ZC1_lat_masked) & (grad_at_ZC1 > 0.0)

    # then get the next zero crossing for when we need it
    # first define regions poleward of zero crossing
    ZC2_lat_masked = np.full_like(ZC1_lat_masked, np.nan)
    needs_ZC2 = ~pelat_increases_at_ZC
    if np.any(needs_ZC2):
        lat_after_ZC = lat_masked > ZC1_lat_masked[needs_ZC2][..., None]
        pelat_after_ZC = np.where(lat_after_ZC, pe[..., mask][needs_ZC2], np.nan)
        ZC2_lat_masked[needs_ZC2] = TropD_Calculate_ZeroCrossing(
            pelat_after_ZC, lat_masked, lat_uncertainty=lat_uncertainty
        )

    # if the gradient is increasing poleward, use it, otherwise, use the next
    Phi = np.where(pelat_increases_at_ZC, ZC1_lat_masked, ZC2_lat_masked)

    return Phi


@hemisphere_handler
def TropD_Metric_PSI(
    V: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "Psi_500",
    lat_uncertainty: float = 0.0,
    field_type: str = "V",
    threshold: Optional[float] = 0.1,
) -> np.ndarray:
    """
    TropD Mass Streamfunction (PSI) Metric

    Latitude of the subtropical zero crossing of the meridional mass streamfunction.
    The input field can be either zonal-mean meridional wind or a precomputed mass
    streamfunction.

    Parameters
    ----------
    V : numpy.ndarray (dim1, ..., lat, lev) or (dim1, ..., lev, lat)
        N-dimensional zonal-mean meridional wind when ``field_type="V"`` or a
        precomputed mass streamfunction when ``field_type="PSI"``. The final two
        axes may be ordered either as ``(..., lat, lev)`` or ``(..., lev, lat)``.
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray (lev,)
        vertical level array in hPa
    method : {"Psi_500", "Psi_500_10Perc", "Psi_<p1>_<p2>", "Psi_<p1>_<p2>_<x>Perc", "Psi_<p1>_<p2>_<x>Perc_center2d", "Psi_<p1>_<p2>_<x>Perc_profile", "Psi_500_Int", "Psi_Int"},
    optional
        Method of determining which Psi zero crossing to return, by default "Psi_500":

        * "Psi_500": Zero crossing of the streamfunction (Psi) at 500hPa
        * "Psi_500_10Perc": Crossing of ``threshold`` times the extremum value of Psi
                            in each hemisphere at the 500hPa level. The historical
                            method name is retained and defaults to ``threshold=0.1``.
        * "Psi_<p1>_<p2>": Zero crossing of Psi vertically averaged between the
                           ``p1`` and ``p2`` hPa levels, e.g. ``"Psi_300_700"``
        * "Psi_<p1>_<p2>_<x>Perc": Latitude where the layer-mean streamfunction
                                   between ``p1`` and ``p2`` hPa decreases to
                                   ``x`` % of the Hadley-cell-center value from
                                   the full ``(lat, lev)`` field. Omitting the
                                   suffix defaults to the ``center2d`` variant
                                   below, so ``"Psi_500_800_5Perc"`` remains the
                                   Hill-style default. In this repository, if
                                   the resulting latitude lies poleward of
                                   60N or is not finite, the metric falls back
                                   to the same layer's standard zero-crossing
                                   edge ``"Psi_<p1>_<p2>"``.
        * "Psi_<p1>_<p2>_<x>Perc_center2d": As above, with the threshold taken
                                            from the 2-D cell-center maximum.
        * "Psi_<p1>_<p2>_<x>Perc_profile": As above, but with the threshold
                                           taken from the layer-mean profile
                                           value at the cell-center latitude.
        * "Psi_500_Int": Zero crossing of the vertically-integrated Psi at 500 hPa
        * "Psi_Int" : Zero crossing of the column-averaged Psi

        The ``Psi_<p1>_<p2>_<x>Perc*`` family follows the Hadley-cell descending-edge
        interpretation discussed by Hill, S. A., S. Bordoni, J. L. Mitchell, and
        J. M. Lora, 2025: *Interpreting Seasonal and Interannual Hadley Cell
        Descending Edge Migrations via the Cell-Mean Rossby Number*,
        *Journal of Climate*, 38, 5505-5520,
        https://doi.org/10.1175/JCLI-D-24-0678.1.

    lat_uncertainty : float, optional
        The minimal distance allowed between the adjacent zero crossings, same units as
        lat, by default 0.0. e.g., for ``lat_uncertainty=10``, this function will return
        NaN if another zero crossing is within 10 degrees of the most equatorward
        zero crossing.
    field_type : {"V", "PSI"}, optional
        Specifies whether the first argument is zonal-mean meridional wind or a
        precomputed mass streamfunction, by default ``"V"``
    threshold : float or None, optional
        Threshold fraction used for the threshold-crossing HC edge. For
        ``method="Psi_500_10Perc"``, this is the threshold applied directly.
        For the other PSI methods, if the zero crossing cannot be found and the
        result would otherwise be NaN, the metric falls back to the latitude
        where Psi first crosses ``threshold`` times ``Pmax`` between the subtropical
        maximum and minimum. Set to ``None`` to disable this fallback. By
        default 0.1.

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional latitudes of the Psi zero crossing
    """

    # type casting/input checking
    V = _coerce_trailing_lat_lev_axes(V, lat, lev, "V")
    lat = np.asarray(lat)
    lev = np.asarray(lev)

    normalized_input = field_type.strip().lower()
    if normalized_input in {"v", "wind", "meridional_wind"}:
        Psi = TropD_Calculate_StreamFunction(V, lat, lev)
    elif normalized_input in {"psi", "streamfunction", "mass_streamfunction"}:
        Psi = V
    else:
        raise ValueError("field_type must be one of 'V' or 'PSI'")

    return _psi_metric_latitude(
        Psi,
        lat,
        lev,
        method=method,
        lat_uncertainty=lat_uncertainty,
        threshold=threshold,
    )


@hemisphere_handler
def TropD_Metric_PSL(
    ps: np.ndarray, lat: np.ndarray, method: str = "peak", **maxlat_kwargs
) -> np.ndarray:
    """
    TropD Sea-level Pressure (PSL) Metric

    Latitude of maximum of the subtropical sea-level pressure

    Parameters
    ----------
    ps : np.ndarray (dim1, ..., lat)
        N-dimensional sea-level pressure
    lat : np.ndarray (lat,)
        latitude array
    method : {"peak", "max"}, optional
        Method for determining latitude of max PSL, by default "peak":

        * "peak": latitude of the maximum of subtropical sea-level pressure (smoothing
                  parameter ``n=30``)
        * "max": latitude of the maximum of subtropical sea-level pressure (smoothing
                 parameter ``n=6``)

    **kwargs : optional
        additional keyword arguments for :py:func:`TropD_Calculate_MaxLat` (not used
        for ``method="fit"``)

        n : int, optional
            Rank of moment used to calculate the location of max, e.g.,
            ``n=1,2,4,6,8,...``, by default 6 if ``method="max"``, 30 if ``method="peak"``

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-1)
        N-1 dimensional latitudes of the subtropical PSL maximum
    """

    ps = np.asarray(ps)
    lat = np.asarray(lat)
    if ps.shape[-1] != lat.size:
        raise ValueError(
            f"last axis of ps had shape {ps.shape[-1]},"
            f" not aligned with lat size {lat.size}"
        )
    if method not in ["max", "peak"]:
        raise ValueError("unrecognized method " + method)

    if method == "peak":
        maxlat_kwargs["n"] = maxlat_kwargs.get("n", 30)
    else:  # max
        maxlat_kwargs["n"] = maxlat_kwargs.get("n", 6)
    # lat should already be last axis
    maxlat_kwargs.pop("axis", None)

    eq_boundary = 15
    polar_boundary = 60
    mask = (lat > eq_boundary) & (lat < polar_boundary)

    Phi = TropD_Calculate_MaxLat(ps[..., mask], lat[mask], **maxlat_kwargs)

    return Phi


@hemisphere_handler
def TropD_Metric_STJ(
    U: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "adjusted_peak",
    n_fit: int = 1,
    **maxlat_kwargs,
) -> np.ndarray:
    #TODO: Amend fit to adjusted_fit for consistency. Deprecate and add warning
    """
    TropD Subtropical Jet (STJ) Metric

    Latitude of the (adjusted) maximum upper-level zonal-mean zonal wind

    Parameters
    ----------
    U : numpy.ndarray (dim1,...,lat,lev,)
        N-dimensional zonal-mean zonal wind
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray (lev,)
        vertical level array in hPa
    method : {"adjusted_peak", "adjusted_max", "core_peak", "core_max", "fit"}, optional
        Method for determing the latitude of the STJ maximum, by default "adjusted_peak":

        * "adjusted_peak": Latitude of maximum (smoothing parameter``n=30``) of [the
                           zonal wind averaged between 100 and 400 hPa] minus [the zonal
                           mean zonal wind at the level closest to 850hPa], poleward of
                           10 degrees and equatorward of the Eddy Driven Jet latitude
        * "adjusted_max": Latitude of maximum (smoothing parameter ``n=6``) of [the
                          zonal wind averaged between 100 and 400 hPa] minus [the zonal
                          mean zonal wind at the level closest to 850hPa], poleward of 10
                          degrees and equatorward of the Eddy Driven Jet latitude
        * "core_peak": Latitude of maximum (smoothing parameter ``n=30``) of the zonal
                       wind averaged between 100 and 400 hPa, poleward of 10 degrees and
                       equatorward of 60 degrees
        * "core_max": Latitude of maximum (smoothing parameter ``n=6``) of the zonal wind
                      averaged between 100 and 400 hPa, poleward of 10 degrees and
                      equatorward of 60 degrees`r`n        * "fit": Latitude of the maximum of [the zonal wind averaged between 100 and 400`r`n          hPa] minus [the zonal mean zonal wind at the level closest to 850 hPa]`r`n          using a quadratic polynomial fit of data from grid points surrounding`r`n          the grid point of the maximum

    n_fit : int, optional
        used when ``method="fit"``, determines the number of points around the max to use
        while fitting, by default 1

    **kwargs : optional
        additional keyword arguments for :py:func:`TropD_Calculate_MaxLat` (not used
        for ``method="fit"``)

        n : int, optional
            Rank of moment used to calculate the location of max, e.g.,
            ``n=1,2,4,6,8,...``, by default 6 if ``method="core_max"`` or
            ``method="adjusted_max"``, 30 if ``method="core_peak"`` or
            ``method="adjusted_peak"``

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional latitudes of the STJ
    Umax : numpy.ndarray (dim1, ..., dimN-2), conditional
        N-2 dimensional  STJ strength, returned if ``method="fit"``
    """

    U = np.asarray(U)
    if U.ndim < 3:
        U = U[None, ...]
    lat = np.asarray(lat)
    lev = np.asarray(lev)
    if U.shape[-2:] != (lat.size, lev.size):
        raise ValueError(
            f"last axes of U had shape {U.shape[-2:]},"
            f" not aligned with grid shape {lat.size, lev.size}"
        )

    if method not in ["adjusted_peak", "core_peak", "adjusted_max", "core_max", "fit"]:
        raise ValueError("unrecognized method " + method)

    layer_400_to_100 = (lev >= 100) & (lev <= 400)
    lev_int = lev[layer_400_to_100]

    # Pressure weighted vertical mean of U
    if lev_int.size > 1:
        u_int = trapezoid(U[..., layer_400_to_100], lev_int, axis=-1) / (
            lev_int[-1] - lev_int[0]
        )
    else:
        u_int = U[..., layer_400_to_100]

    if ("adjusted" in method) or method == "fit":  # adjusted_peak, adjusted_max methods
        idx_850 = find_nearest(lev, 850)
        u = u_int - U[..., idx_850]
    else:  # core_peak, core_max methods
        u = u_int

    eq_boundary = 10
    polar_boundary = 60
    mask = (lat > eq_boundary) & (lat < polar_boundary)

    if method != "fit":
        if "peak" in method:  # adjusted_peak or core_peak have different default
            maxlat_kwargs["n"] = maxlat_kwargs.get("n", 30)
        else:  # adjusted_max or core_max methods
            maxlat_kwargs["n"] = maxlat_kwargs.get("n", 6)
        # lat should already be last axis
        maxlat_kwargs.pop("axis", None)
        u_masked = u[..., mask]
        if "adjusted" in method:  # adjusted_max or adjusted_peak
            (Phi_EDJ,) = TropD_Metric_EDJ(U, lat, lev)
            lat_before_EDJ = lat[mask] < Phi_EDJ[..., None]
            u_masked = np.where(lat_before_EDJ, u_masked, np.nan)

        Phi = TropD_Calculate_MaxLat(u_masked, lat[mask], **maxlat_kwargs)

        return Phi

    else:  # method == 'fit':
        lat_mask = lat[mask]
        Phi, Umax = _quadratic_peak_fit_last_axis(u[..., mask], lat_mask, n_fit=n_fit)

        return Phi, Umax  # type: ignore


@hemisphere_handler
def TropD_Metric_TPB(
    T: np.ndarray,
    lat: np.ndarray,
    lev: np.ndarray,
    method: str = "max_gradient",
    Z: Optional[np.ndarray] = None,
    Cutoff: float = 1.5e4,
    **maxlat_kwargs,
) -> np.ndarray:
    """
    TropD Tropopause Break (TPB) Metric

    Finds the latitude of the tropopause break

    Parameters
    ----------
    T : numpy.ndarray (dim1, ..., lat, lev)
        N-dimensional temperature array (K)
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray (lev,)
        vertical levels array in hPa
    method : {"max_gradient", "cutoff", "max_potemp"}, optional
        Method to identify tropopause break, by default "max_gradient":

        * "max_gradient": The latitude of maximal poleward gradient of the tropopause
                          height

        * "cutoff": The most equatorward latitude where the tropopause crosses
                    a prescribed cutoff value

        * "max_potemp": The latitude of maximal difference between the potential
                        temperature at the tropopause and at the surface

    Z : Optional[numpy.ndarray], optional
        N-dimensional geopotential height array (m), required by ``method="cutoff"``
    Cutoff : float, optional
        Geopotential height cutoff (m) that marks the location of the tropopause break,
        by default 1.5e4, used when ``method="cutoff"``
    **kwargs : optional
        additional keyword arguments for :py:func:`TropD_Calculate_MaxLat` (not used
        for ``method="cutoff"``)

        n : int, optional
            Rank of moment used to calculate the location of max, e.g.,
            ``n=1,2,4,6,8,...``, by default 6 if ``method="max_gradient"``, 30 if ``method="max_pottemp"``

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-2)
        N-2 dimensional latitudes of the tropopause break
    """

    T = np.asarray(T)
    lat = np.asarray(lat)
    lev = np.asarray(lev)
    if T.shape[-2:] != (lat.size, lev.size):
        raise ValueError(
            f"final dimensions on T {T.shape[-2:]} and grid "
            f"coordinates don't match ({lat.size},{lev.size})"
        )
    if method not in ["max_gradient", "max_potemp", "cutoff"]:
        raise ValueError("unrecognized method " + method)
    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
        lat = lat[::-1]
        T = T[..., ::-1, :]
        if Z is not None:
            Z = Z[..., ::-1, :]

    polar_boundary = 60.0
    eq_boundary = 0.0
    mask = (lat > eq_boundary) & (lat < polar_boundary)

    if "max_" in method:  # 'max_gradient' or 'max_potemp'
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
            # redefine mask b/c we have new grid points
            mask = (lat > eq_boundary) & (lat < polar_boundary)
        F = np.where(np.isfinite(F), F, 0.0)

        Phi = TropD_Calculate_MaxLat(F[..., mask], lat[mask], **maxlat_kwargs)

    else:  # method == 'cutoff'
        if Z is None:
            raise ValueError('Z must be provided when method = "cutoff"')
        Pt, Ht = TropD_Calculate_TropopauseHeight(T, lev, Z)

        Phi = TropD_Calculate_ZeroCrossing(Ht[..., mask] - Cutoff, lat[mask])

    return Phi


@hemisphere_handler
def TropD_Metric_UAS(
    U: np.ndarray,
    lat: np.ndarray,
    lev: Optional[np.ndarray] = None,
    method: str = "zero_crossing",
    lat_uncertainty: float = 0.0,
) -> np.ndarray:
    """
    TropD Near-Surface Zonal Wind (UAS) Metric

    Parameters
    ----------
    U : numpy.ndarray (dim1, ..., lat[, lev])
        N-dimensional zonal mean zonal wind array. Also accepts surface wind
    lat : numpy.ndarray (lat,)
        latitude array
    lev : numpy.ndarray, optional (lev,)
        vertical level array in hPa, required if U has final dimension lev
    method : {"zero_crossing"}, optional
        Method for identifying the surface wind zero crossing, by default "zero_crossing".

        * "zero_crossing": the first subtropical latitude where near-surface zonal wind
                           changes from negative to positive

    lat_uncertainty : float, optional
        the minimal distance allowed between adjacent zero crossings in degrees, by
        default 0.0

    Returns
    -------
    Phi : numpy.ndarray (dim1, ..., dimN-2[, dimN-1])
        N-2(N-1) dimensional latitudes of the first subtropical zero crossing of
        near-surface zonal wind
    """

    U = np.asarray(U)
    lat = np.asarray(lat)
    get_lev = lev is not None
    U_grid_shape = U.shape[-2:] if get_lev else U.shape[-1]
    input_grid_shape = (lat.size, lev.size) if get_lev else lat.size  # type: ignore
    if U_grid_shape != input_grid_shape:
        raise ValueError(
            f"last axes of U w/ shape {U_grid_shape},"
            " not aligned with input grid of shape " + str(input_grid_shape)
        )
    if method != "zero_crossing":
        raise ValueError("unrecognized method " + method)

    if get_lev:
        uas = U[..., find_nearest(lev, 850)]  # type: ignore
    else:
        uas = U

    # make latitude vector monotonically increasing
    if lat[-1] < lat[0]:
        uas = uas[..., ::-1]
        lat = lat[::-1]

    # define latitudes of boundaries certain regions
    eq_boundary = 5
    subpolar_boundary = 30
    polar_boundary = 60
    subpolar_mask = (lat > eq_boundary) & (lat < subpolar_boundary)
    mask = (lat > eq_boundary) & (lat < polar_boundary)

    # get subtropical surface wind minimum
    uas_min_lat = TropD_Calculate_MaxLat(-uas[..., subpolar_mask], lat[subpolar_mask])

    # need to look for zero crossing after subtropical minimum,
    lat_after_Umin = lat[mask] > uas_min_lat[..., None]
    ulat_after_Umin = np.where(lat_after_Umin, uas[..., mask], np.nan)
    Phi = TropD_Calculate_ZeroCrossing(
        ulat_after_Umin, lat[mask], lat_uncertainty=lat_uncertainty
    )

    return Phi


# ========================================
# Stratospheric metrics
# Written by Kasturi Shah - August 3 2020
# converted to python by Alison Ming 8 April 2021
# vectorized and refactored by Sam Smith 19 May 22


@hemisphere_handler
def Shah_2020_GWL(
    tracer: np.ndarray, lat: np.ndarray, zonal_mean_tracer=False
) -> np.ndarray:
    """
    Computes the gradient-weighted latitude (GWL) from tracer data
    Reference: Shah et al., JGR-A, 2020
    https://doi.org/10.1029/2020JD033081

    Parameters
    ==========
    tracer : numpy.ndarray (..., lat [, lon])
        N-dimensional array of tracer data for computing gradient. If
        ``zonal_mean_tracer=False``, the last dimension should correspond to the
        longitude axis, otherwise it should be the latitude axis
    latitude : numpy.ndarray (lat,)
        latitude array in degrees
    zonal_mean_tracer : bool, optional
        whether the input tracer data is zonally symmetric (True) or zonally-varying
        (False) (i.e., has a longitude dimension), by default False

    Returns
    =======
    tracer_steep_lat: numpy.ndarray
        N-2(N-1 for ``zonal_mean_tracer=True``) dimenional array of GWL latitudes in the
        SH
    """

    tracer = np.asarray(tracer)
    lat = np.asarray(lat).flatten()
    if not zonal_mean_tracer:
        tracer = tracer.swapaxes(-2, -1)
    if lat.size != tracer.shape[-1]:
        raise ValueError(
            "input array 'lat' should be aligned with "
            f"{'' if zonal_mean_tracer else 'second to '}last axis of tracer"
        )
    if lat[0] > lat[1]:
        lat = lat[::-1]
        tracer = tracer[..., ::-1]

    phi_lat = np.radians(lat)
    phi_mid_lat = 0.5 * (phi_lat[1:] + phi_lat[:-1])

    # calculating gradients
    grad_weight_lat = np.diff(tracer, axis=-1) / np.diff(phi_lat) * np.cos(phi_mid_lat)

    # array of gradient weighted latitudes (...[,nlon])
    GWL_lat = (phi_mid_lat * grad_weight_lat).sum(axis=-1) / grad_weight_lat.sum(
        axis=-1
    )
    # area equivalent GWL width (in degrees)
    if zonal_mean_tracer:
        tracer_steep_lat = np.degrees(GWL_lat)
    else:
        tracer_steep_lat = np.degrees(np.arcsin(np.nanmean(np.sin(GWL_lat), axis=-1)))

    return tracer_steep_lat


@hemisphere_handler
def Shah_2020_1sigma(
    tracer: np.ndarray,
    lat: np.ndarray,
    zonal_mean_tracer=False,
) -> np.ndarray:
    r"""
    Computes the one-sigma width from 3-D tracer data
    Reference: Shah et al., JGR-A, 2020
    https://doi.org/10.1029/2020JD033081

    Parameters
    ==========
    tracer: numpy.ndarray (...,lat[, lon])
        N-dimensional array of tracer data for computing 1:math:`\sigma`-width. If
        ``zonal_mean_tracer=False``, the last dimension should correspond to the
        longitude axis, otherwise it should be the latitude axis
    latitude: numpy.ndarray (lat,)
        array of latitudes in degrees (1-D). If not increasing, it will be sorted
    zonal_mean_tracer : bool, optional
        whether the input tracer data is zonally symmetric (True) or zonally-varying
        (False) (i.e., has a longitude dimension), by default False

    Returns
    =======
    tracer_sigma_lat: numpy.ndarray
        N-2(N-1 for ``zonal_mean_tracer=True``) dimenional array of
        1:math:`\sigma`-widths in the NH
    """

    tracer = np.atleast_2d(tracer)
    lat = np.asarray(lat).flatten()
    if not zonal_mean_tracer:
        tracer = tracer.swapaxes(-2, -1)
    if lat.size != tracer.shape[-1]:
        raise ValueError(
            "input array 'lat' should be aligned with second to last axis of tracer"
        )
    if lat[0] > lat[1]:
        lat = lat[::-1]
        tracer = tracer[..., ::-1]

    phi_lat = np.broadcast_to(np.radians(lat), tracer.shape)

    # finding ranges of 70 degs latitude with largest tracer values
    nlats_70deg = round(70.0 / (lat[1] - lat[0])) + 1
    tracer_70deg_totals = fftconvolve(
        np.where(np.isfinite(tracer), tracer, 0.0),
        np.ones((tracer.ndim - 1) * (1,) + (nlats_70deg,)),
        "valid",
        axes=-1,
    )
    max_70deg_starts = np.argmax(tracer_70deg_totals, axis=-1)[..., None]
    bands_70deg = (np.arange(lat.size) >= max_70deg_starts) & (
        np.arange(lat.size) < max_70deg_starts + nlats_70deg
    )
    tracer_banded = np.ma.masked_where(~bands_70deg, tracer)
    # mean and std of 70deg bands
    mean_70deg = tracer_banded.mean(axis=-1).filled(0.0)
    std_70deg = tracer_banded.std(axis=-1, ddof=1).filled(0.0)
    threshold = (mean_70deg - std_70deg)[..., None]
    sigma_width_lat = (
        np.ma.masked_where(
            ~((tracer < threshold) & (phi_lat > phi_lat[..., :1])),
            phi_lat,
            copy=False,
        )
        .min(axis=-1)
        .filled(np.nan)
    )
    # area equivalent latitude (in degrees)
    if zonal_mean_tracer:
        tracer_sigma_lat = np.degrees(sigma_width_lat)
    else:
        tracer_sigma_lat = np.degrees(
            np.arcsin(np.nanmean(np.sin(sigma_width_lat), axis=-1))
        )

    return tracer_sigma_lat


