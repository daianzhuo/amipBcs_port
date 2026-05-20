#!/usr/bin/env python3
"""
process_hadisst.py
==================
Process HadISST1 SST and SIC data to produce AMIP boundary conditions.

Two outputs per variable are written:
  - Observed monthly means   (tos / siconc)
  - Diddled mid-month values (tosbcs / siconcbcs)

The "diddling" step (Taylor et al. 2000) computes mid-month values m_i such
that linear interpolation between consecutive m_i exactly reproduces the
observed monthly mean when integrated over each calendar month.  Climate
models read the diddled files and interpolate in time to obtain daily (or
sub-daily) SST / SIC.

Reference
---------
Taylor, K.E., D. Williamson and F. Zwiers (2000).  The sea surface temperature
and sea ice concentration boundary conditions for AMIP II simulations.
PCMDI Report No. 60, Lawrence Livermore National Laboratory.
https://pcmdi.llnl.gov/report/pdf/60.pdf

Usage
-----
  python process_hadisst.py \\
      --sst  HadISST_sst.nc[.gz]  \\
      --ice  HadISST_ice.nc[.gz]  \\
      [--start-year 1870]          \\
      [--end-year   2025]          \\
      [--outdir     ./output]

Output files (CF-compliant netCDF-4):
  tos_HadISST_187001-202512.nc        monthly-mean SST (degC)
  siconc_HadISST_187001-202512.nc     monthly-mean SIC (%)
  tosbcs_HadISST_187001-202512.nc     mid-month diddled SST (degC)
  siconcbcs_HadISST_187001-202512.nc  mid-month diddled SIC (%)

Dependencies: numpy, scipy, xarray, cftime  (all in base conda)
"""

import argparse
import calendar
import datetime
import gzip
import os
import shutil
import tempfile
import warnings

import cftime
import numpy as np
import xarray as xr
from scipy.linalg import solve_banded

# ---------------------------------------------------------------------------
# Physical constants and defaults
# ---------------------------------------------------------------------------
SST_MIN_C   = -1.8          # Minimum physically plausible SST (°C)
SST_MAX_C   = 45.0          # Maximum physically plausible SST (°C)
SIC_MIN_PCT = 0.0           # Minimum SIC (%)
SIC_MAX_PCT = 100.0         # Maximum SIC (%)
HADISST_FILL = -1000.0      # Sentinel used in HadISST for land / missing
OUTPUT_FILL  = 1.0e20       # Fill value written to output files
NPAD         = 24           # Months of climatological padding for diddling


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def open_nc(path: str) -> xr.Dataset:
    """
    Open a netCDF file, transparently decompressing .gz archives first.
    The temporary file (if created) is cleaned up automatically.
    """
    if path.endswith(".gz"):
        tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        try:
            with gzip.open(path, "rb") as f_in:
                shutil.copyfileobj(f_in, tmp)
            tmp.close()
            ds = xr.open_dataset(tmp.name, use_cftime=True)
        finally:
            # Keep open for xarray lazy loading; register cleanup at exit
            import atexit
            atexit.register(os.unlink, tmp.name)
        return ds
    return xr.open_dataset(path, use_cftime=True)


def subset_time(ds: xr.Dataset, start_year: int, end_year: int) -> xr.Dataset:
    """Select months from start_year-01 through end_year-12."""
    t = ds["time"]
    mask = (t.dt.year >= start_year) & (t.dt.year <= end_year)
    n = int(mask.sum())
    if n == 0:
        raise ValueError(
            f"No data found for {start_year}-{end_year} in this file."
        )
    expected = (end_year - start_year + 1) * 12
    if n < expected:
        warnings.warn(
            f"Requested {expected} months but only {n} are available. "
            "Proceeding with available data."
        )
    return ds.sel(time=mask)


def months_ndays(time_coord: xr.DataArray) -> np.ndarray:
    """Return the number of days in each month of the time coordinate."""
    years  = time_coord.dt.year.values
    months = time_coord.dt.month.values
    return np.array(
        [calendar.monthrange(int(y), int(m))[1] for y, m in zip(years, months)],
        dtype=np.float64,
    )


def midmonth_offsets(ndays: np.ndarray) -> np.ndarray:
    """
    Cumulative day offset to the centre of each month.

    offset[i] = sum(ndays[0:i]) + ndays[i] / 2
    """
    cum = np.concatenate([[0.0], np.cumsum(ndays)])
    return cum[:-1] + 0.5 * ndays


# ---------------------------------------------------------------------------
# Diddling: Taylor et al. (2000) tridiagonal algorithm
# ---------------------------------------------------------------------------

def _make_banded_matrix(ndays: np.ndarray) -> np.ndarray:
    """
    Build the scipy banded-matrix for the diddling linear system.

    For N months the system is (i = 0 … N-1):

        alpha_i * m_{i-1} + 3 * m_i + beta_i * m_{i+1} = 4 * M_i

    where, with delta = midmonth_offsets(ndays):

        d_left  = delta[i] - delta[i-1]
        d_right = delta[i+1] - delta[i]
        alpha_i = d_left  / (d_left + d_right)
        beta_i  = d_right / (d_left + d_right)

    Neumann (zero-gradient) boundary conditions are applied at both ends:
        i = 0  :  set d_left[0]  = d_right[0]  => alpha_0 = 0.5
                  fold into diagonal: diag[0] += alpha_0
        i = N-1:  set d_right[-1] = d_left[-1] => beta_{N-1} = 0.5
                  fold into diagonal: diag[-1] += beta_{N-1}

    Returns ab (shape 3 x N) in scipy solve_banded format:
        ab[0, j] = A[j-1, j]   (super-diagonal; ab[0,0] unused)
        ab[1, j] = A[j, j]     (main diagonal)
        ab[2, j] = A[j+1, j]   (sub-diagonal;  ab[2,-1] unused)
    """
    N = len(ndays)
    delta = midmonth_offsets(ndays)

    # Distances between consecutive mid-month points
    d = np.diff(delta)                # length N-1

    d_left  = np.empty(N)
    d_right = np.empty(N)
    d_left[1:]  = d       # d_left[i]  = delta[i] - delta[i-1]   for i >= 1
    d_right[:-1] = d      # d_right[i] = delta[i+1] - delta[i]   for i <= N-2

    # Neumann BC: mirror the neighbour distance at each boundary
    d_left[0]   = d_right[0]
    d_right[-1] = d_left[-1]

    d_sum  = d_left + d_right
    alpha  = d_left  / d_sum    # coefficient on m_{i-1} in equation i
    beta   = d_right / d_sum    # coefficient on m_{i+1} in equation i

    ab = np.zeros((3, N))
    ab[1, :] = 3.0
    ab[1, 0]  += alpha[0]    # Neumann BC left
    ab[1, -1] += beta[-1]    # Neumann BC right

    # super-diagonal: ab[0, j] = beta[j-1]  for j = 1 … N-1
    ab[0, 1:] = beta[:-1]
    # sub-diagonal:   ab[2, j] = alpha[j+1] for j = 0 … N-2
    ab[2, :-1] = alpha[1:]

    return ab


def _climo_pad(data: np.ndarray, npad: int = NPAD) -> np.ndarray:
    """
    Prepend and append climatological padding to a (time, ...) array.

    Padding months are taken from the mean annual cycle computed over the
    first / last min(2 years, half the series) of data.
    """
    N = data.shape[0]
    assert N % 12 == 0, "Data length must be a multiple of 12 for padding."
    n_use = min(24, N // 2)   # months used to build the climatology (max 2 yrs)

    def _monthly_climo(chunk):
        # chunk shape: (n_use, ...), n_use must be divisible by 12
        assert chunk.shape[0] % 12 == 0
        ny = chunk.shape[0] // 12
        return chunk.reshape(ny, 12, *chunk.shape[1:]).mean(axis=0)  # (12, ...)

    start_climo = _monthly_climo(data[:n_use])   # (12, ...)
    end_climo   = _monthly_climo(data[-n_use:])  # (12, ...)

    n_reps = int(np.ceil(npad / 12))
    start_pad = np.tile(start_climo, [n_reps] + [1] * (data.ndim - 1))[:npad]
    end_pad   = np.tile(end_climo,   [n_reps] + [1] * (data.ndim - 1))[:npad]

    return np.concatenate([start_pad, data, end_pad], axis=0)


def _ndays_pad(ndays: np.ndarray, npad: int = NPAD) -> np.ndarray:
    """
    Extend ndays array with padding months that mirror the first / last months.
    (Used so the banded-matrix is built with realistic month lengths.)
    """
    assert len(ndays) % 12 == 0
    start_pad = np.tile(ndays[:12], int(np.ceil(npad / 12)))[:npad]
    end_pad   = np.tile(ndays[-12:], int(np.ceil(npad / 12)))[:npad]
    return np.concatenate([start_pad, ndays, end_pad])


def diddle(
    monthly: np.ndarray,
    ndays:   np.ndarray,
    vmin:    float,
    vmax:    float,
    npad:    int = NPAD,
) -> np.ndarray:
    """
    Apply the Taylor et al. (2000) diddling algorithm to a (time, space)
    array of monthly means.

    Efficiently solves the tridiagonal system for all spatial columns
    simultaneously using scipy.linalg.solve_banded.

    Parameters
    ----------
    monthly : ndarray, shape (N_time, N_space)
        Monthly mean values.  Land / missing columns should be masked out
        before calling (they are not touched here).
    ndays   : ndarray, shape (N_time,)
        Number of days in each month.
    vmin, vmax : float
        Physical minimum / maximum for clipping (after solving).
    npad    : int
        Months of climatological padding added to each end to reduce
        boundary effects.  Interior results are then extracted.

    Returns
    -------
    midmonth : ndarray, shape (N_time, N_space)
        Mid-month values suitable for model boundary conditions.
    """
    N_time, N_space = monthly.shape

    # 1. Add climatological padding
    padded_data  = _climo_pad(monthly, npad)   # (N_time + 2*npad, N_space)
    padded_ndays = _ndays_pad(ndays,   npad)   # (N_time + 2*npad,)

    # 2. Build the banded matrix (same coefficient structure for all columns)
    ab = _make_banded_matrix(padded_ndays)

    # 3. RHS = 4 * M
    rhs = 4.0 * padded_data   # (N_time + 2*npad, N_space)

    # 4. Solve all spatial columns simultaneously
    #    solve_banded accepts b of shape (N, K)
    midmonth_padded = solve_banded((1, 1), ab, rhs)   # (N_time + 2*npad, N_space)

    # 5. Extract interior (remove padding)
    midmonth = midmonth_padded[npad: npad + N_time, :]

    # 6. Apply physical constraints
    midmonth = np.clip(midmonth, vmin, vmax)

    return midmonth


# ---------------------------------------------------------------------------
# Variable-specific processing
# ---------------------------------------------------------------------------

def process_variable(
    da:         xr.DataArray,
    ndays:      np.ndarray,
    var_id:     str,          # "tos" or "siconc"
    out_dir:    str,
    period_str: str,
) -> None:
    """
    Process one HadISST variable (SST or SIC):
      1. Clean fill values / apply physical limits
      2. Convert units to CMIP convention
      3. Diddle to produce mid-month values
      4. Write both obs (monthly mean) and bcs (mid-month) netCDF files

    Parameters
    ----------
    da        : xarray DataArray with dims (time, latitude, longitude)
    ndays     : days-per-month array matching da.time
    var_id    : "tos" or "siconc"
    out_dir   : directory for output files
    period_str: e.g. "187001-202512"
    """
    assert var_id in ("tos", "siconc")
    is_sst = (var_id == "tos")

    # ── 1. Identify land / missing mask ──────────────────────────────────
    # HadISST uses -1000 as a sentinel for land / permanent ice / missing.
    # We build a 2-D mask from the first time step (land does not move).
    first = da.isel(time=0).values
    land_mask = np.isnan(first) | (first < HADISST_FILL * 0.5)  # True where land/missing

    # Replace HadISST fill with NaN throughout
    data = da.values.copy().astype(np.float64)
    data[data < HADISST_FILL * 0.5] = np.nan

    N_time, N_lat, N_lon = data.shape
    N_space = N_lat * N_lon

    # ── 2. Unit conversion & physical limits ─────────────────────────────
    if is_sst:
        # HadISST SST is in °C; output stays in °C (CMIP/AMIP convention for tosbcs)
        data = np.where(np.isnan(data), np.nan,
                        np.clip(data, SST_MIN_C, SST_MAX_C))
        vmin, vmax = SST_MIN_C, SST_MAX_C
        units_out  = "degC"
        long_name_obs = "Sea Surface Temperature"
        long_name_bcs = "Constructed mid-month Sea Surface Temperature"
        std_name_obs  = "sea_surface_temperature"
        std_name_bcs  = "sea_surface_temperature"
        bcs_id      = "tosbcs"
        sic_comment = None
        data_proc   = data
    else:
        # HadISST SIC is a fraction [0, 1]; CMIP convention is percent
        # Values between 0 and 1 are valid; -1000 = missing; some files
        # encode SIC slightly outside [0,1] due to interpolation artefacts.
        data_pct = np.where(np.isnan(data), np.nan, data * 100.0)
        data_pct = np.where(np.isnan(data_pct), np.nan,
                            np.clip(data_pct, SIC_MIN_PCT, SIC_MAX_PCT))
        vmin, vmax = SIC_MIN_PCT, SIC_MAX_PCT
        units_out  = "%"
        long_name_obs = "Sea Ice Area Fraction"
        long_name_bcs = "Constructed mid-month Sea-ice area fraction"
        std_name_obs  = "sea_ice_area_fraction"
        std_name_bcs  = "sea_ice_area_fraction"
        sic_comment   = "Percentage of grid cell covered by sea ice"
        bcs_id    = "siconcbcs"
        data_proc = data_pct

    # ── 3. Diddle: solve for mid-month values ─────────────────────────────
    print(f"  Diddling {var_id} ({N_time} months × {N_lat}×{N_lon} grid) …")

    # Flatten to (time, space); replace NaN (land) with 0 for the solver,
    # then mask out after.
    flat = data_proc.reshape(N_time, N_space)
    land_flat = land_mask.reshape(N_space)
    ocean_pts = ~land_flat

    # Solve only for ocean columns (avoids propagating fill values)
    if ocean_pts.any():
        ocean_data = flat[:, ocean_pts]
        # Some coastal / sea-ice points may still have isolated NaNs;
        # fill those with the time-mean of that point before diddling.
        col_means = np.nanmean(ocean_data, axis=0)
        # Some columns (e.g. semi-permanent sea-ice) are all-NaN; fall back to 0
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask  = np.isnan(ocean_data)
        ocean_data = np.where(nan_mask, col_means[np.newaxis, :], ocean_data)
        midmonth_ocean = diddle(ocean_data, ndays, vmin, vmax)
    else:
        raise RuntimeError("No ocean points found in the data!")

    # Reconstruct full grid
    midmonth_flat = np.full((N_time, N_space), OUTPUT_FILL, dtype=np.float32)
    midmonth_flat[:, ocean_pts] = midmonth_ocean.astype(np.float32)
    midmonth = midmonth_flat.reshape(N_time, N_lat, N_lon)

    # Also reconstruct obs on full grid (NaN -> fill)
    obs_full = np.where(np.isnan(data_proc), OUTPUT_FILL, data_proc).astype(np.float32)

    # ── 4. Build time coordinates ─────────────────────────────────────────
    # Mid-month time: exact centroid = month_start + ndays/2
    # e.g. January (31 days) → Jan 16 12:00 UTC  (matches CMOR/PCMDI convention)
    years  = da.time.dt.year.values
    months = da.time.dt.month.values
    mid_times = np.array([
        cftime.DatetimeGregorian(int(y), int(m), 1)
        + datetime.timedelta(days=float(nd) / 2.0)
        for y, m, nd in zip(years, months, ndays)
    ])

    # Month-start and month-end for time_bnds
    t_starts = np.array([
        cftime.DatetimeGregorian(int(y), int(m), 1)
        for y, m in zip(years, months)
    ])
    t_ends = np.array([
        cftime.DatetimeGregorian(
            int(y) + (1 if int(m) == 12 else 0),
            1 if int(m) == 12 else int(m) + 1,
            1,
        )
        for y, m in zip(years, months)
    ])
    time_bnds_vals = np.stack([t_starts, t_ends], axis=1)   # (N_time, 2)

    # ── 5. Write output files ─────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    lat_coord = da["lat"]
    lon_coord = da["lon"]
    lat_name  = "lat"
    lon_name  = "lon"

    # Build lat/lon bounds if not already present
    def _make_1d_bnds(coord):
        vals = coord.values
        d = np.diff(vals)
        d_ext = np.concatenate([[d[0]], d, [d[-1]]])
        lo = vals - 0.5 * d_ext[:-1]
        hi = vals + 0.5 * d_ext[1:]
        return np.stack([lo, hi], axis=1)

    def _get_or_make_bnds(coord, ds):
        bnds_name = coord.attrs.get("bounds")
        if bnds_name and bnds_name in ds:
            return ds[bnds_name].values
        return _make_1d_bnds(coord)

    lat_bnds = _get_or_make_bnds(lat_coord, da)
    lon_bnds = _get_or_make_bnds(lon_coord, da)

    history_str = (
        f"Created {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} "
        f"by process_hadisst.py from HadISST1"
    )

    global_attrs = {
        "title": "HadISST1 AMIP boundary conditions",
        "source": "UK Met Office Hadley Centre HadISST version 1.1",
        "institution": "Met Office Hadley Centre",
        "references": (
            "Rayner et al. (2003), doi:10.1029/2002JD002670; "
            "Taylor et al. (2000) PCMDI Report 60"
        ),
        "history": history_str,
        "Conventions": "CF-1.9",
        "frequency": "mon",
        "mip_era": "CMIP7",
    }

    def _write_nc(data_arr, time_arr, t_bnds, vid, lname, sname, obs_or_bcs, extra_attrs=None):
        fname = f"{vid}_HadISST_{period_str}.nc"
        fpath = os.path.join(out_dir, fname)

        var_attrs = {
            "standard_name": sname,
            "long_name": lname,
            "units": units_out,
            "cell_methods": (
                "time: mean (interval: 1 month)" if obs_or_bcs == "obs"
                else "time: point"
            ),
            "cell_measures": "area: areacello",
        }
        if extra_attrs:
            var_attrs.update(extra_attrs)

        ds_out = xr.Dataset(
            {
                vid: xr.DataArray(
                    data_arr,
                    dims=["time", lat_name, lon_name],
                    attrs=var_attrs,
                ),
                "time_bnds": xr.DataArray(
                    t_bnds,
                    dims=["time", "bnds"],
                ),
                f"{lat_name}_bnds": xr.DataArray(
                    lat_bnds, dims=[lat_name, "bnds"]
                ),
                f"{lon_name}_bnds": xr.DataArray(
                    lon_bnds, dims=[lon_name, "bnds"]
                ),
            },
            coords={
                "time": xr.DataArray(
                    time_arr,
                    dims=["time"],
                    attrs={
                        "standard_name": "time",
                        "long_name": "time",
                        "bounds": "time_bnds",
                        "axis": "T",
                    },
                ),
                lat_name: lat_coord.assign_attrs({
                    "bounds": f"{lat_name}_bnds",
                    "axis": "Y",
                }),
                lon_name: lon_coord.assign_attrs({
                    "bounds": f"{lon_name}_bnds",
                    "axis": "X",
                }),
            },
            attrs=global_attrs,
        )

        encoding = {
            vid: {"dtype": "float32", "zlib": True, "complevel": 4, "_FillValue": OUTPUT_FILL},
            "time": {"units": "days since 1870-01-01", "calendar": "gregorian", "dtype": "float64"},
            "time_bnds": {"units": "days since 1870-01-01", "calendar": "gregorian", "dtype": "float64"},
            f"{lat_name}_bnds": {"dtype": "float64"},
            f"{lon_name}_bnds": {"dtype": "float64"},
        }
        ds_out.to_netcdf(fpath, encoding=encoding)
        print(f"  Written: {fpath}")

    extra = {"comment": sic_comment} if not is_sst else None

    # obs: monthly means, time at mid-month, with time_bnds
    _write_nc(obs_full,  mid_times, time_bnds_vals,
              var_id, long_name_obs, std_name_obs, "obs", extra_attrs=extra)

    # bcs: mid-month diddled values; time bounds span the same calendar month
    _write_nc(midmonth,  mid_times, time_bnds_vals,
              bcs_id, long_name_bcs, std_name_bcs, "bcs", extra_attrs=extra)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Process HadISST1 SST and SIC for AMIP boundary conditions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sst",        required=True, help="Path to HadISST_sst.nc[.gz]")
    p.add_argument("--ice",        required=True, help="Path to HadISST_ice.nc[.gz]")
    p.add_argument("--start-year", type=int, default=1870,
                   help="First year to process")
    p.add_argument("--end-year",   type=int, default=2025,
                   help="Last year to process")
    p.add_argument("--outdir",     default="./output",
                   help="Directory for output files")
    return p.parse_args()


def main():
    args = parse_args()

    period_str = f"{args.start_year}01-{args.end_year}12"
    print(f"Processing HadISST1  {args.start_year}-01 to {args.end_year}-12")
    print(f"Output directory: {args.outdir}")

    # ── Load SST ──────────────────────────────────────────────────────────
    print("\nLoading SST …")
    ds_sst = open_nc(args.sst)

    # Detect variable name (HadISST uses 'sst')
    sst_var = next(
        (v for v in ds_sst.data_vars if "sst" in v.lower() or "tos" in v.lower()),
        list(ds_sst.data_vars)[0],
    )
    print(f"  SST variable: {sst_var!r}")
    ds_sst = subset_time(ds_sst, args.start_year, args.end_year)
    da_sst = ds_sst[sst_var]

    # Normalise dimension names to lat / lon
    rename = {}
    for old in da_sst.dims:
        if old.lower() in ("lat", "latitude"):
            rename[old] = "lat"
        elif old.lower() in ("lon", "longitude"):
            rename[old] = "lon"
    if rename:
        da_sst = da_sst.rename(rename)

    # Ensure lat is ascending (S→N) to match CMIP convention
    if da_sst["lat"].values[0] > da_sst["lat"].values[-1]:
        da_sst = da_sst.isel(lat=slice(None, None, -1))

    # Ensure lon is in 0–360 range (reorder if stored as −180–180)
    if da_sst["lon"].values[0] < 0:
        da_sst = da_sst.assign_coords(lon=(da_sst["lon"] % 360)).sortby("lon")

    ndays = months_ndays(da_sst.time)
    N_time = len(ndays)
    if N_time % 12 != 0:
        # Trim to complete years
        N_trim = (N_time // 12) * 12
        warnings.warn(
            f"Data length {N_time} is not a multiple of 12. "
            f"Trimming to {N_trim} months."
        )
        da_sst = da_sst.isel(time=slice(0, N_trim))
        ndays  = ndays[:N_trim]

    print(f"  {len(ndays)} months loaded")

    # ── Load SIC ──────────────────────────────────────────────────────────
    print("\nLoading SIC …")
    ds_ice = open_nc(args.ice)

    ice_var = next(
        (v for v in ds_ice.data_vars
         if any(k in v.lower() for k in ("sic", "ice", "siconc"))),
        list(ds_ice.data_vars)[0],
    )
    print(f"  SIC variable: {ice_var!r}")
    ds_ice = subset_time(ds_ice, args.start_year, args.end_year)
    da_ice = ds_ice[ice_var]

    rename = {}
    for old in da_ice.dims:
        if old.lower() in ("lat", "latitude"):
            rename[old] = "lat"
        elif old.lower() in ("lon", "longitude"):
            rename[old] = "lon"
    if rename:
        da_ice = da_ice.rename(rename)

    if da_ice["lat"].values[0] > da_ice["lat"].values[-1]:
        da_ice = da_ice.isel(lat=slice(None, None, -1))

    if da_ice["lon"].values[0] < 0:
        da_ice = da_ice.assign_coords(lon=(da_ice["lon"] % 360)).sortby("lon")

    # Match length to SST (both cover the same period)
    if len(da_ice.time) > len(da_sst.time):
        da_ice = da_ice.isel(time=slice(0, len(da_sst.time)))

    # ── Process SST ───────────────────────────────────────────────────────
    print("\nProcessing SST (tos / tosbcs) …")
    process_variable(da_sst, ndays, "tos",    args.outdir, period_str)

    # ── Process SIC ───────────────────────────────────────────────────────
    print("\nProcessing SIC (siconc / siconcbcs) …")
    process_variable(da_ice, ndays, "siconc", args.outdir, period_str)

    print("\nDone.")


if __name__ == "__main__":
    main()
