#!/usr/bin/env python3
"""
demo_diddle.py
==============
Demonstrate the Taylor et al. (2000) diddling algorithm using synthetic
SST and SIC datasets with cosine climatology, a long-term trend, and noise.

Produces a two-panel figure analogous to Figure 1 of Taylor et al. (2000):

  Top panel    – SST
  Bottom panel – SIC

Each panel shows, over a configurable window (default 3 years):

  • Synthetic "observed" weekly scatter — the underlying signal sampled at 7-day
    intervals with added noise, standing in for real ocean observations
  • Naive method:   daily time series from linearly interpolating between raw
                    monthly mean values placed at mid-month (traditional approach)
  • Diddled method: daily time series from linearly interpolating between the
                    adjusted mid-month values computed by diddle()

Usage
-----
  python demo_diddle.py [--total-years N] [--window-start Y]
                        [--window-years W] [--seed S] [--no-plot]
"""

import argparse
import calendar
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from process_hadisst import diddle, midmonth_offsets, fill_void

# ── Physical bounds (same as process_hadisst) ────────────────────────────────
SST_MIN, SST_MAX = -1.8,  45.0   # °C
SIC_MIN, SIC_MAX =  0.0, 100.0   # %


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def _triangle_pos(phase: np.ndarray, amplitude: float) -> np.ndarray:
    """
    Triangular bump active only in the positive cosine phase (|phase| < π/2).

    Peaks at `amplitude` when phase = 0, falls linearly to 0 at phase = ±π/2,
    and is exactly 0 outside that interval.
    """
    return np.maximum(0.0, amplitude * (1.0 - np.abs(phase) / (np.pi / 2)))


def make_sst_series(years: np.ndarray, months: np.ndarray, rng) -> np.ndarray:
    """
    Cosine annual cycle plus a triangular summer boost, warming trend, and noise.

    Climatology: 20 °C mean, ±8 °C cosine amplitude, peaking in August,
                 plus a ~2 °C triangular boost during the positive phase
                 (May–Nov), zero in the negative phase (Nov–May).
    Trend:       +0.3 °C / decade.
    Noise:       σ = 0.9 °C.
    """
    n = len(years)
    phase = 2.0 * np.pi * (months - 8) / 12.0
    climo = 20.0 + 8.0 * np.cos(phase) + _triangle_pos(phase, amplitude=2.0)
    trend = 0.3 / 120.0 * np.arange(n)
    noise = rng.normal(0.0, 0.9, n)
    return np.clip(climo + trend + noise, SST_MIN, SST_MAX)


def make_sic_series(years: np.ndarray, months: np.ndarray, rng) -> np.ndarray:
    """
    Cosine annual cycle plus a triangular winter boost, declining trend, and noise.

    Climatology: 35 % mean, ±30 % cosine amplitude, peaking in March,
                 plus a ~5 % triangular boost during the positive phase
                 (Dec–Jun), zero in the negative phase (Jun–Dec).
    Trend:       −5 % / decade.
    Noise:       σ = 2 %.
    """
    n = len(years)
    phase = 2.0 * np.pi * (months - 3) / 12.0
    climo = 35.0 + 30.0 * np.cos(phase) + _triangle_pos(phase, amplitude=5.0)
    trend = -5.0 / 120.0 * np.arange(n)
    noise = rng.normal(0.0, 2.0, n)
    return np.clip(climo + trend + noise, SIC_MIN, SIC_MAX)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_daily_series(
    midmonth_vals: np.ndarray,
    midpts: np.ndarray,
    total_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate `midmonth_vals` (placed at `midpts`) onto every
    calendar day in the series.

    Returns (day_centres, values) where day_centres[i] = i + 0.5.
    """
    day_centres = np.arange(0.5, total_days, 1.0)
    values = np.interp(day_centres, midpts, midmonth_vals)
    return day_centres, values


def make_weekly_obs(
    diddled_vals: np.ndarray,
    midpts: np.ndarray,
    total_days: int,
    noise_std: float,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic weekly observations by sampling the diddled daily
    signal at 7-day intervals and adding Gaussian noise.

    Returns (week_day_centres, obs_values).
    """
    week_days = np.arange(3.5, total_days, 7.0)
    signal    = np.interp(week_days, midpts, diddled_vals)
    obs       = signal + rng.normal(0.0, noise_std, len(week_days))
    return week_days, obs


def print_stats(
    label: str,
    monthly: np.ndarray,
    diddled: np.ndarray,
    ndays: np.ndarray,
    midpts: np.ndarray,
    units: str,
) -> None:
    """Print RMSE and max absolute error for both methods."""
    boundaries = np.concatenate([[0.0], np.cumsum(ndays)])
    N = len(ndays)

    def monthly_mean_of_daily(midvals):
        rec = np.empty(N)
        for i in range(N):
            days = np.arange(boundaries[i] + 0.5, boundaries[i + 1], 1.0)
            rec[i] = np.interp(days, midpts, midvals).mean()
        return rec

    naive_rec   = monthly_mean_of_daily(monthly)
    diddled_rec = monthly_mean_of_daily(diddled)

    ne = naive_rec   - monthly
    de = diddled_rec - monthly

    print(f"\n{'─'*52}")
    print(f"  {label}  ({units})")
    print(f"{'─'*52}")
    print(f"  Naive   RMSE={np.sqrt(np.mean(ne**2)):.4f}  max|e|={np.abs(ne).max():.4f}")
    print(f"  Diddled RMSE={np.sqrt(np.mean(de**2)):.2e}  max|e|={np.abs(de).max():.2e}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_taylor_style(
    sst_monthly: np.ndarray,
    sst_diddled: np.ndarray,
    sst_obs_days: np.ndarray,
    sst_obs_vals: np.ndarray,
    sic_monthly: np.ndarray,
    sic_diddled: np.ndarray,
    sic_obs_days: np.ndarray,
    sic_obs_vals: np.ndarray,
    ndays: np.ndarray,
    years: np.ndarray,
    win_start: int,      # first month index of display window (0-based)
    win_n_months: int,   # number of months in window
    start_year: int,
) -> plt.Figure:
    """
    Build the Taylor-style two-panel comparison figure.

    Each panel shows synthetic weekly observations, the naive daily
    interpolation, and the diddled daily interpolation, over `win_n_months`
    months beginning at month index `win_start`.
    """
    midpts = midmonth_offsets(ndays)
    total_days = int(np.sum(ndays))
    boundaries = np.concatenate([[0.0], np.cumsum(ndays)])

    win_end = win_start + win_n_months
    day_lo  = boundaries[win_start]
    day_hi  = boundaries[win_end]

    ref_date = datetime.date(start_year, 1, 1)

    def day_to_date(d):
        return ref_date + datetime.timedelta(days=d)

    cases = [
        ("SST", sst_monthly, sst_diddled, sst_obs_days, sst_obs_vals, "°C"),
        ("SIC", sic_monthly, sic_diddled, sic_obs_days, sic_obs_vals, "%"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle(
        "Synthetic boundary-condition comparison\n"
        "Weekly obs (dots)  |  Naive linear interp (blue dashed)  |  "
        "Diddled interp (red solid)",
        fontsize=10,
    )

    for ax, (label, monthly, diddled, obs_days, obs_vals, units) in zip(axes, cases):
        # ── Daily series ─────────────────────────────────────────────────────
        day_c, naive_daily   = build_daily_series(monthly, midpts, total_days)
        _,     diddled_daily = build_daily_series(diddled, midpts, total_days)

        # ── Mask everything to the display window ────────────────────────────
        win_mask  = (day_c >= day_lo) & (day_c <= day_hi)
        win_days  = np.array([mdates.date2num(day_to_date(d)) for d in day_c[win_mask]])

        obs_mask  = (obs_days >= day_lo) & (obs_days <= day_hi)
        obs_dates = np.array([mdates.date2num(day_to_date(d)) for d in obs_days[obs_mask]])

        # ── Weekly observations ───────────────────────────────────────────────
        ax.scatter(obs_dates, obs_vals[obs_mask],
                   color="black", s=12, zorder=3, label="Weekly observations")

        # ── Interpolated daily curves ─────────────────────────────────────────
        ax.plot(win_days, naive_daily[win_mask],
                color="steelblue", lw=1.2, ls="--", zorder=4,
                label="Traditional (linear interp between monthly means)")
        ax.plot(win_days, diddled_daily[win_mask],
                color="crimson",   lw=1.4, ls="-",  zorder=5,
                label="New method (linear interp between diddled mid-month values)")

        # ── Formatting ────────────────────────────────────────────────────────
        ax.set_ylabel(f"{label} ({units})", fontsize=10)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.set_xlim(
            mdates.date2num(day_to_date(day_lo)),
            mdates.date2num(day_to_date(day_hi)),
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", ls=":", lw=0.5, alpha=0.5)

    axes[0].set_title(
        f"Years {years[win_start]}–{years[win_end - 1]}  "
        f"(synthetic cosine climatology + trend + noise)",
        fontsize=9,
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fill_void visual test
# ---------------------------------------------------------------------------

def demo_fill_void(rng, n_lat: int = 24, n_lon: int = 48) -> plt.Figure:
    """
    Construct a tiny synthetic SST field on a (n_lat × n_lon) grid,
    stamp a 'continent' of NaN cells in the middle, apply fill_void,
    and show before / after side by side.

    The continent covers roughly the central third of both axes.
    A second time step with different noise is included to verify that
    fill_void operates independently per time step.
    """
    # ── Synthetic ocean SST ──────────────────────────────────────────────────
    lat = np.linspace(-90, 90, n_lat)
    lon = np.linspace(0, 360, n_lon, endpoint=False)
    _, lat2d = np.meshgrid(lon, lat)

    # Simple zonal gradient + latitude dependence, two time steps
    base  = 28.0 - 0.3 * np.abs(lat2d)
    noise = rng.normal(0.0, 0.5, (2, n_lat, n_lon))
    data  = np.stack([base, base], axis=0) + noise   # (2, n_lat, n_lon)

    # ── Stamp a fake continent (central ~third) ───────────────────────────────
    r0, r1 = n_lat // 3, 2 * n_lat // 3
    c0, c1 = n_lon // 3, 2 * n_lon // 3
    data[:, r0:r1, c0:c1] = np.nan

    # ── Apply fill_void ───────────────────────────────────────────────────────
    filled = fill_void(data)

    # ── Verify: no NaN remaining, ocean values unchanged ─────────────────────
    assert not np.isnan(filled).any(), "fill_void left NaN cells!"
    ocean_mask = ~np.isnan(data[0])
    np.testing.assert_allclose(
        filled[0][ocean_mask], data[0][ocean_mask],
        err_msg="fill_void modified existing ocean values"
    )
    print("\nfill_void test passed: no NaN remaining, ocean values unchanged.")

    # ── Figure ────────────────────────────────────────────────────────────────
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    cmap = "RdYlBu_r"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        "fill_void test  —  synthetic SST (°C), time step 0\n"
        "Central block set to NaN to simulate a land continent",
        fontsize=10,
    )

    im0 = axes[0].imshow(data[0], origin="lower", aspect="auto",
                         cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Before fill_void  (NaN = white)")

    im1 = axes[1].imshow(filled[0], origin="lower", aspect="auto",
                         cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("After fill_void")

    diff = filled[0] - np.where(np.isnan(data[0]), filled[0], data[0])
    im2 = axes[2].imshow(diff, origin="lower", aspect="auto",
                         cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title("Difference (filled − original)\nzero everywhere except continent")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, label="SST (°C)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="SST (°C)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, label="Δ SST (°C)")

    for ax in axes:
        ax.set_xlabel("Longitude index")
        ax.set_ylabel("Latitude index")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Taylor et al. (2000) diddling demo with synthetic data"
    )
    p.add_argument("--total-years",   type=int, default=20,
                   help="Total years of synthetic data to generate (default: 20)")
    p.add_argument("--window-start",  type=int, default=4,
                   help="Year (0-indexed from series start) to begin the display "
                        "window (default: 4, i.e. 5th year)")
    p.add_argument("--window-years",  type=int, default=3,
                   help="Number of years to show in the figure (default: 3)")
    p.add_argument("--seed",          type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--no-plot",       action="store_true",
                   help="Skip matplotlib figure; only print statistics")
    return p.parse_args()


def main():
    args = parse_args()

    rng      = np.random.default_rng(args.seed)
    n_months = args.total_years * 12
    START    = 2000

    years  = np.array([START + i // 12 for i in range(n_months)])
    months = np.array([1 + i % 12       for i in range(n_months)])

    ndays = np.array(
        [calendar.monthrange(int(y), int(m))[1] for y, m in zip(years, months)],
        dtype=np.float64,
    )

    sst_monthly = make_sst_series(years, months, rng)
    sic_monthly = make_sic_series(years, months, rng)

    midpts = midmonth_offsets(ndays)

    print(f"Synthetic dataset: {args.total_years} years ({n_months} months), seed={args.seed}")

    sst_diddled = diddle(sst_monthly[:, np.newaxis], ndays, SST_MIN, SST_MAX)[:, 0]
    sic_diddled = diddle(sic_monthly[:, np.newaxis], ndays, SIC_MIN, SIC_MAX)[:, 0]

    print_stats("SST", sst_monthly, sst_diddled, ndays, midpts, "°C")
    print_stats("SIC", sic_monthly, sic_diddled, ndays, midpts, "%")

    if args.no_plot:
        return

    total_days = int(np.sum(ndays))
    sst_obs_days, sst_obs_vals = make_weekly_obs(sst_diddled, midpts, total_days, 1.5, rng)
    sic_obs_days, sic_obs_vals = make_weekly_obs(sic_diddled, midpts, total_days, 6.0, rng)
    sic_obs_vals = np.clip(sic_obs_vals, SIC_MIN, SIC_MAX)

    win_start    = args.window_start * 12
    win_n_months = args.window_years * 12

    fig = plot_taylor_style(
        sst_monthly, sst_diddled, sst_obs_days, sst_obs_vals,
        sic_monthly, sic_diddled, sic_obs_days, sic_obs_vals,
        ndays, years,
        win_start=win_start,
        win_n_months=win_n_months,
        start_year=START,
    )

    out = "demo_diddle.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved → {out}")

    fig_fv = demo_fill_void(rng)
    out_fv = "demo_fill_void.png"
    fig_fv.savefig(out_fv, dpi=150)
    print(f"Figure saved → {out_fv}")

    plt.show()


if __name__ == "__main__":
    main()
