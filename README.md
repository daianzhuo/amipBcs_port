# HadISST1 AMIP Boundary Condition Processing

Process HadISST1 SST and SIC into AMIP-format boundary conditions for CMIP
climate model simulations, covering January 1870 – December 2025.

## What this produces

| File | Contents |
|------|----------|
| `tos_HadISST_187001-202512.nc` | Observed monthly-mean SST in Kelvin |
| `siconc_HadISST_187001-202512.nc` | Observed monthly-mean SIC in % |
| `tosbcs_HadISST_187001-202512.nc` | **Diddled** mid-month SST in K (use this for model BCs) |
| `siconcbcs_HadISST_187001-202512.nc` | **Diddled** mid-month SIC in % (use this for model BCs) |

"Diddled" means mid-month values computed by the Taylor et al. (2000)
algorithm so that linear interpolation between consecutive mid-month values
reproduces the observed monthly mean exactly.  Most AMIP-mode climate models
expect this form.

---

## Step 1 – Set up the Python environment

The Python environemnt should have the following prerequisite packages 
installed: `numpy`, `scipy`, `xarray`, `cftime`.  If you prefer an isolated
environment:

```bash
conda env create -f environment.yml
conda activate hadisst
```

---

## Step 2 – Download HadISST1 data

The files are provided as gzip-compressed netCDF by the UK Met Office.
Download both using:

```bash
# SST
curl -O https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz

# Sea-ice concentration
curl -O https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_ice.nc.gz
```

Or with `wget`:

```bash
wget https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz
wget https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_ice.nc.gz
```

The script can read `.nc` files and can handle `.nc.gz` files directly, so 
there is no need to decompress them first. 

---

## Step 3 – Run the processing script

```bash
python process_hadisst.py \
    --sst  HadISST_sst.nc.gz \
    --ice  HadISST_ice.nc.gz \
    --start-year 1870 \
    --end-year   2025 \
    --outdir     ./output
```

All four output files will be written to `./output/`.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sst` | *(required)* | Path to `HadISST_sst.nc[.gz]` |
| `--ice` | *(required)* | Path to `HadISST_ice.nc[.gz]` |
| `--start-year` | `1870` | First year of output |
| `--end-year` | `2025` | Last year of output |
| `--outdir` | `./output` | Output directory (created if absent) |

---

## Step 4 – Verify the output

Quick sanity check with Python:

```python
import xarray as xr

ds = xr.open_dataset("output/tosbcs_HadISST_187001-202512.nc")
print(ds)
print(ds.tosbcs.sel(time="1979-07-15", method="nearest").values.mean())
# Should be roughly 300 K for the global ocean mean in July 1979
```

Or with `ncdump`:

```bash
ncdump -h output/tosbcs_HadISST_187001-202512.nc
```

---

## Algorithm notes

### Diddling (Taylor et al. 2000)

For each calendar month *i* with observed mean *M_i*, the mid-month value
*m_i* is found by solving the tridiagonal system:

    α_i · m_{i-1}  +  3 · m_i  +  β_i · m_{i+1}  =  4 · M_i

where

    α_i = Δ_left  / (Δ_left + Δ_right)
    β_i = Δ_right / (Δ_left + Δ_right)

and Δ_left, Δ_right are the day-distances to the neighbouring mid-month
points.  This guarantees that piecewise-linear interpolation between
consecutive *m_i* integrates to *M_i* over every month.

Twenty-four months of climatological padding are added at each end of the
series before solving, which suppresses boundary artefacts.

Physical limits are enforced by clipping after the solve:

| Variable | Minimum | Maximum |
|----------|---------|---------|
| SST | −1.8 °C (271.35 K) | 45 °C (318.15 K) |
| SIC | 0 % | 100 % |

### Units

| Variable | Input (HadISST) | Output |
|----------|-----------------|--------|
| SST | °C | Kelvin (K) |
| SIC | fraction 0–1 | percent (%) |

---

## References

- Rayner, N.A. et al. (2003). Global analyses of sea surface temperature, sea
  ice, and night marine air temperature since the late nineteenth century.
  *J. Geophys. Res.* 108, 4407. doi:10.1029/2002JD002670

- Taylor, K.E., D. Williamson and F. Zwiers (2000). The sea surface temperature
  and sea ice concentration boundary conditions for AMIP II simulations.
  PCMDI Report No. 60, Lawrence Livermore National Laboratory.
  https://pcmdi.llnl.gov/report/pdf/60.pdf
