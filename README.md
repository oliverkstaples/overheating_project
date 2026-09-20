# Indoor overheating in English homes (EFUS 2017)

Code for a three-part analysis of indoor overheating using hourly living-room and bedroom
temperature records from the Energy Follow-Up Survey 2017 (EFUS; monitoring May–September
2018), modelled for **London** and the **South West** of England.

1. **Dataset review** – Jupyter notebooks describing the temperature records and dwelling
   sample for each region, room and period.
2. **Models** – hourly linear mixed-effects models of indoor on outdoor temperature (random
   intercept and slope by dwelling; i.i.d. or AR(1) residuals, `nlme`), plus daily
   overheating models (LME, and GEE with an AR(1) working correlation) with EPC band and
   building characteristics.
3. **Climate projection** – the fitted GEE applied to UKCP18 12 km regional RCP8.5 daily
   maximum temperatures (16 ensemble members, to 2080) to project future exceedance.

Overheating is defined with **fixed indoor thresholds** (26, 27 and 28 °C) on occupied
hours: more than 3% of 07:00–21:00 (living room) or 1% of 22:00–06:59 (bedroom). It is not
the adaptive TM52/TM59 criterion. The climate predictor is the 2-day running mean of the
daily maximum outdoor temperature (2DMMT).

## Layout

```
R/         hourly LME scripts (efus_mm_*.R) and daily LME/GEE models (efus_overheating_models.R)
python/    daily dataset builder, model plots, climate projection, extraction scripts
*.ipynb    Part 1 dataset-review notebooks and Part 2 LME diagnostics notebooks
```

Scripts come in two arms (London = Government Office Region 7, South West = region 8), each
for living room and bedroom, for August (`1month`) and May–September (`4month`).

## Setup

```bash
conda env create -f environment.yml    # Python + R
conda activate efus
Rscript -e 'install.packages("clubSandwich", repos = "https://cloud.r-project.org")'  # not on conda-forge
```

Figures use the `scienceplots` "science" style, which needs a LaTeX installation.
Notebooks are stored without outputs; run them to regenerate figures.

## Data (not included)

No data is committed to this repository. To run the pipeline, place the following in the
project root (all are git-ignored):

| File | Contents |
|---|---|
| `efus_indoor_outdoor_livingroom.parquet`, `efus_indoor_outdoor_bedroom.parquet` | Hourly `CaseID, hour, T_in, Location, T_out`, built from the raw EFUS sensor CSVs by `python/efus_extract_*.py` |
| `test_data/ukda_9434_csv_r/csv/selected_interview_responses_caseid.csv` | Per-dwelling characteristics from the EFUS interview data (`dwtype_efus`, `dwage_efus`, `WallType2x_efus`, `InsulatedWalls_efus`, `FullyDblGlz_efus`, `floor6x_efus`, `EPceeb12e_efus`, `gorEHS_efus`, `AnyCooling`) |
| `test_data/RCM/tasmax_rcp85_land-rcm_uk_region_*_day_*.nc` | UKCP18 12 km regional RCP8.5 daily `tasmax`, one file per ensemble member (Met Office) |

EFUS data are available from the UK Data Service under their licence terms.

## Running

Run everything from the project root. Python scripts take `--room livingroom|bedroom`,
`--region london|southwest` and `--subdir august|""`. The R script uses **`--key=value`**
syntax (a space-separated flag is silently ignored). For August, pass `--months 8`;
otherwise the dataset defaults to May–September.

```bash
# Hourly LME models (London, living room, August) -> diagnostics/
Rscript R/efus_mm_london_1month.R

# Daily overheating dataset -> models -> plots (London, living room, August)
python python/efus_overheating_datasets.py --room livingroom --region london --subdir august --months 8
Rscript R/efus_overheating_models.R --room=livingroom --region=london --subdir=august --months=8
python python/efus_overheating_model_plots.py     --room livingroom --region london --subdir august
python python/efus_overheating_impact_profiles.py --room livingroom --region london --subdir august

# Climate projection -> analysis/.../climate_projection.csv and .svg
python python/efus_climate_projection.py --room livingroom --region london --subdir august
```

Outputs are written to `analysis/`, `diagnostics/` and `plots/` (created on first run).

## Known issues

- `python/efus_extract_*.py` and the `*_diagnostics.ipynb` notebooks contain hardcoded
  paths from the original Linux machine (`/home/teaching/...`); edit them before running.
- Only `R/efus_mm_london_1month.R` has been refactored to i.i.d. + AR(1) residuals. The
  other `efus_mm_*` scripts still fit a custom SAR(1,24) correlation structure that was
  found to be mis-converged; treat those SAR results with caution.
- `environment.yml` was rebuilt from the code's imports rather than exported from the
  original environment.

## Status

Research code, work in progress; a report is in preparation.
