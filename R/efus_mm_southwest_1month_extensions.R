#!/usr/bin/env Rscript
# efus_mm_southwest_1month_extensions.R
#
# Extension models ME1–ME7 for the Southwest August subset of EFUS 2017.
# Improves on the M1–M7 static LME in efus_mm_london_1month.R using more
# physically realistic representations of the diurnal temperature cycle.
#
# Physical motivation
# -------------------
# Static models M2/M6/M7 use only the first Fourier harmonic (sin_h, cos_h)
# to represent time-of-day.  The outdoor temperature profile departs from a
# pure sinusoid in two distinct ways:
#
#   1. NON-SINUSOIDAL SHAPE  (2nd Fourier harmonic)
#      The diurnal outdoor temperature cycle is asymmetric: the morning
#      rise is steeper and the evening decline slower than a single
#      sinusoid predicts — following the Parton-Logan (1981, Agric.
#      Meteorol. 23:205-216) family of models.  South African harmonic
#      analysis of diurnal temperature finds that the first two harmonics
#      together account for >98 % of the daily variance (Linacre 1971,
#      S. Afr. Geogr. J. 53:1).  Adding the 2nd harmonic
#          sin(4πt/24),  cos(4πt/24)
#      captures the asymmetry with just 2 extra parameters.
#
#   2. TIME-VARYING OUTDOOR SENSITIVITY  (1st harmonic × T_out interaction)
#      Solar irradiance during daylight hours amplifies the indoor–outdoor
#      coupling through window heat gain; nocturnal thermal decoupling
#      reduces it.  In the ISO 13786 / harmonic thermal-admittance
#      framework this appears as a frequency-dependent transfer function
#      (Campbell & Norman 1998, "An Introduction to Environmental
#      Biophysics").  In the LME it is captured by letting the slope
#      vary sinusoidally:
#          β₁(t) = β₁₀ + γ_s·sin(ωt) + γ_c·cos(ωt)
#      which is equivalent to including the 2 interaction terms
#          sin_T := sin_h × T_out_c,   cos_T := cos_h × T_out_c.
#
# New predictors (added to df_m before fitting)
#   sin2_h  : sin(4π·hour/24)   — 2nd Fourier harmonic
#   cos2_h  : cos(4π·hour/24)
#   sin_T   : sin_h × T_out_c   — time-varying slope (1st harmonic)
#   cos_T   : cos_h × T_out_c
#
# Models
#   ME1 : T_in ~ T_out_c + sin_h + cos_h + sin2_h + cos2_h
#              [M2 + 2nd harmonic;  +2 params vs M2]
#   ME2 : T_in ~ T_out_c + sin_h + cos_h + sin_T + cos_T
#              [M2 + time-varying sensitivity;  +2 params]
#   ME3 : T_in ~ T_out_c + sin_h + cos_h + sin2_h + cos2_h + sin_T + cos_T
#              [M2 + both extensions;  +4 params; best parametric approx to M3]
#   ME4 : ME3 + binary building characteristics       (parallel to M4/M6)
#   ME5 : ME3 + binary + categorical bldg chars       (parallel to M5/M7)
#   ME6 : ME1 + binary building characteristics
#   ME7 : ME1 + binary + categorical bldg chars
#
# Reference models M2 and M3 are re-fitted here for direct ΔAIC comparison.
# All models use the same random structure: random = ~ T_out_c | dwelling.

suppressPackageStartupMessages({
  library(arrow)
  library(nlme)
  library(dplyr)
  library(lubridate)
})

# ── Root path ─────────────────────────────────────────────────────────────────
args        <- commandArgs(trailingOnly = FALSE)
script_file <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_file) == 0 || nchar(script_file) == 0) {
  ROOT <- getwd()
} else {
  ROOT <- normalizePath(file.path(dirname(script_file), ".."))
}

# ── Load data ─────────────────────────────────────────────────────────────────
cat("Loading parquet ...\n")
df <- arrow::read_parquet(file.path(ROOT, "efus_indoor_outdoor_livingroom.parquet"))
df <- as.data.frame(df)

df <- df[df$T_in  >= -10 & df$T_in  <= 40 &
         df$T_out >= -10 & df$T_out <= 40, ]

BLDG_COLS <- c("CaseID", "dwtype_efus", "dwage_efus", "WallType2x_efus",
               "InsulatedWalls_efus", "FullyDblGlz_efus", "floor6x_efus",
               "EPceeb12e_efus", "gorEHS_efus", "AnyCooling")
bldg <- read.csv(
  file.path(ROOT,
            "test_data/ukda_9434_csv_r/csv/selected_interview_responses_caseid.csv"),
  stringsAsFactors = FALSE
)[, BLDG_COLS]

df <- merge(df, bldg, by = "CaseID", all.x = TRUE)
df <- df[complete.cases(df[, BLDG_COLS[-1]]), ]
for (col in BLDG_COLS[-1]) df[[col]] <- as.integer(df[[col]])

df <- df[df$gorEHS_efus == 8, ]
df <- df[lubridate::month(df$hour) == 8L, ]

T_OUT_MEAN     <- mean(df$T_out)
df$T_out_c     <- df$T_out - T_OUT_MEAN
df$hour_of_day <- lubridate::hour(df$hour)

# ── First harmonic (same as M2/M3) ────────────────────────────────────────────
df$sin_h <- sin(2 * pi * df$hour_of_day / 24)
df$cos_h <- cos(2 * pi * df$hour_of_day / 24)

# ── Extension predictors ──────────────────────────────────────────────────────
#
# 2nd Fourier harmonic: captures the asymmetric diurnal shape (steeper morning
# rise, gentler evening decline) that the 1st harmonic misses.
df$sin2_h <- sin(4 * pi * df$hour_of_day / 24)
df$cos2_h <- cos(4 * pi * df$hour_of_day / 24)

# Time-varying slope: encodes sinusoidal modulation of the outdoor sensitivity
#   β₁(t) = β₁₀ + γ_s·sin(ωt) + γ_c·cos(ωt)
# Daytime solar gain strengthens the indoor–outdoor link; night decouples it.
df$sin_T <- df$sin_h * df$T_out_c
df$cos_T <- df$cos_h * df$T_out_c

# ── Subset: dwellings with ≥ 20 obs, in observation order ─────────────────────
n_per_dw <- tapply(df$CaseID, df$CaseID, length)
df_m     <- df[df$CaseID %in% names(n_per_dw)[n_per_dw >= 20], ]
df_m$dwelling <- as.character(df_m$CaseID)
df_m <- df_m[order(df_m$dwelling, df_m$hour), ]
df_m <- droplevels(df_m)

cat(sprintf("Dataset: %d obs, %d dwellings\n\n",
            nrow(df_m), length(unique(df_m$dwelling))))

# ── Building characteristic dummies ──────────────────────────────────────────
df_m$CavityWall <- as.integer(df_m$WallType2x_efus == 2)

BIN <- paste("CavityWall", "InsulatedWalls_efus",
             "FullyDblGlz_efus", "AnyCooling", sep = " + ")
CAT <- paste("factor(dwtype_efus)", "factor(dwage_efus)",
             "factor(floor6x_efus)", "factor(EPceeb12e_efus)", sep = " + ")

# ── Fit helpers ───────────────────────────────────────────────────────────────
ctrl_static <- lmeControl(opt = "optim",  maxIter = 500, msMaxIter = 500,
                          tolerance = 1e-6, niterEM = 50)

fit_static <- function(fixed_formula) {
  lme(fixed   = fixed_formula, data = df_m,
      random  = ~ T_out_c | dwelling,
      method  = "ML", control = ctrl_static)
}

# ── Reference models (M2 and M3 re-fitted for ΔAIC comparison) ───────────────
cat("Fitting M2  (reference: T_out_c + sin_h + cos_h) ...\n")
m2_ref <- fit_static(T_in ~ T_out_c + sin_h + cos_h)
cat(sprintf("  AIC = %.2f\n", AIC(m2_ref)))

cat("Fitting M3  (reference: T_out_c + factor(hour_of_day)) ...\n")
m3_ref <- fit_static(T_in ~ T_out_c + factor(hour_of_day))
cat(sprintf("  AIC = %.2f\n", AIC(m3_ref)))

# ── ME1–ME3: base extension models ────────────────────────────────────────────
cat("\nFitting ME1 (M2 + 2nd harmonic) ...\n")
me1 <- fit_static(T_in ~ T_out_c + sin_h + cos_h + sin2_h + cos2_h)
cat(sprintf("  AIC = %.2f  (ΔAIC vs M2 = %+.2f)\n",
            AIC(me1), AIC(me1) - AIC(m2_ref)))

cat("Fitting ME2 (M2 + time-varying sensitivity) ...\n")
me2 <- fit_static(T_in ~ T_out_c + sin_h + cos_h + sin_T + cos_T)
cat(sprintf("  AIC = %.2f  (ΔAIC vs M2 = %+.2f)\n",
            AIC(me2), AIC(me2) - AIC(m2_ref)))

cat("Fitting ME3 (M2 + 2nd harmonic + time-varying sensitivity) ...\n")
me3 <- fit_static(T_in ~ T_out_c + sin_h + cos_h + sin2_h + cos2_h + sin_T + cos_T)
cat(sprintf("  AIC = %.2f  (ΔAIC vs M2 = %+.2f)\n",
            AIC(me3), AIC(me3) - AIC(m2_ref)))

# ── ME4–ME7: extension models with building characteristics ───────────────────
cat("\nFitting ME4 (ME3 + binary bldg) ...\n")
me4 <- fit_static(as.formula(paste(
  "T_in ~ T_out_c + sin_h + cos_h + sin2_h + cos2_h + sin_T + cos_T +", BIN)))
cat(sprintf("  AIC = %.2f  (ΔAIC vs ME3 = %+.2f)\n",
            AIC(me4), AIC(me4) - AIC(me3)))

cat("Fitting ME5 (ME3 + binary + categorical bldg) ...\n")
me5 <- fit_static(as.formula(paste(
  "T_in ~ T_out_c + sin_h + cos_h + sin2_h + cos2_h + sin_T + cos_T +",
  BIN, "+", CAT)))
cat(sprintf("  AIC = %.2f  (ΔAIC vs ME3 = %+.2f)\n",
            AIC(me5), AIC(me5) - AIC(me3)))

cat("Fitting ME6 (ME1 + binary bldg) ...\n")
me6 <- fit_static(as.formula(paste(
  "T_in ~ T_out_c + sin_h + cos_h + sin2_h + cos2_h +", BIN)))
cat(sprintf("  AIC = %.2f  (ΔAIC vs ME1 = %+.2f)\n",
            AIC(me6), AIC(me6) - AIC(me1)))

cat("Fitting ME7 (ME1 + binary + categorical bldg) ...\n")
me7 <- fit_static(as.formula(paste(
  "T_in ~ T_out_c + sin_h + cos_h + sin2_h + cos2_h +", BIN, "+", CAT)))
cat(sprintf("  AIC = %.2f  (ΔAIC vs ME1 = %+.2f)\n",
            AIC(me7), AIC(me7) - AIC(me1)))

# ── Variance-component extractor ──────────────────────────────────────────────
var_components <- function(mod) {
  re_cov   <- as.matrix(nlme::getVarCov(mod, type = "random.effects"))
  sig2_u0  <- re_cov[1, 1]
  sig2_u1  <- re_cov[2, 2]
  sig_u01  <- re_cov[1, 2]
  sig2_eps <- mod$sigma^2
  list(sig2_u0 = sig2_u0, sig2_u1 = sig2_u1, sig_u01 = sig_u01,
       sig2_eps = sig2_eps, rho = NA_real_, Phi = NA_real_)
}

# ── Summary comparison table ──────────────────────────────────────────────────
all_models <- list(
  M2  = m2_ref,
  M3  = m3_ref,
  ME1 = me1,
  ME2 = me2,
  ME3 = me3,
  ME4 = me4,
  ME5 = me5,
  ME6 = me6,
  ME7 = me7
)
vc_list <- lapply(all_models, var_components)

# ΔAIC reference: M2 for base models; ME3 for ME4/ME5; ME1 for ME6/ME7
delta_ref <- list(
  M2  = NA,     M3  = "M2",
  ME1 = "M2",  ME2 = "M2",  ME3 = "M2",
  ME4 = "ME3", ME5 = "ME3",
  ME6 = "ME1", ME7 = "ME1"
)

cat("\n")
cat(strrep("=", 96), "\n")
cat("MODEL COMPARISON: reference (M2, M3)  |  base extensions (ME1-ME3)  |  + bldg chars (ME4-ME7)\n")
cat(strrep("=", 96), "\n")
cat(sprintf("  %-5s  %12s  %12s  %12s  %8s  %8s  %8s  %6s\n",
            "Model", "log-lik", "AIC", "ΔAIC", "σ_u0", "σ_u1", "σ_ε", "params"))
cat("  ", strrep("-", 90), "\n")

n_fixed <- function(mod) length(fixef(mod))
re_params <- 4L  # σ_u0, σ_u1, corr(u0,u1), σ_ε
for (nm in names(all_models)) {
  mod  <- all_models[[nm]]
  vc   <- vc_list[[nm]]
  ref  <- delta_ref[[nm]]
  daic <- if (!is.null(ref) && !is.na(ref))
            sprintf("%+12.2f", AIC(mod) - AIC(all_models[[ref]]))
          else sprintf("%12s", "—")
  np   <- n_fixed(mod) + re_params
  cat(sprintf("  %-5s  %12.2f  %12.2f  %s  %8.4f  %8.4f  %8.4f  %6d\n",
              nm, as.numeric(logLik(mod)), AIC(mod), daic,
              sqrt(vc$sig2_u0), sqrt(vc$sig2_u1), sqrt(vc$sig2_eps), np))
  if (nm %in% c("M3", "ME3")) cat("  ", strrep("-", 90), "\n")
}
cat(strrep("=", 96), "\n")

# ── Fixed effects tables ──────────────────────────────────────────────────────
pstars <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**",
                      ifelse(p < 0.05, "*", "")))

print_fe <- function(nm, mod) {
  vc   <- vc_list[[nm]]
  corr <- vc$sig_u01 / sqrt(vc$sig2_u0 * vc$sig2_u1)
  fe   <- summary(mod)$tTable

  cat("\n")
  cat(strrep("=", 72), "\n")
  cat(sprintf("FIXED EFFECTS — %s\n", nm))
  cat(strrep("=", 72), "\n")
  cat(sprintf("  %-40s %9s  %8s  %9s\n", "Parameter", "Coef", "SE", "p"))
  cat("  ", strrep("-", 68), "\n")

  hour_rows <- grepl("hour_of_day", rownames(fe))
  for (i in seq_len(nrow(fe))) {
    if (hour_rows[i]) next
    cat(sprintf("  %-40s %9.4f  %8.4f  %9.4f  %s\n",
                rownames(fe)[i],
                fe[i,"Value"], fe[i,"Std.Error"], fe[i,"p-value"],
                pstars(fe[i,"p-value"])))
  }
  if (any(hour_rows)) {
    hr <- fe[hour_rows, "Value"]
    pk <- sub("factor\\(hour_of_day\\)","", rownames(fe)[hour_rows][which.max(hr)])
    tr <- sub("factor\\(hour_of_day\\)","", rownames(fe)[hour_rows][which.min(hr)])
    cat(sprintf("  %-40s  range [%+.3f, %+.3f]  peak h=%s  trough h=%s\n",
                sprintf("factor(hour_of_day) [%d dummies]", sum(hour_rows)),
                min(hr), max(hr), pk, tr))
  }
  cat(sprintf("\n  Random:  σ_u0=%.4f  σ_u1=%.4f  corr(u0,u1)=%+.3f  σ_ε=%.4f\n",
              sqrt(vc$sig2_u0), sqrt(vc$sig2_u1), corr, sqrt(vc$sig2_eps)))
}

for (nm in names(all_models)) print_fe(nm, all_models[[nm]])

# ── Implied peak-hour of the combined diurnal component ───────────────────────
# For models with harmonic terms, compute at which hour the mean indoor
# temperature is highest (marginalising over T_out_c = 0).
implied_peak <- function(mod, nm) {
  fe   <- fixef(mod)
  nms  <- names(fe)
  b_s1 <- if ("sin_h"  %in% nms) fe["sin_h"]  else 0
  b_c1 <- if ("cos_h"  %in% nms) fe["cos_h"]  else 0
  b_s2 <- if ("sin2_h" %in% nms) fe["sin2_h"] else 0
  b_c2 <- if ("cos2_h" %in% nms) fe["cos2_h"] else 0
  hrs  <- 0:23
  diurnal <- b_s1 * sin(2*pi*hrs/24) + b_c1 * cos(2*pi*hrs/24) +
             b_s2 * sin(4*pi*hrs/24) + b_c2 * cos(4*pi*hrs/24)
  pk <- hrs[which.max(diurnal)]
  cat(sprintf("  %-5s : implied diurnal peak at %02d:00  (range %.3f deg C)\n",
              nm, pk, max(diurnal) - min(diurnal)))
}

cat("\nImplied diurnal peak (T_out_c = 0, fixed effects only):\n")
for (nm in names(all_models)) implied_peak(all_models[[nm]], nm)

# ── Export diagnostics ─────────────────────────────────────────────────────────
cat("\nExporting diagnostics...\n")
diag_dir <- file.path(ROOT, "diagnostics")
dir.create(diag_dir, showWarnings = FALSE)

diag_df <- df_m[, c("dwelling", "hour", "T_in", "T_out", "T_out_c",
                     "hour_of_day", "sin_h", "cos_h", "sin2_h", "cos2_h",
                     "sin_T", "cos_T")]
for (nm in names(all_models)) {
  mod <- all_models[[nm]]
  diag_df[[paste0("fitted_", nm)]] <- as.numeric(fitted(mod))
  diag_df[[paste0("resid_",  nm)]] <- as.numeric(residuals(mod, type = "response"))
  diag_df[[paste0("nresid_", nm)]] <- as.numeric(residuals(mod, type = "normalized"))
}
arrow::write_parquet(diag_df,
  file.path(diag_dir, "southwest_august_extensions_diagnostics.parquet"))

ranef_df <- data.frame(dwelling = rownames(ranef(me1)))
for (nm in names(all_models)) {
  re <- ranef(all_models[[nm]])
  ranef_df[[paste0("u0_", nm)]] <- re[rownames(ranef_df), 1]
  ranef_df[[paste0("u1_", nm)]] <- if (ncol(re) >= 2) re[rownames(ranef_df), 2] else NA_real_
}
write.csv(ranef_df,
  file.path(diag_dir, "southwest_august_extensions_ranef.csv"), row.names = FALSE)

aic_df <- data.frame(
  model  = names(all_models),
  aic    = sapply(all_models, AIC),
  loglik = sapply(all_models, function(m) as.numeric(logLik(m)))
)
write.csv(aic_df,
  file.path(diag_dir, "southwest_august_extensions_aic.csv"), row.names = FALSE)

vc_df <- data.frame(
  model   = names(all_models),
  sig_u0  = sapply(vc_list, function(v) sqrt(v$sig2_u0)),
  sig_u1  = sapply(vc_list, function(v) sqrt(v$sig2_u1)),
  sig_eps = sapply(vc_list, function(v) sqrt(v$sig2_eps)),
  rho     = NA_real_,
  phi     = NA_real_
)
write.csv(vc_df,
  file.path(diag_dir, "southwest_august_extensions_varcomp.csv"), row.names = FALSE)

cat("  -> diagnostics/southwest_august_extensions_diagnostics.parquet\n")
cat("  -> diagnostics/southwest_august_extensions_ranef.csv\n")
cat("  -> diagnostics/southwest_august_extensions_aic.csv\n")
cat("  -> diagnostics/southwest_august_extensions_varcomp.csv\n")

cat("\nDone.\n")
