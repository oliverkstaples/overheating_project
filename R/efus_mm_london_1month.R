#!/usr/bin/env Rscript
# efus_mm_london_1month.R
#
# Fits static LME (M1-M7) and AR(1) LME (M8-M14) on the London August subset
# of EFUS 2017.
#
# Models
#   M1  : T_in ~ T_out_c                        (random intercept + slope)
#   M2  : T_in ~ T_out_c + sin_h + cos_h
#   M3  : T_in ~ T_out_c + factor(hour_of_day)
#   M4  : M3 + binary building characteristics  (CavityWall, InsulatedWalls, FullyDblGlz, AnyCooling)
#   M5  : M4 + categorical building characteristics (dwtype, dwage, floor6x, EPceeb12e)
#   M6  : M2 + binary building characteristics
#   M7  : M6 + categorical building characteristics
#   M8  : M1 + AR(1) residuals        a_t = rho*a_{t-1} + eta_t
#   M9  : M2 + AR(1) residuals
#   M10 : M3 + AR(1) residuals
#   M11 : M4 + AR(1) residuals        (hour dummies + binary)
#   M12 : M5 + AR(1) residuals        (hour dummies + binary + categorical)
#   M13 : M6 + AR(1) residuals        (sin/cos + binary)
#   M14 : M7 + AR(1) residuals        (sin/cos + binary + categorical)

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

df <- df[df$gorEHS_efus == 7, ]
df <- df[lubridate::month(df$hour) == 8L, ]

T_OUT_MEAN      <- mean(df$T_out)
df$T_out_c      <- df$T_out - T_OUT_MEAN
df$hour_of_day  <- lubridate::hour(df$hour)
df$sin_h        <- sin(2 * pi * df$hour_of_day / 24)
df$cos_h        <- cos(2 * pi * df$hour_of_day / 24)


n_per_dw <- tapply(df$CaseID, df$CaseID, length)
df_m     <- df[df$CaseID %in% names(n_per_dw)[n_per_dw >= 20], ]
df_m$dwelling <- as.character(df_m$CaseID)
df_m <- df_m[order(df_m$dwelling, df_m$hour), ]
df_m <- droplevels(df_m)

cat(sprintf("Dataset: %d obs, %d dwellings\n\n",
            nrow(df_m), length(unique(df_m$dwelling))))

# ── Fit helpers ───────────────────────────────────────────────────────────────
ctrl_static <- lmeControl(opt = "optim",  maxIter = 500, msMaxIter = 500,
                          tolerance = 1e-6, niterEM = 50)
ctrl_ar1    <- lmeControl(opt = "nlminb", maxIter = 500, msMaxIter = 500,
                          tolerance = 1e-6, niterEM = 50)

fit_static <- function(fixed_formula) {
  lme(fixed   = fixed_formula, data = df_m,
      random  = ~ T_out_c | dwelling,
      method  = "ML", control = ctrl_static)
}

fit_ar1 <- function(fixed_formula) {
  lme(fixed       = fixed_formula, data = df_m,
      random      = ~ T_out_c | dwelling,
      correlation = corAR1(form = ~ 1 | dwelling),
      method      = "ML", control = ctrl_ar1)
}

# ── Constant-only null models (T_in ~ 1, same random + correlation structure) ──
cat("Fitting M0_static (null, static) ...\n"); m0_static <- fit_static(T_in ~ 1)
cat(sprintf("  AIC = %.2f\n", AIC(m0_static)))

# ── M1-M3 ─────────────────────────────────────────────────────────────────────
cat("Fitting M1 ...\n"); m1 <- fit_static(T_in ~ T_out_c)
cat(sprintf("  AIC = %.2f\n", AIC(m1)))
cat("Fitting M2 ...\n"); m2 <- fit_static(T_in ~ T_out_c + sin_h + cos_h)
cat(sprintf("  AIC = %.2f\n", AIC(m2)))
cat("Fitting M3 ...\n"); m3 <- fit_static(T_in ~ T_out_c + factor(hour_of_day))
cat(sprintf("  AIC = %.2f\n", AIC(m3)))

# ── Building characteristic variables ─────────────────────────────────────────
df_m$CavityWall <- as.integer(df_m$WallType2x_efus == 2)

BIN <- paste("CavityWall", "InsulatedWalls_efus",
             "FullyDblGlz_efus", "AnyCooling", sep = " + ")
CAT <- paste("factor(dwtype_efus)", "factor(dwage_efus)",
             "factor(floor6x_efus)", "factor(EPceeb12e_efus)", sep = " + ")

# ── M4-M7: static + building characteristics ──────────────────────────────────
cat("\nFitting M4  (M3 + binary bldg) ...\n")
m4 <- fit_static(as.formula(paste("T_in ~ T_out_c + factor(hour_of_day) +", BIN)))
cat(sprintf("  AIC = %.2f\n", AIC(m4)))
cat("Fitting M5  (M4 + categorical bldg) ...\n")
m5 <- fit_static(as.formula(paste("T_in ~ T_out_c + factor(hour_of_day) +", BIN, "+", CAT)))
cat(sprintf("  AIC = %.2f\n", AIC(m5)))
cat("Fitting M6  (M2 + binary bldg) ...\n")
m6 <- fit_static(as.formula(paste("T_in ~ T_out_c + sin_h + cos_h +", BIN)))
cat(sprintf("  AIC = %.2f\n", AIC(m6)))
cat("Fitting M7  (M6 + categorical bldg) ...\n")
m7 <- fit_static(as.formula(paste("T_in ~ T_out_c + sin_h + cos_h +", BIN, "+", CAT)))
cat(sprintf("  AIC = %.2f\n", AIC(m7)))

# ── Constant-only AR(1) null ──────────────────────────────────────────────────
cat("\nFitting M0_ar1 (null, AR1) ...\n"); m0_ar1 <- fit_ar1(T_in ~ 1)
cat(sprintf("  AIC = %.2f\n", AIC(m0_ar1)))

# ── M8-M10: AR(1), no building characteristics ────────────────────────────────
cat("\nFitting M8  (M1 + AR1) ...\n");  m8  <- fit_ar1(T_in ~ T_out_c)
cat(sprintf("  AIC = %.2f\n", AIC(m8)))
cat("Fitting M9  (M2 + AR1) ...\n");  m9  <- fit_ar1(T_in ~ T_out_c + sin_h + cos_h)
cat(sprintf("  AIC = %.2f\n", AIC(m9)))
cat("Fitting M10 (M3 + AR1) ...\n"); m10 <- fit_ar1(T_in ~ T_out_c + factor(hour_of_day))
cat(sprintf("  AIC = %.2f\n", AIC(m10)))

# ── M11-M14: AR(1) + building characteristics ─────────────────────────────────
cat("\nFitting M11 (M4 + AR1) ...\n")
m11 <- fit_ar1(as.formula(paste("T_in ~ T_out_c + factor(hour_of_day) +", BIN)))
cat(sprintf("  AIC = %.2f\n", AIC(m11)))
cat("Fitting M12 (M5 + AR1) ...\n")
m12 <- fit_ar1(as.formula(paste("T_in ~ T_out_c + factor(hour_of_day) +", BIN, "+", CAT)))
cat(sprintf("  AIC = %.2f\n", AIC(m12)))
cat("Fitting M13 (M6 + AR1) ...\n")
m13 <- fit_ar1(as.formula(paste("T_in ~ T_out_c + sin_h + cos_h +", BIN)))
cat(sprintf("  AIC = %.2f\n", AIC(m13)))
cat("Fitting M14 (M7 + AR1) ...\n")
m14 <- fit_ar1(as.formula(paste("T_in ~ T_out_c + sin_h + cos_h +", BIN, "+", CAT)))
cat(sprintf("  AIC = %.2f\n", AIC(m14)))

# ── Variance-component extractor (handles static / AR1) ───────────────────────
var_components <- function(mod) {
  re_cov   <- as.matrix(nlme::getVarCov(mod, type = "random.effects"))
  sig2_u0  <- re_cov[1, 1]
  sig2_u1  <- re_cov[2, 2]
  sig_u01  <- re_cov[1, 2]
  sig2_eps <- mod$sigma^2
  cs       <- mod$modelStruct$corStruct
  if (is.null(cs)) {
    rho <- NA_real_
  } else {
    rho <- coef(cs, unconstrained = FALSE)[[1]]
  }
  Phi <- NA_real_
  list(sig2_u0 = sig2_u0, sig2_u1 = sig2_u1, sig_u01 = sig_u01,
       sig2_eps = sig2_eps, rho = rho, Phi = Phi)
}

# ── Summary comparison table ──────────────────────────────────────────────────
all_models <- list(
  M0_static = m0_static,
  M1=m1, M2=m2, M3=m3, M4=m4, M5=m5, M6=m6, M7=m7,
  M0_ar1 = m0_ar1,
  M8=m8, M9=m9, M10=m10, M11=m11, M12=m12, M13=m13, M14=m14
)
vc_list    <- lapply(all_models, var_components)

ref_static <- list(M4="M3", M5="M3", M6="M2", M7="M2",
                   M8="M1", M9="M2", M10="M3",
                   M11="M10", M12="M10", M13="M9", M14="M9")

cat("\n")
cat(strrep("=", 108), "\n")
cat("MODEL COMPARISON: static (M1-M3)  |  static+bldg (M4-M7)  |  AR(1) (M8-M10)  |  AR(1)+bldg (M11-M14)\n")
cat(strrep("=", 108), "\n")
cat(sprintf("  %-6s  %12s  %12s  %12s  %8s  %8s  %8s  %8s  %8s\n",
            "Model", "log-lik", "AIC", "ΔAIC", "σ_u0", "σ_u1", "σ_ε", "ρ", "Φ"))
cat("  ", strrep("-", 102), "\n")

for (nm in names(all_models)) {
  mod  <- all_models[[nm]]
  vc   <- vc_list[[nm]]
  daic <- if (nm %in% names(ref_static))
            sprintf("%+12.2f", AIC(mod) - AIC(all_models[[ref_static[[nm]]]]))
          else sprintf("%12s", "—")
  fmt_p <- function(x) if (is.na(x)) sprintf("%8s","—") else sprintf("%8.4f", x)
  cat(sprintf("  %-6s  %12.2f  %12.2f  %s  %8.4f  %8.4f  %8.4f  %s  %s\n",
              nm, as.numeric(logLik(mod)), AIC(mod), daic,
              sqrt(vc$sig2_u0), sqrt(vc$sig2_u1), sqrt(vc$sig2_eps),
              fmt_p(vc$rho), fmt_p(vc$Phi)))
  if (nm %in% c("M3", "M7", "M10")) cat("  ", strrep("-", 102), "\n")
}
cat(strrep("=", 108), "\n")

# ── Fixed effects tables ──────────────────────────────────────────────────────
pstars <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**",
                      ifelse(p < 0.05, "*", "")))

print_fe <- function(nm, mod) {
  vc   <- vc_list[[nm]]
  corr <- vc$sig_u01 / sqrt(vc$sig2_u0 * vc$sig2_u1)
  fe   <- summary(mod)$tTable

  cor_str <- ""
  if (!is.na(vc$rho)) cor_str <- sprintf("  AR(1): ρ=%.4f", vc$rho)
  cat("\n")
  cat(strrep("=", 72), "\n")
  cat(sprintf("FIXED EFFECTS — %s%s\n", nm, cor_str))
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

# ── Observation-level total variance ─────────────────────────────────────────
#
# For random = ~ T_out_c | dwelling, each observation i has:
#   Var(u0 + u1*x_i) = sig2_u0 + x_i^2*sig2_u1 + 2*x_i*cov(u0,u1)
#
# Averaging over observed x values gives the mean random-effect variance,
# which is then added to the residual variance to get total observation-level
# variance.  This is more accurate than sig2_u0 + sig2_u1 + sig2_eps because
# the random-slope contribution depends on T_out_c and cov(u0, u1).
obs_total_variance <- function(mod, data) {
  re_cov     <- as.matrix(nlme::getVarCov(mod, type = "random.effects"))
  x          <- data$T_out_c
  Z          <- cbind(1, x)
  rand_var_i <- rowSums((Z %*% re_cov) * Z)
  mean(rand_var_i) + mod$sigma^2
}

# ── Export diagnostics for Python ─────────────────────────────────────────────
cat("\nExporting diagnostics...\n")
diag_dir <- file.path(ROOT, "diagnostics")
dir.create(diag_dir, showWarnings = FALSE)

# Residuals and fitted values: one row per observation
diag_df <- df_m[, c("dwelling", "hour", "T_in", "T_out", "T_out_c", "hour_of_day")]
for (nm in names(all_models)) {
  mod <- all_models[[nm]]
  diag_df[[paste0("fitted_",  nm)]] <- as.numeric(fitted(mod))
  diag_df[[paste0("resid_",   nm)]] <- as.numeric(residuals(mod, type = "response"))
  diag_df[[paste0("nresid_",  nm)]] <- as.numeric(residuals(mod, type = "normalized"))
}
arrow::write_parquet(diag_df,
  file.path(diag_dir, "london_august_diagnostics.parquet"))

# Random effects (BLUPs): one row per dwelling
ranef_df <- data.frame(dwelling = rownames(ranef(m1)))
for (nm in names(all_models)) {
  re <- ranef(all_models[[nm]])
  ranef_df[[paste0("u0_", nm)]] <- re[rownames(ranef_df), 1]
  ranef_df[[paste0("u1_", nm)]] <- if (ncol(re) >= 2) re[rownames(ranef_df), 2] else NA_real_
}
write.csv(ranef_df,
  file.path(diag_dir, "london_august_ranef.csv"), row.names = FALSE)

# AIC table
aic_df <- data.frame(
  model  = names(all_models),
  aic    = sapply(all_models, AIC),
  loglik = sapply(all_models, function(m) as.numeric(logLik(m)))
)
write.csv(aic_df, file.path(diag_dir, "london_august_aic.csv"), row.names = FALSE)

# Variance components
vc_df <- data.frame(
  model         = names(all_models),
  sig_u0        = sapply(vc_list, function(v) sqrt(v$sig2_u0)),
  sig_u1        = sapply(vc_list, function(v) sqrt(v$sig2_u1)),
  sig_eps       = sapply(vc_list, function(v) sqrt(v$sig2_eps)),
  cov_u01       = sapply(vc_list, function(v) v$sig_u01),
  total_obs_var = sapply(all_models, obs_total_variance, data = df_m),
  rho           = sapply(vc_list, function(v) ifelse(is.na(v$rho), NA_real_, v$rho)),
  phi           = sapply(vc_list, function(v) ifelse(is.na(v$Phi), NA_real_, v$Phi))
)
write.csv(vc_df, file.path(diag_dir, "london_august_varcomp.csv"), row.names = FALSE)

cat("  -> diagnostics/london_august_diagnostics.parquet\n")
cat("  -> diagnostics/london_august_ranef.csv\n")
cat("  -> diagnostics/london_august_aic.csv\n")
cat("  -> diagnostics/london_august_varcomp.csv\n")

cat("\nDone.\n")
