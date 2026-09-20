# =============================================================================
# efus_overheating_models_meannight.R
#
# GEE models for the mean-temperature night criterion (bedroom only).
#
# Response:  night_mean_overheat_{26,27}  (binary: night mean T_in >= threshold)
# Predictor: T_2DMMT (2-day running mean of outdoor Tmax) — projectable from GCM
# Period:    May–Sep (all 5 months)
# Regions:   London, Southwest
#
# Run from project root:
#   Rscript R/efus_overheating_models_meannight.R --region=london
#   Rscript R/efus_overheating_models_meannight.R --region=southwest
# =============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(geepack)
  library(lubridate)
})

args_cli   <- commandArgs(trailingOnly = TRUE)
parse_arg  <- function(name, default) {
  m <- regmatches(args_cli, regexpr(paste0("(?<=--", name, "=)\\S+"), args_cli, perl=TRUE))
  if (length(m)) m[1] else default
}
REGION     <- parse_arg("region", "london")
MONTHS     <- 5:9
THRESHOLDS <- c(26L, 27L)

ROOT <- normalizePath(".")

reg_pre  <- if (REGION == "london") "" else REGION
path_parts <- Filter(nchar, c(reg_pre, "bedroom/mean_criterion"))
ANALYSIS   <- file.path(ROOT, "analysis", paste(path_parts, collapse="/"))
dir.create(ANALYSIS, showWarnings = FALSE, recursive = TRUE)

# ── Load ─────────────────────────────────────────────────────────────────────
cat("Loading data...\n")
df <- as.data.frame(read_parquet(file.path(ANALYSIS, "daily_overheating.parquet")))
df <- df[lubridate::month(df$date) %in% MONTHS, ]
cat(sprintf("  %d dwelling-night rows, %d dwellings\n",
            nrow(df), length(unique(df$CaseID))))

# ── Factor coding ─────────────────────────────────────────────────────────────
df$EPC     <- factor(df$EPceeb12e_efus, levels = c(1L, 2L, 3L, 4L))
df$dwtype  <- factor(df$dwtype_efus,    levels = c(6L, 1L, 2L, 3L, 4L, 5L))
df$dwage   <- factor(df$dwage_efus,     levels = c(7L, 1L, 2L, 3L, 4L, 5L, 6L))
df$cooling <- factor(df$AnyCooling,     levels = c(0L, 1L))

df$EPC     <- droplevels(df$EPC)
df$dwtype  <- droplevels(df$dwtype)
df$dwage   <- droplevels(df$dwage)
df$cooling <- droplevels(df$cooling)

df <- df[order(df$CaseID, df$date), ]

# ── Centre T_2DMMT ────────────────────────────────────────────────────────────
T2DMMT_mean  <- mean(df$T_2DMMT, na.rm = TRUE)
df$T_2DMMT_c <- df$T_2DMMT - T2DMMT_mean
cat(sprintf("  T_2DMMT centre: %.2f degC\n", T2DMMT_mean))

# ── GEE helper ────────────────────────────────────────────────────────────────
safe_geeglm <- function(response_str, predictor, df_fit, family, corstr, id_var) {
  formulas <- list(
    paste(response_str, "~", predictor, "* EPC + dwtype + dwage + cooling"),
    paste(response_str, "~", predictor, "* EPC + cooling"),
    paste(response_str, "~", predictor, "* EPC"),
    paste(response_str, "~", predictor, "+ EPC"),
    paste(response_str, "~", predictor)
  )
  # Drop rows with NA in any column used across all formulas before fitting
  key_cols <- c(response_str, predictor, "EPC", "dwtype", "dwage", "cooling", id_var)
  key_cols <- intersect(key_cols, names(df_fit))
  df_fit   <- df_fit[complete.cases(df_fit[, key_cols]), , drop = FALSE]
  id_vec   <- df_fit[[id_var]]
  last_fit <- NULL
  for (i in seq_along(formulas)) {
    fml <- tryCatch(as.formula(formulas[[i]]), error = function(e) NULL)
    if (is.null(fml)) next
    fit <- tryCatch(
      geeglm(fml, id = id_vec, family = family,
             corstr = corstr, data = df_fit, std.err = "san.se"),
      error = function(e) { cat(sprintf("    formula %d error: %s\n", i, conditionMessage(e))); NULL }
    )
    if (is.null(fit)) next
    last_fit <- fit
    if (max(abs(coef(fit)), na.rm = TRUE) < 100) {
      if (i > 1) cat(sprintf("    ** fallback formula %d used: %s\n", i, formulas[[i]]))
      return(fit)
    }
    cat(sprintf("    formula %d diverged (max|coef|=%.2e), trying simpler\n",
                i, max(abs(coef(fit)), na.rm = TRUE)))
  }
  cat("    ** all formulas diverged; skipping model\n")
  NULL
}

gee_coef_table <- function(fit, model_name) {
  s   <- summary(fit)$coefficients
  data.frame(
    model    = model_name,
    term     = rownames(s),
    estimate = s[, "Estimate"],
    se_san   = s[, "Std.err"],
    p_san    = s[, "Pr(>|W|)"],
    ci_lo    = s[, "Estimate"] - 1.96 * s[, "Std.err"],
    ci_hi    = s[, "Estimate"] + 1.96 * s[, "Std.err"],
    row.names = NULL
  )
}

# ── GEE: night_mean_overheat_{26,27} ~ T_2DMMT_c * EPC + ... ────────────────────
cat("\n── GEE: night_mean_overheat (binary logit, AR1) ────────────────────────────\n")

gee_mean <- list()
for (thr in THRESHOLDS) {
  outcome <- paste0("night_mean_overheat_", thr)
  mname   <- paste0("gee_mean_", thr)
  cat(sprintf("  Fitting %s...\n", mname))
  fit <- safe_geeglm(outcome, "T_2DMMT_c", df, binomial(link = "logit"), "ar1", "CaseID")
  if (is.null(fit)) { cat(sprintf("  Skipping %s (failed to converge)\n", mname)); next }
  cat("  Coefficients:\n"); print(round(summary(fit)$coefficients, 4))
  gee_mean[[mname]] <- fit
}

# ── Prediction grids ──────────────────────────────────────────────────────────
cat("\n── GEE prediction grids ────────────────────────────────────────────────\n")

t2d_seq   <- seq(min(df$T_2DMMT, na.rm=TRUE), max(df$T_2DMMT, na.rm=TRUE), length.out=80)
pred_base <- expand.grid(T_2DMMT = t2d_seq, EPC_level = levels(df$EPC))
pred_base$T_2DMMT_c <- pred_base$T_2DMMT - T2DMMT_mean
pred_base$EPC        <- factor(pred_base$EPC_level, levels = levels(df$EPC))
pred_base$dwtype     <- factor(6L, levels = levels(df$dwtype))
pred_base$dwage      <- factor(7L, levels = levels(df$dwage))
pred_base$cooling    <- factor(0L, levels = levels(df$cooling))

all_grids <- lapply(names(gee_mean), function(nm) {
  pg       <- pred_base
  pg$model <- nm
  pg$prob  <- predict(gee_mean[[nm]], newdata = pg, type = "response")
  pg[, c("model", "T_2DMMT", "EPC_level", "prob")]
})
pred_grid_df <- do.call(rbind, all_grids)

# ── Save ─────────────────────────────────────────────────────────────────────
cat("\n── Saving outputs ──────────────────────────────────────────────────────\n")

gee_coefs <- do.call(rbind, lapply(names(gee_mean),
                                   function(nm) gee_coef_table(gee_mean[[nm]], nm)))
write.csv(gee_coefs, file.path(ANALYSIS, "gee_coefs.csv"), row.names = FALSE)
cat("  Saved: gee_coefs.csv\n")

gee_fitted <- do.call(rbind, lapply(names(gee_mean), function(nm) {
  thr <- sub("gee_mean_", "", nm)
  data.frame(model = nm, CaseID = df$CaseID, date = df$date,
             fitted   = fitted(gee_mean[[nm]]),
             observed = df[[paste0("night_mean_overheat_", thr)]])
}))
write.csv(gee_fitted, file.path(ANALYSIS, "gee_fitted.csv"), row.names = FALSE)
cat("  Saved: gee_fitted.csv\n")

write.csv(pred_grid_df, file.path(ANALYSIS, "gee_pred_grid.csv"), row.names = FALSE)
cat("  Saved: gee_pred_grid.csv\n")

write.csv(
  data.frame(variable = "T_2DMMT", centre = T2DMMT_mean),
  file.path(ANALYSIS, "climate_centres.csv"), row.names = FALSE
)
cat("  Saved: climate_centres.csv\n")

cat("\nDone.\n")
