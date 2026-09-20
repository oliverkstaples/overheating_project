# =============================================================================
# efus_overheating_models.R
#
# LME (nlme) and GEE (geepack) models for overheating impact profiles.
#
# Key design decisions:
#   - corAR1 is NOT used for LME: AR1 phi≈0.88 at daily scale causes
#     random-intercept/slope variance to collapse to zero (identification
#     conflict). Residual autocorrelation is handled instead by CR2
#     cluster-robust standard errors (clubSandwich).
#
#     Why phi inflates variance — the equation:
#       For e_t = phi * e_{t-1} + eps_t with Var(eps_t) = sigma^2_eps,
#       the marginal (long-run) variance is
#
#           Var(e_t) = sigma^2_eps / (1 - phi^2)
#
#       At phi = 0.9:  Var(e_t) = sigma^2_eps / (1 - 0.81) = sigma^2_eps / 0.19
#                               ~ 5.26 * sigma^2_eps
#
#       The long-run variance is 5x the innovation variance.  As phi -> 1 the
#       AR(1) process behaves like a slowly drifting mean — identical to a static
#       random intercept u_j.  REML cannot separate the two, so the random-effect
#       variance collapses to zero and the residual absorbs all between-dwelling
#       variation.
#
#   - corAR1 IS retained for GEE: GEE has no random effects to collapse,
#     so AR1 working correlation only affects efficiency, not validity.
#   - dwage_efus is included as a confounder: EPC band correlates strongly
#     with construction era, which drives the apparent EPC C+ overheating
#     paradox without age adjustment.
#   - dwage reference = 7 (Post-1990): the most modern era is chosen as
#     reference so all coefficients show deviation from the newest stock.
#
# Inputs:  analysis/daily_overheating.parquet
# Outputs: analysis/lme_coefs.csv       (model-based + CR2 SEs and CIs)
#          analysis/lme_varcomp.csv
#          analysis/lme_fitted.csv
#          analysis/gee_coefs.csv
#          analysis/gee_fitted.csv
#          analysis/gee_pred_grid.csv
#          analysis/climate_centres.csv
#          analysis/lme_acf_diag.csv    (residual ACF diagnostic)
#
# Run from project root: Rscript efus_overheating_models.R
# =============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(nlme)
  library(geepack)
  library(lubridate)
  library(clubSandwich)
})

# ── Configuration (overridable via command-line args) ─────────────────────────
# Usage: Rscript efus_overheating_models.R [--region=london] [--room=livingroom]
#                                          [--subdir=] [--months=5,6,7,8,9]
#   --region : london | southwest  (default: london)
#   --room   : livingroom | bedroom (default: livingroom)
#   --subdir : time-based suffix, e.g. "august" (default: "")
#   --months : comma-separated months (default: "5,6,7,8,9")
args_cli   <- commandArgs(trailingOnly = TRUE)
parse_arg  <- function(name, default) {
  m <- regmatches(args_cli, regexpr(paste0("(?<=--", name, "=)\\S+"), args_cli, perl=TRUE))
  if (length(m)) m[1] else default
}
REGION     <- parse_arg("region", "london")
ROOM       <- parse_arg("room",   "livingroom")
SUBDIR_ARG <- parse_arg("subdir", "")
MONTHS_ARG <- parse_arg("months", "5,6,7,8,9")
MONTHS     <- as.integer(strsplit(MONTHS_ARG, ",")[[1]])
THRESHOLDS <- c(26L, 27L, 28L)

ROOT <- normalizePath(".")

# Build ANALYSIS path: analysis/[region/][room/][subdir]  (london has no prefix)
reg_pre  <- if (REGION == "london") "" else REGION
room_sub   <- if (SUBDIR_ARG != "") file.path(ROOM, SUBDIR_ARG) else ROOM
path_parts <- Filter(nchar, c(reg_pre, room_sub))
ANALYSIS   <- file.path(ROOT, "analysis", paste(path_parts, collapse="/"))
dir.create(ANALYSIS, showWarnings = FALSE, recursive = TRUE)

# ── Load and filter ───────────────────────────────────────────────────────────
cat("Loading data...\n")
df <- as.data.frame(read_parquet(file.path(ANALYSIS, "daily_overheating.parquet")))
df <- df[lubridate::month(df$date) %in% MONTHS, ]
cat(sprintf("  %d dwelling-day rows, %d dwellings, months: %s\n",
            nrow(df), length(unique(df$CaseID)),
            paste(MONTHS, collapse = ",")))

# ── Factor coding ─────────────────────────────────────────────────────────────
# EPC:    reference = 1  (C+, best performance)
# dwtype: reference = 6  (Flat — highest prior overheating risk)
# dwage:  reference = 7  (post-1990 — newest stock; levels = c(7,1,...,6) below)
# cooling: reference = 0 (no cooling)
df$EPC     <- factor(df$EPceeb12e_efus, levels = c(1L, 2L, 3L, 4L))
df$dwtype  <- factor(df$dwtype_efus,    levels = c(6L, 1L, 2L, 3L, 4L, 5L))
df$dwage   <- factor(df$dwage_efus,     levels = c(7L, 1L, 2L, 3L, 4L, 5L, 6L))
df$cooling <- factor(df$AnyCooling,     levels = c(0L, 1L))

# Drop unused factor levels (e.g. EPC4 absent in South West) — geepack
# geeglm() silently tolerates missing levels for binary responses but errors
# on cbind() binomial-counts responses if any factor has unused levels.
df$EPC     <- droplevels(df$EPC)
df$dwtype  <- droplevels(df$dwtype)
df$dwage   <- droplevels(df$dwage)
df$cooling <- droplevels(df$cooling)

# ── Sort by dwelling then date (required for rolling quantities and GEE AR1) ──
df <- df[order(df$CaseID, df$date), ]

# ── Centre outdoor climate predictors ────────────────────────────────────────
T2DMMT_mean   <- mean(df$T_2DMMT,        na.rm = TRUE)
minTout_mean  <- mean(df$daily_min_Tout,  na.rm = TRUE)
meanTout_mean <- mean(df$daily_mean_Tout, na.rm = TRUE)

df$T_2DMMT_c  <- df$T_2DMMT        - T2DMMT_mean
df$minTout_c  <- df$daily_min_Tout  - minTout_mean
df$meanTout_c <- df$daily_mean_Tout - meanTout_mean

cat(sprintf("  Outdoor centres: T_2DMMT=%.2f  min_Tout=%.2f  mean_Tout=%.2f\n",
            T2DMMT_mean, minTout_mean, meanTout_mean))

# ── Indoor 2DMMnT: 2-day running mean of daily min indoor temperature ─────────
# Right-aligned: T_2DMMnT[t] = (daily_min_Tin[t-1] + daily_min_Tin[t]) / 2
# NA on the first day of each dwelling's record.
df$T_2DMMnT <- ave(df$daily_min_Tin, df$CaseID, FUN = function(x) {
  n <- length(x)
  c(NA_real_, (x[-n] + x[-1]) / 2)
})
T_2DMMnT_mean <- mean(df$T_2DMMnT, na.rm = TRUE)
df$T_2DMMnT_c <- df$T_2DMMnT - T_2DMMnT_mean

cat(sprintf("  Indoor centre:   T_2DMMnT=%.2f\n", T_2DMMnT_mean))

# ── Derived columns ───────────────────────────────────────────────────────────
for (thr in THRESHOLDS) {
  df[[paste0("logdh_",      thr)]] <- log1p(df[[paste0("daily_dh_",           thr)]])
  df[[paste0("exceed_n_",   thr)]] <- as.integer(
    round(df[[paste0("daily_exceed_prop_", thr)]] * df$occ_hours))
  df[[paste0("non_exceed_", thr)]] <- df$occ_hours - df[[paste0("exceed_n_", thr)]]
}

# =============================================================================
# Helpers
# =============================================================================

LME_CTRL <- lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200)

# Normalised-residual ACF, mean across dwellings (returns vector length max_lag)
mean_resid_acf <- function(fit, data, id_col = "CaseID",
                           idx_col = "day_index", max_lag = 10L) {
  r   <- residuals(fit, type = "normalized")
  ids <- data[[id_col]]
  ord <- data[[idx_col]]
  rows <- tapply(seq_along(r), ids, function(ii) {
    x <- r[ii][order(ord[ii])]
    if (length(x) < max_lag + 3L) return(rep(NA_real_, max_lag))
    acf(x, lag.max = max_lag, plot = FALSE)$acf[2L:(max_lag + 1L)]
  })
  mat <- do.call(rbind, Filter(function(a) !anyNA(a), rows))
  colMeans(mat, na.rm = TRUE)
}

# Fixed-effect coefficient table with both model-based and CR2 SEs/CIs/p.
lme_coef_table <- function(fit, model_name, data) {
  tt  <- summary(fit)$tTable
  nr  <- nrow(tt)
  dfs <- tt[, "DF"]

  cr2 <- tryCatch(
    coef_test(fit, vcov = "CR2", cluster = data$CaseID),
    error = function(e) { message("  [CR2 failed] ", conditionMessage(e)); NULL }
  )

  data.frame(
    model     = model_name,
    term      = rownames(tt),
    estimate  = tt[, "Value"],
    se_mod    = tt[, "Std.Error"],
    p_mod     = tt[, "p-value"],
    ci_lo_mod = tt[, "Value"] - qt(0.975, dfs) * tt[, "Std.Error"],
    ci_hi_mod = tt[, "Value"] + qt(0.975, dfs) * tt[, "Std.Error"],
    se_cr2    = if (!is.null(cr2)) cr2$SE       else rep(NA_real_, nr),
    df_cr2    = if (!is.null(cr2)) cr2$df_Satt  else rep(NA_real_, nr),
    p_cr2     = if (!is.null(cr2)) cr2$p_Satt   else rep(NA_real_, nr),
    ci_lo_cr2 = if (!is.null(cr2))
                  cr2$beta - qt(0.975, cr2$df_Satt) * cr2$SE
                else rep(NA_real_, nr),
    ci_hi_cr2 = if (!is.null(cr2))
                  cr2$beta + qt(0.975, cr2$df_Satt) * cr2$SE
                else rep(NA_real_, nr),
    row.names = NULL
  )
}

# Variance-component table (always includes corr column, NA if absent).
lme_vc_table <- function(fit, model_name) {
  vc      <- VarCorr(fit)
  col_nms <- colnames(vc)
  data.frame(
    model     = model_name,
    component = rownames(vc),
    variance  = suppressWarnings(as.numeric(vc[, "Variance"])),
    std_dev   = suppressWarnings(as.numeric(vc[, "StdDev"])),
    corr      = if ("Corr" %in% col_nms)
                  suppressWarnings(as.numeric(vc[, "Corr"]))
                else rep(NA_real_, nrow(vc)),
    stringsAsFactors = FALSE
  )
}

# GEE coefficient table with sandwich SEs and stable CR2 columns (always NA
# for GEE — clubSandwich does not support geeglm objects reliably).
gee_coef_table <- function(fit, model_name) {
  s   <- summary(fit)$coefficients
  tbl <- data.frame(
    model     = model_name,
    term      = rownames(s),
    estimate  = s[, "Estimate"],
    se_san    = s[, "Std.err"],
    p_san     = s[, "Pr(>|W|)"],
    ci_lo     = s[, "Estimate"] - 1.96 * s[, "Std.err"],
    ci_hi     = s[, "Estimate"] + 1.96 * s[, "Std.err"],
    row.names = NULL
  )
  tbl
}

# Common model formula components
EPC_DW_COVARS <- "+ dwtype + dwage + cooling"

# =============================================================================
# LME 1 — daily_max_Tin
#   random: intercept + slope on T_2DMMT_c  (no corAR1)
#   interaction: T_2DMMT_c × EPC
# =============================================================================
cat("\n── LME: daily_max_Tin ──────────────────────────────────────────────────\n")

m_max <- tryCatch(
  lme(daily_max_Tin ~ T_2DMMT_c * EPC + dwtype + dwage + cooling,
      random  = ~ T_2DMMT_c | CaseID,
      data    = df, method = "REML", control = LME_CTRL),
  error = function(e) {
    message("  RS model failed (", conditionMessage(e), "); using pdDiag")
    lme(daily_max_Tin ~ T_2DMMT_c * EPC + dwtype + dwage + cooling,
        random  = list(CaseID = pdDiag(~ T_2DMMT_c)),
        data    = df, method = "REML", control = LME_CTRL)
  }
)
acf_max <- mean_resid_acf(m_max, df)
cat(sprintf("  Residual ACF (lag1=%.3f lag2=%.3f) — handled by CR2\n",
            acf_max[1], acf_max[2]))
cat("  Variance components:\n"); print(VarCorr(m_max))
cat("  Fixed effects:\n"); print(round(summary(m_max)$tTable, 4))

# =============================================================================
# LME 2 — daily_min_Tin
#   random: intercept + slope on minTout_c  (no corAR1)
#   interaction: minTout_c × EPC
# =============================================================================
cat("\n── LME: daily_min_Tin ──────────────────────────────────────────────────\n")

m_min <- tryCatch(
  lme(daily_min_Tin ~ minTout_c * EPC + dwtype + dwage + cooling,
      random  = ~ minTout_c | CaseID,
      data    = df, method = "REML", control = LME_CTRL),
  error = function(e) {
    message("  RS model failed (", conditionMessage(e), "); using pdDiag")
    lme(daily_min_Tin ~ minTout_c * EPC + dwtype + dwage + cooling,
        random  = list(CaseID = pdDiag(~ minTout_c)),
        data    = df, method = "REML", control = LME_CTRL)
  }
)
acf_min <- mean_resid_acf(m_min, df)
cat(sprintf("  Residual ACF (lag1=%.3f lag2=%.3f) — handled by CR2\n",
            acf_min[1], acf_min[2]))
cat("  Variance components:\n"); print(VarCorr(m_min))
cat("  Fixed effects:\n"); print(round(summary(m_min)$tTable, 4))

# =============================================================================
# LME 3+ — log1p(daily_dh_t)
#   random: intercept only (zero-inflation makes slope unstable)
#   interaction: meanTout_c × EPC  (no corAR1)
# =============================================================================
lme_dh <- list()

for (thr in THRESHOLDS) {
  cat(sprintf("\n── LME: log1p(daily_dh_%d) ─────────────────────────────────────────────\n", thr))
  outcome <- paste0("logdh_", thr)
  mname   <- paste0("lme_logdh", thr)

  m_fit <- lme(
    as.formula(paste(outcome, "~ meanTout_c * EPC + dwtype + dwage + cooling")),
    random  = ~ 1 | CaseID,
    data    = df, method = "REML", control = LME_CTRL
  )
  acf_v <- mean_resid_acf(m_fit, df)
  cat(sprintf("  Residual ACF (lag1=%.3f lag2=%.3f) — handled by CR2\n",
              acf_v[1], acf_v[2]))
  cat("  Fixed effects:\n"); print(round(summary(m_fit)$tTable, 4))
  lme_dh[[mname]] <- m_fit
}

# =============================================================================
# GEE helper: automatic formula fallback when coefficients diverge
# Full model: ~ predictor * EPC + dwtype + dwage + cooling
# Fallback 1: ~ predictor * EPC + cooling   (drop building-type/age terms)
# Fallback 2: ~ predictor * EPC             (minimal; rare)
# Divergence criterion: any |coef| > 1e6
# =============================================================================
safe_geeglm <- function(response_str, predictor, df_fit, family, corstr, id_var) {
  formulas <- list(
    paste(response_str, "~", predictor, "* EPC + dwtype + dwage + cooling"),
    paste(response_str, "~", predictor, "* EPC + cooling"),
    paste(response_str, "~", predictor, "* EPC"),
    paste(response_str, "~", predictor, "+ EPC"),
    paste(response_str, "~", predictor)
  )
  id_vec <- df_fit[[id_var]]
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

# =============================================================================
# GEE — daily_overheat_t  (binary, logit, corstr AR1)
# =============================================================================
cat("\n── GEE: daily_overheat ─────────────────────────────────────────────────────\n")

gee_overheat <- list()
for (thr in THRESHOLDS) {
  outcome <- paste0("daily_overheat_", thr)
  mname   <- paste0("gee_overheat_", thr)
  cat(sprintf("  Fitting %s...\n", mname))
  fit <- safe_geeglm(outcome, "T_2DMMT_c", df, binomial(link = "logit"), "ar1", "CaseID")
  if (is.null(fit)) { cat(sprintf("  Skipping %s (failed to converge)\n", mname)); next }
  cat("  Coefficients:\n"); print(round(summary(fit)$coefficients, 4))
  gee_overheat[[mname]] <- fit
}

# =============================================================================
# GEE — daily_exceed_prop_t  (binomial counts, logit, corstr AR1)
# =============================================================================
cat("\n── GEE: daily_exceed_prop ──────────────────────────────────────────────\n")

gee_exc <- list()
for (thr in THRESHOLDS) {
  n_col  <- paste0("exceed_n_",   thr)
  nn_col <- paste0("non_exceed_", thr)
  mname  <- paste0("gee_excprop_", thr)
  cat(sprintf("  Fitting %s...\n", mname))
  resp <- paste0("cbind(", n_col, ", ", nn_col, ")")
  fit <- safe_geeglm(resp, "T_2DMMT_c", df, binomial(link = "logit"), "ar1", "CaseID")
  if (is.null(fit)) { cat(sprintf("  Skipping %s (failed to converge)\n", mname)); next }
  cat("  Coefficients:\n"); print(round(summary(fit)$coefficients, 4))
  gee_exc[[mname]] <- fit
}

# =============================================================================
# GEE — daily_overheat_t with T_2DMMnT predictor (sustained indoor night heat)
# Rows with NA T_2DMMnT (first day per dwelling) are dropped.
# =============================================================================
cat("\n── GEE: daily_overheat vs T_2DMMnT ────────────────────────────────────────\n")

df_nin <- df[!is.na(df$T_2DMMnT_c), ]

gee_nin_overheat <- list()
for (thr in THRESHOLDS) {
  outcome <- paste0("daily_overheat_", thr)
  mname   <- paste0("gee_nin_overheat_", thr)
  cat(sprintf("  Fitting %s...\n", mname))
  fit <- safe_geeglm(outcome, "T_2DMMnT_c", df_nin, binomial(link = "logit"), "ar1", "CaseID")
  if (is.null(fit)) { cat(sprintf("  Skipping %s (failed to converge)\n", mname)); next }
  cat("  Coefficients:\n"); print(round(summary(fit)$coefficients, 4))
  gee_nin_overheat[[mname]] <- fit
}

# =============================================================================
# GEE prediction grids — P(overheating criterion) vs T_2DMMT × EPC  (outdoor)
#                         P(overheating criterion) vs T_2DMMnT × EPC (indoor night)
# Reference: Flat (dwtype=6), post-1990 (dwage=7), no cooling
# =============================================================================
cat("\n── GEE prediction grids ────────────────────────────────────────────────\n")

t2d_seq   <- seq(min(df$T_2DMMT), max(df$T_2DMMT), length.out = 80)
pred_base <- expand.grid(T_2DMMT = t2d_seq, EPC_level = levels(df$EPC))
pred_base$T_2DMMT_c <- pred_base$T_2DMMT - T2DMMT_mean
pred_base$EPC        <- factor(pred_base$EPC_level, levels = levels(df$EPC))
pred_base$dwtype     <- factor(6L, levels = levels(df$dwtype))
pred_base$dwage      <- factor(7L, levels = levels(df$dwage))
pred_base$cooling    <- factor(0L, levels = levels(df$cooling))

all_grids <- lapply(names(gee_overheat), function(nm) {
  pg       <- pred_base
  pg$model <- nm
  pg$prob  <- predict(gee_overheat[[nm]], newdata = pg, type = "response")
  pg[, c("model", "T_2DMMT", "EPC_level", "prob")]
})
pred_grid_df <- do.call(rbind, all_grids)

t2n_seq  <- seq(min(df$T_2DMMnT, na.rm = TRUE), max(df$T_2DMMnT, na.rm = TRUE), length.out = 80)
pred_nin <- expand.grid(T_2DMMnT = t2n_seq, EPC_level = levels(df$EPC))
pred_nin$T_2DMMnT_c <- pred_nin$T_2DMMnT - T_2DMMnT_mean
pred_nin$EPC     <- factor(pred_nin$EPC_level, levels = levels(df$EPC))
pred_nin$dwtype  <- factor(6L, levels = levels(df$dwtype))
pred_nin$dwage   <- factor(7L, levels = levels(df$dwage))
pred_nin$cooling <- factor(0L, levels = levels(df$cooling))

all_nin_grids <- lapply(names(gee_nin_overheat), function(nm) {
  pg       <- pred_nin
  pg$model <- nm
  pg$prob  <- predict(gee_nin_overheat[[nm]], newdata = pg, type = "response")
  pg[, c("model", "T_2DMMnT", "EPC_level", "prob")]
})
pred_nin_grid_df <- do.call(rbind, all_nin_grids)

# =============================================================================
# Collect and save
# =============================================================================
cat("\n── Saving outputs ──────────────────────────────────────────────────────\n")

# LME coefficients (model-based + CR2)
lme_coefs <- rbind(
  lme_coef_table(m_max, "lme_max_tin", df),
  lme_coef_table(m_min, "lme_min_tin", df),
  do.call(rbind, lapply(names(lme_dh),
                        function(nm) lme_coef_table(lme_dh[[nm]], nm, df)))
)
write.csv(lme_coefs, file.path(ANALYSIS, "lme_coefs.csv"), row.names = FALSE)
cat("  Saved: analysis/lme_coefs.csv\n")

# LME variance components
lme_vc <- rbind(
  lme_vc_table(m_max, "lme_max_tin"),
  lme_vc_table(m_min, "lme_min_tin"),
  do.call(rbind, lapply(names(lme_dh),
                        function(nm) lme_vc_table(lme_dh[[nm]], nm)))
)
write.csv(lme_vc, file.path(ANALYSIS, "lme_varcomp.csv"), row.names = FALSE)
cat("  Saved: analysis/lme_varcomp.csv\n")

# LME fitted values
lme_fitted <- rbind(
  data.frame(model = "lme_max_tin", CaseID = df$CaseID, date = df$date,
             fitted = fitted(m_max), observed = df$daily_max_Tin),
  data.frame(model = "lme_min_tin", CaseID = df$CaseID, date = df$date,
             fitted = fitted(m_min), observed = df$daily_min_Tin),
  do.call(rbind, lapply(THRESHOLDS, function(thr) {
    nm <- paste0("lme_logdh", thr)
    oc <- paste0("logdh_",    thr)
    data.frame(model = nm, CaseID = df$CaseID, date = df$date,
               fitted = fitted(lme_dh[[nm]]), observed = df[[oc]])
  }))
)
write.csv(lme_fitted, file.path(ANALYSIS, "lme_fitted.csv"), row.names = FALSE)
cat("  Saved: analysis/lme_fitted.csv\n")

# GEE coefficients
gee_coefs <- rbind(
  do.call(rbind, lapply(names(gee_overheat),
                        function(nm) gee_coef_table(gee_overheat[[nm]], nm))),
  do.call(rbind, lapply(names(gee_exc),
                        function(nm) gee_coef_table(gee_exc[[nm]],  nm))),
  do.call(rbind, lapply(names(gee_nin_overheat),
                        function(nm) gee_coef_table(gee_nin_overheat[[nm]], nm)))
)
write.csv(gee_coefs, file.path(ANALYSIS, "gee_coefs.csv"), row.names = FALSE)
cat("  Saved: analysis/gee_coefs.csv\n")

# GEE fitted values
gee_fitted <- rbind(
  do.call(rbind, lapply(names(gee_overheat), function(nm) {
    thr <- sub("gee_overheat_", "", nm)
    data.frame(model = nm, CaseID = df$CaseID, date = df$date,
               fitted   = fitted(gee_overheat[[nm]]),
               observed = df[[paste0("daily_overheat_", thr)]])
  })),
  do.call(rbind, lapply(names(gee_exc), function(nm) {
    thr <- sub("gee_excprop_", "", nm)
    data.frame(model = nm, CaseID = df$CaseID, date = df$date,
               fitted   = fitted(gee_exc[[nm]]),
               observed = df[[paste0("daily_exceed_prop_", thr)]])
  })),
  do.call(rbind, lapply(names(gee_nin_overheat), function(nm) {
    thr <- sub("gee_nin_overheat_", "", nm)
    data.frame(model = nm, CaseID = df_nin$CaseID, date = df_nin$date,
               fitted   = fitted(gee_nin_overheat[[nm]]),
               observed = df_nin[[paste0("daily_overheat_", thr)]])
  }))
)
write.csv(gee_fitted, file.path(ANALYSIS, "gee_fitted.csv"), row.names = FALSE)
cat("  Saved: analysis/gee_fitted.csv\n")

# GEE prediction grids
write.csv(pred_grid_df, file.path(ANALYSIS, "gee_pred_grid.csv"),
          row.names = FALSE)
cat("  Saved: analysis/gee_pred_grid.csv\n")

write.csv(pred_nin_grid_df, file.path(ANALYSIS, "gee_pred_grid_nin.csv"),
          row.names = FALSE)
cat("  Saved: analysis/gee_pred_grid_nin.csv\n")

# Climate centres
write.csv(
  data.frame(variable = c("T_2DMMT", "daily_min_Tout", "daily_mean_Tout",
                           "T_2DMMnT"),
             centre   = c(T2DMMT_mean, minTout_mean, meanTout_mean,
                          T_2DMMnT_mean)),
  file.path(ANALYSIS, "climate_centres.csv"), row.names = FALSE
)
cat("  Saved: analysis/climate_centres.csv\n")

# Residual ACF diagnostic
acf_diag <- rbind(
  data.frame(model = "lme_max_tin", lag = seq_along(acf_max), acf = acf_max),
  data.frame(model = "lme_min_tin", lag = seq_along(acf_min), acf = acf_min)
)
write.csv(acf_diag, file.path(ANALYSIS, "lme_acf_diag.csv"), row.names = FALSE)
cat("  Saved: analysis/lme_acf_diag.csv\n")

cat("\nDone.\n")
