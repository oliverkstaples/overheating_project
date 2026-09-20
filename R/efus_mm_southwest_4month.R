#!/usr/bin/env Rscript
# efus_mm_southwest_4month.R
#
# Fits static LME (M1-M7), AR(1) LME (M8-M10), and SAR(1,24) LME (M11-M15)
# on the Southwest May–September subset of EFUS 2017.
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
#   M11 : M1 + SAR(1,24) residuals    a_t = rho*a_{t-1} + Phi*a_{t-24} + eta_t
#   M12 : M2 + SAR(1,24) residuals
#   M13 : M3 + SAR(1,24) residuals
#   M14 : M5 + SAR(1,24) residuals    (hour dummies + binary + categorical bldg chars)
#   M15 : M7 + SAR(1,24) residuals    (sin/cos     + binary + categorical bldg chars)

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
df <- df[lubridate::month(df$hour) %in% 5:9, ]

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

# ── M8-M10: AR(1) ─────────────────────────────────────────────────────────────
cat("\nFitting M8  (M1 + AR1) ...\n");  m8  <- fit_ar1(T_in ~ T_out_c)
cat(sprintf("  AIC = %.2f\n", AIC(m8)))
cat("Fitting M9  (M2 + AR1) ...\n");  m9  <- fit_ar1(T_in ~ T_out_c + sin_h + cos_h)
cat(sprintf("  AIC = %.2f\n", AIC(m9)))
cat("Fitting M10 (M3 + AR1) ...\n"); m10 <- fit_ar1(T_in ~ T_out_c + factor(hour_of_day))
cat(sprintf("  AIC = %.2f\n", AIC(m10)))

# ══════════════════════════════════════════════════════════════════════════════
# SAR(1,24) custom corStruct
#
# Residual model:   a_{ij} = rho * a_{i-1,j} + Phi * a_{i-24,j} + eta_{ij}
#
# Parameters (rho, Phi) are stored in atanh-space (unconstrained) during
# optimisation; converted back with tanh.  Stationarity requires that the AR
# polynomial  1 - rho*z - Phi*z^24  has all roots outside the unit circle.
# We enforce a practical sufficient condition |rho| + |Phi| < 0.999 inside
# coef<- so that ARMAacf never receives a non-stationary specification.
# ══════════════════════════════════════════════════════════════════════════════

corSAR24 <- function(value = c(0.5, 0.2), form = ~1) {
  stopifnot(length(value) == 2, all(abs(value) < 1))
  attr(value, "formula") <- form
  attr(value, "fixed")   <- FALSE
  class(value)           <- c("corSAR24", "corStruct")
  value
}

# Theoretical ACF via Yule-Walker (ARMAacf); AR order = max(24, maxlag)
.sar24_acf <- function(rho, Phi, maxlag) {
  p  <- max(24L, as.integer(maxlag))
  ar <- numeric(p)
  ar[1L]  <- rho
  ar[24L] <- Phi
  suppressWarnings(
    tryCatch(
      stats::ARMAacf(ar = ar, lag.max = maxlag),
      error = function(e) { v <- numeric(maxlag + 1L); v[1L] <- 1; v }
    )
  )
}

# coef(): return atanh-transformed (unconstrained=TRUE) or natural values
coef.corSAR24 <- function(object, unconstrained = TRUE, ...) {
  val <- as.double(object)[1:2]
  if (unconstrained) atanh(val)
  else { names(val) <- c("Rho", "Phi"); val }
}

# coef<-(): enforce stationarity: scale (rho, Phi) if |rho|+|Phi| >= 0.999
`coef<-.corSAR24` <- function(object, ..., value) {
  constrained <- tanh(as.double(value)[1:2])
  total <- abs(constrained[1]) + abs(constrained[2])
  if (total >= 0.999) constrained <- constrained * (0.999 / total)
  object[1:2] <- constrained
  object
}

Initialize.corSAR24 <- function(object, data, ...) {
  form     <- attr(object, "formula")
  grp_form <- nlme::getGroupsFormula(form)
  grps <- if (!is.null(grp_form)) {
    droplevels(interaction(data[, all.vars(grp_form), drop = FALSE]))
  } else {
    factor(rep(1L, nrow(data)))
  }
  attr(object, "groups")    <- grps
  # Use Dim.corStruct to build the proper list (N, M, sumLenSq, len, start)
  # that corFactor.corStruct expects when it calls Dim(object) later.
  attr(object, "Dim")       <- nlme:::Dim.corStruct(object, grps)
  attr(object, "covariate") <- lapply(
    split(seq_len(nrow(data)), grps, drop = TRUE), seq_along)
  object
}

corMatrix.corSAR24 <- function(object,
                               covariate = nlme::getCovariate(object),
                               corr = TRUE, ...) {
  if (is.null(covariate)) stop("corSAR24: not initialized")
  p   <- coef(object, unconstrained = FALSE)
  rho <- p["Rho"]; Phi <- p["Phi"]
  val <- lapply(covariate, function(idx) {
    n <- length(idx)
    stats::toeplitz(.sar24_acf(rho, Phi, n - 1L))
  })
  if (length(val) == 1L) val[[1L]] else val
}

print.corSAR24 <- function(x, ...) {
  p <- coef(x, unconstrained = FALSE)
  cat(sprintf("SAR(1,24): rho = %.4f,  Phi = %.4f\n", p["Rho"], p["Phi"]))
  invisible(x)
}

# ── Fast corFactor for corSAR24 ───────────────────────────────────────────────
#
# Without this method, nlme falls back to corFactor.corStruct which calls
# corMatrix() → dense n×n Toeplitz → dense O(n³) Cholesky at every optimizer
# step.  For n=744 and 69 groups that is ~10 hours per model.
#
# Here we use the exact Durbin-Levinson innovation filter:
#   • startup rows t = 2..25:  dense 24×24 block from DL recursion  O(24²)
#   • stationary rows t ≥ 26:  3 entries per row (diag, lag-1, lag-24)  O(n)
# Total: O(n) per group.  nlme's subsequent O(n²) operations then dominate,
# giving ≈ 5–10 min for M11-M13 instead of hours.
#
# Return format (matches corFactor.corAR1):
#   numeric vector of length Σ nᵢ²  (column-major L matrices concatenated)
#   with  attr(., "logDet") = Σ sum(log(diag(Lᵢ)))
#   where  Lᵢ^T Lᵢ = Cᵢ⁻¹  (Lᵢ lower-triangular Cholesky of group i's
#   inverse correlation matrix).

.dl_recursion <- function(acf, max_order) {
  # Durbin-Levinson steps 0..max_order.
  # acf[1]=ρ₀=1, acf[k]=ρ_{k-1}.  Returns list(phi, d) where
  #   phi[[m]] = order-m prediction coefficients (length m)
  #   d[m+1]   = innovation variance dₘ  (d[1]=d₀=1)
  n_steps  <- min(as.integer(max_order), length(acf) - 1L)
  phi_list <- vector("list", n_steps)
  d_vec    <- numeric(n_steps + 1L)
  d_vec[1L] <- 1.0
  if (n_steps < 1L) return(list(phi = phi_list, d = d_vec))

  kappa         <- acf[2L]
  phi_list[[1L]] <- kappa
  d_vec[2L]     <- 1.0 - kappa * kappa

  for (m in seq_len(n_steps - 1L) + 1L) {
    prev  <- phi_list[[m - 1L]]
    s     <- sum(prev * rev(acf[2L:m]))        # Σ φⱼ^{(m-1)} ρ_{m-j}
    kappa <- (acf[m + 1L] - s) / d_vec[m]
    phi_list[[m]] <- c(prev - kappa * rev(prev), kappa)
    d_vec[m + 1L] <- d_vec[m] * (1.0 - kappa * kappa)
  }
  list(phi = phi_list, d = d_vec)
}

corFactor.corSAR24 <- function(object, ...) {
  covariate <- nlme::getCovariate(object)
  if (is.null(covariate)) stop("corSAR24: not initialized")

  pars  <- coef(object, unconstrained = FALSE)
  rho   <- pars["Rho"]
  Phi   <- pars["Phi"]

  acf24 <- .sar24_acf(rho, Phi, 24L)          # length 25; [1]=1=ρ₀
  sig2  <- max(1.0 - rho * acf24[2L] - Phi * acf24[25L], 1e-10)
  isq   <- 1.0 / sqrt(sig2)                   # 1/√σ²_η for stationary rows

  dl <- .dl_recursion(acf24, 24L)             # startup block, converges at m=24

  # Pre-allocate the full output vector once (avoids the ~3× peak allocation
  # that the lapply+unlist approach causes, which OOMs on large datasets).
  ns     <- lengths(covariate)
  result <- double(sum(ns * ns))              # Σ nᵢ² doubles, initialised to 0
  logDet <- 0.0
  pos    <- 1L                                # current write position (1-indexed)

  for (i in seq_along(covariate)) {
    n   <- ns[i]
    n2  <- n * n
    L   <- matrix(0.0, n, n)

    L[1L, 1L] <- 1.0                          # row 1: marginal distribution

    # Rows 2..min(n,25): exact startup via Durbin-Levinson
    t_end <- min(n, 25L)
    for (t in seq_len(t_end - 1L) + 1L) {
      m   <- t - 1L
      phi <- dl$phi[[m]]
      isd <- 1.0 / sqrt(max(dl$d[m + 1L], 1e-10))
      L[t, t]            <- isd
      L[t, (t - 1L):1L]  <- -phi * isd       # L[t,t-j] = -φⱼ/√dₘ
    }

    # Rows 26..n: stationary innovation filter (vectorised)
    if (n >= 26L) {
      tr <- 26L:n
      L[cbind(tr, tr)]        <- isq
      L[cbind(tr, tr - 1L)]   <- -rho * isq
      L[cbind(tr, tr - 24L)]  <- -Phi * isq
    }

    logDet <- logDet + sum(log(diag(L)))
    result[pos:(pos + n2 - 1L)] <- as.double(L)  # copy into pre-allocated vector
    pos <- pos + n2
    # L is now eligible for GC; only `result` needs to stay alive
  }

  attr(result, "logDet") <- logDet
  result
}

# O(n) band-multiply: y = t(L) %*% x for the SAR(1,24) L, applied group-by-group.
# t(L)[s,j] = L[j,s]: for j >= s (upper triangular).
# Non-zeros:
#   diagonal j=s:    isd_st[s] (startup s<=25) or isq (s>=26)
#   lag-1 from stat: L[s+1,s] = -rho*isq  (when s+1 >= 26, i.e. s >= 25)
#   lag-24 from stat: L[s+24,s] = -Phi*isq (when s+24 >= 26, i.e. s >= 2)
#   startup off-diag: L[t,s] = -phi_t[t-s]*isd_t for 2<=t<=25, s<t
recalc.corSAR24 <- function(object, conLin, ...) {
  covariate <- nlme::getCovariate(object)
  if (is.null(covariate)) stop("corSAR24: not initialized")
  pars  <- coef(object, unconstrained = FALSE)
  rho   <- pars["Rho"]
  Phi   <- pars["Phi"]
  acf24 <- .sar24_acf(rho, Phi, 24L)
  sig2  <- max(1.0 - rho * acf24[2L] - Phi * acf24[25L], 1e-10)
  isq   <- 1.0 / sqrt(sig2)
  dl    <- .dl_recursion(acf24, 24L)
  isd_st <- 1.0 / sqrt(pmax(dl$d[1L:25L], 1e-10))  # isd for startup rows 1..25

  Xy  <- conLin$Xy
  off <- 0L

  for (i in seq_along(covariate)) {
    n  <- length(covariate[[i]])
    ix <- off + seq_len(n)
    x  <- Xy[ix, , drop = FALSE]       # n × p
    y  <- matrix(0.0, n, ncol(x))

    # Diagonal
    t_up <- min(n, 25L)
    y[seq_len(t_up), ] <- x[seq_len(t_up), , drop = FALSE] * isd_st[seq_len(t_up)]
    if (n >= 26L) y[26L:n, ] <- x[26L:n, , drop = FALSE] * isq

    # Stationary lag-1 off-diagonal: y[s] -= rho*isq * x[s+1], s >= 25, s <= n-1
    if (n >= 26L) {
      s2 <- n - 1L
      y[25L:s2, ] <- y[25L:s2, ] - rho * isq * x[26L:n, , drop = FALSE]
    }

    # Stationary lag-24 off-diagonal: y[s] -= Phi*isq * x[s+24], s >= 2, s <= n-24
    if (n >= 26L) {
      s2 <- n - 24L
      if (2L <= s2)
        y[2L:s2, ] <- y[2L:s2, ] - Phi * isq * x[26L:n, , drop = FALSE]
    }

    # Startup off-diagonals: L[t,s] = -phi_t[t-s]*isd_t for t in 2..min(25,n), s < t
    t_max <- min(n, 25L)
    if (t_max >= 2L) {
      for (t in 2L:t_max) {
        phi_t <- dl$phi[[t - 1L]]
        isd_t <- isd_st[t]
        xt    <- x[t, ]
        for (s in 1L:(t - 1L))
          y[s, ] <- y[s, ] - phi_t[t - s] * isd_t * xt
      }
    }

    Xy[ix, ] <- y
    off <- off + n
  }
  conLin$Xy <- Xy
  conLin
}

# ── Fit SAR(1,24) helper ─────────────────────────────────────────────────────
ctrl_sar24 <- lmeControl(opt = "nlminb", maxIter = 100, msMaxIter = 50,
                          tolerance = 1e-3, niterEM = 10)

fit_sar24 <- function(fixed_formula, rho0 = 0.7, Phi0 = 0.2) {
  lme(fixed       = fixed_formula, data = df_m,
      random      = ~ T_out_c | dwelling,
      correlation = corSAR24(c(rho0, Phi0), form = ~ 1 | dwelling),
      method      = "ML",
      control     = ctrl_sar24)
}

# ── Constant-only SAR(1,24) null ─────────────────────────────────────────────
cat("\nFitting M0_sar24 (null, SAR24) ...\n"); m0_sar24 <- fit_sar24(T_in ~ 1)
cat(sprintf("  AIC = %.2f\n", AIC(m0_sar24)))

# ── M11-M13: SAR(1,24) ────────────────────────────────────────────────────────
cat("\nFitting M11 (M1 + SAR24) ...\n"); m11 <- fit_sar24(T_in ~ T_out_c)
cat(sprintf("  AIC = %.2f\n", AIC(m11)))
cat("Fitting M12 (M2 + SAR24) ...\n")
m12 <- fit_sar24(T_in ~ T_out_c + sin_h + cos_h)
cat(sprintf("  AIC = %.2f\n", AIC(m12)))
cat("Fitting M13 (M3 + SAR24) ...\n")
m13 <- fit_sar24(T_in ~ T_out_c + factor(hour_of_day))
cat(sprintf("  AIC = %.2f\n", AIC(m13)))

# ── M14-M15: SAR(1,24) + building characteristics ─────────────────────────────
# Warm-start (rho, Phi) from the converged SAR models with matching ToD structure.
.p14 <- coef(m13$modelStruct$corStruct, unconstrained = FALSE)
cat("Fitting M14 (M5 + SAR24) ...\n")
m14 <- fit_sar24(
  as.formula(paste("T_in ~ T_out_c + factor(hour_of_day) +", BIN, "+", CAT)),
  rho0 = .p14["Rho"], Phi0 = .p14["Phi"])
cat(sprintf("  AIC = %.2f\n", AIC(m14)))

.p15 <- coef(m12$modelStruct$corStruct, unconstrained = FALSE)
cat("Fitting M15 (M7 + SAR24) ...\n")
m15 <- fit_sar24(
  as.formula(paste("T_in ~ T_out_c + sin_h + cos_h +", BIN, "+", CAT)),
  rho0 = .p15["Rho"], Phi0 = .p15["Phi"])
cat(sprintf("  AIC = %.2f\n", AIC(m15)))

# ── Variance-component extractor (handles static / AR1 / SAR24) ───────────────
var_components <- function(mod) {
  re_cov   <- as.matrix(nlme::getVarCov(mod, type = "random.effects"))
  sig2_u0  <- re_cov[1, 1]
  sig2_u1  <- re_cov[2, 2]
  sig_u01  <- re_cov[1, 2]
  sig2_eps <- mod$sigma^2
  cs       <- mod$modelStruct$corStruct
  if (is.null(cs)) {
    rho <- NA_real_; Phi <- NA_real_
  } else if (inherits(cs, "corSAR24")) {
    p   <- coef(cs, unconstrained = FALSE)
    rho <- unname(p["Rho"]); Phi <- unname(p["Phi"])
  } else {
    rho <- coef(cs, unconstrained = FALSE)[[1]]; Phi <- NA_real_
  }
  list(sig2_u0 = sig2_u0, sig2_u1 = sig2_u1, sig_u01 = sig_u01,
       sig2_eps = sig2_eps, rho = rho, Phi = Phi)
}

obs_total_variance <- function(mod, data) {
  re_cov     <- as.matrix(nlme::getVarCov(mod, type = "random.effects"))
  x          <- data$T_out_c
  Z          <- cbind(1, x)
  rand_var_i <- rowSums((Z %*% re_cov) * Z)
  mean(rand_var_i) + mod$sigma^2
}

# ── Summary comparison table ──────────────────────────────────────────────────
all_models <- list(
  M0_static = m0_static,
  M1=m1, M2=m2, M3=m3, M4=m4, M5=m5, M6=m6, M7=m7,
  M0_ar1 = m0_ar1,
  M8=m8, M9=m9, M10=m10,
  M0_sar24 = m0_sar24,
  M11=m11, M12=m12, M13=m13, M14=m14, M15=m15
)
vc_list    <- lapply(all_models, var_components)

ref_static <- list(M4="M3", M5="M3", M6="M2", M7="M2",
                   M8="M1", M9="M2", M10="M3",
                   M11="M1", M12="M2", M13="M3",
                   M14="M5", M15="M7")

cat("\n")
cat(strrep("=", 108), "\n")
cat("MODEL COMPARISON: static (M1-M3)  |  static+bldg (M4-M7)  |  AR(1) (M8-M10)  |  SAR(1,24) (M11-M13)  |  SAR+bldg (M14-M15)\n")
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
  if (nm %in% c("M3", "M7", "M10", "M13")) cat("  ", strrep("-", 102), "\n")
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
  if (!is.na(vc$rho) && is.na(vc$Phi)) cor_str <- sprintf("  AR(1): ρ=%.4f", vc$rho)
  if (!is.na(vc$Phi))                  cor_str <- sprintf("  SAR(1,24): ρ=%.4f  Φ=%.4f",
                                                           vc$rho, vc$Phi)
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
  file.path(diag_dir, "southwest_may_sep_diagnostics.parquet"))

# Random effects (BLUPs): one row per dwelling
ranef_df <- data.frame(dwelling = rownames(ranef(m1)))
for (nm in names(all_models)) {
  re <- ranef(all_models[[nm]])
  ranef_df[[paste0("u0_", nm)]] <- re[rownames(ranef_df), 1]
  ranef_df[[paste0("u1_", nm)]] <- if (ncol(re) >= 2) re[rownames(ranef_df), 2] else NA_real_
}
write.csv(ranef_df,
  file.path(diag_dir, "southwest_may_sep_ranef.csv"), row.names = FALSE)

# AIC table
aic_df <- data.frame(
  model  = names(all_models),
  aic    = sapply(all_models, AIC),
  loglik = sapply(all_models, function(m) as.numeric(logLik(m)))
)
write.csv(aic_df, file.path(diag_dir, "southwest_may_sep_aic.csv"), row.names = FALSE)

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
write.csv(vc_df, file.path(diag_dir, "southwest_may_sep_varcomp.csv"), row.names = FALSE)

cat("  -> diagnostics/southwest_may_sep_diagnostics.parquet\n")
cat("  -> diagnostics/southwest_may_sep_ranef.csv\n")
cat("  -> diagnostics/southwest_may_sep_aic.csv\n")
cat("  -> diagnostics/southwest_may_sep_varcomp.csv\n")

cat("\nDone.\n")
