#!/usr/bin/env Rscript

pkg <- normalizePath("myglm-3")
source("tests/fixtures/myglm_c4d934f_bsa.R")
original_bsachgpt <- bsachgpt
source(file.path(pkg, "R", "bsa.R"))

make_data <- function(with_change) {
  set.seed(91)
  n <- 200L
  p <- 20L
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_left <- c(rep(3, 5), rep(0, p - 5))
  beta_right <- if (with_change) c(rep(-3, 5), rep(0, p - 5)) else beta_left
  eta <- c(
    X[1:100, , drop = FALSE] %*% beta_left,
    X[101:200, , drop = FALSE] %*% beta_right
  )
  list(X = X, Y = as.integer(eta > 0))
}

check_case <- function(with_change) {
  dat <- make_data(with_change)
  old <- original_bsachgpt(
    dat$X, dat$Y, kmax = 2L, c1 = 0.18, m1 = 20L, delta = 0.1
  )
  new <- bsa_chgpt(
    dat$X, dat$Y, kmax = 2L, c1 = 0.18, m1 = 20L, delta = 0.1
  )
  stopifnot(identical(
    as.integer(new$cplocation.estimator),
    as.integer(old$cplocation.estimator)
  ))
  new
}

strong <- check_case(TRUE)
stopifnot(identical(as.integer(strong$cplocation.estimator), c(1L, 100L, 200L)))
null <- check_case(FALSE)
stopifnot(identical(as.integer(null$cplocation.estimator), c(1L, 200L)))
cat(
  "myglm3 bsa conformance: strong=",
  paste(strong$cplocation.estimator, collapse = ","),
  "; null=",
  paste(null$cplocation.estimator, collapse = ","),
  "\n",
  sep = ""
)
