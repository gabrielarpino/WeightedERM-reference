library(glmnet)
library(MASS)

# Preserved from gabrielarpino/myglm-2 commit c4d934f for conformance testing.
bsachgpt <- function(X, Y, kmax, c1, m1, delta) {
  n <- length(Y)
  v <- length(Y)
  u <- 1
  p <- length(X[1, ])
  alpha <- c(u, v)
  hs <- vector(mode = "numeric", length = 0L)
  shat <- vector(mode = "numeric", length = 0L)
  alphalength <- vector(mode = "numeric", length = 0L)
  for (z in 1:kmax) {
    alphalength[z] <- length(alpha)
    for (i in 1:(length(alpha) - 1)) {
      for (s in (alpha[i] + m1):(alpha[i + 1] - m1)) {
        if ((alpha[i] + m1) > (alpha[i + 1] - m1)) {
          break
        }
        hs[1:(alpha[i] + m1)] <- rep(10000, (alpha[i] + m1))
        betahat33 <- matrix(0, nrow = (alpha[i + 1] - m1), ncol = p)
        betahat55 <- matrix(0, nrow = (alpha[i + 1] - m1), ncol = p)
        lam_l <- c1 * (
          sqrt(2 * log(2 * p) / (s - alpha[i] + 1)) +
            log(2 * p) / (s - alpha[i] + 1)
        )
        betahat22 <- as.vector(coef(
          glmnet::glmnet(
            X[alpha[i]:s, ], Y[alpha[i]:s],
            intercept = FALSE, family = "binomial"
          ),
          s = lam_l
        ))
        betahat33[s, ] <- betahat22[-1]
        lam_r <- c1 * (
          sqrt(2 * log(2 * p) / (alpha[i + 1] - s)) +
            log(2 * p) / (alpha[i + 1] - s)
        )
        betahat44 <- as.vector(coef(
          glmnet(
            X[(s + 1):alpha[i + 1], ], Y[(s + 1):alpha[i + 1]],
            intercept = FALSE, family = "binomial"
          ),
          s = lam_r
        ))
        betahat55[s, ] <- betahat44[-1]
        I <- rep(1, alpha[i + 1] - alpha[i] + 1)
        lam_full <- c1 * (
          sqrt(2 * log(2 * p) / (alpha[i + 1] - alpha[i] + 1)) +
            log(2 * p) / (alpha[i + 1] - alpha[i] + 1)
        )
        betahat66 <- as.vector(coef(
          glmnet(
            X[alpha[i]:alpha[i + 1], ], Y[alpha[i]:alpha[i + 1]],
            intercept = FALSE, family = "binomial"
          ),
          s = lam_full
        ))
        hs[alpha[i]] <- sum(
          log(I + exp(X[alpha[i]:alpha[i + 1], ] %*% betahat66[-1])) -
            Y[alpha[i]:alpha[i + 1]] *
              (X[alpha[i]:alpha[i + 1], ] %*% betahat66[-1])
        ) / n + delta * lam_full
        I1 <- rep(1, s - alpha[i] + 1)
        I2 <- rep(1, alpha[i + 1] - s)
        hs[s] <- sum(
          log(I1 + exp(X[alpha[i]:s, ] %*% betahat33[s, ])) -
            Y[alpha[i]:s] * (X[alpha[i]:s, ] %*% betahat33[s, ])
        ) / n + delta * lam_l +
          sum(
            log(I2 + exp(X[(s + 1):alpha[i + 1], ] %*% betahat55[s, ])) -
              Y[(s + 1):alpha[i + 1]] *
                (X[(s + 1):alpha[i + 1], ] %*% betahat55[s, ])
          ) / n + delta * lam_r
      }
      shat[i] <- which.min(hs)
      if (shat[i] != alpha[i]) {
        alpha <- sort(c(alpha, shat[i]))
        alpha <- alpha[!duplicated(alpha)]
      }
    }
    alphalength[z + 1] <- length(alpha)
    if (alphalength[z] == alphalength[z + 1]) {
      break
    }
  }
  list(
    cpnumber.estimator = length(alpha) - 2,
    cplocation.estimator = alpha
  )
}
