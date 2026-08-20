library(glmnet)
library(MASS)

#' This function provides change point estimation by Binary Segmentation Algorithm
#' for the high dimensional generalized linear regression model.
#' @export
#' @param X an n*p data matrix, n = number of observations, p = dimension
#' @param Y an n-length response vector
#' @param kmax the maximum number of change points to search for
#' @param c1 a tuning constant for the penalty
#' @param m1 the minimal segment length (in samples)
#' @param delta the fractional minimal length (so that m1 = ceiling(n * delta))
#' @return A list with components\n#'   • cpnumber.estimator: estimated number of change points\n#'   • cplocation.estimator: vector of the estimated change-point locations
#' @importFrom stats coef
#' @importFrom glmnet glmnet
bsa_chgpt <- function(X, Y, kmax, c1, m1, delta) {
  message(">> Running UPDATED bsa_chgpt()")
  message("   Inputs: kmax=", kmax, "  c1=", c1, "  m1=", m1, "  delta=", delta)

  n     <- length(Y)
  p     <- ncol(X)
  alpha <- c(1, n)      # initial segmentation at 1 and n

  for (z in seq_len(kmax)) {
    old_alpha <- alpha
    for (i in seq_len(length(old_alpha) - 1)) {
      a <- old_alpha[i]
      b <- old_alpha[i + 1]
      if ((a + m1) > (b - m1)) next

      losses <- rep(Inf, b)

      # No-split candidate, restored from the authors' original implementation
      # (myglm-2 commit c4d934f). Passing lambda directly is the current
      # package's numerically stable equivalent of the original path fit plus
      # coef(..., s = lam_full) interpolation.
      len_full <- b - a + 1
      lam_full <- c1*(sqrt(2*log(2*p)/len_full) + log(2*p)/len_full)
      fit_full <- glmnet(X[a:b, ], Y[a:b], intercept=FALSE,
                         family="binomial", lambda=lam_full)
      beta_full <- as.vector(coef(fit_full))[-1]
      I_full <- rep(1, len_full)
      losses[a] <- sum(log(I_full + exp(X[a:b, ] %*% beta_full)) -
                       Y[a:b] * (X[a:b, ] %*% beta_full)) / n + delta*lam_full

      for (s in (a + m1):(b - m1)) {
        len_l <- s - a + 1
        lam_l <- c1*( sqrt(2*log(2*p)/len_l) + log(2*p)/len_l )
        fit_l <- glmnet(X[a:s, ], Y[a:s], intercept=FALSE, family="binomial", lambda=lam_l)
        beta_l <- as.vector(coef(fit_l))[-1]
        I_l    <- rep(1, len_l)
        loss_l <- sum(log(I_l + exp(X[a:s, ] %*% beta_l)) -
                      Y[a:s] * (X[a:s, ] %*% beta_l)) / n + delta*lam_l

        len_r <- b - s
        lam_r <- c1*( sqrt(2*log(2*p)/len_r) + log(2*p)/len_r )
        fit_r <- glmnet(X[(s+1):b, ], Y[(s+1):b], intercept=FALSE, family="binomial", lambda=lam_r)
        beta_r <- as.vector(coef(fit_r))[-1]
        I_r    <- rep(1, len_r)
        loss_r <- sum(log(I_r + exp(X[(s+1):b, ] %*% beta_r)) -
                      Y[(s+1):b] * (X[(s+1):b, ] %*% beta_r)) / n + delta*lam_r

        losses[s] <- loss_l + loss_r
      }

      s_best <- which.min(losses)
      if (s_best != a) {
        alpha <- sort(unique(c(alpha, s_best)))
      }
    }
    if (identical(alpha, old_alpha)) {
      message(">> No new splits at iteration ", z, "; stopping early.")
      break
    }
  }

  cpnumber.estimator   <- length(alpha) - 2
  cplocation.estimator <- alpha
  message(">> bsa_chgpt complete: found ", cpnumber.estimator, " change points at ", paste(alpha, collapse=", "))
  list(cpnumber.estimator = cpnumber.estimator,
       cplocation.estimator = cplocation.estimator)
}


#' This function provides change point estimation by Binary Segmentation Algorithm
#' for the high dimensional generalized linear regression model,
#' enforcing exactly kmax change points.
#' @export
#' @param X an n*p data matrix, n = number of observations, p = dimension
#' @param Y an n-length response vector
#' @param kmax the exact number of change points to estimate
#' @param c1 a tuning constant for the penalty
#' @param m1 the minimal segment length (in samples)
#' @param delta the fractional minimal length (so that m1 = ceiling(n * delta))
#' @return A list with components\n#'   • cpnumber.estimator: estimated number of change points (should equal kmax)\n#'   • cplocation.estimator: vector of the estimated change-point locations
#' @importFrom stats coef
#' @importFrom glmnet glmnet
bsa_exactchgpt <- function(X, Y, kmax, c1, m1, delta) {
  message(">> Running UPDATED bsa_exactchgpt()")
  message("   Inputs: kmax=", kmax, "  c1=", c1, "  m1=", m1, "  delta=", delta)

  n     <- length(Y)
  p     <- ncol(X)
  alpha <- c(1, n)

  for (z in kmax:kmax) {
    old_alpha <- alpha
    for (i in seq_len(length(old_alpha) - 1)) {
      a <- old_alpha[i]
      b <- old_alpha[i + 1]
      if ((a + m1) > (b - m1)) next

      losses <- rep(Inf, b)
      # full-segment loss
      len_full <- b - a + 1
      lam_full <- c1*( sqrt(2*log(2*p)/len_full) + log(2*p)/len_full )
      fit_full <- glmnet(X[a:b, ], Y[a:b], intercept=FALSE, family="binomial", lambda=lam_full)
      beta_full <- as.vector(coef(fit_full))[-1]
      I_full    <- rep(1, len_full)
      losses[a] <- sum(log(I_full + exp(X[a:b, ] %*% beta_full)) -
                       Y[a:b] * (X[a:b, ] %*% beta_full)) / n + delta*lam_full

      # split candidates
      for (s in (a + m1):(b - m1)) {
        len_l <- s - a + 1
        lam_l <- c1*( sqrt(2*log(2*p)/len_l) + log(2*p)/len_l )
        fit_l <- glmnet(X[a:s, ], Y[a:s], intercept=FALSE, family="binomial", lambda=lam_l)
        beta_l <- as.vector(coef(fit_l))[-1]
        I_l    <- rep(1, len_l)
        loss_l <- sum(log(I_l + exp(X[a:s, ] %*% beta_l)) -
                      Y[a:s] * (X[a:s, ] %*% beta_l)) / n + delta*lam_l

        len_r <- b - s
        lam_r <- c1*( sqrt(2*log(2*p)/len_r) + log(2*p)/len_r )
        fit_r <- glmnet(X[(s+1):b, ], Y[(s+1):b], intercept=FALSE, family="binomial", lambda=lam_r)
        beta_r <- as.vector(coef(fit_r))[-1]
        I_r    <- rep(1, len_r)
        loss_r <- sum(log(I_r + exp(X[(s+1):b, ] %*% beta_r)) -
                      Y[(s+1):b] * (X[(s+1):b, ] %*% beta_r)) / n + delta*lam_r

        losses[s] <- loss_l + loss_r
      }

      s_best <- which.min(losses)
      if (s_best != a) {
        alpha <- sort(unique(c(alpha, s_best)))
      }
    }
    if (identical(alpha, old_alpha)) {
      message(">> No new splits (exact) on forced iteration; stopping.")
      break
    }
  }

  cpnumber.estimator   <- length(alpha) - 2
  cplocation.estimator <- alpha
  message(">> bsa_exactchgpt complete: found ", cpnumber.estimator,
          " change points at ", paste(alpha, collapse=", "))
  list(cpnumber.estimator = cpnumber.estimator,
       cplocation.estimator = cplocation.estimator)
}


#' This function wraps around bsachgpt.
#' @export
#' @importFrom MASS mvrnorm
#' @importFrom stats rbinom rnorm runif
#' @param n p=dimension
#' @param p n=observation
#' @param Sigma pxp covariance of rows of X
#' @param kmax the maximum change point number.
#' @param delta the minimal interval length specified by users.
#' @return  A list including: the estimated change point number, the estimated
#'   change point locations.
bsawrapper<-function(n, p, Sigma, kmax, delta){
  #=================== initial settings====================
  # n=200       # sample size
  # p=10    # data dimension
  # Sigma<-matrix(0,nrow = p,ncol = p)   #covariance matrix
  # for (i in 1:p){
  #   for (j in 1:p){
  #     Sigma[i,j]<-0.6^(abs(i-j))
  #   }
  # }
  print("Entered bsawrapper")
  tau0<-0.5  # true change point location
  # kmax=6   # upper bound of the number of change point
  #==== tuning parameter(specifed by users)==========
  # delta=0.1  # fractional length of the shortest interval
  # (i.e. distance between nearest adjacent changepoints as a fraction of n)
  c1<-0.18 # lambda related parameter
  #===============signal jump size==================
  m1=ceiling(n*delta)   # size of the shortest interval
  signaljump<-20*sqrt(log(p)/(delta*n))  # signal jump
  signalsupport<-ceiling(log(p))
  signalsupport.range<-0.3*p

  #===============generating change point model settings============
  beta1<-rep(0,p)   #  regression coefficient 1
  beta2<-rep(0,p)   #  regression coefficient 2
  for (i in sample(1:(signalsupport.range),signalsupport)) {
    beta1[i]<-runif(1,min=0,max=2)
  }
  for (i in sample(1:(signalsupport.range),signalsupport)) {
    beta2[i]<-runif(1,min=0,max=2)+runif(1,min=0.5,max=signaljump)
  }

  X<-mvrnorm(n,rep(0,p),Sigma)  # design matrix
  error<-rnorm(n)      # error term
  Y1<-vector(mode = "numeric",length=0L)  # response variable before change point
  Y2<-vector(mode = "numeric",length=0L)  # response variable after change point
  O1<-vector(mode = "numeric",length=0L)
  O2<-vector(mode = "numeric",length=0L)
  R1=X[1:(tau0*n),]%*%beta1+error[1:(tau0*n)]
  R2=X[(tau0*n+1):n,]%*%beta2+error[(tau0*n+1):n]
  for (i in 1:(n/2)){
    O1[i]<-exp(R1[i])/(1+exp(R1[i]))
    O2[i]<-exp(R2[i])/(1+exp(R2[i]))
    Y1[i]=rbinom(1,1,O1[i])
    Y2[i]=rbinom(1,1,O2[i])
  }

  Y=c(Y1,Y2)   # response variable


  #=======estimate change points by bsa=======
  reg<-bsa_chgpt(X,Y,kmax,c1,m1,delta)
  cpnumber<-reg$cpnumber.estimator
  cplocation<-reg$cplocation.estimator
  #=========output==========
  print("Reached end of bsawrapper")
  print(cpnumber)
  print(cplocation)
  list<-list(cpnumber=cpnumber,cplocation=cplocation)
  return(list)
}



#' This function wraps around bsa.chgpt.
#' @export
#' @importFrom MASS mvrnorm
#' @importFrom stats rbinom rnorm runif
#' @param X an n*p data matrix, n=observation, p=dimension
#' @param Y an n dimensional data vector, n=observation
#' @param kmax the maximum change point number.
#' @param delta the minimal interval length specified by users.
#' @return  A list including: the estimated change point number, the estimated
#'   change point locations.
bsa_wrapper<-function(X, Y, kmax, delta){
  #=================== initial settings====================
  # n=200       # sample size
  # p=10    # data dimension
  # Sigma<-matrix(0,nrow = p,ncol = p)   #covariance matrix
  # for (i in 1:p){
  #   for (j in 1:p){
  #     Sigma[i,j]<-0.6^(abs(i-j))
  #   }
  # }
  print("Entered bsa_wrapper")
  n<-length(Y)
  # tau0<-0.5  # true change point location
  # kmax=6   # upper bound of the number of change point
  #==== tuning parameter(specifed by users)==========
  # delta=0.1  # fractional length of the shortest interval
  # (i.e. distance between nearest adjacent changepoints as a fraction of n)
  c1<-0.18 # lambda related parameter
  #===============signal jump size==================
  m1=ceiling(n*delta)   # size of the shortest interval

  #=======estimate change points by bsa=======
  reg<-bsa_chgpt(X,Y,kmax,c1,m1,delta)
  cpnumber<-reg$cpnumber.estimator
  cplocation<-reg$cplocation.estimator
  #=========output==========
  print("Reached end of bsa_wrapper")
  print(cpnumber)
  print(cplocation)
  list<-list(cpnumber=cpnumber,cplocation=cplocation)
  return(list)
}



#' This function wraps around bsa.exactchgpt.
#' @export
#' @importFrom MASS mvrnorm
#' @importFrom stats rbinom rnorm runif
#' @param X an n*p data matrix, n=observation, p=dimension
#' @param Y an n dimensional data vector, n=observation
#' @param kmax the maximum change point number.
#' @param delta the minimal interval length specified by users.
#' @return  A list including: the estimated change point number, the estimated
#'   change point locations.
bsa_exact_wrapper<-function(X, Y, kmax, delta){
  #=================== initial settings====================
  # n=200       # sample size
  # p=10    # data dimension
  # Sigma<-matrix(0,nrow = p,ncol = p)   #covariance matrix
  # for (i in 1:p){
  #   for (j in 1:p){
  #     Sigma[i,j]<-0.6^(abs(i-j))
  #   }
  # }
  print("Entered bsa_exact_wrapper")
  n<-length(Y)
  # tau0<-0.5  # true change point location
  # kmax=6   # upper bound of the number of change point
  #==== tuning parameter(specifed by users)==========
  # delta=0.1  # fractional length of the shortest interval
  # (i.e. distance between nearest adjacent changepoints as a fraction of n)
  c1<-0.18 # lambda related parameter
  #===============signal jump size==================
  m1=ceiling(n*delta)   # size of the shortest interval

  #=======estimate change points by bsa=======
  reg<-bsa_exactchgpt(X,Y,kmax,c1,m1,delta)
  cpnumber<-reg$cpnumber.estimator
  cplocation<-reg$cplocation.estimator
  #=========output==========
  print("Reached end of bsa_exact_wrapper")
  print(cpnumber)
  print(cplocation)
  list<-list(cpnumber=cpnumber,cplocation=cplocation)
  return(list)
}