# the following packages are required
# install.packages("readr")
# install.packages("pracma")
# devtools::install_github("tobiaskley/inferchange")
# devtools::install_github("haeran-cho/fnets")
# devtools::install_github("cykbennie/fbi")

library(inferchange)

library(fbi)

# fredmd.csv (monthly data, containing the period from 1970-01 to 2024-01) to be downloaded from https://research.stlouisfed.org/econ/mccracken/fred-databases/
data <- fredmd("fredmd.csv")
data <- as.data.frame(data)
data$date <- format(as.Date(data$date, "%Y-%m-%d"), "%Y-%m")

# to remove many NAs at the beginning
# we are using only a balanced dataset
data <- data[data$date > "1959-12", ]
data <- data[, !is.na(data[1, ])]
data <- data[, apply(data, 2, function(z){ sum(is.na(z)) }) == 0]

x0 <- t(as.matrix(data[, - 1]))
x0 <- x0[!(rownames(x0) %in% c("NONBORRES", "M1SL")), ]

dim(x0) # n = 774, p = 109

r <- 6

# data standardisation 

x <- x0 - apply(x0, 1, mean)
sc <- apply(x, 1, sd)
sx <- x / sc

# truncation for removing possibly anomalous obs (cf. Barigozzi et al (2024) https://arxiv.org/abs/2407.09390)
tau <- quantile(abs(sx), .999)
tx <- sx * (abs(sx) <= tau) + tau * sign(sx) * (abs(sx) > tau)

Gamma <- tx %*% t(tx) / dim(tx)[2]
sv <- svd(Gamma, nu = r, nv = 0)
f <- t(sv$u) %*% tx / sqrt(dim(tx)[1])

# one-step prediction model

y <- tx[rownames(sx) == "INDPRO", -1]
X <- cbind(t(f), t((tx - sv$u %*% t(sv$u) %*% tx)[!(rownames(sx) %in% c("INDPRO")), ]))
X <- X[- nrow(X), ]

fredmd_data <- list(X = X, y = y, date = data$date[-length(data$date)])
save(fredmd_data, file = "fredmd_data.RData")

# change point analysis
out <- McScan(X, y, method = 'auto', post = FALSE)
plot(out)$plt_path
out$cp
data$date[out$cp]

# for cp1
s <- 1
k <- 170
e <- 605

# for cp2
s <- 170 + 1
k <- 605
e <- 717

# for cp3
s <- 605 + 1
k <- 717
e <- dim(X)[1]

# bootstrap ci

set.seed(12345)
ci_boot <- ci_delta(X = X[s:e, ], y = y[s:e], k = k - s + 1, M = 9999, alpha = .1, bootstrap = TRUE)
plot(ci_boot)
print(ind <- which(ci_boot$ci[, 1] * ci_boot$ci[, 2] > 0)) # rejected hypotheses H_0i: delta_i = 0
diff(ci_boot$ci[1, ]) # width

# sampling-based ci

set.seed(12345)
ci_noboot <- ci_delta(X = X[s:e, ], y = y[s:e], k = k - s + 1, M = 9999, alpha = .1, bootstrap = FALSE)
plot(ci_noboot)
print(ind <- which(ci_noboot$ci[, 1] * ci_noboot$ci[, 2] > 0)) # rejected hypotheses H_0i: delta_i = 0
diff(ci_noboot$ci[1, ]) # width

