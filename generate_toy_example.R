set.seed(76)

# number of observations to simulate
n <- 100

# Using a correlation matrix (let's assume that all variables
# have unit variance
M <- matrix(
  c(1, 0.7, 0.7, 0.5,
    0.7, 1, 0.6, 0.3,
    0.7, 0.6, 1, 0.3,
    0.5, 0.3, 0.3, 1),
  nrow=4, ncol=4)

# Cholesky decomposition
L <- chol(M)
nvars <- dim(L)[1]

# R chol function produces an upper triangular version of L
# so we have to transpose it.
# Just to be sure we can have a look at t(L) and the
# product of the Cholesky decomposition by itself
#t(L)
#t(L) %*% L

# Random variables that follow an M correlation matrix
r <- t(L) %*% matrix(rnorm(nvars*n), nrow = nvars, ncol = n) %>%
  t()
colnames(r) <- c("weight", "height", "inseam", "age")
rdata <- r %>%
  as_tibble() %>%
  select(inseam, weight, height, age)

# Write to CSV
write_csv(rdata, path = "toy_example.csv")


