library(tidyverse)
library(glmnet)
library(broom)
library(modelr)
library(ISLR)


# 1. Load Credit data set from ISLR package, convert to tibble, select subset
# of variables, and only 20 rows!
set.seed(76)
credit <- ISLR::Credit %>%
  as_tibble() %>%
  select(Balance, Income, Limit, Rating, Cards, Age, Education, Married) %>%
  sample_n(20)
credit


# 2. Define model formula where credit card balance (debt) is the outcome variable
model_formula <-
  "Balance ~ Income + Limit + Rating + Cards + Age + Education + Married" %>%
  as.formula()

# Fit unregularized multiple regression model and output regression table. The
# unregularized beta-hat coefficients are in the estimate column. Recall that
# this is one "extreme". REMEMBER THESE VALUES!!!
lm(model_formula, data = credit) %>%
  tidy(conf.int = TRUE)

# Recall the other "extreme" is a model that is completely regularized, meaning
# you use none of the predictors, so that y_hat is simply the mean balance.
# REMEMBER THIS VALUE AS WELL!!!
mean(credit$Balance)


# 3. Based on the above model formula, create "model matrix" representation of
# the predictor variables. Note:
# -the model_matrix() function conveniently converts all categorical predictors
# to numerical ones using one-hot encoding
# -we remove the first column corresponding to the intercept because it is
# simply a column of ones.
x_matrix <- credit %>%
  modelr::model_matrix(model_formula, data = .) %>%
  select(-`(Intercept)`) %>%
  as.matrix()

# Compare the original data to the model matrix. What is different?
credit
x_matrix


# 4.a) Fit a LASSO model. Note the inputs
# -Instead of inputing a model formula, you input the corresponding x_matrix and
# outcome variable
# -Setting alpha = 1 sets the regularization method to be LASSO. Setting it to be 0
# sets the regularization method to be "ridge regression", another regulization
# method that we don't have time to cover in this class
# -lambda is complexity/tuning parameter whose value we specify. Here let's
# specify 10, an arbitrarily chosen value
LASSO_fit_a <- glmnet(x = x_matrix, y = credit$Balance, alpha = 1, lambda = 10)
LASSO_fit_a

# Unfortunately the output isn't that informative. Let's use a wrapper function
# that yields a more informative output:
get_LASSO_coefficients <- function(LASSO_fit){
  beta_hats <- LASSO_fit %>%
    broom::tidy(return_zeros = TRUE) %>%
    select(term, estimate, lambda) %>%
    arrange(desc(lambda))
  return(beta_hats)
}
get_LASSO_coefficients(LASSO_fit_a)

# For that value of lambda = 10, we have the beta-hat coefficients that minimizes
# the equation using numerical optimization. Observe how all the
# beta-hats have been shrunk while the beta-hat for Limit variable has been
# "shrunk" to 0 and hence is dropped from the model. Compare above output with
# previously seen "unregularized" regression results
lm(model_formula, data = credit) %>%
  tidy(conf.int = TRUE)


# 4.b) Fit a LASSO model considering TWO lambda tuning/complexity parameters at
# once and look at beta-hats
lambda_inputs <- c(10, 1000)
LASSO_fit_b <- glmnet(x = x_matrix, y = credit$Balance, alpha = 1, lambda = lambda_inputs)
get_LASSO_coefficients(LASSO_fit_b)

# The above output is in tidy/long format, which makes it hard to compare beta-hats
# for both lambda values. Let's convert it to wide format and compare the beta-hats
get_LASSO_coefficients(LASSO_fit_b) %>%
  spread(lambda, estimate)

# Notice how for the larger lambda, all non-intercept beta-hats have been shrunk
# to 0. All that remains is the intercept, whose value is the mean of the y.
# This is because lambda = 1000 penalizes complexity more harshly.


# 4.c) Fit a LASSO model with several lambda tuning/complexity parameters at once
# and look at beta-hats
lambda_inputs <- seq(from = 0, to = 1000)
lambda_inputs
LASSO_fit_c <- glmnet(x = x_matrix, y = credit$Balance, alpha = 1, lambda = lambda_inputs)

# Since we are now considering several possible values of lambda tuning parameter
# let's visualize instead:
get_LASSO_coefficients(LASSO_fit_c) %>%
  # Plot:
  ggplot(aes(x = lambda, y = estimate, col = term)) +
  geom_line() +
  labs(x = "lambda", y = "beta-hat")

# Observe:
# -For lambda = 0 i.e. no complexity penalization, the values of the beta-hats
# are the unregularized lm() multiple regression values from earlier.
# i.e. one extreme
# -At around lambda = 500, all the beta-hat slopes for all our predictor variables
# have been shrunk to 0 and all that remains is the intercept, which is the mean
# y value in our training set. i.e. the other extreme

# However a typical LASSO plot doesn't show the intercept since it is a beta-hat
# value that is not a candidate to be shrunk to zero, so let's remove it from
# our plot:
get_LASSO_coefficients(LASSO_fit_c) %>%
  filter(term != "(Intercept)") %>%
  # Plot:
  ggplot(aes(x = lambda, y = estimate, col = term)) +
  geom_line() +
  labs(x = "lambda", y = "beta-hat")

# It's hard to see in what order the beta-hats get shrunk to 0, so let's zoom-in
# the plot a bit
get_LASSO_coefficients(LASSO_fit_c) %>%
  filter(term != "(Intercept)") %>%
  # Plot:
  ggplot(aes(x = lambda, y = estimate, col = term)) +
  geom_line() +
  labs(x = "lambda", y = "beta-hat") +
  coord_cartesian(xlim=c(0, 500), ylim = c(-10, 10))

# The results are a little compressed on the left-end of the x-axis, so
# let's rescale the x-axis to be on a log10 scale:
get_LASSO_coefficients(LASSO_fit_c) %>%
  filter(term != "(Intercept)") %>%
  # Plot:
  ggplot(aes(x = lambda, y = estimate, col = term)) +
  geom_line() +
  labs(x = "lambda (log10-scale)", y = "beta-hat") +
  coord_cartesian(xlim=c(1, 500), ylim = c(-10, 10)) +
  scale_x_log10()

# Ask yourself, in what order are the variables being shrunk to 0?


# 4.d) Fit a LASSO model with a narrower search grid of lambda tuning/complexity
# parameter values AND such that the lambdas are spaced by multiplicative powers
# of 10, instead of additive differences, and look at beta-hats
lambda_inputs <- 10^seq(from = -5, to = 3, length = 100)
summary(lambda_inputs)
LASSO_fit_d <- glmnet(x = x_matrix, y = credit$Balance, alpha = 1, lambda = lambda_inputs)

# Plot all beta-hats with lambda on log10-scale
LASSO_coefficients_plot <- get_LASSO_coefficients(LASSO_fit_d) %>%
  filter(term != "(Intercept)") %>%
  # Plot:
  ggplot(aes(x = lambda, y = estimate, col = term)) +
  geom_line() +
  labs(x = "lambda (log10-scale)", y = "beta-hat") +
  scale_x_log10()
LASSO_coefficients_plot

# Zoom-in. In what order to the beta-hat slopes get shrunk to 0?
LASSO_coefficients_plot +
  coord_cartesian(xlim = c(10^0, 10^3), ylim = c(-2, 2))


# 5. However, how do we know which lambda value to use? Should we set it to
# yield a less complex or more complex model? Let's use the glmnet package's
# built in crossvalidation functionality, using the same search grid of
# lambda_input values:
lambda_inputs <- 10^seq(from = -5, to = 3, length = 100)
LASSO_CV <- cv.glmnet(
  x = x_matrix,
  y = credit$Balance,
  alpha = 1,
  lambda = lambda_inputs,
  nfolds = 10,
  type.measure = "mse"
)
LASSO_CV

# Alas that output is not useful, so let's broom::tidy() it
LASSO_CV %>%
  broom::tidy() %>%
  rename(mse = estimate)

# What is te smallest estimated mse?
LASSO_CV %>%
  broom::tidy() %>%
  rename(mse = estimate) %>%
  arrange(mse)

# The lambda_star is in the top row. We can extract this lambda_star value from
# the LASSO_CV object:
lambda_star <- LASSO_CV$lambda.min
lambda_star

# What do the all these values mean? For each value of the lambda
# tuning/complexity parameter, let's plot the estimated MSE generated by
# crossvalidation:
CV_plot <- LASSO_CV %>%
  broom::tidy() %>%
  rename(mse = estimate) %>%
  arrange(mse) %>%
  # plot:
  ggplot(aes(x = lambda)) +
  geom_point(aes(y = mse)) +
  scale_x_log10() +
  labs(x = "lambda (log10-scale)", y = "Estimated MSE")
CV_plot

# Zoom-in:
CV_plot +
  coord_cartesian(xlim=c(10^(-2), 10^2), ylim = c(40000, 50000))

# Mark the lambda_star with dashed blue line
CV_plot +
  coord_cartesian(xlim=c(10^(-2), 10^2), ylim = c(40000, 50000)) +
  geom_vline(xintercept = lambda_star, linetype = "dashed", col = "blue")


# 6. Now mark lambda_star in beta-hat vs lambda plot:
LASSO_coefficients_plot +
  geom_vline(xintercept = lambda_star, linetype = "dashed", col = "blue")

# zoom-in:
LASSO_coefficients_plot +
  geom_vline(xintercept = lambda_star, linetype = "dashed", col = "blue") +
  coord_cartesian(ylim = c(-3, 3))

# What are the beta_hat values resulting from lambda_star? Which are shrunk to 0?
get_LASSO_coefficients(LASSO_fit_d) %>%
  filter(lambda == lambda_star)


# 7. Get predictions from f_hat LASSO model using lambda_star
credit <- credit %>%
  mutate(y_hat_LASSO = predict(LASSO_fit_d, newx = x_matrix, s = lambda_star)[,1])
credit


# 8. Train/test framework
credit_train <- credit %>%
  slice(1:10)
credit_test <- credit %>%
  slice(11:20) %>%
  # Remove outcome variable for test set
  select(-Balance)

# model matrix representation of predictor variables for training set:
x_matrix_train <- credit_train %>%
  modelr::model_matrix(model_formula, data = .) %>%
  select(-`(Intercept)`) %>%
  as.matrix()

# model matrix representation of predictor variables for test set:
x_matrix_test <- credit_test %>%
  modelr::model_matrix(model_formula, data = .) %>%
  select(-`(Intercept)`) %>%
  as.matrix()

# The previous didn't work b/c there is no outcome variable Balance in test as
# specified in model_formula. The solution is to create a temporary dummy
# variable of 1's (or any value); it makes no difference since ultimately we
# only care about x values.
x_matrix_test <- credit_test %>%
  # Create temporary outcome variance just to get model matrix to work:
  mutate(Balance = 1) %>%
  modelr::model_matrix(model_formula, data = .) %>%
  select(-`(Intercept)`) %>%
  as.matrix()

# Fit/train model to training set using arbitrarily chosen lambda = 1-
LASSO_fit_train <- glmnet(x = x_matrix_train, y = credit_train$Balance, alpha = 1, lambda = 10)

# Predict y_hat's for test data using model and same lambda = 10.
credit_test <- credit_test %>%
  mutate(y_hat_LASSO = predict(LASSO_fit_train, newx = x_matrix_test, s = 10)[,1])
credit_test
