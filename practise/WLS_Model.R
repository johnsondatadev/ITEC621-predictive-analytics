library(MASS)

data("Boston")

head(Boston)

# Step 1: Fit the OLS (Regression) Model
lm.ols <- lm(medv ~ ., data = Boston)

# Step 2: Generate the absolute values for the residuals
abs.res <- abs(residuals(lm.ols))

# Step 3: Fit a model for the absolute residuals as a function of the fitted values
# Predicting the absolute values of the residuals with the fitted values
lm.abs.res <- lm(abs.res ~ fitted(lm.ols))

# Step 4: Get the inverse square of the fitted values of the new model
wts <- 1 / fitted(lm.abs.res) ^ 2

# Fit the WLS model with the weights argument in the usual OLS model syntax
lm.wls <- lm(medv ~ ., data = Boston, weights = wts)

# Get the summary output
summary(lm.wls)