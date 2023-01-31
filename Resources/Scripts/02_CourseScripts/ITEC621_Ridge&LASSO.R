########################################
# ITEC 621 Ridge and LASSO Regressions #
########################################

# Filename: ITEC621_RidgeLASSO.R
# Prepared by J. Alberto Espinosa
# Last updated on 3/14/2022

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: the material covered in this script ASSUMES that you have gone through the self-paced course KSB-999 R Overview for Business Analytics and are thoroughly familiar with the concepts illustrated in the RWorkshopScripts.R script. If you have not completed this workshop and script exercises, please stop here and do that first.


########################################
#                INDEX                 #
########################################

## Dimensionality

## Ridge Regression
# Optimal Ridge Lambda
# Predicting with Ridge

## LASSO Regression
# Optimal LASSO Lambda
# Predicting with LASSO

########################################
#             END OF INDEX             #
########################################


## Dimensionality

# Some (the book authors) refer to this issue as the "curse of dimensionality". As you add more and more predictors to a model, and as you add complexity to the model specification (e.g., high polynomials) the model will fit the data better and fit indices will improve. In some cases, the fit may even be perfect. For example, if you grow a tree until it has the same number of branches as you have data points, the fit will be perfect because the model will touch every single data point. Similarly, if you fit a smoothing spline model, you can set your parameters so that the resulting regression model touches every single point in the dataset.

# In addition, complex modesl with lots of predictors, multicollinearity will begin to escalate, some times rapidly.

# Naturally, these are major concerns in predictive modeling because we are "over-fitting" the model. This causes fit indices like R-squared, MSE and deviance to improve "artificially". However, if we test the model with different data, overfitted models will have higher variance, and these predictions will not be as accurate. 

# Overfitting and multi-collinearity are two of the many problems of dimensionality. The methods presented in this section are ideally suited to address dimensionality issues. There are many types of methods that address and correct for dimensionality. The 2 types of methods we will cover in this class are: (1) Regularization or Shrinkage (covered here); and (2) Dimension Reduction (covered in the next section).

# Regularization, Shrinkage and Penalized Methods. 

# These three methods are one and the same. People use different names, but they all refer to families of methods in which coefficients are artificially shrunk to reduce multicollinearity. When you shrink the coefficients, you are actually "biasing" the coefficients, but on the upside shrinkage reduces the model variance.

# How much do you shrink the coefficients? It depends. The shrinkage factor lambda is a "tuning parameter", meaning that you can shrink coefficients to various degrees and then select the shrinkage that yields the best cross-validation testing results (i.e., best predictive accuracy and lowest variance)

# In this section we explore popular regularization approaches: Ridge Regressopm and LASSO (Least Absolute Shrinkage and Selection Operator). They both aim to shrink the coefficients, such that less important variables are not necessarily removed, but they have a minimal influence as predictors. The magnitude of the shrinkage can be controlled with a tuning parameter called "shinkage factor" or simply "Lambda".


## Ridge Regression

# Ridge regression computes different coefficients than OLS, so they tend to be more biased than OLS. But by shrinking coefficients that may cause multi-collinearity, we reduce the variance of the model.

# OLS vs. Ridge

# The relationship between the Ridge and OLS coefficients is: Beta(Ridge) = Beta(OLS)/(1+Lambda), where Lambda is a tuning parameter that can take any value:

# If Lambda = 0, ridge and OLS regressions produce the exact same coefficients. 1/(1+Lambda) is the "shrinkage" factor: As Lambda gets larger, the coefficients get smaller (i.e., less important). If Lambda is extremely large, then all coefficients get "crushed" to almost nothing. Ridge does not crush coefficients to 0 (except when Lambda is infinite), but LASSO has a mathematical algorithm that shrinks coefficients to 0 when they are small enough.

# Also, because the Lambda shrinkage factor is not applied on the intercept, extreme values of Lambda yield the null model (just the intercept).

# Because Lambda causes the coefficients to shrink, they are "biased" compared to the unbiased OLS coefficients. So, Ridge regression will generally imrprove the MSE accuracy, but beyond a certain point (usually around 10) the coefficients will be too biased to be useful. Generally speaking Ridge regression works best when OLS estimates have high variance.

# GOAL: find the best value of lambda that minimizes the cross-validation MSE to identify the optimal amount of shrinkage.

# Let's run a Ridge regression with several lambdas and find the best lambda that minimizes the 10-Fold cross validation MSE.

# Note about glmnet(): it was developed by the ISLR textbook lead authors Trevor Hastie and Rob Tibshirani, along with two of their colleagues, Jerome Friedman and Noha Simon. See:

browseURL("https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html")

library(glmnet) # Contains functions for Ridge and LASSO
library(ISLR) # Contains the Hitters data set

# Like we did above, we need to omit records with missing data

Hitters <- na.omit(Hitters) 

# Let's minimize use of scientific notation

options(scipen = 4)

# Note: this package has a different syntax and it requires defining an X matrix (predictors) and a Y vector (response values) because this package does not use the y ~ x1 + x2 etc. syntax. we can use the model.matrix(){stats} function to create model matrices with just the predictor variables. Also, we need to remove qualitative variables because glmnet() only takes numerical data.

# IMPORTANT: the book does not explain this, but note that we add [,-1] at the end. This so because the model.matrix() function will include a column full of 1's as the first column, which represents the intercept. If you don't remove the first column, your Ridge models will have 2 intercepts and the coefficients will be slightly off. Because Ridge is a standardized and centered regression model, it is best to remove the intercept. The [,-1] removes that first column from the x matrix and yields a Ridge regression model without an intercept. 

# The [,-1] removes that first column full of 1's from the x matrix.

x <- model.matrix(Salary ~ ., data = Hitters)[, -1]
x # Take a look

y <- Hitters$Salary # y vector with just the outcome variable
y # Take a look

# alpha = 0 fits a Ridge Regression for a variety of descending lambda values
# alpha = 1 to fit a LASSO regression
# 0 < alpha < 1 to fit an elastic net (combination Ridge and LASSO)

ridge.mod <- glmnet(x, y, alpha = 0) 

ridge.mod # take a look (%dev is the fraction of deviance explained for 100 different lambdas - change with "nlambda" argument)

# In glmnet(), use family=“binomial” for logit; and family=“poisson” for count data models.

# Note: Ridge regression standardizes the coefficients to avoid issues of scale invariance (i.e., OLS coefficients are scale invariant, so if we change inches with feet, the coefficients change proportionally; because of the shrinkage factor, Ridge coefficients are not scale invariant, thus the need to standardize the regression). However, Ridge regression can be run without standardizing coefficients simply by specifying "standardize=FALSE".

# l1 Norm (LASSO) vs. l2 Norm (Ridge)

# l2 Norm is the square root of the sum of the squared coefficients in a Ridge regression. As lambda increases, the coefficients shring more and more, proportionally.l2 Norm measures the total amount of shrinkage provided by the particular lambda value used. The smaller l2, the larger the shrinkage. l2 Norm = square root of the sum of the model's squared coefficients.

# - l1 Norm: is the sum of the absolute values of the coefficients 
#      in a LASSO regression. Like l2, it measures the total amount
#      of shrinkage obtained by a particular lambda value used. The
#      smaller l1 the larger the shrinkage

# You can plot the l2 Norm (note that l2 Norm is erroneously labeled l1 Norm)

plot(ridge.mod, label = T)

# To compute the l2 Norm for any lambda, for example 50

l2.norm.50 <- sqrt(sum(coef(ridge.mod)[-1, 50] ^ 2)) 
l2.norm.50

# To get the specific Ridge coefficients for a value of lambda:

coef(ridge.mod, s = 0) # No shrinkage, same as OLS
coef(ridge.mod, s = 50) # Some lambda=50
coef(ridge.mod, s = 10 ^ 20) # Extreme shrinkage, coefficients close to 0


# Optimal Ridge Lambda

# Now let's find the best lambda value that minimizes the cross-validation MSE

# The cv.glmnet(){glmnet} function does cross-validation testing

RNGkind(sample.kind = "default") # To use the R default RNG
set.seed(1) # To get repeatable results

cv.10Fold <- cv.glmnet(x, y, alpha = 0) # 10-Fold is the default
# Use the parameter nfolds = xx to change the number of folds

# List all lambdas and corresponding MSE's

round(cbind("Lambda" = cv.10Fold$lambda, 
            "10-Fold MSE" = cv.10Fold$cvm),
      digits = 3)

plot(cv.10Fold) # Plot all lambdas vs. MSEs

# Find the best lambda that minimizes 10FCV MSE

best.lambda <- cv.10Fold$lambda.min 
best.lambda # Check it out
log(best.lambda) # Spot it in the plot

# Extracting Ridge Coefficients (no p-values)

coef(ridge.mod, s = best.lambda) # From the Ridge object
coef(cv.10Fold, s = "lambda.min") # Or the cv.10Fold object

# Find the smallest cross validation MSE's

min.mse <- min(cv.10Fold$cvm)

cbind("Best Lambda" = best.lambda, 
      "Log(Lambda)" = log(best.lambda), 
      "Best 10FCV MSE" = min.mse)

# Extracting Coefficients for other Lambda values, for example 50

coef(ridge.mod, s = 50) # From the Ridge object


# Predicting with Ridge

# We can use the predict() function to obtain Ridge regression coefficients, which we can then use to make predictions with new data.

# Let's find the Ridge regression coefficients for the best lambda

predict(ridge.mod, s = best.lambda, type = "coefficients")

# IMPORTANT: with shrinkage methods like Ridge regressions, p-values and R-Squared values are no longer relevant, because all coefficients are retained in the model, althoug shrunk. The actual magnitude of the coefficients tells you how important the corresponding variable is in the prediction.

# Also, remember that the coefficients are standardized, so they all measure the same thing -- i.e., how many standard deviations does the response variable change when the respective predictor changes by 1 standard deviation.

# Now let's use the predict() function to make actual predictions

# To do predictions with Ridge regression, you need to store the predictor values you want to use for the prediction in a "new X" matrix, and use the predict() function with the "newx" parameter. For the purposes of this example, let's just draw a random test sample with 10% of the existing data set, using the best lambda we found above.

set.seed(1)

test <-sample(1:nrow(x), 0.10 * nrow(x))

ridge.pred <- predict(ridge.mod, s = best.lambda, newx = x[test,])
ridge.pred # Take a look

# You now know most of what you need to know about Ridge regression. But you can do clever things with the glmnet() function:

# If you want a single value of lambda, for example 0 (OLS):

ridge.mod.0 <- glmnet(x, y, alpha = 0, lambda = 0) 
coef(ridge.mod.0) # take a look

# If you set lambda very large you will obtain a model very close to the null model (just the intercept)

ridge.mod.large <- glmnet(x, y, alpha = 0, lambda = 1000000)
coef(ridge.mod.large) # take a look

# You can also fit models with a specific sequence of lambda values

ridge.mod.various <- glmnet(x, y, alpha=0, 
                            lambda = c(100000, 10000, 1000, 100, 10, 5, 0)) 

coef(ridge.mod.various) # Take a look at the coefficients
options(scipen = 4) # To omit the scientific notation
ridge.mod.various$lambda # Take a look at the lambdas

# Take a look at the 3rd. model only

ridge.mod.various$lambda[3] # List the third lambda
coef(ridge.mod.various)[, 3] # And the respective coefficients

# Note that the coefficients are listed from highest to lowest shrinkage (starting with lambda=100000), so it is customary to list a decreasing sequence of lambda values.

# Another clever way to try various Lambda values in a series of Ridge regressions is to create a large sequence of numbers and use them as lambdas. For example the command below generates 100 values stored in the object named "lambdas.100". The sequence takes values from 10 to the 10th power to 10 to the -2 power..

lambdas.100 <- 10 ^ seq(10, -2, length = 100) 
lambdas.100 # Take a look

# This command fits a model for each of the 100 lambda values in grid

ridge.mod.100 <- glmnet(x,y,alpha = 0,lambda = lambdas.100) 
ridge.mod.100 # Check it out

# Note: the ridge regression object is a matrix with one row for each predictor and one column for each value of lambda:

coef(ridge.mod.100)
str(ridge.mod.100) # To further inspect of the ridge.mod object
dim(coef(ridge.mod.100)) # Shows the coefficient matrix dimensions
# 20 (variables) x100 (lambdas) matrix

plot(ridge.mod.100) # Plot L2 Norm against the coefficients

# Example: 50th Lambda

ridge.mod.100$lambda[50] # List the 50th lambda
coef(ridge.mod.100)[, 50] # And its respective coefficients

l2.norm.100 <- sqrt(sum(coef(ridge.mod.100)[-1, 50] ^ 2)) 
l2.norm.100

# Example: 60th Lambda

ridge.mod$lambda[60]
coef(ridge.mod)[, 60]
sqrt(sum(coef(ridge.mod)[-1, 60] ^ 2)) # l2 Norm

# Notice that the l2 Norm 50th Lambda are much smaller (i.e., shrunk) than those of the 60th Lambda (remember that the lambdas are sorte in descending order, so the 50th lambda is larger than the 60th)

# Ridge and Logistic Regression

# You can fit a Ridge shrinkage model on a Logistic Regression simply by adding the family="binomial" attribute to both the glmnet() and cv.glmnet() functions. Selecting the best lambda is identical to quantitative Ridge models. The coefficient interpretation in Ridge Logistic is similar to the Logistic Regression interpretation based on log-odds effects.

# Let's run the same model we ran with Logistic earlier with the heart disease data

library(glmnet) # Contains functions for Ridge and LASSO
heart <- read.table("Heart.csv", sep = ",", head = T)

x <- model.matrix(chd ~ .,data = heart)[, -1]
y <- heart$chd

# Ridge Logit model:

ridge.logit <- glmnet(x, y, alpha = 0, family = "binomial") 

# Find best lambda:

set.seed(1)

cv.10Fold.logit <- cv.glmnet(x, y, alpha = 0, family = "binomial")

# Tecnical Note: the $cvm property below provides a CV MSE value for Ridge and CV Deviance or Binomial (2LL) for Ridge Logit. For Ridge Logit, you can change the measure.type to report different CV deviance measures. For example, the function below with measure.type = "class" will use the classification error as the deviance measure. This error is provided as the Area Under the ROC Curve (AUC) (see lecture on classification models). You can try this and plot it on your own.

# cv.10Fold.logit.class <- cv.glmnet(x, y, alpha=0, family="binomial", measure.type="class")

cbind("Lambda" = cv.10Fold.logit$lambda, 
      "10-Fold CV Deviance" = cv.10Fold.logit$cvm)

plot(cv.10Fold.logit)

best.lambda.logit <- cv.10Fold.logit$lambda.min 
best.lambda.logit
log(best.lambda.logit) # Spot it in the plot

min.mse.logit <- min(cv.10Fold.logit$cvm)

cbind("Best Lambda"=best.lambda.logit, 
      "Log(Lambda)"=log(best.lambda.logit), 
      "Best 10FCV Deviance" = min.mse.logit)

# Display coefficients

predict(ridge.logit, s = best.lambda.logit, type = "coefficients")

# Or

ridge.coef <- coef(ridge.logit, s = best.lambda.logit)
ridge.coef
cbind(ridge.coef, exp(ridge.coef))

# Plain Logit with no Shrinkage
ridge.0 <- coef(ridge.logit, s = 0)
ridge.0
cbind(ridge.0, exp(ridge.0))


## LASSO Regression

# alpha = 1 fits a LASSO model for a variety of descending lambda values
# alpha = 0 fits a Ridge Regression 
# 0 < alpha < 1 to fit an elastic net (combination Ridge and LASSO)

# IMPORTANT: the one difference between Ridge regression and LASSO is that coefficents can shrink significatnly with Ridge regression, but they never become 0 (except when lambda is infinite). In contrasts, some LASSO coefficients can shrink to 0.

# Repeating a few commands for convenience

library(glmnet) # Contains functions for Ridge and LASSO
library(ISLR) # Contains the Hitters data set

Hitters <- na.omit(Hitters) 

x <- model.matrix(Salary ~ ., data = Hitters)[, -1]
y <- Hitters$Salary # y vector with just the outcome variable

# Fit the LASSO model

lasso.mod <- glmnet(x, y, alpha=1) 
lasso.mod # take a look (%dev is the fraction of deviance explained for 100 different lambdas - change with "nlambda" argument)

# In glmnet() use family=“binomial” for logit; and family=“poisson” for count data

# Note: Like Ridge regression, LASSO also standardizes the coefficients to avoid issues of scale invariance (i.e., OLS coefficients are scale invariant).

# Also, while we compute the l2 Norm in Ridge regression, the LASSO regression yields the l1 Norm instead, which is the sum of the absolute values of the coefficients. Like with l2 Norm, l1 Norm measures the total amount of shrinkage obtained by a particular lambda value used. The smaller l1 the larger the shrinkage

plot(lasso.mod) # Plot the l1 Norm against the coefficients

# To get the specific LASSOe coefficients for a value of lambda:

coef(lasso.mod, s=0) # No shrinkage, same as OLS
coef(lasso.mod, s=50) # Some lambda=50
coef(lasso.mod, s=10^20) # Extreme shrinkage, coefficients close to 0


# Optimal LASSO Lambda

# Now let's find the best lambda value that minimizes the cross-validation MSE

# The cv.glmnet(){glmnet} function does cross-validation testing

RNGkind(sample.kind = "default") # To use the R default RNG
set.seed(1) # To get repeatable results

cv.10Fold.lasso <- cv.glmnet(x, y, alpha=1) # 10-Fold is the default
# Use the parameter nfolds=xx to change the number of folds

# List all lambdas and corresponding MSE's
cbind("Lambda" = cv.10Fold.lasso$lambda, 
      "10 Fold MSE" = cv.10Fold.lasso$cvm)

plot(cv.10Fold.lasso) # Plot all lambdas vs. MSEs

# Find the best lambda that minimizes 10FCV MSE

best.lambda.lasso <- cv.10Fold.lasso$lambda.min 
best.lambda.lasso # Check it out
log(best.lambda.lasso) # Spot it in the plot

# Ridge coefficients for the best.lambda

coef(lasso.mod, s = best.lambda.lasso) # With the LASSO object
coef(cv.10Fold.lasso, s = "lambda.min") # Or the CV object

# Find the smallest cross validation MSE's

min.mse.lasso <- min(cv.10Fold.lasso$cvm)

cbind("Best Lambda" = best.lambda.lasso, 
      "Log(Lambda)" = log(best.lambda.lasso), 
      "Best 10FCV MSE" = min.mse.lasso)


# Predicting with LASSO

# We can use the predict() function to obtain LASSO regression coefficients, which we can then use to make predictions with new data.

# Let's find the LASSO coefficients for the best lambda

lasso.coef <- predict(lasso.mod, 
                      s = best.lambda.lasso,
                      type = "coefficients")

lasso.coef <- coef(lasso.mod, 
                   s = best.lambda.lasso)

lasso.coef

# Notice how some coefficients shrunk to 0 (which was not the case with Ridge)

# To omit the 0 coefficients:

lasso.coef[lasso.coef!=0]

# IMPORTANT: as with Ridge, p-values and R-Squared values are no longer relevant with LASSO.

# Also, remember that the coefficients are standardized, so they all measure the same thing -- i.e., how many standard deviations does the response variable change when the respective predictor changes by 1 standard deviation.

# Now let's use the predict() function to make actual predictions

# To do predictions with Ridge regression, you need to store the predictor values you want to use for the prediction in a "new X" matrix, and use the predict() function with the "newx" parameter. For the purposes of this example, let's just draw a random test sample with 10% of the existing data set, using the best lambda we found above.

set.seed(1)
test=sample(1:nrow(x), 0.10 * nrow(x)) 

lasso.pred <- predict(lasso.mod, 
                      s = best.lambda.lasso, 
                      newx = x[test,])
lasso.pred # Take a look

# LASSO and Logistic Regression

# You can fit a Ridge shrinkage model on a Logistic Regression simply by adding the family="binomial" attribute to both the glmnet() and cv.glmnet() functions. Selecting the best lambda is identical to quantitative Ridge models. The coefficient interpretation in Ridge Logistic is similar to the Logistic Regression interpretation based on log-odds effects.

# Let's run the same model we ran with Logistic earlier with the heart disease data

library(glmnet) # Contains functions for Ridge and LASSO
heart <- read.table("Heart.csv", sep = ",", head = T)

x <- model.matrix(chd~., data = heart)[, -1]
y <- heart$chd

# Ridge Logit model:

lasso.logit <- glmnet(x, y, alpha = 1, family = "binomial") 

# Find best lambda:

set.seed(1)
cv.10Fold.logit <- cv.glmnet(x, y, alpha = 1, 
                             family = "binomial")

cbind("Lambda" = cv.10Fold.logit$lambda, 
      "10-Fold MSE" = cv.10Fold.logit$cvm)

plot(cv.10Fold.logit)

best.lambda.logit <- cv.10Fold.logit$lambda.min 
best.lambda
log(best.lambda) # Spot it in the plot

min.mse.logit <- min(cv.10Fold.logit$cvm)

cbind("Best Lambda" = best.lambda.logit, 
      "Log(Lambda)" = log(best.lambda.logit), 
      "Best 10FCV MSE" = min.mse.logit)

# Display coefficients

predict(lasso.logit, 
        s = best.lambda.logit, 
        type="coefficients")

# Or

lasso.coef <- coef(lasso.logit, 
                   s = best.lambda.logit)

lasso.coef

lasso.both <- round(cbind(lasso.coef, 
                          exp(lasso.coef)),
                    digits = 3)

colnames(lasso.both) <- c("Log-Odds", "Odds")

lasso.both

# Plain Logit with no Shrinkage

lasso.0 <- coef(lasso.logit, s=0)
lasso.0

lasso.both <- round(cbind(lasso.0, 
                          exp(lasso.0)),
                    digits = 3)

colnames(lasso.both) <- c("Log-Odds", "Odds")

lasso.both
