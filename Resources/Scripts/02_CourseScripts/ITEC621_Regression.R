###########################################
#      ITEC 621 Regression Refresher      #
###########################################

# Filename: ITEC621_Regression.R
# Prepared by J. Alberto Espinosa
# Last updated on 1/17/2022

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: the material covered in this script ASSUMES that you have gone through the self-paced course KSB-999 R Overview for Business Analytics and are thoroughly familiar with the concepts illustrated in the RWorkshopScripts.R script. If you have not completed this workshop and script exercises, please stop here and do that first.

#######################################################
#                       INDEX                         #
#######################################################

## Simple Linear Regression
## Regression with a Dummy Variable

## Multiple Linear Regression
# ANOVA in Regressions

## Weighted Least Squares (WLS) Regression
# Testing for Heteroskedasticity 
# Fitting WLS
# Iterative Re-Weighted Least Squares (IRLS)

## Generalized Linear Method (GLM)
## Logistic with GLM

#######################################################
#                  END OF INDEX                       #
#######################################################


## Simple Linear Regression

# Most of the material covered on regression in this course is covered in the ISLR book. This section of the script contain most of the code in the ISLR book, but I have added additional code (for example, to  compute collinearity statistics) and a substantial amount of comments.

options(scipen = "4") # Limit the use of scientific notation

# Modern Applied Statistics with S {MASS}
library(MASS) # Contains the Boston data set

# The Boston data set contains data for the Boston area housing market

?Boston # Let's check the variables in the data file
names(Boston) # View the data names
head(Boston) # View a few records

# The lm(){stats} function fits linear models based on the OLS regression model. The lm() model requires the outcome variable, followed by the ~ operator, followed by the predictors, separated with a "+" sign, and the data attribute to indicate the data set to be used for fitting the model. 

# Let's start with the null model with no predictors

lm.fit.null <- lm(medv ~ 1, data = Boston) # Use 1 for no predictors
summary(lm.fit.null) # The null model
mean(Boston$medv) # Mean = intercept in the null model

# a simple regregression model has only one predictor so the "+" is not used. For example, a simple regression model for Boston housing data for housing values (medv) as a function of % low status population (lstat)

# Now let's add one predictor

lm.fit <- lm(medv ~ lstat, data = Boston) # Using the fixed data set Boston

# Alternatively,
attach(Boston) # you can attach the data first
lm.fit <- lm(medv ~ lstat) # And then run the regression

lm.fit # Display the regression output object
summary(lm.fit) # Display the summary of results
names(lm.fit) # List all the properties of the lm.fit object
coef(lm.fit) # Coefficients only
confint(lm.fit) # Confidence intervals only

# The object produced by the lm() function contains tons of information, stored in attributes prefixed with $. You can view the full contents with the str() function:

str(lm.fit) # So, for example, use:

lm.fit$coefficients # To extract regression coefficients
lm.fit$residuals # To extract error terms or residuals
lm.fit$fitted.values # To extract predicted or fitted values

# The summary() is also an object with useful information

summary(lm.fit)$r.squared # R-Squared
summary(lm.fit)$adj.r.squared # Adjusted R-Squared
summary(lm.fit)$coefficients["lstat","Pr(>|t|)"] # p-value for lstat

# Important regression plots

plot(lm.fit) # Display 4 important regression diagnostic plots

# You need to hit <Return> at the R console to see all the plots
# Or, select the one you want to see:

plot(lm.fit, which=2)

# Or, plot all of them in 4 quadrants

par(mfrow=c(2,2)) # Divide the output into a 2x2 frame by row
# par(mfcol=c(2,2)) # Or, divide the output into a 2x2 frame by column
plot(lm.fit)
par(mfrow=c(1,1)) # Restore the output window

# Inspecting regression plots is very important in OLS regression modeling:

# (1) Residual plot (inspect dispersion of residuals)
# (2) Residual qqplot (inspect normality of resituals)
# (3) Standardized residual (square root) plot -- same as (1) but residuals are divided by their standard deviations
# (4) shows the influence of outliers -- observations that fall outside the dotted lines (Cook's distance) have too much influence on the slope of the regression line.

# Making predictions with the fitted model

# Get predicted values for all data points with confidence intervals
predict(lm.fit, interval="confidence") # lwr and upr bounds of 95% confidence interval

# Get predicted values for 3 new data points
predict(lm.fit, data.frame(lstat=(c(5,10,15))), interval="confidence") # or
predict(lm.fit, data.frame(lstat=(c(5,10,15))), interval="prediction")

# Note: interval="confidence" provides the range of values representing a 95% probability (by default) that the model will predict values within that interval. If the interval does not include 0, then the coefficients are significant at the p < 0.05 level. 

# To get other confidence levels, e.g. 99%, use the level= attribute

predict(lm.fit, data.frame(lstat=(c(5,10,15))), interval="confidence", level=0.99)

# In contrast, interval="prediction" provides the 95% probability that an "actual prediction" will fall in the interval range. Generally, prediction intervals are larger that confidence intervals because they need to account for the variance in Y.

# Technical note on fitting and predicting models. For now, we are using the full data set (e.g., Boston) to fit the model and to make predictions. Later on, we will split the data set into training (e.g., Boston[train,]) and test (e.g., Boston[-train,]) subsamples, and then fit or train the model with the training subsample and test it with the test subsample.

# Plot the data

plot(Boston$lstat, Boston$medv) # Plot the (x,y) data
abline(a=mean(Boston$medv), b=0, lwd=3) # Line at the mean of medv
abline(lm.fit) # Draw the regression line on the plot
abline(lm.fit,lwd=3) # Change the line weight to 3
abline(lm.fit,lwd=2, col="red", lty="dotted") # Change the line's color and weight

# Change other line attributes

plot(Boston$lstat, Boston$medv,col="red") 
plot(Boston$lstat, Boston$medv,pch=20)
plot(Boston$lstat, Boston$medv,pch="+")

# Quick simple regression examples

library(MASS) # Contains the Boston dataset
lm.simple=lm(medv~rm, data=Boston)
summary(lm.simple)

# Note: you can use the attach() function to load the dataset into the work environment, and then avoid having to prefix the variables with Boston$

attach(Boston)

plot(rm, medv, main = "Regression for Boston House Values", xlab = "No Rooms", ylab = "Median Value $000")

abline(lm.simple, col="red" )

# You can inspect the coefficients visually too:

require(coefplot)
coefplot(lm.simple)

# Simple regression model example with baseball player salaries

library(ISLR) # Contains the Hitter data set
lm.fit <- lm(Salary ~ HmRun, data=Hitters) # Using home runs to predict salaries
summary(lm.fit) # Home runs has a positive significant effect

# Let's plot the regression, first the data points
plot(Salary ~ HmRun, data=Hitters, cex=0.5, xlab="X (Predictor)", ylab="Y (Outcome)")
abline(a=mean(Hitters$Salary), b=0) # Line at the mean of medv
abline(lm.fit, col="red") # Draw the regression line on the plot


## Regression with a Dummy Variable

options(scipen = "4") # Limit the use of scientific notation

# Price is a continuous variable and US is a binary variable (Yes=1, No=0). The coefficient for USYes indicates that, keeping price constant, on average, a US store sells 1.199 thousand dollars more in car seats than non-US stores.

library(ISLR) # Contains the "Carseats" data set
?Carseats # Explore the Car seats data set
lm.fit <- lm(Sales ~ Price + US, data=Carseats)
summary(lm.fit)

# You can inspect the coefficients visually too:

library(coefplot)
coefplot(lm.fit)


## Multiple Linear Regression

options(scipen = "4") # Limit the use of scientific notation

library(MASS) # Needed for the Boston data set
?Boston
# Multiple regression model with 3 predictors on the median value of houses in the Boston area. The model syntax is the same as for simple linear models, but now we have more than one predictor, separated with the "+" operator

lm.fit <- lm(medv ~ lstat + crim + age + chas, 
             data = Boston) 

summary(lm.fit) # Display regression output summary

# You can inspect the coefficients visually too:

library(coefplot)
coefplot(lm.fit)

# To run a multiple regression on all available variables use the dot "."

lm.fit <- lm(medv ~ ., data = Boston) 

summary(lm.fit) # Review the results
coefplot(lm.fit)

# To view the results with a correlation matrix

summary(lm.fit, correlation = T) 

# Also, 

coef(lm.fit) # Display only the regression coefficients
confint(lm.fit) # Display only the confidence intervals

# Inspect regression plots:

par(mfrow=c(2,2))
plot(lm.fit)
par(mfrow=c(1,1))


# ANOVA in Regressions

options(scipen = "4") # Limit the use of scientific notation

# The "anova()" function is useful when comparing nested linear models (i.e., one model is the subset of another):

library(ggplot2) # Contains the diamonds dataset

lm.null <- lm(price~1, data=diamonds) # Null model
lm.small <- lm(price~carat, data=diamonds) # Small model
lm.large <- lm(price~carat+clarity, data=diamonds) # Large model

anova(lm.null, lm.small, lm.large) # Compare 3 nested models

# p-value is significant -> lm.large has more predictive power than lm.small, and lm.small has more predictive power than lm.null. In other words, carats improve the predictive power of the null model, but adding "clarity" as a predictor improves the explanatory power of the small.


## Weighted Least Squares (WLS) Regression 


# Testing for Heteroskedasticity 

# This material is not in the ISLR textbook

# The OLS model assumes that residuals are even throughout the regression line. When you fit an OLS model, you need to inspect the first regression plot, which shows residuals against fitted values. The residuals should show a cloud of even data throughout. If it doesn't, then you need to test for "heteroskedasticity" or uneven residuals. If some residuals are much larger than others, then when you square them (to get the sum of error squares), the squared values will be even larger and those observations will pull the regression line disproportionately. 

# When hetoroskedasticy is present, we need to weight down the squared residuals and, instead of getting the regression line that minimizes the sum of squared errors, we fit a regression line that minimizes the "weighted" sum of error squares.

# There are various weights that can be used for WLS, but the best is usually to use the inverse of the OLS error squares, so that we weight down the sum proportionally to the size of the error squared. Let's go through this methdod.

library(MASS) # To read the Boston housing market data
library(lmtest) # Contains bptest() and more

options(scipen=4)

# If you plan to use the regression formula in various parts of the script, it helps to store the the formula in an object. This is a regression formula for median house values on all other variables

lm.ols <-lm(medv ~ ., data = Boston) # Fit the model using all predictors for medv
summary(lm.ols) # Take a look 

# Inspect the residual plot and, if you suspect that your model suffers from heteroscedasticity (i.e., uneven error variance), you should test for heteroscedasticity with the Breush-Pagan bptest(){lmtest} test. If the p-value is significant this means that the errors squared are highly correlated with the predicted values, which means that the errors increase or decrease systematically. That is, the errors are heteroskedsastic.

plot(lm.ols, which=1) # Uneven residuals, suggestomg heteroskedasticity

# Then, perform a Breusch-Pagan test for Heteroscedasticity

bptest(lm.ols, data = Boston) 
# p-value is significant, so heteroscedasticity is present

# Heteroskedasticity does not affect bias of the estimators, but the model is no longer BLUE. That is, it is not the most efficient (i.e., least variance) estimator. To correct for heteroskedasticity we use WLS, which is more efficent.


# Fitting WLS

# There are many ways to compute the weighted sum of errors squared, but the most popular, and probably most effective weighting scheme is to use the actual errors of the OLS model to weight down the sum. However, with heteroskedasticity, the residuals will be all over the place, that is with high variance, so it is more effective and stable to use predicted residuals, rather than actual residuals for the weighted sum calculation. But since residuals can be positive or negative, it is better to use the magnitude of the residual, that is, the absolute value of the residual, i.e., 1/fitted(abs(residuals))^2. Let's do it slowly in steps:

# Step 1: Fit an OLS model 

lm.ols <- lm(medv ~ ., data = Boston) # Re-fitted here for convenience
plot(lm.ols, which = 1) # Uneven residuals, suggestomg heteroskedasticity

# Step 2: fit another linear model using the fitted (i.e., predicted) values of the OLS model above as predictors of the absolute value of the residuals as the response variable

lm.abs.res <- lm(abs(residuals(lm.ols)) ~ fitted(lm.ols))

plot(fitted(lm.ols), abs(residuals(lm.ols))) # Take a look
abline(lm.abs.res, col="red") # Draw regression line

# The formulas above may be a bit confusing. I'm doing the same thing below, but step by step to make it more clear.

abs.res <- abs(residuals(lm.ols)) # Compute the absolute value of the OLS residuals
fitted.ols <- fitted(lm.ols) # Compute the OLS fitted values
lm.abs.res <- lm(abs.res ~ fitted.ols) # Regress abs.res on fitted.ols

plot(fitted.ols, abs.res) # Take a look
abline(lm.abs.res, col="red") # Draw regression line

# Step 3: Compute a weight vector using the inverse of these predicted residuals squared

wts <- 1/fitted(lm.abs.res)^2

# Step 4: Use this weight vector in the WLS model

lm.wls <- lm(medv ~ ., data=Boston, weights=wts)
summary(lm.wls)


# Alternative Weighting Method using Squared Residuals

# The method described above is very popular because the weights are computed from the actual errors, but using their absolute values to make them all positive and practitioners agree that this yields a low variance model. But this is not the only weighting method available. For example, you could use squared errors, rather than the absolute value of the errors, yielding a different weighting scheme. 

# Step 1: Fit an OLS model. We start with the same lm.ols model above

# Step 2: We now predict squared errors, rather than absolute values. 

# That is, we fit another linear model using the fitted (i.e., predicted) values of the OLS model above as predictors of the squared residuals as the response variable

lm.res2 <- lm(residuals(lm.ols)^2 ~ fitted(lm.ols))

plot(fitted(lm.ols), residuals(lm.ols)^2) # Take a look
abline(lm.res2, col="red") # Draw regression line

# Step 3: Compute a weight vector using the inverse of these predicted residuals squared. Since we fitted squared errors, we don't need to square them again

wts2 <- 1/fitted(lm.res2)

# Step 4: Use this weight vector in the WLS model

lm.wls2 <- lm(medv~., data=Boston, weights=wts2)
summary(lm.wls2) # Similar results to lm.wls above


# Iterative Re-Weighted Least Squares (IRLS)

# There are many other methods available for least sum of "weighted" least squares. The WLS method described above is pretty standard. But there are other methods and various weighting methods. Iterative Re-Weighted Least Squares (IRLS) is another popular method to compute "robust" estimators when heteroscedasticity is present.

# The MASS library has a "Residual Linear Model function rlm(), which does OLS with robust residuals. This method is equivalent to WLS, with one difference. Because the weights are calculated from residuals, but the final residuals are dependent on these weights, the IRLS solves this issue by itereating the model several times, calculating new residuals each time, re-weighting the model, and so on -- i.e., running WLS multiple times in iterations until the residuals don't change any more (the model converges). 

# For practical purposes the WLS method explained above is sufficient. However, IRLS is very effective at dealing with "valid outliers" which are outliers that pull the regression line in one direction or another, but if the data is valid you are not supposed to remove this outlier from the data. The rlm() function uses various weighting methods. Let's run the model above with rlm()

rlm.fit=rlm(lm.formula, data=Boston)
summary(rlm.fit) # Take a look 

require(coefplot)
coefplot(lm.wls)


## Generalized Linear Method (GLM)

# The lm() function fits a linear model using OLS. It assumes that the residuals are normally distributed. The outcome variable does not have to be normally distributed, strictly, as long as the residuals are normally distributed. However, when the outcome variable is not normally distibuted, the residuals are almost never normally distributed. Since you can inspect the normality of the outcome variable before you run your regression model, this is generally one of the first things to do.

# If the outcome variable is not normally distributed, but you can identify the type of distribution it follows (e.g., poisson, logistic), then you can use the glm() function instead of the lm() to fit your model.

# The glm() function works just like the linear models function lm(), except that it can be used to fit models for a wider range of distributions, not just normal. 

library(MASS) # Needed for the Boston data set

lm.fit <- lm(medv ~ lstat + age, data = Boston) # OLS 
summary(lm.fit) # Check it out

# Now try OLS with glm() using family=gaussian(link="identity")
# link = "identify" means "no transformation", leave as is.

glm.fit = glm(medv ~ lstat + age, 
              data = Boston, 
              family = gaussian(link = "identity"))

summary(glm.fit) # Notice that the results are identical

# You can derive the R-square of a GLM model as follows

res.dev <- glm.fit$deviance
null.dev <- glm.fit$null.deviance
dev.Rsq <- (null.dev-res.dev)/null.dev # Deviance explained by the model
print(dev.Rsq, digits = 4)


## Logistic with GLM

# For example, let's read the South Africa heart disease data set from the authors' data site:

heart <- read.table("Heart.csv", sep = ",", head = T)

# Dataset documentation
browseURL("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.info.txt")

# Note: the ISLR text authors have tons of data sets available at:
browseURL("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/")
# To read this data, use the command above but change the respective 
# .data file name for the file you are interested.

# Now let's fit a logistic regression model (more on this later)

# chd is 1 if patient has coronary heart disease

heart.fit <- glm(chd ~ ., 
                 data = heart, 
                 family = binomial(link = "logit")) 

# Notice that the syntax is very similar to the lm() syntax, but we can now specify the distribution family for the response variable. This requires that we specify two things:

# 1. The distribution of the outcome variable. In this case, we know that the outcome variable is binary, 0 or 1, so we use family = binomial

# 2. The link function, which is used to transform the outcome variable, if needed. For logistic regression, the outcome variable, cannot be 0 or 1, so we need to transform it using the logistic function. We do that with the link = "logit" parameter.

summary(heart.fit) # Take a look

# The coefficients represent the "change" in log-odds (i.e., logit) that the outcome variable is 1. A positive coefficients indicates how much the log-odds (and the odds) of Y being 1 increase, when the variable goes up by 1. A negative coefficient indicates how much the log-odds decrease. 

# Convert Log-Odds

log.odds <- coef(heart.fit)
odds <- exp(coef(heart.fit))
prob <- odds/(1+odds)
round(cbind(log.odds, odds, prob), digits = 3)

# You can inspect the coefficients visually too:

require(coefplot)
coefplot(heart.fit)

