###############################
# ITEC 621 Variable Selection #
###############################

# Filename: ITEC621_VariableSelection.R
# Prepared by J. Alberto Espinosa
# Last updated on 2/11/2022

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: the material covered in this script ASSUMES that you have gone through the self-paced course KSB-999 R Overview for Business Analytics and are thoroughly familiar with the concepts illustrated in the RWorkshopScripts.R script. If you have not completed this workshop and script exercises, please stop here and do that first.


#######################################################
#                       INDEX                         #
#######################################################

### Multicollinearity

## Condition Index (CI)
## Variance Inflation Factors (VIFs)

### Model Selection

## Subset Comparison

## Step Methods 
# Overview
# Forward Selection w/step() Function
# Backward Selection w/step() Function
# Stepwise Selection w/step() Function

# (FYI Only)

## Best Subset Selection
## Forward and Backward Selection using regsubsets()

## Choosing Among Models using cross-validation

#######################################################
#                  END OF INDEX                       #
#######################################################


### Multicollinearity


# We cover dimensionality methods in section 7 below. But a central concern with dimensionality is multicollinearity. When selecting predictors for a model, it is important to monitor the multicollinearity statistics of the resulting models.

# As discussed earlier, predictors should be highly correlated with the outcome variable, but not with each other. To the extent possible, they should be somewhat independent, or uncorrelated. Some correlation is tolerable, but high correlation causes multicollinearity, which is a big dimensionality problem. 

# High multicollinearity causes a lot of variance in the model and if it is too high, the model may be too unstable to be reliable. If you drop a few observations, the resulting coefficients should not change much. If they change substantiall, chances are that your model suffers from high multicollinearity.

# Multicollinearity is tested in two ways:

# 1. For the entire model, using the Condition Index (CI). The CI evaluates the overall linear association of the predictors for the ENTIRE MODEL.

# 2. If the CI shows that the model is multicollinear, you can inspect which predictors contribute the most to collinearity by inspecting the Variance Inflation Variable (VIF) of EACH VARIABLE

# We discuss both methods below.


## Condition Index (CI)


# Condition Index (CI): general collinearity statistic for the full model
#    CI<30 -- collinearity is not a concern
#    30<CI<50 -- Collinearity may be a concern, but may be OK
#    50<CI -- Collinearity is severe -- must fix

# Note: we will see this later with machine learning, but when collinearity is severe, some coefficients may still be highly significant and you may get a false sense of confidence that your results are OK. But the problem is that the standard errors of the coefficients are very high and these coefficients become unstable. With severe collinearity, if you drop a few variables the coefficients will change substantially. We will learn how to test for this with machine learning and cross validation

# Note: install.packages("klaR") once if you haven't done it yet.

library(klaR) # Contains the cond.index() function to compute the CI
library(MASS) # Contains the Boston housing market data set

# Run multiple regression on all available variables

lm.fit <- lm(medv ~ ., data = Boston) # Boston data is in {MASS} library
summary(lm.fit)

collin.diag <- cond.index(lm.fit, data = Boston)
collin.diag

# Technical Note: the condition index is computed from the eigenvalues of the correlation matrix. A predictor matrix with p variables has p eigenvalues. The CI is the square root of the ratio of the highest to lowest eigenvalue in the correlation matrix of the predictors. But the cond.index() function reports all the ratios as a vector of P elements, from the smallest (= 1) to the largest (Pth). You only need to pay attention to the largest ratio. Since the eigenvalues are ordered from smallest to largest, you only need to look a the last ratio. To display the largest ration, i.e., the CI of the model, you can do this.

collin.diag[length(collin.diag)] # Index to select the last element

# Let's try a reduced model

lm.fit.red <- lm(medv ~ crim + indus + rm + age + lstat, 
                 data = Boston)

summary(lm.fit.red)

collin.diag.red <- cond.index(lm.fit.red, data = Boston)
collin.diag.red[length(collin.diag.red)]


#### ASIDE ####

# The code below is old code that I don't use in the course because you can no longer install the {perturb} package. The code will work fine if you already have this package installed.

# The {perturb} package we have been using is no longer available in the CRAN site. You can also install the {perturb} package by downloading the last available version of the package from CRAN, but this is a bit complicated, and I also recommend discontinuing the use of deprecated packages. So, if you can't use {perturb}, just use the library and code provided above.

library(perturb) # Contains the colldiag() function to compute the CI
library(MASS) # Contains the Boston housing market data set

# Run multiple regression on all available variables

lm.fit <- lm(medv ~ ., data=Boston) # Boston data is in {MASS} library
summary(lm.fit)

# The colldiag(){perturb} function provides CI collinearity diagnostics

collin.diag <- colldiag(mod=lm.fit, 
                        scale=FALSE, center=FALSE, 
                        add.intercept=TRUE)

collin.diag # Check out the severe collinearity
collin.diag[1] # Display just the CI in the first column

# Sometimes standardized regressions reduce collinearity: use scale=TRUE or center=TRUE view collinearity diagnostics with standardized and/or centered variables

collin.diag = colldiag(mod=lm.fit, 
                       scale=TRUE, center=FALSE, 
                       add.intercept=TRUE)

collin.diag[1] # Check out the high but less severe collinearity

# If the CI is high but the VIF's (see below) are not so high, then the intercept may be the culprit. Some times the intercept is highly collinear with a variable that stays relatively flat with respect to Y. If this is the case, you can center variables or standardize. Check the condition index when the intercept is removed:

collin.diag <- colldiag(mod = lm.fit, 
                        scale = TRUE, center = FALSE, 
                        add.intercept = FALSE)

collin.diag[1] # Not so severe any more

# Let's try a reduced model

lm.fit.red <- lm(medv ~ crim + indus + rm + age + lstat, data=Boston)
summary(lm.fit.red)

collin.diag.red <- colldiag(mod=lm.fit.red, 
                            scale=TRUE, center=FALSE, 
                            add.intercept=TRUE)

collin.diag.red[1] # CI with all variables is still severe CI = 1017.38


#### \end of ASIDE ####


## Variance Inflation Factors (VIFs)


# Variance Inflation Factors (VIF): each variable's contribution to multicollinearity

library(car) # Contains the vif() function
# {car} is the "Companion to Applied Regression" package

# Once you determine the collinearity is a problem, as indicated by the CI the next step is to evaluate the VIF for each variable, to see which variable contributes the most. Think of the VIF of a variable as the R-square you would get by regressing that predictor on all other predictor. But, instead of yielding a 0-1 value (like the R-Square) the VIF converts this R-Square into a formula that provides a factor of how much the standard error is amplified due to its contribution to multicollinearity:

# VIF < 5 -- No concern, little or no contribution to collinearity
# 5 < VIF < 10 -- Some concern, but tolerable
# 10 < VIF -- The variable contributes significantly to collinearity
#             The coefficients have a high standard error and are 
#             unstable and unreliable.

# Print variance inflation factors with the vif(){car} function

vif(lm.fit) # Full model
vif(lm.fit.red) # Reduced model (CORRECTION, used to say "reduced")

# To model all predictors (.) except age (-):

lm.fit1 <- lm(medv ~ . -age, data=Boston)
summary(lm.fit1)

# The update function allows you to add or remove variables from the model

lm.fit1 <- update(lm.fit, ~ . -age) 
summary(lm.fit1)
vif(lm.fit1)


### Model Selection


## Subset Comparison

# Comparing Large vs. Reduced Models with an F-Test (Not on the textbook)

# The sections below explore a number of methods to select variables to include in a model. Often the selection will be between a smaller model (i.e., "reduced" or "restricted" model) and one that includes additional variables (i.e., "full" or "large" model). Reduced models are more biased, but large models can have problems of multi-dimensionality, like multicollinearity. 

# So, it is a balancing act. A commonly used approach is to select the best predictor variables based on business knowledge and then test adding a few additional variables. When new variables are entered into a model they should be tested with an F-Test to see if thee added variables provide "significant" additional predictive power. If the test fails, then they should be removed. There is an easy way to do this with the ANOVA function.

# Let's use the baseball data to illustrate

library(ISLR) # Needed for the Hitters data set
head(Hitters) # Baseball data
Hitters <- na.omit(Hitters) # To omit records with missing variables

# First fit both models, reduced and full (e.g.):

# Reduced

lm.reduced <- lm(Salary ~ AtBat + Hits + Walks, data=Hitters)
summary(lm.reduced) # Take a look

# Large

lm.large <- lm(Salary ~ AtBat + Hits + Walks + 
                        Division + PutOuts, 
                        data=Hitters)

summary(lm.large) # Take a look

# Use the anova() function to compare models, which performs an F-Test

anova(lm.reduced, lm.large) 
# Added variables add significant explanatory power

# Note: when the F-Test is significant, most of the coefficients of the added variables should be significant too

# Let's try a larger (i.e., full) model by adding the variable "Errors"

lm.full <- lm(Salary ~ AtBat + Hits + Walks + 
                       Division + PutOuts + Errors, 
                       data=Hitters)

summary(lm.full) # Take a look

# We can now compare all 3 models

anova(lm.reduced, lm.large, lm.full) 

# Added variable does not add significant explanatory power and it is no coincidence that the p-value for Errors is not significant


## Step Methods


# Overview

# An alternative way (recommended) to run stepwise regressions is the step() function. These will work with both lm() or glm() functions:

library(ISLR) # Contains the Hitters data set
options(scipen=4) 

# First, specify the smallest (null) and largest (full) models

# Null model with only the intercept (specifying the number 1 as a predictor fits a model with just the intercept and no other predictors)

Hitters.null <- lm(Salary ~ 1, data=Hitters) 

# Full mode with ALL the variables

Hitters.full <- lm(Salary ~., data=Hitters)

# Inspect both models:

summary(Hitters.null)
summary(Hitters.full)

# Technical note: in these examples, we use the step() function to go from the Null to the Full model, or from the Full to the Null model, using a forward, backwards or stepwise process. However, this method doesn't have to start with the Null model or end with the Full model. You could define a Small model (e.g., Hitters.small), with a minimal set of predictors that MUST be included in the model, and end with some large model (e.g., Hitters.large) and do a forward, backwards or stepwise search between these two.

# step(){stats} Function: it performs forwards, backwards and stepwise variable selection between a "reduced" and "larger" model. As long as the reduced model is a subset of the full model, it will work. Generally, we use the "null" model (just the intercept) as the reduced model, but it doesn't have to be. A couple of important comments about the step() function:

# Variables are included or excluded based on whether the AIC increases or decreases. This roughly coincides with p-values of around 0.15 in the coefficients (p<0.15 includes the variable; p>0.15 excludes the variable). You can change the inclusion/exclusion significance level by adding a parameter k in the step function. The default is k=2, which is approximately equivalent to p<0.15. If you want tighter significance levels you can use higher values of k. For example, k=2.7 yields approximately p<0.10.

# To find exactly how k relates to p-values, use the chi-square distribution function.

qchisq(0.01, 1, lower.tail = F) # k value for p<0.01 is 6.634
qchisq(0.05, 1, lower.tail = F) # k value for p<0.05 is 3.841
qchisq(0.10, 1, lower.tail = F) # k value for p<0.10 is 2.705

# Because of all the issues associated with forward, backwards and stepwise variable selection, the consensus is that you should only use these as exploratory and preliminary approaches. Therefore, there is no need to change the default k value.

# Note: the step(){stats} function works with both lm() and glm() objects. We illustrate it with lm() objects here.


# Forward Selection w/step() Function

# Start with NULL model as a baseline

Hitters.forward <- step(Hitters.null, 
                        scope = list(lower = Hitters.null, 
                                   upper = Hitters.full),
                        direction = "forward", 
                        test = "F")

summary(Hitters.forward)


# Forward Selection but with a smaller p-value. Let's try k=3

Hitters.forward.3 <- step(Hitters.null, 
                          scope = list(lower = Hitters.null, 
                                     upper = Hitters.full),
                          direction = "forward", 
                          test = "F", 
                          k = 3)

summary(Hitters.forward.3)

# Backward Selection w/step() Function -- start with the Full model and go in a backward direction:

Hitters.backward <- step(Hitters.full, 
                         scope = list(lower = Hitters.null, 
                                      upper = Hitters.full),
                         direction = "backward", 
                         test = "F")

summary(Hitters.backward)


# Stepwise Selection w/step() Function

# With Stepwise variable selection, you can start with the Full model and proceed in a backward-forward direction, or you can start with the Null model and proceed in a forward-backward direction.

# We re-fit the null and full models for convenience

library(ISLR) # Contains the Hitters data set

Hitters.null <- lm(Salary ~ 1, data = Hitters) 
Hitters.full <- lm(Salary ~ ., data = Hitters)

# Let's try Stepwise, starting with the Null model and going Forward-Backward

Hitters.step.forward <- step(Hitters.null, 
                             scope = list(lower = Hitters.null, 
                                          upper = Hitters.full),
                             direction = "both", 
                             test = "F")

summary(Hitters.step.forward)

Hitters.step.backward <- step(Hitters.full, 
                              scope = list(lower = Hitters.null, 
                                           upper = Hitters.full), 
                         direction = "both", 
                         test = "F")

summary(Hitters.step.backward)

# We can now treat the results like any lm() object:

summary(Hitters.forward)
summary(Hitters.backward)
summary(Hitters.step.forward)
summary(Hitters.step.backward)

# Note, you can use step() with GLM too (OLS, Logit, etc.), not just LM

Hitters.glm.null <- glm(Salary ~ 1, data = Hitters) 
Hitters.glm.full <- glm(Salary ~., data = Hitters)

Hitters.glm.step.backward <- step(Hitters.glm.full, 
                                  scope = list(lower = Hitters.glm.null, 
                                               upper = Hitters.glm.full), 
                                  direction = "both", 
                                  test = "F")

summary(Hitters.glm.step.backward)


###### ASIDE

# The next few sections are FYI only. We don't use Best Subset Selection in this class, but it is a valid and popular method described in the textbook. It accomplishes similar results than the Step Methods above, but I find the Step Methods more informational. But you can explore this on your own by following the discussion in the ISLR book.


## Best Subset Selection


library(ISLR) # Contains the Hitters data set
names(Hitters) # Baseball data
head(Hitters)

# Let's explore the data set

dim(Hitters) # Check the number of records and variables

# Count empty salary records (i.e., is.na=TRUE)

sum(is.na(Hitters$Salary)) 

Hitters <- na.omit(Hitters) # To omit records with missing variables

# Note: this is how the ISLR text authors do it. I would have created a new table named something like HittersFullData, just to distinguish the two. This example works fine for this data set, but it can be problematic with tables with many variables, where there are always missing values in various fields

sum(is.na(Hitters)) 
# The sum should be 0 because there should be no more empty fields

# Subset selection using the {leaps} package

library(leaps) # Contains the regsubsets() function for subset selection

# regsubset(){leaps} searches for the best predictive subsets

# Let's fit the full model first

regfit.full <- regsubsets(Salary ~ ., Hitters) 

# Note: best model is based on the lowest RSS; the syntax is identical to regression lm()

options(width=140) # To prevent output wrapping

summary(regfit.full)

# Yields 8 subsets, with the respective variables marked with "*". Unless specified by "nvmax" the max number of variables in the subset is 8 by default.

summary(regfit.full)$rsq # Let's inspect the respective R-squares
# Naturally, R-squares go up as more variables are added to the model

# Now, let's try up to 19 variables in the subset:

regfit.full <- regsubsets(Salary ~ .,data = Hitters, nvmax = 19) 
reg.summary <- summary(regfit.full) # Let's save the summary in an object
reg.summary

options(width=80) # Back to wrapping at 80 characters

names(reg.summary) # Check out the subset data names
reg.summary$rsq # Let's inspect R-squares

# Let's plot the subset stats

par(mfrow = c(2,2)) # 2 x 2 layout

# RSS

plot(reg.summary$rss, 
     xlab = "Number of Variables", 
     ylab = "RSS", 
     type = "l")

# Adjusted R-Square

plot(reg.summary$adjr2, 
     xlab = "Number of Variables", 
     ylab = "Adjusted RSq", 
     type = "l")

# This command (like which.min) finds the maximum (minimum) value in a vector

which.max(reg.summary$adjr2) 

# Let's draw a single red dot on record 11, which is the max

points(11, reg.summary$adjr2[11], 
       col = "red", cex = 2, pch = 20) 

# Cp

plot(reg.summary$cp,
     xlab = "Number of Variables", 
     ylab = "Cp",
     type = 'l')

which.min(reg.summary$cp)

points(10, reg.summary$cp[10], 
       col = "red", cex = 2, pch = 20)

# BIC

plot(reg.summary$bic,
     xlab = "Number of Variables", 
     ylab = "BIC",
     type = 'l')

which.min(reg.summary$bic)

points(6, reg.summary$bic[6], 
       col = "red", cex = 2, pch = 20)

# The regsubsets() function has a built in plot() command to display subset variables (ordered worst to best)

?plot.regsubsets
par(mfrow = c(1,1)) # Back to 1 x 1 layout

plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")
coef(regfit.full, 6) # List coefficients for model #6


# Getting more data out of regsubsets()

# The regsubsets() object does not have the richer set of statistics that the lm() object provides. But there is a way to feed the regsubset() object model into an lm() model using the {HH} package (Heiberger & Holland), which is a general statistics package

install.packages("HH") # If not already installed
library(HH) # HH works well with regsubsets()
# Note: we will use lm.regsubsets{HH} to pass the model object to lm()
library(ISLR) # Contains the Hitters data set

# First, get the subsets as usual with regsubsets()

regfit.full <- regsubsets(Salary ~ ., Hitters) 

# Then create the corresponding summary object

regfit.sum <- summary(regfit.full) 
regfit.sum

# For example, to get an lm() object for model #3:

regfit.lm.3 <- lm.regsubsets(regfit.full, 3) 
summary(regfit.lm.3)

# IMPORTANT: the lm.regsubsets() function does not work well with categorical variables. In this case it is best to write lm() model by hand, for example:

which.min(regfit.sum$bic) # Yields model #6

# Now read the variables with "*" and enter them in the lm() model:

regfit.sum 
summary(lm(Salary ~ AtBat + Hits + Walks +
                    CRBI + Division + PutOuts, 
           data = Hitters))


## Forward and Backward Selection using regsubsets()

# This command uses the "exhaustive" method by default

regfit.full <- regsubsets(Salary ~ ., Hitters, 
                          nvmax = 19) 

summary(regfit.full)

# Forward selection

regfit.fwd <- regsubsets(Salary ~ ., 
                         data = Hitters, 
                         nvmax = 19, 
                         method = "forward")

summary(regfit.fwd)

# Backward selection

regfit.bwd <- regsubsets(Salary ~ ., 
                         data = Hitters, 
                         nvmax = 19, 
                         method = "backward")

summary(regfit.bwd)

# Comparing coefficients:

coef(regfit.full, 7)
coef(regfit.fwd, 7)
coef(regfit.bwd, 7)


## Choosing Among Models using cross-validation


# Complete this section on your own

# This code section comes from the textbook. It is a bit involved and contains a loop and some matrix algebra to do cross validation. You should be able to follow this on your own following my comments. But if you with to do cross valiadation, I recommend fitting your candidate models with glm() as we did earlier and testing for cross validation with cv.glm(){book} as we did earlier. It is a lot simpler.

library(ISLR) # Contains the Hitters data set

# Note: it is important that we remove observations with incomplete data using the na.omit() function. Some of the cross-validation routines below won't work if the number of observations don't match the number of predicted values, and this can happen when there are NA values in the data set.

Hitters=na.omit(Hitters)

# We need a train and a test set -- we do this randomly in this example

set.seed(1) # To get repeatable results

# Create a sample of TRUE and FALSE values for the number of rows in the Hitters table with replacement

train <- sample(c(T, F), nrow(Hitters), rep = T)
train # Check out the results

# Use the remaining observations for the test set

test=(!train) # Opposite of the training set
test # Check out the results

# Fit the model on the training subset

regfit.best <- regsubsets(Salary ~ ., 
                          data = Hitters[train,], 
                          nvmax = 19) 

summary(regfit.best) # Check it out

# We first need to make a model X matrix from the test data

test.mat <- model.matrix(Salary ~ ., data = Hitters[test,]) 

# Note: the model.matrix function builds an X matrix (i.e., predictor variable matrix) from the data

test.mat # Check out the text X matrix

# What follows is too tedious, but in essence, it is a loop that runs 
# 19 times, one for each of the regsubset() models.

# We first create an initial vector with 19 NA's

val.errors=replicate(19, NA) 
val.errors # Check out this vector

# Note: in the loop below the number of rows in the test set has to match the number of rows in the set predicted below. This will only happen if we eliminate all the NA's with the na.omit() function as we did above.

for(i in 1:19){
  # Get the coefficients for the i-th model
  coefi = coef(regfit.best, id = i) 
  
  # Note: in the next command we use the "%*%" operator, 
  # which is a matrix operator used to multiply 2 
  # "conformable" matrices -- i.e., matrices that that 
  # can be multiplied. FYI, 2 matrices can only be  
  # multiplied if the number of columns of the first 
  # matrix match the number of rows of the second matrix 
  # (e.g., M1[mxn] %*% M2[nxk]). Again, R is vectorized, 
  # so when you multiply 2 matrices with a straigh R 
  # multiplication, e.g., M1*M2, the resulting matrix will
  # contain the elements of M1 multiplied by the 
  # respective alements of M2. If what you need is a true 
  # matrix multiplication you need to use the matrix 
  # operator M1%*%M2
  
  pred <- test.mat[, names(coefi)] %*% coefi 
  
  # The command above multiplies the X values from the test
  # matrix that match coefficient names times the respective 
  # coefficients for model i. This is necessary because 
  # there is no predict() function for regsubsets(). 
  # Remember that the prediction is X times the beta 
  # coefficients. 
  
  # Now, let's compute the MSE for the i-th model
  
  val.errors[i] <- mean((Hitters$Salary[test] - pred)^2) 
}

# To see how the i-th pass of the loop works, check it out, for example for i=6

coefi <- coef(regfit.best, id =6) # Coefficients for the 6th model
coefi # Take a look

# Matrix-multiply the X times the coefficients

pred <- test.mat[, names(coefi)] %*% coefi 
pred # Take a look

# To compute the MSE for the 6th model (i = 6):

val.errors[6] <- mean((Hitters$Salary[test] - pred)^2) 
val.errors[6] # Take a look

# To see all 19 MSE values computed by the loop above

val.errors 
which.min(val.errors) # Now find the smallest of the 19 MSE's

# The result is model 10
coef(regfit.best, 10) # Get the coefficients for that model

# The following creates a function to do the same steps above. Not necessary for this class, but for informational purposes only:

predict.regsubsets <- function(object, newdata, id,...){
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form,newdata)
  coefi <- coef(object,id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}

# The final predictive model to be used should be re-fitted using the full data set, not just the training set. But because the cross validation found the 10-variable model to be the best, we now just focus on finding the best 10-variable model
