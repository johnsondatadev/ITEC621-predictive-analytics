#############################
# ITEC 621 Machine Learning #
#############################

# Filename: ITEC621_MachineLearning.R
# Prepared by J. Alberto Espinosa
# Last updated on 2/19/2022

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

#######################################################
#                       INDEX                         #
#######################################################

## Set Seed and R's Random Number Generator (RNG)
## Random Splitting Cross Validation (RSCV)
## Random Splitting CV with Re-Sampling
## Leave-One-Out Cross-Validation (LOOCV)
## k-Fold Cross-Validation (KFCV)
## Bootstrap Regression
## The {caret} Package

#######################################################
#                  END OF INDEX                       #
#######################################################

options(scipen=4) # Limit scientific notation


## Set Seed and R's Random Number Generator (RNG)


# In machine learning and cross-validation we do a lot of random sampling and re-sampling to train and test models. Random sampling is done through random number generators (like the random number tables in the back of statistic books). To draw a random sample you start at an arbitrary number in the table, called the "seed", and start drawing numbers in sequence, starting with the seed. To set the seed, we use the `set.seed()`.

# IMPORTANT Technical Note:** Related to random sampling, R discovered problems with the `sample()` function, caused by some inconsistencies in the `set.seed()` random number generator (RNG). They corrected this with R version 3.6.0 and higher. For the most part, you can use `set.seed()` without worrying about these inconsistencies, because you will generate a random sample any way. However, your results may be different than mine, depending on the version of R you are running and depending on what is the default "random number generartor (RNG)" in your R installation. The differences are usually due to the various rounding methods used by the RNG. But for your results to match mine, we all need to set the RNG to the same default. Still, your numbers may not exactly match mine and that is OK, as long as they are not radically different. 

# To increase the chances of getting the same results, let's all set the same RNG default this way (when you start R or at the top of your ML or CV script): 
  
RNGkind(sample.kind = "default") # To use the R default RNG

# Once you've done this, then set the seed to any number you wish. To re-sample with different samples, change the seed. To get repeatable results, use the same seed. To get the same results than mine, use the same seed I used. In the example below I use a seed of 1, but it could be 10, 35, or any number you wish.

set.seed(1)


## Random Splitting Cross Validation (RSCV)


# Let start by doing CV ourselves. Later, we will let the various R functions do it for us. But it helps to learn to do it with our own code. Let's use the Auto data set:

library(ISLR) # Contains the Auto data set
head(Auto)

?Auto # Take a look at the variables

# Let's start by counting the number of records in the Auto data set

nrow(Auto) # 392 observations in the data set

# We are about to draw random samples, so let's set the seed

set.seed(10) # Arbitrary seed

# Suppose you want to train your model with 70% of the data and test it with the remaining 30%. Let's start by generating randomw "index" vector named "train", containing 70% of the 392 row numbers from the Auto data set nrow(Auto):

trsize <- 0.7
train <- sample(nrow(Auto), trsize * nrow(Auto))
train # Take a look
length(train) # Count the numbers in the index vector, 274 or 70% of 392

# You can try 0.6 and 0.8 times nrow(Auto) for different sub-samples.

# Now let's "train" (i.e., fit) the model on the train data. We can do this in 3 (or more) ways:

# 1. With the `subset=index vector` parameter

lm.fit.train <- lm(mpg ~ horsepower, 
                   data = Auto, 
                   subset = train)

# 2. By indexing the Auto data set to select just the train observations

lm.fit.train <- lm(mpg ~ horsepower, 
                   data = Auto[train, ])

# 3. By creating separate train and test subsets, and training the model with the train subset:

Auto.train <- Auto[train, ]
Auto.test <- Auto[-train, ]

lm.fit.train <- lm(mpg ~ horsepower, data = Auto.train)

# Technical Note: when using CV we rarely need to inspect the summary of the models because we will use the models for prediction testing, not for interpretation. Once the final model is selected, you can then re-fit the selected model with the full data set and view the model summary() then. Nevertheless, let's take a quick look:

summary(lm.fit.train) # All 3 methods yield the same results

# Let's use the trained model to make predictions with the test subset:

pred.mpg <- predict(lm.fit.train, Auto.test) 

# Let's compute the "Test MSE" (i.e., deviance) of the model based on the mean of the squared differences between the actual and predicted values of mpg in the test subset:

mse.test <- mean( (Auto.test$mpg - pred.mpg ) ^ 2) # Or
mse.test # Check out the deviance

# To recap, in the example above, we trained the model with the train sub-sample. We then used the trained model to make predictions with the test sub-sample. This is a good test because we use different data to train and test the model. If we test the model instead with the same data used to train the model, we can compute the train MSE, which is not useful, because we are testing with the same data used to build the model. But just to illustrate, let's give it a try:

# For illustration purposes only, but you should NEVER use the train.mse to evaluate a model:

pred.mpg.tr <- predict(lm.fit.train, Auto.train) 

# Let's compute the "Test MSE" (i.e., deviance) of the model based on the mean of the squared differences between the actual and predicted values of mpg in the test subset:

mse.train <- mean( (Auto.train$mpg - pred.mpg.tr ) ^ 2) # Or
mse.train # Check out the deviance

# Let's display the test and train MSE's side by side:

cbind("MSE Test" = mse.test, "MSE Train" = mse.train)

# Notice that the Test MSE is larger than the train MSE. Why is that? It is because we are calculating the MSE on the same data we use to train the model. That is, the train error under-estimates the test error. That is, in general (but not always, depending on the random samples), MSE Train < MSE Test.

# Technical issue: due to random sampling, depending on the seed used, MSE Train may in a few cases be larger than the MSE Test. If this happens, change the seed.

# Let's do the same with a square polynomial model

lm.fit2 <- lm(mpg ~ poly(horsepower, 2), data = Auto.train) 
lm.fit2.mse <- mean( (Auto.test$mpg - predict(lm.fit2, Auto.test) ) ^ 2)
lm.fit2.mse

# And also with a cubic polynomial model

lm.fit3 <- lm(mpg ~ poly(horsepower, 3), data = Auto.train)
lm.fit3.mse <- mean( (Auto.test$mpg - predict(lm.fit3, Auto.test) ) ^ 2)
lm.fit3.mse

# Let's look at the 3 Test MSE's together

mse.all = c("Linear Model" = mse.test, 
            "MSE Poly 2"=lm.fit2.mse, 
            "MSE Poly 3"=lm.fit3.mse)
mse.all

# Which model is the best, based on CV testing? The answer should be obvious.


## Random Splitting CV with Re-Sampling

  
# It is always a good idea to re-sample a few times and then compute an average of the Test MSE results. A single sample may be a lucky or unlucky sample, but re-sampling a few times will keep things more random. If you re-sample, you may want to  change the seed each time. Let's re-sample 10 times with a random seed each time with a loop:

mse.test <- replicate(10, 0) # Initialize a vector with 10 elements with 0 values
for (i in 1:10) 
{ 
  sd <- sample(1:1000, 1) # Draw a random seed between 1 and 1000
  set.seed(sd) # Set the seed to this random value
  
  # Draw the train and test sub-samples
  trsize <- 0.7
  train <- sample(nrow(Auto), trsize * nrow(Auto))
  Auto.train <- Auto[train, ]
  Auto.test <- Auto[-train, ]
  
  # Train your model
  lm.fit.train <- lm(mpg ~ horsepower, data = Auto.train)
  
  # Compute the Test MSE for the i-th sub-sample and store in mse.test vector
  mse.test[i] <- mean( (Auto.test$mpg - predict(lm.fit.train, Auto.test) )^2) 
}

print(mse.test, digits = 4) # This will show all 10 MSE's calculated in the loop
mean(mse.test) # Average the resulting 10 Test MSEs above


## Leave-One-Out Cross-Validation (LOOCV)


# LOOCV can be done with the cv.glm() function in the {boot} package, which works well with the glm() function

library(boot) # Contains the cv.glm() function
library(ISLR) # Contains the Auto data set
attach(Auto)

# Note that the model below is a plain OLS model, but we need to use the glm() function because the cv.glm() requires a glm() object. Note: there is also an equivalent cv.lm(){DAAG} -- try on your own library(DAAG); cv.lm(data=DataSetName, fitted.lm, m=3)

lm.fit <- lm(mpg ~ horsepower, data = Auto)
summary(lm.fit)

# With identical result to

glm.fit <- glm(mpg ~ horsepower, data = Auto)
summary(glm.fit)

# The cv.glm() function requires a glm() object
# cv.glm,() Produces a list with the MSE results

?cv.glm()

cv.loo <- cv.glm(Auto, glm.fit) # Create LOOCV object (the default methoe)

# Note: the cv.glm() function above usually has a third argument indicating the value of K which is the number of partitions for K-Fold validation. When omitted, as in this case, K is set to n (total observations) by default, thus applying LOOCV

# The delta component contains the MSE cross validation results 

cv.loo$delta 

# Note: delta has 2 numbers and they should be almost identical. If not, see below. The first delta value is the actual raw cross-validation MSE. The second one is some bias-corrected value we won't be using. To list just the raw CV MSE:

print(cv.loo$delta[1], digits = 5)

# Let's write a for loop to do 5 polinomials and storing results in a vector

# First, let's initialize the cv.error vector with 5 zeros

cv.error <- replicate(5, 0) 
cv.error # Check it out

# Note: the loop below may take a little while to run

for (i in 1:5){ 
  glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto) # fit for polinomial i
  cv.error[i] <- cv.glm(Auto, glm.fit)$delta[1] # cv.error for polinomial i
}

# Check out the vector with MSE values for each of the 5 polinomials

cv.error # Looks like the 5th polynomial is the best, but the second is not bad 


## k-Fold Cross-Validation (KFCV)

# Let's start with a simple k-fold CV for, say K=10

library(boot) # Contains the cv.glm() function
library(ISLR) # Contains the Auto data set
attach(Auto)

glm.fit <- glm(mpg ~ horsepower, data = Auto) # Fit a glm model
summary(glm.fit)

# Compute the 10FCV:

# Same as with LOOCV, but specifying K=10 for 10FCV
cv.10K <- cv.glm(Auto, glm.fit, K = 10) 

# The first $delta value is the raw cross-validation estimate of the prediction test error, that is the Test MSE

print(cv.10K$delta[1], digits = 5)

# Let's look at LOOCV and 10KCV together

mse.both <- c("MSE LOOCV" = cv.loo$delta[1], 
              "MSE 10FCV" = cv.10K$delta[1])

print(mse.both, digits = 5)

# Now let's do a loop to fit 10 polynomial models and calculate the respective cross validation MSE's.

set.seed(17) # Set arbitrarily to 17, can be any number

# Initialze error vector to 10 zeros for a 10 polinomial for loop this time

cv.error.10 <- replicate(10,0) 
cv.error.10 # Check it out

# Note: the for loop below runs 10 regression models, one for each polinomial, so it may take a while too

for (i in 1:10){
  glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
  cv.error.10[i] <- cv.glm(Auto, glm.fit, K = 10)$delta[1] # 10-Fold validation
  # Note: K is the number of folds, usually 5 to 10.
}

cv.error.10 # Check out the MSE's for the 10 different polinomials

# Looks like the 6th polynomial is the best (this may change with different samplings), but I think such a model would be too complex to interpret. You should use this polynomial if your goal is prediction accuracy, but you would be better off with the second polynomial if your goal is interpretation.

# Now let's try 4 different values of K=5, 10, 15, 20 for a simple linear regression

set.seed(1) # Set arbitrarily to 1, can be any number
cv.error.4 <- replicate(4, 0) # Initialze error vector with 4 zeros
cv.error.4 # Check it out

# Initialze another vector to keep the values of K for plotting

cv.k=replicate(4, 0) 
cv.k # Check it out

glm.fit <- glm(mpg ~ horsepower, data = Auto) # Fit a glm model
summary(glm.fit)

# Note: the for loop below runs 4 regression models, one for each value of K

for (i in 1:4){
  cv.k[i] <- i * 5
  cv.error.4[i] <- cv.glm(Auto, glm.fit, K = i * 5)$delta[1]
} 

# i * 5-Fold (5, 10, 15, 20) Validation

cv.k # Check out the K values
cv.error.4 # Check out the MSE's for the 10 different polinomials
cbind(cv.k, cv.error.4) # List K values with their MSE side by side

plot(cv.k, 
     cv.error.4,
     "l",  # "l" yields a line
     ylab = "MSE", 
     xlab = "K-Fold") 

# How about LOOCV?

cv.error.LOOCV <- cv.glm(Auto, glm.fit)$delta[1]
cv.error.LOOCV 
# Not much better than K=10 and slightly worse than K=20


## Bootstrap Regression


# Bootstrapping is used for many statistical methods and it is based on re-sampling the data with replacement many times, and using the resulting sample statistics as indicators of the population statistic. The nice thing about boostrapping is that, even if the data is not normally distributed, if you sample the data many times and compute the mean every time, the sample means will be normally distributed. So this is a useful method when the distribution of the data is unknown.

# There are many packages like {boot}, {Boot} and {caret}, with powerful libraries and functions for bootstrapping. {Boot} actually uses {boot} but makes fitting models easier. You can explore these packages and functions on your own. In this section I will illustrate how to write a simple script for bootstrapping.

# In regression, you can use bootstrapping for many things. For example, you can bootstrap several samples and compute regression coefficients and fit statistics each time and then average them out. For example, this allows you to compute the mean of a regression coefficient, along with its standard deviation, standard error, etc. Since bootstrap re-samples with replacement, there will be a few data points left out of the sample each time. Well, these data points can be used for cross-validation testing. 

# Let's work out an example:

# First let's load the {ISLR} library, which contains the Auto data set.

library(ISLR)

# Let's run a simple OLS model to predict vehicles' miles per gallon (mpg) using horsepower and weight as predictors

lm.fit <- lm(mpg ~ horsepower + weight, data = Auto)
summary(lm.fit) # Take note of the coefficients and R-square

# Now, let's bootstrap this model. First let set the number of bootstrap samples we wish to draw. I place this at the top of the code, so that the number of samples can be easily changed.

R <- 200 # You can change this number if you want more or less re-samples

# Let's create (initialize) a vector for each statistic we wish to compute, with R zeros. Say, we want to compute the R-Squared, the coefficients for horsepower and weight and the cross-validation test mse.

rsq <- rep(0, R) # Vector with R zeros for R-squares
hrsp.coef <- rep(0, R) # Horsepower coefficient vector
wgt <- rep(0, R) # # Weight coefficient vector
mse.test <- rep(0, R) # Mean squared error vector

# Now, let's write a loop with R cycles

for (i in 1:R) 
{ 
  # Set a different seed in each loop
  sd <- sample(1:1000, 1) # Draw a random seed between 1 and 1000
  set.seed(sd) # Set the seed to this random value
  
  # Bootstrap a train sub-sample with the same number of observations
  # as the Auto data set, but with replacement.
  train <- sample(nrow(Auto), replace = T) # Train index vector
  
  # Generate the train and test sub-samples
  Auto.train <- Auto[train, ]
  Auto.test <- Auto[-train, ]
  
  # Train your model with the train sub-sample
  lm.fit.train <- lm(mpg ~ horsepower + weight, data = Auto.train)
  
  # Read and store the R-square and the 2 coefficients in the 
  # respective i-th vectors
  
  rsq[i] <- summary(lm.fit.train)$r.squared
  hrsp.coef[i] <- lm.fit.train$coefficients["horsepower"]
  wgt[i] <- lm.fit.train$coefficients["weight"]
  
  # Now, let's compute the Test MSE
  mse.test[i] <- mean( (Auto.test$mpg - predict(lm.fit.train, Auto.test) )^2) 
}

# Let's view the result of each bootstrap

print(cbind("R-Squared" = rsq, 
            "Horsepower" = hrsp.coef, "Weight" = wgt, 
            "Test MSE" = mse.test), digits = 4)

# Now let's print the means and standard deviations of the bootstrapped results. Note: the standard errors of the coefficients are their standard deviations across the bootstrapped samples. A coefficient +/- (2 * Std Error) gives the 95% confidence interval.

results <- rbind(
                 cbind(mean(rsq), 
                       mean(hrsp.coef), 
                       mean(wgt), 
                       mean(mse.test)),
                 cbind(sd(rsq), 
                       sd(hrsp.coef),
                       sd(wgt), 
                       sd(mse.test)) )

rownames(results) <- c("Mean", "Std Error")
colnames(results) <- c("R-Squared", "Horesepower", "Weight", "Test MSE")
print(results, digits = 4)


## The {caret} Package


# The {caret} package is a companion to the book "Applied Predictive Modeling" book by Kuhn and Johnson:

browseURL("http://appliedpredictivemodeling.com/")

# It contains several useful functions for machine learning. The train() function is particularly useful to do cross validation with various models and methods by just changing a few parameters. The documentation for this package can be found at:

browseURL("http://topepo.github.io/caret/index.html")

# Let's load the {caret} and {ISLR) packages first

library(caret)
library(ISLR) # Contains the Auto data set

# Now let's fit an lm() model using the train() function in {caret}. The default cross-validation method is bootstrapping with 25 samples with replacement

set.seed(1) # Set the seed

lm.fit.caret <- train(mpg ~ horsepower + weight, 
                      data = Auto, 
                      method = "lm")

lm.fit.caret # Reports RMSE and R squared

# The default cross validation is done with bootstrapping with 25 samples

lm.fit.caret$results$RMSE # This is where RMSE is stored
lm.fit.caret$results$RMSE^2 # To get the MSE

summary(lm.fit.caret) # Same as lm() results

# More data from lm.fit.caret

lm.fit.caret$resample # Display fit stats for each sample
lm.fit.caret$results # Display all bootstrap results

# The {caret} package has many methods you can use:
browseURL("http://topepo.github.io/caret/train-models-by-tag.html")

# Also, the trainControl() function within train() offers many re-sampling methods, such as "boot" (default), LOOCV, CV, etc.)

? trainControl() # Check it out

# For example, to change the default number of bootstrap samples to, for example 100

lm.fit.caret.100 <- train(mpg ~ horsepower + weight, 
                          data = Auto, 
                          method = "lm", 
                          trControl = trainControl(number = 100))

lm.fit.caret.100$results # Display bootstrap results for 100 samples

# To change the cross-validation method to 10FCV:

lm.fit.caret.10FCV <- train(mpg ~ horsepower + weight, 
                            data = Auto, 
                            method = "lm", 
                            trControl = trainControl(method = "cv", 
                                                     number = 10))

lm.fit.caret.10FCV # Check the RMSE
lm.fit.caret.10FCV$results # All 10FCV results
lm.fit.caret.10FCV$results$RMSE ^ 2 # To get the MSE

summary(lm.fit.caret.10FCV) # lm() regression output

# Now let's quickly change this to LOOCV

lm.fit.caret.LOOCV <- train(mpg ~ horsepower + weight, 
                            data = Auto, 
                            method = "lm", 
                            trControl = trainControl(method = "loocv"))

lm.fit.caret.LOOCV # Check the RMSE
lm.fit.caret.LOOCV$results # All LOOCV results
lm.fit.caret.LOOCV$results$RMSE ^ 2 # To get the MSE

summary(lm.fit.caret.LOOCV)

# Now let's quickly change this to a Random Forest tree model with 10FCV

# We will be using the "rf" method to fit a random forest tree model. This requires loading the {randomForest} library because train() uses existing model fitting libraries and functions, which must be loaded to work. 

# In many instances, if you don't load the necessary libraries, the train() function will load them for you. But it is better to load the respective package for the necesary model specified in the "method" attribute ("rf" in this case).

library(randomForest)

rf.fit.caret.10FCV <- train(mpg ~ horsepower + weight, 
                            data = Auto, 
                            method = "rf", 
                            trControl = trainControl(method ="cv", 
                                                     number = 10))

rf.fit.caret.10FCV # Check the RMSE
rf.fit.caret.10FCV$results # All Random Forest results
rf.fit.caret.10FCV$results$RMSE # To get the RMSE
rf.fit.caret.10FCV$results$RMSE^2 # To get the MSE

summary(rf.fit.caret.10FCV)

# Another way to specify the train control is to store the controls in another object, e.g., "my.ctrl" so that they can be easily changed.

my.ctrl <- trainControl(method="cv", 
                        number=10) # Define & store the control

rf.fit.caret.10FCV <- train(mpg ~ horsepower + weight, 
                            data = Auto, 
                            method="rf", 
                            trControl = my.ctrl) # Then use it

rf.fit.caret.10FCV # Check the RMSE (changed a little due to random sampling)


# Neural Network 

# Note: see R scripts for Neural Networks for a more complete set of scripts

library(neuralnet)

nn.fit <- train(mpg ~ horsepower + weight, 
                data = Auto, 
                method = "neuralnet")

nn.fit # Check the RMSE

nn.fit$results # All Neural Network results
nn.fit$results$RMSE # To get the RMSE
nn.fit$results$RMSE ^ 2 # To get the MSE

summary(nn.fit)

