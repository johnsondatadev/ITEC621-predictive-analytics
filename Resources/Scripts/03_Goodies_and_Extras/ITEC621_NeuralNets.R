############################
# ITEC 621 Neural Networks #
############################

# Filename: ITEC621_NeuralNets.R
# Prepared by J. Alberto Espinosa
# Last updated on 3/20/2022 

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: the material covered in this script ASSUMES that you have gone through the self-paced course KSB-999 R Overview for Business Analytics and are thoroughly familiar with the concepts illustrated in the RWorkshopScripts.R script. If you have not completed this workshop and script exercises, please stop here and do that first.

#######################################################
#                       INDEX                         #
#######################################################

## Multiple Linear Regression (for comparison)
## neuralnet() Function and Arguments Overview
## Neural Networks -- Quantitative Outcome
## Predicting and Testing -- Quantitative Outcome
## Neural Networks -- Classification - Outcome
## Predicting and Testing -- Classification - Outcome
## Confusion Matrix

#######################################################
#                  END OF INDEX                       #
#######################################################


## Multiple Linear Regression (for comparison)

library(MASS) # Needed for the Boston data set
formula.medv <- medv ~ lstat + crim + age + chas

# The data needs to be normalized to train neural networks. This is so because the scale of the data can influence the random initial weights, which will then influence all subsequent adjustments in each epoch. While this is not necessary with OLS regressions, we normalize the data for all models, including OLS so that we can compare results. We use the Max-Min normalization method, which is popular with neural network training. See explanation on normalization below.

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Let's normalize the Boston dataset:

Boston.n <- as.data.frame(lapply(Boston, normalize))

# Let's split the data into train and test subsamples

RNGkind(sample.kind = "default")
set.seed(1)

tr.size <- 0.7
train <- sample(1:nrow(Boston.n), tr.size * nrow(Boston.n))

# Train & test subsample s

Boston.n.train <- Boston.n[train,]
Boston.n.test <- Boston.n[-train,]

lm.fit <- lm(formula.medv, data = Boston.n.train) 

summary(lm.fit) # Display regression output summary

# Compare actual against predicted values

data.frame("Actual" = Boston.n.test$medv, 
           "Predicted" = predict(lm.fit, Boston.n.test))

mse.lm <- mean( (Boston.n.test$medv - predict(lm.fit, Boston.n.test))^2 )
mse.lm # Show the results


## neuralnet() Function and Arguments Overview


# Notes on neuralnet() key attributes (values shown are the default):

# hidden = 1 -- a vector the number of nodes in each layer, e.g., hidden=3 has 1 layer with 3 nodes; hidden=c(4,2) has 2 layers, the first one with 4 nodes and the second one with 2

# threshold = 0.01 -- is an approximation threshold used in gradient descent derivatives. The default value is 0.1 and the smallest value needed for neuralnet() to work is 0.01. If the model does not converge, increase the threshold progressively to 0.1, 0.2, etc. If the model does not converge still, increase it to 1, 10, 100, etc. 

# stepmax = 10^5 -- is the maximum number of steps for training a neural network. The model training will stop if the model has not converged after the stepmax. Increasing the value of stepmax will make it more likely that the model will converge, but it will take substantially longer. 

# rep = 1 -- the number of times you want the neuralnet training to run. Usually, 1 is sufficient, but if you want more randomness and are willing to wait, you can use more reps.

# algorithm = "rprop+" -- is the internal algorithm to calculate the neural network. Available methods include: "rprop+" (resilient back propagation), "rprop-", "backprop" (back propagation), "sag", "slr" 

# err.fct = "sse" -- is the method used to calculate the error. "ce" is cross-entropy which can be used for classification models.

# act.fct = "logistic" -- the activation function, "logistic" and "tanh" are popular functions for classification models

# linear.output = T -- leave default as T for quantitative models or change to F for classification models, along with the act.fct activation function

# Normalizing the Data

# The weights of the model are initialized randomly and updated in each epoch during training. Thus the scale of the variables used can result in a slow or unstable training process, and can cause the model not to converge. Therefore, it is advised to normalize the data before training a neural network. Normalizing means re-scaling the data to a (-1 to 1) scale. You can use z-scores (i.e., center the variables and divide by their standard deviation), but this messes up the dummy variables. If you don't have any dummy variables, normalizing with z scores is OK. If there are dummy variables, you can normalize all other predictors, except dummy variables, or use other popular normalization methods, like Max-Min, in which the variable is transformed as a deviation (difference) from the minimum value for that variable, divided by the largest difference between maximum and minimum values. This can be accomplished with a simple function (already done above, repeated here for convenience):

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Let's normalize the Boston dataset:

Boston.n <- as.data.frame(lapply(Boston, normalize))

# Important Note about Dummy and categorical variables

# neuralnet() requires quantitative predictors. If you have a factor dummy variable, convert it to numeric. If you have a categorical variable, transform the variable to the respective binary variables, in numeric format. This can be done by hand or using the model.matrix() function. 

# For example, if your data set is called my.data and the model formula is y ~ x1 + x2 + x3 + etc., and some of these predictors are categorical, this command will convert the categorical x's into the respective binary variables: 

# my.data.q <- model.matrix(~ x1 + x2 + x3 + etc.)

# But you then need to add the outcome to the dataset as follows:

# my.data.q$y <- my.data$y


## Neural Networks -- Quantitative Outcome

library(neuralnet) # First, load the {neuralnet} library

# 1 Neuron, 1 Layer

net.fit.1 <- neuralnet(formula.medv, 
                       data = Boston.n, 
                       hidden = 1, 
                       threshold = 0.01)

plot(net.fit.1)

# 2 Neurons, 1 Layer

net.fit.2 <- neuralnet(formula.medv, 
                       data = Boston.n, 
                       hidden = 2, 
                       threshold = 0.01)

plot(net.fit.2)

# 1 Layer w/4 Neurons; 1 Layer w/2 Neurons
# If the algorithm does not converge increase the threshold to 0.1, 0.2, or even larger and/or increase the stepmax to a large value, for example 10^5. Note that this will increase the computational time substantially.

net.fit.4.2 <- neuralnet(formula.medv, 
                         data = Boston.n, 
                         hidden = c(4,2), 
                         threshold = 0.01, 
                         stepmax = 10^5)

plot(net.fit.4.2)


## Predicting and Testing -- Quantitative Outcome

# Splitting train and test sample. Note: it is not necessary to split the sample into train and test subsets to fit a neural network because back propagation algorithms will do that when training the model. We split the data here to extract a test only to illustrate how to make predictions with neural networks.

# Repeat train/test subsample for convenience

RNGkind(sample.kind = "default")
set.seed(1)

tr.size <- 0.7
train <- sample(1:nrow(Boston.n), tr.size * nrow(Boston.n))

Boston.n.train <- Boston.n[train,]
Boston.n.test <- Boston.n[-train,]

net.fit.4.2 <- neuralnet(formula.medv, 
                         data = Boston.n.train, 
                         hidden = c(4,2), 
                         threshold = 0.01, 
                         stepmax = 10^5)

plot(net.fit.4.2)

# We use the {neuralnet}predict() function and test sub-sample to make predictions

pred.nnet <- predict(net.fit.4.2, Boston.n.test) 
pred.nnet # Display predictions

data.frame("Actual" = Boston.n.test$medv, 
           "Predicted" = pred.nnet, 
           "Difference" = Boston.n.test$medv - pred.nnet)

mse.nnet <- mean( (Boston.n.test$medv - pred.nnet) ^ 2)
mse.nnet # Display the test MSE

cbind("OLS MSE" = mse.lm, 
      "Neural Net MSE" = mse.nnet) # Neural Net is more accurate


## Neural Networks -- Classification Outcome

heart <- read.table("Heart.csv", 
                    sep = ",", 
                    header = T, 
                    stringsAsFactors = T) 

# Need to convert factor variables to numeric

heart$famhist <- as.numeric(heart$famhist)

# Need to normalize the data. For a classification model, it is best to use the Max-Min normalization:

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

heart.n <- as.data.frame(lapply(heart, normalize))

# Specifying a formula object, for convenience

formula.chd <- chd ~ sbp + tobacco + ldl + 
                     adiposity + famhist + typea + 
                     obesity + alcohol + age

# Note: the neuralnet() algorithms below may take a long time to run and will not necessary converge. You can manipulate the threshold and stepmax if you are having issues getting your model to converge, or you can reduce the number of hidden layers and nodes. Of course, increasing the stepmax and threshold increase substantially the amount of time needed to estimate the neural net.

heart.nnet.1 <- neuralnet(formula.chd, 
                          data = heart.n, 
                          hidden = 1, 
                          threshold = 0.01)

plot(heart.nnet.1)

heart.nnet.2 <- neuralnet(formula.chd, 
                          data = heart.n, 
                          hidden = 2, 
                          threshold = 0.01)

plot(heart.nnet.2)

heart.nnet.4 <- neuralnet(formula.chd, data = heart.n, 
                          hidden = 4, threshold = 0.01)

plot(heart.nnet.4)

# Threshold of 0.01 not always converge, so I increased it to 0.5

heart.nnet.4.2 <- neuralnet(formula.chd, 
                            data = heart.n, 
                            hidden = c(4,2), 
                            threshold = 0.5)

plot(heart.nnet.4.2)

# Threshold of 0.01 not always converge, so I increased it to 0.5

heart.nnet.4.2 <- neuralnet(formula.chd, 
                            data = heart.n, 
                            hidden = c(4,2), 
                            threshold = 0.5, 
                            act.fct = "logistic")

plot(heart.nnet.4.2)


## Predicting and Testing -- Classification Outcome

# Splitting train and test sample

RNGkind(sample.kind = "default")
set.seed(1)

tr.size <- 0.7
train <- sample(1:nrow(heart.n), tr.size * nrow(heart.n))

heart.n.train <- heart.n[train,]
heart.n.test <- heart.n[-train,]

# Training

heart.nnet.4.2 <- neuralnet(formula.chd, 
                            data = heart.n.train, 
                            hidden = c(4,2), 
                            threshold = 0.5, 
                            act.fct = "logistic")

plot(heart.nnet.4.2)

# Predicting

# Compute predictions as probabilities

pred.prob <- predict(heart.nnet.4.2, heart.n.test) 

# Convert probabilities to classifications

thresh <- 0.5 # Set the classification threshold

pred.prob.class <- ifelse(pred.prob > thresh, 1, 0) 

cbind("Probability" = pred.prob, 
      "Classification" = pred.prob.class, 
      "Actual" = heart.n.test$chd) # Display predictions


## Confusion Matrix

confmat <- table("Predicted" = pred.prob.class, 
                 "Actual" = heart.n.test$chd) 

confmat # Display matrix

# Compute fit stats

TruN <- confmat[1,1] # True negatives
TruP <- confmat[2,2] # True positives
FalN <- confmat[1,2] # False negatives
FalP <- confmat[2,1] # False positives
TotN <- TruN + FalP # Total negatives
TotP <- TruP + FalN # Total positives
Tot <- TotN + TotP # Total

paste(TruN, TruP, FalN, FalP, TotN, TotP, Tot) # To double check the computations

# Now let's use these to compute accuracy and error rates

Accuracy.Rate <- (TruN + TruP) / Tot
Accuracy.Rate # Check it out

Error.Rate <- (FalN + FalP) / Tot
Error.Rate # Check it out

# Sensitivity -- rate of correct positives

Sensitivity <- TruP / TotP # Proportion of correct positives
Sensitivity # Check it out

# Specificity -- rate of correct negatives

Specificity <- TruN / TotN # Proportion of correct negatives
Specificity

# False Positive Rate = 1 - specificity (useful for ROC curves)

FalP.Rate <- 1 - Specificity
FalP.Rate

nnet.rates.50 <- round(c(Accuracy.Rate, Error.Rate, 
                         Sensitivity, Specificity, 
                         FalP.Rate), 
                       digits = 3)

names(nnet.rates.50) <- c("Accuracy", "Error", 
                          "Sensitivity", "Specificity", 
                          "False Positives")

nnet.rates.50

# Try on your own -- change the classification threshold from prob > 0.5 to > 0.3 and >0.7 and compute the respective confusion matrices and fit statistics and see how sensitivity and specificity change. Also, try fitting a classification tree and a logistic model with the same data and compare the fit statistics across all 3 models.

# pred.prob.class <- ifelse(pred.prob > 0.3, 1, 0) # Convert probs to classifications with a 0.3 threshold
