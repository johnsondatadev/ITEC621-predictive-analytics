##################
# ITEC 621 Trees #
##################

# Filename: ITEC621_Trees.R
# Prepared by J. Alberto Espinosa
# Last updated on 1/17/2022

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: the material covered in this script ASSUMES that you have gone through the self-paced course KSB-999 R Overview for Business Analytics and are thoroughly familiar with the concepts illustrated in the RWorkshopScripts.R script. If you have not completed this workshop and script exercises, please stop here and do that first.


#######################################################
#                       INDEX                         #
#######################################################

## Regression Trees

# Controlling Tree Size with mindev
# Controlling Tree Size with Cross-Validation

## Classification Trees

# Pruning Classification Trees
# Tree Splitting Based on Gini Index of "Purity"
# Cross Validation with Subset Sampling
# ROC Curves and Classification Trees

## Bootstrap Aggregation -- Bagging

# Bagging Tree Predictions

## Random Forest

## Boosted Trees

#######################################################
#                  END OF INDEX                       #
#######################################################


# Decision trees are mainly used for classification models, but they can also be used for association problems with regression trees. We focus primarily on classification trees, but we cover regression trees below for completeness.


## Regression Trees

library(MASS) # Contains the Boston data set
library(tree) # Needed to fit decision trees

# Say, for example, that you have 2 predictors. A regression tree will find which of the two variables divides the observations more accurately into two portions, and at which value of that variable, and then assigns a predicted value for each of the two portions equal to their respective response value mean. It then takes each of the two portions and further subdivides them, either with the same variable or the other variable, whichever separates the portion better. 

# The criteria for evaluating which variable to use for the split and at which value is based on minimizing the MSE. The process continues with further subdivisions of the data. Each sub-division portion is called a "leaf" and the splitting point is called a "node". In principle, one could continue the data splitting until each data point is a leaf, but this would overfit the data. The methods below help decide where to stop "growing" the tree using cross validation methods.

tree.boston <- tree(medv ~ ., Boston) 

# The summary shows 9 terminal nodes and the resulting residual mean deaviance (MSE) of 13.55

tree.boston # Check the specific splitting points:

# Get a quick summary of variables used, terminal nodes and mean deviance (mean of sum of squared errors)

summary(tree.boston) 

# Now let's visualize the tree

plot(tree.boston) # Plot the tree
text(tree.boston, pretty = 0) # Let's make it pretty and add labels

# Note: we will learn later how to grow and prune trees to achieve the lowest cross-validation MSE and thus, the tree size with the highest predictive accuracy. But, for now, if you wish to control the size of the tree, you can do so by changing the "mindev" attribute.


# Controlling Tree Size with mindev

# By default, mindev=0.01. This means that tree() will stop splitting a branch if the resulting deviance (MSE) is less than 0.01 (1%) of the deviance at the root of the tree. Increase this value to get fewer branches, and decrease it to get more branches.

# Small Tree

tree.boston.small <- tree(medv ~ ., 
                          Boston, 
                          mindev = 0.1) 

plot(tree.boston.small) # Plot the tree
text(tree.boston.small, pretty = 0) # Let's make it pretty and add labels

# Large Tree

tree.boston.large <- tree(medv ~ ., 
                          Boston, 
                          mindev = 0.005) 

plot(tree.boston.large) # Plot the tree
text(tree.boston.large, pretty = 0) # Let's make it pretty and add labels


# Controlling Tree Size with Cross-Validation

# Cross validation of tree results. Let's explore the CV for various pruned trees

set.seed(1)
cv.boston <- cv.tree(tree.boston) 

# Notes: 

# The cv.tree() function does 10FCV and minimizes the deviance measured as the MSE in regression trees (and 2LL for classification trees). You can change the 10-Fold to other folds with the attribute K =n n (number of folds)

# Let's plot tree size (number of terminal nodes) vs. deviance

plot(cv.boston$size, 
     cv.boston$dev, 
     xlab = "Tree Size", ylab = "10FCV Test SSE",
     type = 'b') 

# Notes: type='b' is for "both", points and lines
#        The $dev value in the plot is the sum of squared errors SSE

# If you prefer to plot the MSE instead

plot(cv.boston$size, 
     cv.boston$dev / nrow(Boston), 
     xlab = "Tree Size", ylab = "10FCV Test MSE",
     type = 'b')

cbind("Size" = cv.boston$size, "Deviance" = cv.boston$dev)

min.dev <- min(cv.boston$dev) # Find the tree size with smallest deviance
min.dev

best.ind <- which(cv.boston$dev == min.dev) # Tree with best CV deviance
best.ind

best.size <- cv.boston$size[best.ind] # Tree size with best CV deviance
best.size

# Notice that the lowest deviance is for the most complex tree w/9 leaves
# The tree is already at 9 leaves, but let's prune it anyway

prune.boston <- prune.tree(tree.boston, best = best.size) 

plot(prune.boston)
text(prune.boston, pretty = 0)

# Just to illustrate, let's prune to just 5 terminal nodes (or regions, or leaves), just to illlustrate

prune.boston <- prune.tree(tree.boston, best = 5) 

plot(prune.boston)
text(prune.boston, pretty = 0)

# If instead of using cv.tree to do cross-validation, you prefer to do sub-sample splitting manually, with training and test data:

# Let's illustrate a regression tree with the Boston data set

set.seed(1) # To get replicable results

trsize <- 0.7

train <- sample(1:nrow(Boston), trsize * nrow(Boston)) # Train index

Boston.train <- Boston[train,] # Train sub-sample
Boston.test <- Boston[-train,] # Test sub-sample

# Fit a regression tree on the train data

tree.boston <- tree(medv ~ ., Boston.train)

# Then test the fitted model on the test data (i.e., -train)

yhat <- predict(tree.boston, newdata = Boston.test)

plot(yhat, Boston.test$medv) # Let's plot predicted vs. actual
abline(0, 1) # And draw a line

mse.tree <- mean( (yhat - Boston.test$medv) ^ 2 ) # And calculate the MSE
mse.tree


## Classification Trees

# Classification trees work just like regression trees, but the response variable is binary (i.e., a classification). While decision are generally not as precise as logistic regression models, and despite the fact that they are not very useful for interpretation there is an abundance of sophisticated decision tree methods (e.g., Bootstrap Aggregation, Random Forests, Boosting, etc.), which can be quite accurate for prediction. 

# These tree methods differ on things like: how the tree branching happens; how much to grow a tree; when to prune the tree; which variables to use for splitting leaves; etc. We discuss some of these methods below. But first we start with just plain trees.

# install.packages("tree") # If not installed already

library(ISLR) # Contains the Carseats data set
library(tree) # Needed to fit classification trees

# Carseats is simulated data set of child car seat sales in 400 stores

attach(Carseats) 
head(Carseats) # Take a quick look at the data

# For classification trees, if the response variable is not binary, we need to convert it to binary using some criteria,  for example Sales<=8K

# ifelse is a useful function for this

High <- ifelse(Sales <= 8, "No", "Yes") # Storing the results in object "High"
High <- as.factor(High) # Some R versions need this conversion

# Add the results of "High" to the data

Carseats <- data.frame(Carseats, High) 
head(Carseats) # Check out that the "High" variable was added

# Let's fit a tree on all the data, except Sales 

tree.carseats <- tree(High ~ . -Sales, Carseats)

# Inspect the number of knots, deviance and training error

summary(tree.carseats) 

# Notes: 

# 1. The deviance is based on the log likelyhood function (2LL), the smaller the better. The residual mean deviance is the deviance divided by the number of observations minus terminal nodes. By itself, deviance or residual mean deviance are not very meaningful fit statistic, but are excellent to compare various tree modesl -- the model with the smallest deviance (between actual and predicted values is better)

# 2. The misclassification error is more meaningful. It is the total number of misclassified observations, relative to the total number of observations.

# Now let's analyze the tree visually

plot(tree.carseats) # Display tree
text(tree.carseats, pretty = 0) # Display data labels and make it pretty

tree.carseats # Display the data for every leaf


# Pruning Classification Trees

# cv.tree() does the cross validation of the various pruned trees

# Note: FUN=prune.misclass uses the misclassification error for cross validation and prunning. Othewise, the default is deviance.

cv.carseats <- cv.tree(tree.carseats, FUN = prune.misclass)
cv.carseats # Inspect the basic cv.tree() object results

# Note: "dev" reports the cross validation error rate, miss-classification in this case

cbind("Tree Size" = cv.carseats$size, "Misclass" = cv.carseats$dev)
# Note the tree with the smallest cross validation error ("dev")

# Let's inspect the tree visually

# Let's plot CV for each tree size, misclassification in this case

plot(cv.carseats, type = "b") # type="b" plots "both", lines and dots

# Missclassification is lowest at 22 nodes, but flattens around 12 nodes, so let's prune to 7 nodes (leaves)

prune.carseats <- prune.misclass(tree.carseats, best = 12) 

plot(prune.carseats) # Plot the tree
text(prune.carseats, pretty = 0) # With labels


# Tree splitting based on Gini index of "Purity"

# The default leaf splitting method is "missclassification"

tree.carseats <- tree(High ~ . -Sales, 
                      Carseats)

summary(tree.carseats) 

# But you can split leaves using the Gini index

tree.carseats <- tree(High ~ . -Sales, 
                      Carseats, 
                      split = "gini")

summary(tree.carseats) # Notice the higher missclassification error


# Cross Validation with Subset Sampling

# The method above fits a tree model with the entire data, but then the cv.tree() function does 10-Fold Cross Validation to find the optimal tree size to prube to. This is good if you are only using a tree model for your predictions. However, it is customary to try various classification models (e.g., logistic, LDA, etc.) and pick the one with the lowest cross-validation deviance. In this case we need to use similar cross validation methods across models. Since we have used subset sampling for the other classification models above, let's do the same here so that we can compare.

library(tree) # Needed to fit classification trees

heart <- read.table("Heart.csv", 
                    sep = ",", 
                    header = T, 
                    stringsAsFactors = T) 

heart$chd <- as.factor(heart$chd) # chd is numeric, need to convert to factor

# Compute index vectors for train and test sub-samples

set.seed(1)

trsize <- 0.7
train <- sample(1:nrow(heart), trsize * nrow(heart))

# Create train and test sub-samples

heart.train <- heart[train,]
heart.test <- heart[-train,]

# Let's check the sample sizes

nrow(heart)
nrow(heart.train)
nrow(heart.test)

# Fit the tree on the train data

heart.tree.train = tree(chd ~ ., data = heart.train)

# Inspect the results

heart.tree.train # See tree results
summary(heart.tree.train) # Basic tree results

plot(heart.tree.train) # Plot the tree
text(heart.tree.train, pretty = 0) # Add labels, pretty messy tree

# Now let's make predictions with the train model and the test subsample

# We will need classification predictions (0,1) for the confusion matrix

heart.tree.pred.class <- predict(heart.tree.train, 
                                 heart.test, 
                                 type = "class")

# But later we will need classification probabilities for the ROC curves

heart.tree.pred.prob <- predict(heart.tree.train, heart.test)

# Confusion Matrix

confmat <- table("Predicted" = heart.tree.pred.class, "Actual" = heart.test$chd) 
confmat

TruN <- confmat[1,1] # True negatives
TruP <- confmat[2,2] # True positives
FalN <- confmat[1,2] # False negatives
FalP <- confmat[2,1] # False positives
TotN <- TruN + FalP # Total negatives
TotP <- TruP + FalN # Total positives
Tot <- TotN + TotP # Total

# Take a look

c("TruN"=TruN, "TruP"=TruP, "FalN"=FalN, "FalP"=FalP, 
  "TotN"=TotN, "TotP"=TotP, "Tot"=Tot)

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

tree.rates.50 <- c(Accuracy.Rate, Error.Rate, 
                   Sensitivity, Specificity, FalP.Rate)

names(tree.rates.50)=c("Accuracy Rate", "Error Rate", 
                       "Sensitivity", "Specificity", "False Positives")

tree.rates.50

# Let's predict classes with the classification threshhold prob(chd = 1) > 0.6

thresh <- 0.6 # Set the threshold 
heart.tree.pred.class.60 <- ifelse(heart.tree.pred.prob[,2] > thresh, 1, 0)

# Confusion matrix with this new threshold

confmat <- table("Predicted" = heart.tree.pred.class.60, 
                 "Actual" = heart.test$chd) 

confmat

TruN <- confmat[1,1] # True negatives
TruP <- confmat[2,2] # True positives
FalN <- confmat[1,2] # False negatives
FalP <- confmat[2,1] # False positives
TotN <- TruN + FalP # Total negatives
TotP <- TruP + FalN # Total positives
Tot <- TotN + TotP # Total

# Take a look

c("TruN" = TruN, "TruP" = TruP, 
  "FalN" = FalN, "FalP" = FalP, 
  "TotN"=TotN, "TotP"=TotP, 
  "Tot"=Tot)

# Now let's use these to compute accuracy and error rates

Accuracy.Rate <- (TruN + TruP) / Tot
Error.Rate <- (FalN + FalP) / Tot
Sensitivity <- TruP / TotP # Proportion of correct positives
Specificity <- TruN / TotN # Proportion of correct negatives
FalP.Rate <- 1 - Specificity

tree.rates.60 <- c(Accuracy.Rate, Error.Rate, 
                   Sensitivity, Specificity, FalP.Rate)

names(tree.rates.60) <- c("Accuracy Rate", "Error Rate", 
                          "Sensitivity", "Specificity", "False Positives")

tree.rates.60

# Both thresholds together

rbind(tree.rates.50, tree.rates.60)


# ROC Curves and Classification Trees

library(ROCR)

# For the prediction() function, we use the predicted values from heart.tree.pred.prob above. But this object is matrix with 2 columns. The first has the probability of being 0 and the second has the probability of being 1. We need the second column, which we can extract with the [,2] index. 

pred <- prediction(heart.tree.pred.prob[,2], heart.test$chd) 
perf <- performance(pred, "tpr", "fpr")

plot(perf, colorize = T)

auc <- performance(pred,"auc") # Compute the AUC

auc.name <- auc@y.name[[1]] # Retrieve the AUC label text
auc.value <- round(auc@y.values[[1]], digits = 3) # Retrieve the AUC value, rounded

paste(auc.name, "is", auc.value) # Display AUC label and value -- not so great


#### Aside ####

# This is the cross validation example in the textbook, which is similar to what I did above. You can review this on your own, if you wish

set.seed(2)

# Take 200 observations for the training set (you can try other training/test samplings)

train <- sample(1:nrow(Carseats), 200) 

Carseats.train <- Carseats[train,]
Carseats.train # Take a look

Carseats.test <- Carseats[-train,] # And the rest for the test set.
Carseats.test # Take a look

# Let's create the "High.test" object from the "High (Yes/No)" object we computed for all the response values

High.test <- High[-train]

# Note, the command above does not yield the opposite of High. Rather it creates the High.test vector with all the observations that are not in the training set (i.e., the test set). Remember that the brackets [] are used as an index, not for math operations

# Let's do some machine learning

# Fit the model on the training set

tree.carseats <- tree(High ~ . -Sales, 
                      Carseats, 
                      subset = train)

tree.carseats <- tree(High ~ . -Sales, 
                      Carseats.train) # Alternatively

# Predict the test set

tree.pred <- predict(tree.carseats, Carseats.test, type = "class") 

table(tree.pred, High.test) # Confusion matrix
(86 + 57) / 200 # Accuracy rate = 71.3%

# Let's explore pruning the tree

set.seed(3)

# cv.tree() does the cross validation of the various pruned trees

# Note: FUN=prune.misclass uses the misclassification error for cross validation and prunning. Othewise, the default is deviance

cv.carseats <- cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)

# Note: "dev" reports the cross validation error rate
cv.carseats 
# Note the tree with the smallest cross validation error ("dev")

# Let's inspect the tree visually

par(mfrow=c(1, 2)) # Let's split the graphics screen

# Let's plot CV for each tree size, deviance = miss-classification

plot(cv.carseats, type = "b") 

# Now prune tree back to 9 nodes

prune.carseats <- prune.misclass(tree.carseats, best = 9) 

plot(prune.carseats) # Plot the tree
text(prune.carseats, pretty = 0) # With labels

# Let's evaluate the tree

# Predict on the test set

tree.pred <- predict(prune.carseats, Carseats.test, type = "class") 

table(tree.pred, High.test) # Confusion matrix
(94+60)/200 
# 77%, which is an improvement over the full tree and more interpretable

#### /Aside ####


## Bootstrap Aggregation -- Bagging

# The tree methods illustrated in the remainder of this section rely on the modeling of more than one tree and then aggregating the result. The methods differ in what varies from tree to tree.

# Bagging stands for "bootstrap aggregation". Bootstrap means to draw a random sample to fit a tree, and then repeat this many times with new samples with replacement. Each tree includes all variables in the model and a subset of observations. The results are then aggregated. The rationale behind this method is that bootstrapping and aggregating reduces variance and tends to produce more accurate and stable results than single trees. The main tuning parameter in Bagging is the size of the random sample and the number of trees fitted and aggregated.

# As we discuss below, Bagging is a special case of Random Forest. The main difference is that in Random Forest, the number of variables m used to fit the individual trees is a subset of all the available variables p, so m<=p. In Bagging, m=p, thus a special case of Random Forest. Therefore, both methods use the same library and function randomForest(){randomForest}.

library(randomForest) # install "randomForest" package if you haven't
?randomForest()

library(MASS) # Contains the Boston housing data set

# Let's re-create the Boston training subset we created above. No need to do this if you have not quit R since you computed the training sample above. Otherwise, re-creat "train"

set.seed(1) # To get repeatable results

trsize <- 0.7

train <- sample(1:nrow(Boston), trsize * nrow(Boston)) 
Boston.train <- Boston[train,]

# Let's fit a Bagging model (i.e., Random Forest with all variables included in the training data. 

# IMPORTANT: we are fitting a bagged regression tree, but the same method applies to bagged classification trees

bag.boston <- randomForest(medv ~ ., 
                           data = Boston.train, 
                           mtry = 13, importance = T)

plot(bag.boston) # Notice that the MSE error flattens after about 100 bootstrapped trees

# Note: mtry = 13 tells Random Forest to use all 13 predictors, which is the full set, thus a Bagging model with m=p (the number of variables in each estimaged tree m is equal to the number of total predictors in the model.

# FYI, as we will see below, Random Forest trees generate random trees with different m variables each time. The variables used is random too. The total number of predictors is p. Thus a model with m<p is called "Random Forest", whereas a model with m=p, in which all the predictors are used in each tree estimation, is called "Bootstrap Aggregation".

# The importance attribute is set to TRUE to obtain each variable's importance in reducing error

bag.boston # Take a quick look

# Notice that the model was fit for 500 trees, each with 13 predictors (which are all the predictors, thus a bagging tree)

# One shortcoming of trees is that there are no coefficients or p-values to ascertain which variables have stronger effects. The importance() function helps overcome this problem by displaying the importance of each variable to the tree model.

# Variable Importance Plot
varImpPlot(bag.boston) 

# See the actual values
importance(bag.boston)

# The 2 values of importance reported above are:

# (1) Mean increase (over all trees) in accuracy (% MSE  
#     explained) when the variable is added to the the model

# (2) Mean increase (over all trees) in "purity" when the  
#     tree is split by that variable

# Higher values are best for either


# Bagging Tree Predictions

Boston.test <- Boston[-train,]

bag.pred <- predict(bag.boston, newdata = Boston.test) 

plot(bag.pred, 
     Boston.test$medv, 
     xlab = "Predicted", 
     ylab = "Actual") # Plot pred vs. actual

abline(0,1) # Draw a 45 degree line (intercept=0; slope=1)

mse.bag <- mean((bag.pred - Boston.test$medv)^2) # Get the mean squared error
mse.bag

# Notice that the MSE is almost 1/2 of the regression tree MSE

# To fit a model with a different number of trees, use the ntree = 25 (for 25 trees, or any other number of trees)

bag.boston.25 <- randomForest(medv ~ ., data=Boston.train, mtry=13, ntree=25)

bag.pred.25 <- predict(bag.boston.25, newdata=Boston.test)

plot(bag.boston.25)
mean((bag.pred.25 - Boston.test$medv)^2)

# Notice that the MSE is a bit higher than with 500 trees


## Random Forest

# Note: Again, Bagging is a special case of Random Forest. In Random Forest we use m sample predictors from a set of p available predictors such that m<=p. We also vary the m predictors from tree to tree to to reduce the correlation among the trees. The limitation of Bagging is that all trees are fitted with the same predictors, so results are likely to be somewhat correlated. Random Forest with m<p helps correct for this because every tree will be different.

# We use the same randomForest() function, but specify a smaller number of sampled predictors with "mtry" than the number of available predictors, i.e., m<p. If not specified, m=p/3 is the default.

library(randomForest) # if not loaded already
library(MASS) # For Boston housing data set, if not loaded already

# I did this above already, repeating here for convenience

set.seed(1) # To get replicable results

trsize <- 0.7
train <- sample(1:nrow(Boston), trsize * nrow(Boston)) 

Boston.train <- Boston[train,]
Boston.test <- Boston[-train,]

rf.boston <- randomForest(medv ~ ., 
                          data = Boston.train, 
                          mtry = 6, 
                          importance = T)

rf.boston # Inspect the results

plot(rf.boston) # Notice how MSE declines as more trees are sampled
# It seems like 150 trees may be sufficient.

varImpPlot(rf.boston) # We can also plot the results
importance(rf.boston) # To view the importance of each variable

# The results show that house size (rm) and overall community wealth (lstat) and are the most important predictors.

# Let's do predictions with the test data

rf.pred <- predict(rf.boston, newdata = Boston.test) 

plot(rf.pred, Boston.test$medv, 
     xlab="Predicted", 
     ylab="Actual") # Plot pred vs. actual

abline(0,1) # Draw a line

mse.rf <- mean( (rf.pred - Boston.test$medv)^2 )
mse.rf

# Note that the MSE for this model is even smaller than for the Bagged model


## Boosted Trees


library(gbm) # Generalized Boosted Regression Models 
library(MASS) # Contains the Boston data set

# Like Bagging and Random Forest, Boosting models fit several trees and aggregate the result. Unlike Bagging and Random Forest, Boosting does not fit several random trees, but it fits an initial tree and then fit another one to explain the residuals (errors), then again, etc. 

# Bagging and Random Forest are considered "fast" learning methods because the best model is generated in the first few samples and subsequent trees may or may not improve the MSE, whereas Boosting is considered to be a "slow" learning method because every new tree builds upon and improves the prior tree. 

# The tuning parameter "lambda" (works just like shrinkage in Ridge and LASSO) controls the speed of learning.

# Aside: to understand this concept, imagine that you run an OLS regression with certain variables and you get some fairly large residuals (i.e., errors). The residual values represent the portion of the outcome values that are not explained by the model. You can then build another OLS regression model to explain (i.e., predict) those residuals. This new regression will explain some of the error variance, but will also yield new errors (smaller than the first ones, because some of the variance in the errors is already explained with the second model). Then you can fit a third regression model to explain the new residuals, and so on. You can then aggregate all the regression models, which on the aggregate, will have small residuals. Boosting applies this concept when generating trees.

set.seed(1) # To get replicable results

# Let's fit a Boosting model on the Boston data. Use:

# - distribution = "gaussian" (i.e., normal distribution) for regression trees

# - distribution = "bernoulli" for classification trees

# Let's fit a model with 5000 trees, limiting the depth of each tree to 4 and using all available predictors

boost.boston <- gbm(medv ~ ., data = Boston, 
                    distribution = "gaussian", 
                    shrinkage = 0.01,
                    cv.folds = 10,
                    n.trees = 5000, 
                    interaction.depth = 4) # 4 tree splits

boost.boston

# Number of boosted trees with smallest CV Test Error

best.num.trees <- which.min(boost.boston$cv.error)

# Smallest CV Test Error

min.10FCV.error <- round(min(boost.boston$cv.error), digits = 4)

# Display result

paste("Min 10FCV Test Error =", min.10FCV.error, 
      "at", best.num.trees, "trees")

# Plot CV Test Error against Number of Trees

plot(boost.boston$cv.error, 
     type = "l",
     xlab = "Number of Boosted Trees", 
     ylab = "10FCV Test MSE")

# Variable Importance or Relative Influence Graph

# The relative influence is normalized, so that the scores add up to 100

summary(boost.boston) # Provides relative influence stats and plot

# Note again, that `lstat` and `rm` are the most important variables, just like with Bagging and Random Forest. Note, the graph may not show all the predictors if the graph is small, but if you zoom it, you should see all predictors.

# Boosted Trees and Partial Dependencies

# Partial dependencies show how a particular predictor affects the outcome variable, holding other predictors constant. It is important to note that this is NOT an effect. You can derive effects from the graph, but an effect is how much Y increases when X increases by 1. In contrast, in the partial dependencies graph, we can see what is the value of Y for a given value of X, holding everything else constant.

# To see how predicted house median values in Boston `medv` vary with `lstat` and `rm`, we can use the `i` vector. The `boost.boston` object has an attribute property called `i`, which is a vector containing the variables used to build the model. To plot how the outcome variable is partially affected by a predictor use the parameter `i =` to specify the predictor of interest.

plot(boost.boston, 
     i = "lstat", 
     ylab = "House Median Value", 
     xlab = "Lower Status Percent of Population")

plot(boost.boston, 
     i = "rm", 
     ylab = "House Median Value", 
     xlab = "Average Number of Rooms per House")


# Cross-Validation with Boosted Trees

# Let's now do Cross-Validation predictions with the test data

set.seed(1)

tr.size <- 0.7
train <- sample(1:nrow(Boston), 
                tr.size * nrow(Boston)) # Train index

Boston.train <- Boston[train,]
Boston.test <- Boston[-train,] 

boost.boston.tr <- gbm(medv ~ ., 
                       data = Boston.train, 
                       distribution = "gaussian", 
                       shrinkage = 0.01, # Lambda
                       cv.folds = 10, # 10FCV
                       n.trees = 5000, # Number of boosted trees
                       interaction.depth = 4) # 4 tree splits       

boost.boston.tr # Basic output

boost.pred <- predict(boost.boston.tr, # Predict with the trained model
                      newdata = Boston.test, # Test with the test subset data
                      n.trees = 5000)

mse.boost <- mean( (boost.pred - Boston.test$medv) ^ 2) # MSE
mse.boost

# Let's compare results. Which method is better

round(cbind("MSE Plain Tree" = mse.tree, 
            "MSE Bagging" = mse.bag, 
            "MSE Random Forest" = mse.rf, 
            "MSE Boosted Tree" = mse.boost),
      digits = 3)

# In this example, the boosted tree model is far superior to the other 3 models, and the plain tree is far inferior to the other 3 models. Bagging and Random Forest are somewhere in between.

# Shrinkage

# Boosting has a similar shrinkage effect, just like Ridge and LASSO regression. The difference is that Ridge and LASSO shrink the regression coefficients, whereas boosting shrinks earlier outcome predictions to weaken the model, to then be further strengthened as it learns from the errors in subsequent models. The shrinkage applies over each tree model, including the first one. A small lambda shrinks he prior tree model predictions more, thus making the early predictions less important for the final aggregated model (i.e., slow learning). Large lambdas give more weight to the initial trees, thus learning fast.

# To vary the shrinkage factor lambda set the `shrinkage =` parameter. The default is 0.01. The default for the interaction tree depth is 1. Let's lower the shrinkage parameter and retain the depth of 1 knot (i.e., a tree "stomp")

shrk <- 0.001 # Let's try a slow learning rate
dpth <- 1 # interaction tree depth

boost.boston.shrk <- gbm(medv ~ ., 
                     data = Boston.train, 
                     distribution = "gaussian", 
                     n.trees = 5000, 
                     interaction.depth = dpth, 
                     shrinkage = shrk, 
                     verbose = F)

# Now let's do predictions with the test data

yhat.boost.pred <- predict(boost.boston.shrk, 
                           newdata = Boston.test, 
                           n.trees = 5000)

mean( (yhat.boost.pred - Boston.test$medv) ^2) 

# Not better


# Boosted Classification Tree

# Classification trees are fitted similarly to regression boosted trees, but with two important differences:

# 1. Use `distribution = "bernoulli" instead of "gaussian"

# 2. Convert the outcome from a factor variable to 0, 1 numeric

# For example, using the Heart.csv data set to predict chd (coronary heart disease)

heart <- read.table("Heart.csv", 
                    sep = ",", 
                    header = T, 
                    stringsAsFactors = T) 

# Create a numeric variable for chd

heart$chd.n <- as.numeric(heart$chd)
head(heart$chd.n)

# Notice that the data was coded as 1-2, not 0-1. We need a quick transformation to 0-1 values

heart$chd.n <- heart$chd.n - 1
head(heart$chd.n) # Now we are good to go

boost.chd <- gbm(chd.n ~ . - chd, # remove chd from the predictors
                 data = heart, 
                 distribution = "bernoulli")

summary(boost.chd)

# age and tobacco are the most important predictors of heart disease. Let's plot their partial correlation. The Y axis depicts the Log-Odds of chd = 1. To convert to odds use exp(Log-Odds)

plot(boost.chd, 
     i = "age", 
     ylab = "Log-Odds of Heart Disease", 
     xlab = "Age")

plot(boost.chd, 
     i = "tobacco", 
     ylab = "Log-Odds of Heart Disease", 
     xlab = "Consumed Tobacco")

# For example, the Log-Odds of having coronary heart disease at the age of 60 is about - 0.2, holding everything else constant. The odds are exp(-.02) - 0.82. The probability is Odds / (Odds + 1) = 0.45 or 45%

odds <- round(exp(-0.2), digits = 3)
paste("The odds of coronary heart disease are", odds)

prob <- round(odds / (1 + odds), digits = 3)
paste("The probability of coronary heart disease are", prob)

