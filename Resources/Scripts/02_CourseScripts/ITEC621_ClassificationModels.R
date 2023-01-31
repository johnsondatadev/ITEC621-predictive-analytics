##################################
# ITEC 621 Classification Models #
##################################

# Filename: ITEC621_ClassificationModels.R
# Prepared by J. Alberto Espinosa
# Last updated on 3/27/2022

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: the material covered in this script ASSUMES that you have gone through the self-paced course KSB-999 R Overview for Business Analytics and are thoroughly familiar with the concepts illustrated in the RWorkshopScripts.R script. If you have not completed this workshop and script exercises, please stop here and do that first.


#######################################################
#                       INDEX                         #
#######################################################

## Binomial Logistic Regression
# Transforming Coefficients

## Confusion Matrix
# Compute Confusion Matrix Fit Statistics
# Using prob > 0.60 as the classification threshold (Lambda)

## ROC Curves and Binomial Logistic 
## Mutinomial Logistic Regression

## Linear Discriminant Analysis (LDA)
## ROC Curves and LDA
## Quadratic Discriminant Analysis (QDA)
## ROC Curves and QDA

#######################################################
#                  END OF INDEX                       #
#######################################################

### Logistic Regression, LDA, QDA, and KNN


# Classification models predict outcomes that are either binary or categorical. There are two broad categories of classification models -- association and tree based. In this section we cover association models based on regression and discriminant analysis methods.

# We will be using the Stock Market (Smarket) data set frequently in this section. A few notes are in order.

library(ISLR) # Contains the Smarket data set
attach(Smarket)

?Smarket # Review the variables in this data set.
dim(Smarket) # Retrieve the dimensions of the Smarket data table
# 1250 observations and 9 variables

# This data set does not yield very interesting results in logistic models but the results illustrate a few important points. It also illustrates the use of lag transformations in logistic models. Let's explore the data

names(Smarket)
head(Smarket)

# One important thing to note is that the variable "Direction" is categorical, not binary. So, we have two choices, create a binary variable called "Up" =1 when Direction is up and 0 otherwise. But in the example below we don't make conversions until later and we let R take care of this for us.

summary(Smarket)

# To plot correlation scaterplots of the data
pairs(Smarket) 

# Will give an error because the variable Direction is not numeric
cor(Smarket) 

# Remove that variable from the correlation matrix
cor(Smarket[,-9]) 

# We can use the ggpairs(){GGally} function instead, since it works with categorical variables too.

library(GGally)
ggpairs(Smarket)


## Binomial Logistic Regression

# The outcome in binary predictive logistic model is one of two values: 0 or 1; yes or no; success or failure; approve or disapprove; etc.

# The syntax for logistic regression is the same as lm(){stats}, but it uses the glm(){stats} function rather than the lm() function. glm() stands for "Generalized Linear Model", which covers a number of linear regression models and various transformations of the dependent variable (e.g., logistic). 

# A linear model with no transformation of the dependent variables estimated with glm() and a "gaussian" distribution yields the exact same results as OLS with the lm() function. But glm() supports other distributions and transformations of the dependent variable. The "link" function is the type of transformation of the dependent variable. 

# Different regression methods require specific "model family" and "link" function parameters:

# - Binomial Logistic regression is family=binomial(link="logit")
# - OLS regression is family=gaussian(link="identity")
# - Binomial Probit regression is family=binomial(link="probit")
#   Probit stands for "Probability Unit" and it works just like 
#   logistic, except that it uses a different transformation 
#   function for Y. 
# - Count data models uses family=poisson(link="log")
            
# Example with external data on coronary hearth disease (chd), provided by the ISLR textbook authors

browseURL("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.info.txt") # Note: this file's content and location have changed. I'm using the previous Heart.csv data set, which is on Blackboard.

heart <- read.table("Heart.csv", 
                    sep = ",", 
                    head = T, 
                    stringsAsFactors = T) 

attach(heart)
head(heart)

# Logistic model predicting coronary heart disease

heart.fit = glm(chd ~ ., 
                family = binomial(link="logit"), 
                data = heart)

summary(heart.fit)

# Looks like tobacco, ldl, family history, type a and age are the strongest predictors of coronary heart desease (chd). Interestingly Once you control for these factors obesity and alcohol are not significant predictors. 

# Fit statistics

-2*logLik(heart.fit) # 2LL
deviance(heart.fit) # Should yield the same value
AIC(heart.fit) # 2LL + 2*Number of variables

# Or, all together

c("2LL" = -2 * logLik(heart.fit), 
  "Deviance" = deviance(heart.fit), 
  "AIC" = AIC(heart.fit))

# Transforming Coefficients

log.odds <- coef(heart.fit) # To get just the coefficients
log.odds # Check it out

# To convert Log-Odds to multiplicative change in odds

odds <- exp(coef(heart.fit)) 
odds # Check it out

prob <- odds / (1 + odds) # To convert odds to probabilities
prob # Check it out

round(cbind("Log-Odds" = log.odds, 
            "Odds" = odds, 
            "Probabilities" = prob), # All together
      digits = 3)

# Confidence intervals

# To get the 95% confidence intervals of Log-Odds coefficients

round(confint(heart.fit) , digits = 3)

# (Log-Odds^e) -- the 95% confidence interval of odds

round(exp(confint(heart.fit)), digits = 3)


## Confusion Matrix

# Let's do some Cross-Validation testing with the confusion matrix

# Define the training and test sub-samples

RNGkind(sample.kind="default") # Set the default random number generator
set.seed(1) # Set the random seed to get repeatable results

train <- sample(1:nrow(heart), 0.7 * nrow(heart)) # train sub-sample (index vector)

# Let's create the train and test sub-samples.

heart.train <- heart[train,] # Train sub-sample
nrow(heart.train) # Count the train observations

heart.test <- heart[-train,] # Test sub-sample
nrow(heart.test) # Count the test observations

# Cross Validation with Binary Logistic Predictions
# Fit the model on the training data

heart.fit.train <- glm(chd ~ ., 
                       family = binomial(link = "logit"), 
                       data = heart.train)

summary(heart.fit.train)

# Predicted values using the fitted train model and the test data

# We use type="response" to get "probability" of Y=1

heart.probs.test <- predict(heart.fit.train, 
                            heart.test, 
                            type = "response")

heart.probs.test[1:10] # List first 10


# Confusion matrix for the test data predictions

# Convert probability to prediction, notice that we use the classification threshold of prob > 0.5. As we discussed earlier, we can manipulate this threshold depending and whether we are more interested in sensitivity or specificity.

# We could use the classification threshold 0.5 in the formulas below, but computationally, it is better to store the threshold in a variable (i.e., thresh) to make it easy to change if we want to repeat the calculations with a different threshold. You can change the threshold value below from the console (don't change the script) to experiment and see how the confusion matrix changes. Try different thresholds, but remember to change it back to 0.5 when you are done experimenting

# This works:

heart.pred.test <- ifelse(heart.probs.test > 0.5, 1, 0) 

# But computationally this is better coding and the results are the same:

thresh <- 0.5
heart.pred.test <- ifelse(heart.probs.test > thresh, 1, 0) 

heart.pred.test[1:10] # List first 10

# Quick check to see if prob > 0.5 has a predicted classification of 1

cbind(heart.probs.test, heart.pred.test)[1:10,]

# Cross tabulate Prediction with Actual

conf.mat <- table("Predicted" = heart.pred.test, 
                  "Actual" = heart.test$chd) 

# Let's label rows and columns

colnames(conf.mat) <- c("No", "Yes")
rownames(conf.mat) <- c("No", "Yes")

conf.mat

# Compute Confusion Matrix Fit Statistics

TruN <- conf.mat[1,1] # True negatives
TruP <- conf.mat[2,2] # True positives
FalN <- conf.mat[1,2] # False negatives
FalP <- conf.mat[2,1] # False positives
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

FalseP.Rate <- 1 - Specificity
FalseP.Rate

logit.rates.50 <- c(Accuracy.Rate, Error.Rate, 
                    Sensitivity, Specificity, 
                    FalseP.Rate)

names(logit.rates.50) <- c("Accuracy Rate", "Error Rate", 
                           "Sensitivity", "Specificity", 
                           "False Positives")

print(logit.rates.50, digits = 2)

# Question, can you repeat the modeling above with classification thresholds different than prob > 0.5, say prob > 0.40 and prob > 0.60?

# Answer, copy and paste everything from "# Confusion Matrix" to "logit.rates.50" to a new set of commands and replace "ifelse(heart.probs.test>0.5, 1,0)" with "ifelse(heart.probs.test>0.4, 1,0) (or >0.60)"

# Question, which one is more/less conservative than prob>0.5? Is it prob>0.40 or prob>0.60?

# Answer, prob>0.40 is a less conservative threshold because it will allow more observations to be classified as chd=1 (heart disease). This will lead to more false (misclassified) positives. prob>0.60 is more conservative because it allows fewer observations to be misclassified as positives. However, it will lead to more false negatives. So, it is a tradeoff.

# Using prob > 0.60 as the classification threshold (Lambda)

thresh <- 0.60
heart.pred.test.60 <- ifelse(heart.probs.test > thresh, 1, 0) 

# Cross tabulate Prediction with Actual

conf.mat <- table("Predicted" = heart.pred.test.60, 
                  "Actual" = heart.test$chd)

conf.mat # Check it out

# Let's compute various confusion matrix rates

TruN <- conf.mat[1, 1] # True negatives
TruP <- conf.mat[2, 2] # True positives
FalP <- conf.mat[2, 1] # False positives
FalN <- conf.mat[1, 2] # False negatives
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

logit.rates.60 <- c(Accuracy.Rate, Error.Rate, Sensitivity, Specificity, FalP.Rate)

names(logit.rates.60) <- c("Accuracy Rate", "Error Rate", 
                           "Sensitivity", "Specificity", 
                           "False Positives")

print(logit.rates.60, digits = 2)

# Compare the rates for both thresholds and see the sharp differences

logit.fit.stats.compare <- rbind(logit.rates.50, logit.rates.60)
print(logit.fit.stats.compare, digits = 2)

# Conclusion -- the threshold really matters


## ROC Curves and Binomial Logistic 

# A second package and function for ROC curves

library(ROCR)

# The first step with ROCR is to create a prediction(){ROCR} object using two vectors: (1) the predicted probabilities, computed above, and the actuas values. Notice that we use the index [test] to match the records in the predicted values

pred <- prediction(heart.probs.test, heart.test$chd) 

# The next step is to use the performance(){ROCR} function to create the ROC object. Use "tpr" for True Positive Rate in the vertical axis, and "fpr" for False Positive Rate in the horizontal axis. Other possible values are: "acc"=accuracy; "err"=error rate; "sens"=sensitivity; "spec"=specificity; "auc"=area under the curve 

perf <- performance(pred,"tpr","fpr")
plot(perf, colorize=T)

# Computing the AUC -- also done with the performance() function:

auc <- performance(pred,"auc")

# The performance() object above stores the name of the variable in @y.name[[1]] and the actual AUC in @y.values[[1]]. Note: the use of double brackets [[]] instead of single brackets [] and @ instead of $ to access values is because the performance() object is a "list" not a data frame. Lists use [[]] for indexing values and @ for accessing elements in the list. Also, note that I round the AUC value to keep it short.

auc.name <- auc@y.name[[1]] # Retrieve the AUC name from the AUC list
auc.value <- round(auc@y.values[[1]], digits = 3) # Retrieve the AUC value

paste(auc.name, "is", auc.value) # Display them together


# Another (less powerful) function for ROC curves is roc(){pROC}

library(pROC) # Contains the roc() function to compute ROC curves

# The roc() function takes a regression-like formula y~x, plus a number of parameters. x is a predicted set with probability values (i.e., heart.probs.test, computed above with type="response") and y are the actual values of the response variable (i.e., heart[test,]$chd).

# Notice that we are using the test set and test predictions for this illustration, but you can plot ROC curves for any sub-sample or the entire data set. The important thing is that the number of records in y and x match (as with the [test] set).

# plot=TRUE causes the function to plot the ROC curve. You could omit this plot argument and then plot the curve separately with plot(roc.logit.50)

roc.logit.50 <- roc(heart.test$chd ~ heart.probs.test, plot=T, smooth=T)
roc.logit.50 # You can read off the AUC here


#### Aside ####

# Smarket Example: 

# Another logistic example provided in the textbook. It has less interesting results but it illustrates the method well.

library(ISLR) # Contains the Smarket stock market data set
attach(Smarket)

fit.logit <- glm(Direction ~ Lag1 + Lag2 + Lag3 + 
                             Lag4 + Lag5 + Volume, 
                 data=Smarket, family=binomial(link="logit"))

# Let's look at the coefficient plot

require(coefplot)
coefplot(fit.logit)

# Note that, visually, all the 95% confidence intervals cross the 0 mark, so none of the coefficients are significant whe all lag variables are included in the model.

# Let's look at fit statistics

summary(fit.logit) # Let's look at summary statistics

# Sure enough, none of the coefficients are significant

logLik(fit.logit) # Get the log-likelihood
-2*logLik(fit.logit) # Should be equal to the residual deviance, or just
deviance(fit.logit) # Should yield the same value

# AIC = deviance + penalty, so it should be somewhat higher than the deviance
AIC(fit.logit) 

# The penalty is 2*Number of variables in the model.Smaller deviance and AIC are better; it means that there is a smaller likelihood that the predicted values deviate from actual values (just like MSE), that is, how much it is not explained by the model.

log.odds <- coef(fit.logit) # To get just the coefficients
log.odds # They are very close to 0

# To convert Log-Odds to multiplicative change in odds
odds <- exp(coef(fit.logit))
odds # Naturally, they are very close to 1

# TO convert odds to probabilities
prob <- odds/(1+odds) 
prob # Around 50%, almost like flipping a coin

cbind(log.odds, odds, prob) # Check them out together

# To get the 95% confidence intervals of Log-Odds coefficients
confint(fit.logit)

# exp(x) = e^x -- converts log-odds into odds;The 95% confidence interval of odds:
exp(confint(fit.logit)) 

# 95% confidence interval for the probabilities
exp(confint(fit.logit)) / ( 1 + exp(confint(fit.logit)) ) 

# The predict() function outputs (by default) predicted values for all observations in the training data

probs.logit <- predict(fit.logit, type="response")
# type="response" attribute gives the "probability" of Y=1

probs.logit[1:10] # List first 10

# To display what 0 and 1 mean in the response variable
contrasts(Direction) 

# Let's convert predicted values into a label "Up" if the probability is > 50%, "Down" otherwise

pred.logit = ifelse(probs.logit > 0.5, "Up", "Down")
pred.logit

# Confusion matrix

# To cross tabulate Prediction (probabilities) with Direction (actual)
conf.mat <- table(pred.logit, Direction) 
conf.mat # Check it out

# Let's compute various confusion matrix rates

TruN <- conf.mat[1,1] # True negatives
TruP <- conf.mat[2,2] # True positives
FalP <- conf.mat[2,1] # False positives
FalN <- conf.mat[1,2] # False negatives
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

logit.rates.50 <- c(Error.Rate, Sensitivity, Specificity, FalP.Rate)

names(logit.rates.50) <-
     c("Error Rate", "Sensitivity", "Specificity", "False Positives")

logit.rates.50

# Classification threshold (lambda) for Smarket example

# Note: the glm.probs>0.5 defines the "threshold" for classification of the predictive model.

# We can tune this value depending on the model needs. For example, if we want to be conservative in our predictions a value of glm.probs>0.6 will only classify an observation as "Up" or 1 if the predicted probability is greater than 60%.

pred.logit.cons <- ifelse(probs.logit > 0.60, "Up", "Down")
pred.logit.cons

# To get the "training error rate" (proportion of off diagonal values):

1 - mean(pred.logit.cons == Direction) 
mean(pred.logit.cons != Direction) # Same thing

# Compare with the conservative prediction

conf.mat # Original matrix with threshold > 0.5
conf.mat.cons <- table(pred.logit.cons, Direction) # Conservative matrix
conf.mat.cons # Check out the dramatic changes
# Conclusion -- the threshold really matters

# Mean when prediction values match the actual Direction
mean(pred.logit.cons == Direction) 

# Mean when prediction values DON'T match the actual Direction
# i.e., the Error Rate
mean(pred.logit.cons != Direction) 

# Let's compute various confusion matrix rates

TruN=conf.mat.cons[1,1] # True negatives
TruP=conf.mat.cons[2,2] # True positives
FalP=conf.mat.cons[2,1] # False positives
FalN=conf.mat.cons[1,2] # False negatives
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

logit.rates.60 <- c(Error.Rate, Sensitivity, Specificity, FalP.Rate)

names(logit.rates.60) <- 
    c("Error Rate", "Sensitivity", "Specificity", "False Positives")

logit.rates.60

# Compare the rates for both thresholds and see the sharp differences

logit.fit.stats.compare <- rbind(logit.rates.50, logit.rates.60)
logit.fit.stats.compare

# Smarket Cross-Validation example below from the textbook has less interesting results, but it illustrates the method well

# In this example, we will use data before 2005 for the training subsample and everything else for testing

train <- (Year < 2005) # Creates a vector named "train"

# The train vector above contains TRUE/FAlse values for each observation, TRUE for Year<2005 and FALSE otherwise. This is called a "Boleean" vector. Take a look:

train # Check it out

# Boolean vectors are very useful to partition the data. For example, to creates a holdout set with all obvervations not on the training set

Smarket.2005 <- Smarket[!train,] 

# Inspect the test data set, it has 252 observations and 9 variables
dim(Smarket.2005) 

# Use train to get the data in the training set an !train to get data not in the training set. For example, this creates a vector with the values of the Direction variable for the hold out data set, Which we will use shortly to build the confusion matrix.

Direction.2005 <- Direction[!train] 
Direction.2005 # Check it out

# Let's train the model using only the training data set

fit.2005 <- glm(Direction ~ Lag1 + Lag2 + Lag3 +
                            Lag4 + Lag5 + Volume, 
                data=Smarket, family=binomial(link="logit"), subset=train)

summary(fit.2005)

# Let's predict values in the holdout set
probs.2005 <- predict(fit.2005, Smarket.2005, type="response")

# First, create a test vector with all values equal to "Down"
pred.2005 <- rep("Down", 252) 

# Change the value to "Up" of the predicted probability is > 0.5
pred.2005[probs.2005 > 0.5] = "Up" 

# Confusion matrix predicted vs. actual (i.e., Direction)

table(pred.2005, Direction.2005) 
mean(pred.2005 == Direction.2005) # Correct rate
mean(pred.2005 != Direction.2005) # Error rate

# Let's do some predictions with a reduced model with only Lag1 and Lag2

fit.2005 <- glm(Direction ~ Lag1 + Lag2, 
                data=Smarket, family=binomial(link="logit"), 
                subset=train)

summary(fit.2005)

probs.2005 <- predict(fit.2005, Smarket.2005, type="response")
pred.2005 <- rep("Down",252)
pred.2005[probs.2005 > 0.5] = "Up"

table(pred.2005, Direction.2005)
mean(pred.2005 == Direction.2005) # Correct rate
mean(pred.2005 != Direction.2005) # Error rate

# Let's make a prediction for 3 observations with data on Lag1 and Lag2

predict(fit.2005, 
        newdata=data.frame(Lag1 = c(1.2, 1.5), 
                           Lag2=c(1.1,-0.8)),
        type="response")

#### /Aside ####


## Mutinomial Logistic Regression

# A multinomial logit model is identical to a binomial logit model, except that the categorical outcome variable has more than 2 possible values. For example, suppose that you are trying to predict when people will buy rural, suburban or urban homes. In this case, the dependent variable has 3 possible values. This model can be easily fitted using 2 binomial logit models. First, you create select a baseline or reference value (e.g., rural) and then create dummy variables for the two other categories (i.e., suburban and urban). You then build a binomial logit model with suburban (=1 if suburban, 0 if not) and estimate it. You then do the same with urban (=1 if urban, 0 if not). Each regression will give you the effect on the log odds of someone buying a suburban or rural house respectively.

# This is a valid approach, but since you fit 2 separate models you get separate fit statistics for each of the models. There are statistical routines that can extimate the two binomial logit models together and give you one set of fit statistics for the whole model.

# More generally, a multinomial model is one in which the outcome variable has K categorical values. In our example above K = 3. This model is fit with a multinomial logistic regression composed of K-1 binomial logit models, but estimated together as a single model. There are R packages and functions that will estimate all K-1 models jointly. The most popular ones are:

# multinom(){nnet}; mlogit{mlogit}; and 

# {VGAM} package (Vector Generalized and Additive Models) has the vglm() function (Vector Generalized Linear Model), which runs multinomial logistic and other categorical regressions

# ASIDE: two other packages to fit multinomial logistic models: multinom(){nnet}, glmnet(){glmnet}

library(VGAM) # Contains the vglm() function
library(car) # Contains the Womenlf (Women Labor Force) data set we use below

# Data set about Women's Labor Force Participation
attach(Womenlf)
head(Womenlf)
? Womenlf

# Let's fit a model for labor participation (fulltime, not.work, parttime), based on husband's income in thousands, and presence of children in the household (absence, presence)

# First, notice the attribute refLevel=1, "fulltime"
levels(Womenlf$partic)

# The levels in $partic are: "fulltime" "not.work" "parttime"
# "fulltime" is the refLevel, but you can always change this
# The effect for "not.work" will be suffixed :1 in the summary output
# The effect for "parttime" will be suffixed :2 in the summary output

vglm.fit <- vglm(partic ~ hincome + children, 
                 family = multinomial(refLevel = 1), 
                 data = Womenlf)

# Which corresponds to "fulltime" and can be changed as needed

vglm.summary <- summary(vglm.fit)
vglm.summary

# Interpretation: 

# The coefficients in multinomial logit, just like in binomial logit, show the effect of a 1 unit increase in the predictor variable on the log-odds of the dependent variable. But this is the difference:

# - In binomial logit, the coefficient is about the log-odds of the response variable being 1 (relative to 0, of course)

# - In multinomial logit, the coefficient is about the log-odds of the response variable being in that category, relative to  the reference category (i.e., the response variable left out of the one specified with the "refLevel" attribute)

# Fit Statistics, similar to binomial logistic with glm()

logLik(vglm.fit) # Log-likelihood
deviance(vglm.fit) # Should yield the same value

# This formula yields the same residual deviance

-2 * logLik(vglm.fit) 

AIC(vglm.fit) # Akaike Information Criterion = deviance+2*NumVariables

# All together

c("Log-Likelihood" = logLik(vglm.fit),
   "2LL Deviance" = deviance(vglm.fit), 
   "AIC" = AIC(vglm.fit))

# Let's manipulate the coefficients a bit

coef.log.odds <- coef(vglm.fit) # Log Odds
coef.log.odds # Take a look

# Notice that each variable has 2 values. Since we have 3 categories in the response variable and the first one was left out as the reference variable, the first coefficient is for the log odds of the second response category; and the second one is for the third response category.

# Now let's get the odds and probabilities

coef.odds <- exp(coef.log.odds) # Odds
coef.all <- cbind(coef.log.odds, coef.odds) # Both
coef.all # Tale a look

# Or simply:

cbind("Log-Odds" = coef(vglm.fit), "Odds" = exp(coef.log.odds))

# Same thing -- 2 coefficients for each variable; we would get 3 coefficients if we had 4 response categories, and so on.

# Take hincome:1, for example: on average and holding everything else constant, when the husband’s income goes up by $1K, the log-odds of not working, go up by 0.097, compared to working full time, The odds go up 1.102 times.

# Take hincome:2, for example: On average and holding everything else constant, when the husband’s income goes up by $1K, the log-odds of working part time, go up by 0.104. The odds go up 1.1097 times, compared to working full time.

# Let's find predicted values with the predictvglm() function

pred.log.odds <- predictvglm(vglm.fit, newdata = NULL, se.fit = T)

# Notes: newdata can be used to specify a test or new data set to predict if omitted or NULL the predictions are done on the training set

pred.log.odds # Take a look

# Just like with the coefficients, we get 2 predictions for each observation -- one for the log-odds of being in the second category (compared to the first) and the other for the third category.

# Let's manipulate the predicted values

pred.log.odds$se.fit # Standard errors
pred.log.odds$fitted.values # Predicted values, log odds
pred.odds <- exp(pred.log.odds$fitted.values) # Predicted odds
pred.prob <- pred.odds / (1 + pred.odds) # Predicted probabilities
pred.all <- cbind(pred.log.odds$fitted.values, pred.odds, pred.prob)

# vglm() reports the above data without column names, let's create them

colnames(pred.all) <- c("LogOdds2","LogOdds3", 
                        "Odds2", "Odds3", 
                        "Prob2", "Prob3")

pred.all # Take a look


## Linear Discriminant Analysis (LDA)

require(MASS) # Contains the lda(){MASS} function

heart <- read.table("Heart.csv", sep = ",", head = T) 

# This was the older command, which no longer works: heart <- read.table("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data", sep=",", head=T, row.names=1)

attach(heart)
head(heart)

# Compute index vectors for train and test subsamples

set.seed(1)
train <- sample(1:nrow(heart), 0.7 * nrow(heart))
test <- seq(1:nrow(heart))[-train]

# Fit the model on the train data

heart.fit.lda <- lda(chd ~ ., data = heart[train,])

# Inspect the results

heart.fit.lda 
heart.fit.lda$prior # Check out the prior probabilities

# See the linear discriminant function. Note: heart.fit.lda$scaling contains the LDA coefficients displayed here. Another note, use heart.fit.lda$prior to get the prior distribution of the response variable.

summary(heart.fit.lda) # Not very useful
plot(heart.fit.lda) # Distribution of response classifications

# Let's extract the test subsample

heart.test <- heart[test, ]

# Now let's make predictions with the train model and the test subsample

heart.lda.pred <- predict(heart.fit.lda, heart.test)

heart.lda.pred$posterior # Inspect the respective probabilities
heart.lda.pred$class # Inspect the classifications

# Note: the display above shows the prob of chd=0 and of chd=1. That's why we see 2 columns. The default classification threshold is prob>0.5. That is, any observation with a prob(chd=1>0.5) will be predicted as 1, 0 otherwise

# For most of what we do below, we only need the second column for prob(chd=1):

heart.lda.pred$posterior[,2] # Take a look

# And display the resulting confusion matrices

# Lambda = 0.50

heart.lda.confmat <- table("Predicted" = heart.lda.pred$class, 
                           "Actual"=heart.test$chd) 

# Note: you can use rownames(heart.lda.confmat) <- c("No","Yes") and colnames(heart.lda.confmat) <- c("No","Yes") to display No and Yes instead of 0 and 1.

heart.lda.confmat

TruN <- heart.lda.confmat[1, 1] # True negatives
TruP <- heart.lda.confmat[2, 2] # True positives
FalN <- heart.lda.confmat[1, 2] # False negatives
FalP <- heart.lda.confmat[2, 1] # False positives
TotN <- TruN + FalP # Total negatives
TotP <- TruP + FalN # Total positives
Tot <- TotN + TotP # Total

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

lda.rates.50 <- c(Accuracy.Rate, Error.Rate, 
                  Sensitivity, Specificity, 
                  FalP.Rate)

names(lda.rates.50) <- c("Accuracy Rate", "Error Rate", 
                         "Sensitivity", "Specificity", 
                         "False Positives")

lda.rates.50

# Let's do the same for Lambda=0.60

# Now we can use the data in this probability vector to change the classification threshhold to, say prob(chd=1>0.60)

heart.lda.class.60 <- ifelse(heart.lda.pred$posterior[,2] > 0.6, 1, 0)
heart.lda.class.60 # Take a look how the classifications changed

heart.lda.confmat.60 <- table(heart.lda.class.60, heart.test$chd) 
heart.lda.confmat.60

TruN <- heart.lda.confmat.60[1, 1] # True negatives
TruP <- heart.lda.confmat.60[2, 2] # True positives
FalN <- heart.lda.confmat.60[1, 2] # False negatives
FalP <- heart.lda.confmat.60[2, 1] # False positives
TotN <- heart.lda.confmat.60[1, 1] + heart.lda.confmat.60[2,1] # Total negatives
TotP <- heart.lda.confmat.60[1, 2] + heart.lda.confmat.60[2,2] # Total positives
Tot <- TotN + TotP # Total

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

lda.rates.60 <- c(Accuracy.Rate, Error.Rate, 
                  Sensitivity, Specificity, 
                  FalP.Rate)

names(lda.rates.60) <- c("Accuracy Rate", "Error Rate", 
                         "Sensitivity", "Specificity", 
                         "False Positives")

lda.rates.60

# Both threshhold together

rbind(lda.rates.50, lda.rates.60)

# With Binary Logistic

rbind(logit.rates.50, logit.rates.60, lda.rates.50, lda.rates.60)


## ROC Curves and LDA 

library(ROCR)

# For the prediction() function, we use the predicted values from lda for the prob(chd>1), which is store in the second column of the $posterior variable of the lda predict() object

pred <- prediction(heart.lda.pred$posterior[ , 2], chd[test]) 
perf <- performance(pred,"tpr","fpr")

plot(perf, colorize = T)

auc <- performance(pred,"auc") # Compute the AUC
c(auc@y.name[[1]], auc@y.values[[1]]) # Display the AUC


#### Aside ####

# Smarket Example: this is a less interesting example provided by the textbook:

require(ISLR) # Needed for the Smarket data set
require(MASS) # Contains the lda(){MASS} function
attach(Smarket)

train=(Year<2005) # Creates a vector mamed "train" as we did before
Smarket.2005=Smarket[!train,] # As we did before
Direction.2005=Direction[!train] # As we did before

# Let's fit an LDA model with training data from the Smarket data set using the lda() function

lda.fit=lda(Direction~Lag1+Lag2,data=Smarket,subset=train) 

# An alternative way to specify the training subset

lda.fit=lda(Direction~Lag1+Lag2,data=Smarket,subset=Year<2005) 

lda.fit # Show the output (including the coefficients)

# Let's inspect the plot, showing the LDA combinations that yield Up and Down

plot(lda.fit) 

# Let's now predict with the test data
lda.pred=predict(lda.fit, Smarket.2005)
names(lda.pred) # Print lda.pred attributes for reference
# Note: predict() returns 3 elements:

# (1) $class -- the outcome variable classification

lda.pred$class # Check out the predicted values

# (2) $posterior -- which is actually a matrix, with each column showing the posterior probability that the observation belongs to the first, second, etc. class; and 

head(lda.pred$posterior) # 2 columns, for "Down" and "Up"

# (3) $x which contains the linear discriminants (i.e., linear combination of the coefficients (beta1*lag1+beta2*lag2) reported in lda.fit above. Check the posterior probabilities, in this case it has 2 columns for Down and Up.

head(lda.pred$x) 

# A large $x predicts "Up" and a small x predicts "Down". The cutoff for classification of "Up" vs. "Down" is the mean of the $x values, which should hover around (but not exactly at) 0.

# Let's build the confusion matrix

# Create a vector with just the predicted values of "class"

lda.class=lda.pred$class 

# Confusion matrix with predicted vs. actual (i.e., Direction)
lda.conf.mat <- table(lda.class, Direction.2005) 
lda.conf.mat

TruN=lda.conf.mat[1,1] # True negatives
TruP=lda.conf.mat[2,2] # True positives
FalN=lda.conf.mat[1,2] # False negatives
FalP=lda.conf.mat[2,1] # False positives
TotN=lda.conf.mat[1,1] + lda.conf.mat[2,1] # Total negatives
TotP=lda.conf.mat[1,2] + lda.conf.mat[2,2] # Total positives
Tot=TotN+TotP # Total

# Now let's use these to compute accuracy and error rates

Accuracy.Rate=(TruN+TruP)/Tot
Accuracy.Rate # Check it out

Error.Rate=(FalN+FalP)/Tot
Error.Rate # Check it out

# Sensitivity -- rate of correct positives

Sensitivity=TruP/TotP # Proportion of correct positives
Sensitivity # Check it out

# Specificity -- rate of correct negatives

Specificity=TruN/TotN # Proportion of correct negatives
Specificity

# False Positive Rate = 1 - specificity (useful for ROC curves)

FalP.Rate = 1 - Specificity
FalP.Rate

lda.rates.50=c(Accuracy.Rate, Error.Rate, Sensitivity, Specificity, FalP.Rate)

names(lda.rates.50)=c("Accuracy Rate", "Error Rate", "Sensitivity", "Specificity", "False Positives")

lda.rates.50

# What if we want to change the classification threshold for Up to prob>0.4?

# The posterior probability of being classified as up is in the second column of $posterior:

head(lda.pred$posterior[,2])
lda.pred.40 = ifelse(lda.pred$posterior[,2]>0.4, "Up", "Down")
lda.pred.40

lda.conf.mat.40 <- table(lda.pred.40, Direction.2005) 
lda.conf.mat.40

#### /Aside ####


## Quadratic Discriminant Analysis (QDA)

# The syntax for the qda() function is identical to lda(). Essentially, most of the code below is identical to the LDA code above, except that we use the qda(){MASS} function instead of lda(){MASS}

require(MASS) # Contains the lda(){MASS} function

heart <- read.table("Heart.csv", sep=",", head=T) # This was the older command, which no longer works: heart <- read.table("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data", sep=",", head=T, row.names=1)

attach(heart)
head(heart)

# Compute index vectors for train and test subsamples

set.seed(1)
train <- sample(1:nrow(heart), 0.7 * nrow(heart))
test <- seq(1:nrow(heart))[-train]

# Fit the model on the train data

heart.fit.qda <- qda(chd ~ ., data = heart[train,])

# Inspect the results

heart.fit.qda # See the linear discriminant function
summary(heart.fit.qda) # Not very useful

# Let's extract the test subsample

heart.test = heart[test, ]

# Now let's make predictions with the train model and the test subsample

heart.qda.pred <- predict(heart.fit.qda, heart.test)
heart.qda.pred$class # Inspect the classifications
heart.qda.pred$posterior # Inspect the respective probabilities
heart.qda.pred$posterior[ , 2] # Just the second column for prob(chd=1)

# And display the resulting confusion matrices

# Lambda=0.50

heart.qda.confmat <- table("Predicted" = heart.qda.pred$class, 
                           "Actual"=chd[test]) 

# Note: you can use rownames(heart.qda.confmat) <- c("No","Yes") and colnames(heart.qda.confmat) <- c("No","Yes") to display No and Yes instead of 0 and 1.

heart.qda.confmat

TruN <- heart.qda.confmat[1, 1] # True negatives
TruP <- heart.qda.confmat[2, 2] # True positives
FalN <- heart.qda.confmat[1, 2] # False negatives
FalP <- heart.qda.confmat[2, 1] # False positives
TotN <- heart.qda.confmat[1, 1] + heart.qda.confmat[2,1] # Total negatives
TotP <- heart.qda.confmat[1, 2] + heart.qda.confmat[2,2] # Total positives
Tot <- TotN + TotP # Total

# Now let's use these to compute accuracy and error rates

Accuracy.Rate <- (TruN + TruP) / Tot
Accuracy.Rate # Check it out

Error.Rate <- (FalN + FalP) / Tot
Error.Rate # Check it out

# Sensitivity -- rate of correct positives

Sensitivity <- TruP / TotP # Proportion of correct positives
Sensitivity # Check it out

# Specificity -- rate of correct negatives

Specificity < -TruN / TotN # Proportion of correct negatives
Specificity

# False Positive Rate = 1 - specificity (useful for ROC curves)

FalP.Rate <- 1 - Specificity
FalP.Rate

qda.rates.50 <- c(Accuracy.Rate, Error.Rate, 
                  Sensitivity, Specificity, 
                  FalP.Rate)

names(qda.rates.50) <- c("Accuracy Rate", "Error Rate", 
                         "Sensitivity", "Specificity", 
                         "False Positives")

qda.rates.50

# Let's do the same for Lambda = 0.60

# Now for a classification threshhold of prob(chd = 1 > 0.60)

heart.qda.class.60 <- ifelse(heart.qda.pred$posterior[,2] > 0.6, 1, 0)
heart.qda.class.60 # We will use this later

heart.qda.confmat.60 <- table(heart.qda.class.60, chd[test]) 
heart.qda.confmat.60

TruN <- heart.qda.confmat.60[1, 1] # True negatives
TruP <- heart.qda.confmat.60[2, 2] # True positives
FalN <- heart.qda.confmat.60[1, 2] # False negatives
FalP <- heart.qda.confmat.60[2, 1] # False positives
TotN <- heart.qda.confmat.60[1, 1] + heart.qda.confmat.60[2,1] # Total negatives
TotP <- heart.qda.confmat.60[1, 2] + heart.qda.confmat.60[2,2] # Total positives
Tot <- TotN + TotP # Total

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

qda.rates.60 <- c(Accuracy.Rate, Error.Rate, 
                  Sensitivity, Specificity, 
                  FalP.Rate)

names(qda.rates.60) <- c("Accuracy Rate", "Error Rate", 
                         "Sensitivity", "Specificity", 
                         "False Positives")

qda.rates.60

# Both threshhold together

rbind(qda.rates.50, qda.rates.60)

# Binary Logistic, LDA and QDA Together

rbind(logit.rates.50, logit.rates.60, 
      lda.rates.50, lda.rates.60, 
      qda.rates.50, qda.rates.60)


## ROC Curves and QDA 

# The method and steps for ROC curves in QDA are identical to those of LDA, except that we use a qda() prediction object.

library(ROCR)

pred <- prediction(heart.qda.pred$posterior[ , 2], chd[test]) 
perf <- performance(pred,"tpr","fpr")
plot(perf, colorize = T)

auc <- performance(pred,"auc") # Compute the AUC
c(auc@y.name[[1]], auc@y.values[[1]]) # Display the AUC


#### Aside ####

attach(Smarket)
train=(Year<2005) # As we did before
Smarket.2005=Smarket[!train,] # As we did before
Direction.2005=Direction[!train] # As we did before

qda.fit=qda(Direction~Lag1+Lag2,data=Smarket,subset=train)
qda.fit # It does not provide linear discriminants like LDA
qda.class=predict(qda.fit,Smarket.2005)$class

# QDA confusion matrix

qda.conf <- table(qda.class,Direction.2005)
qda.conf

qda.correct <- mean(qda.class==Direction.2005) # Correct rate
qda.error <- mean(qda.class!=Direction.2005) # Error rate
qda.stats <- c(qda.correct, qda.error)
names(qda.stats) <- c("Correct", "error")
qda.stats

both.stats <- rbind(lda.stats, qda.stats)
both.stats
# QDA does a better job than LDA with this data

#### /Aside ####
