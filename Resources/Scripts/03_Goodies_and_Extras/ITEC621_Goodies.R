###########################
# ITEC 621 Script Goodies #
###########################

# Prepared by J. Alberto Espinosa
# Last updated on 3/28/2022

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: This material is a complement to the ITEC621 R scripts and include material referenced but not covered in the class. The code provided contains ample documentation and is complemented by the Goodies lecture slides.


#######################################################
#                       INDEX                         #
#######################################################

### Survey Analytic Models ####

## Factor Analysis
## Varimax Rotation
## Reliability Analysis and Cronbach's Alpha
## Grouping Variables Into Factors

### Other Transformations

## Box-Cox
## Rank Transformation 

### Other Topics ###

## Interaction Terms - Continuous x Continuous
## Smoothing Spline
## Local Regression
## Generalized Additive Models (GAMs)
## GAM and Logistic Regression
## The Bootstrap

## Support Vector Classifier
## Support Vector Machine
## Support Vector Machine with Multiple Classes

## K-Nearest Neighbors (KNN)
## ROC Curves

## Boosted Trees
## Principal Components for Descriptive Analytics
## K-Means Clustering
## Hierarchical Clustering

#######################################################
#                  END OF INDEX                       #
#######################################################


### Survey Analytic Models ####


## Factor Analysis

# Factor Analysis Using Principal Components (code not in the textbook)

# The goal in factor analysis is to group "observable" variables (e.g., GMAT scores, GPA, other test scores, etc.) into "unobservable" or "latent" factors (e.g., "intelligence"). Factor analysis is a popular method, not only for surveys, but for other applications in marketing, behavioral analytics, etc. For example, we can measure several food item attributes (e.g., fat, calories, carbohydrates, fiber content, popularity, price, etc.). 

# Factor analysis can help group these attributes into "latent" descriptive factors like  "Healthy" (fat, calories, carbohydrates, fiber content) and "Sale Appeal" (popularity, price). Factor analysis looks at the correlation structure of the observable (i.e., recorded) variables and groups them into "unobservable" factors that group together based on their correlation or similarity.

# Principal Components Analysis (PCA) and Common Factor Analysis (CFA) are two very different things, but they are often confused because they are often used together. PCA is about finding an axis in the data, along wich the variance in the data is maximized, then finding the next orthogonal axis with the second highest variance and so on. 

# The idea is the first few components may explain a high proportion of the variance, which can allow us to build a predictive model with k components, rather than p variables (where k<n). If the p variables are largely uncorrelated k will be very close or equal to p and PCA won't be very useful. 

# Conversely, if the p variables are largely correlated, then k will be much smaller than p and pCA will be desirable. CFA usually starts with CFA (thus the confusion) to identify the k orthogonal principal components (there are many methods to extract orthogonal components so PCA is only one, but the most popular method). 

# But in CFA the PCA axes are rotated (with the Varimax rotation methdod being the most popular), so that the axes are no longer orthogonal (i.e., perpendicular), but the rotation aims to find stronger commonality among variables, which can be grouped into factors. CFA is most popular in analytics methods for survey data (i.e., to group answers to the many questions asked into a few groups of correlated variables, or factors).

# The example below illustrates CFA with Varimax Rotated PCA, retaining k=5 components

library(psych) # Contains the principal() function for factor analysi
library(ISLR) # Contains the Hitters data set

attach(Hitters)
names(Hitters)
?Hitters # Explore the variables

# Let's get an example X data frame to illustrate factor analysis

x.Hitters <- data.frame(Hitters$AtBat, Hitters$Hits, Hitters$HmRun, 
                        Hitters$Runs, Hitters$RBI, Hitters$Walks,
                        Hitters$CAtBat,Hitters$CHits,Hitters$CHmRun, 
                        Hitters$CRuns, Hitters$CRBI, Hitters$CWalks)

cor.Hitters <- cor(x.Hitters) # Get a correlation matrix for X first
print(cor.Hitters, digits=3) # Take a look

# Again, the main idea is to reduce P predictors to M factor. But how many factors are good enough? One quick way to decide on the number of factors to extract is to display a scree plot of the correlation matrix, which graphs the eigenvalues for the factors or principal components. Because the average variance on standardized vairables is 1 and eigenvalues are standardized, the average eigenvalue is 1. So, eigenvalue = 1 is typically used as a rule of thumb cutoff point to decide on a factor solution. Take a look:

scree(cor.Hitters, factors = T) # Draw a scree plot for the factors

# Notice that the plot "elbows" around 3 factors. That is, there is a sharp decline in eigenvalues (i.e., variance explained by the respective factor, up to 3 factors, which is just when it crosses the eigenvalue = 1 threshhold). So the best factor solution is M = 3 factors.

# Again, a quick rule to figure out how many factors to extract is to eyeball the Scree Plot. The optimal number of factors is near the elbow and around Eigenvalue = 1

# You can compute the eigenvectors and eigenvalues for any correlation matrix, which will provide the correlation matrix of the data, when the axes are rotated to align with the principal components

eigen.vectors.Hitters <- eigen(cor.Hitters)
eigen.vectors.Hitters # Take a look

# Note: the sum of eigenvector scores squared for any row or column is always 1; i.e., by definition, eigenvectors have a lenght of 1

eigen.values.Hitters <- eigen(cor.Hitters, only.values = T)
eigen.values.Hitters # Take a look

# Now let's do factor analysis on the correlation matrix. First, let's first look at ALL 12 PCA's without rotation (explained shortly).

# Extract all PCA's

factor.Hitters.all <- principal(x.Hitters, nfactors = 12, scores = T) 
factor.Hitters.all # print results

# Notice the factor loadings for each variable. A high factor loading > 0.5 for a given variable and principal component indicates that the variable contributes substantially to that factor. Think of it as a weight. But also notice that the factor loadings are not sequentially ordered

## Varimax Rotation

# Again, the purpose of factor analysis is to group observable variables into unobservable "latent" factors. Principal components helps us do that by finding the various directions in which the data has the highest variance. Two thing generally happen with this method:

# 1. The first factor is based on the first principal components 
#    and, therefore, it is the direction in which the data has 
#    more variance.

# 2. The factor loadings for a given variable (i.e., proportion of 
#    variance of that variable explained by each factor) tend to be
#    a bit more spread out than desirable.

# This is OK when extracting factors, but once we decide that the optimal solution has 4 factors (for example), we can tweak the 4 factors a bit to:

# 1. Spread out the variance across all factors (rather than 
#    having 1 dominant factor); and

# 2. Vary the factor loadings for each variable so that each 
#    variable loads more heavily on a single factor, rather 
#    than evenly spread across factors.

# "Orthogonal" rotations do precisely that. Once a 4-factor solution is selected (for example), those 4 factors are rotated orthogonally (i.e., all factor axes are rotated together) to optimize 1 and 2 above. Intuitively but not exactly, the "Varimax" rotation method is like doing further factor analysis on the extracted factors (only).

# Let's now do a Varimax rotation and force the solution to 5 factors, using rotate="varimax":

factor.Hitters.5 <- principal(x.Hitters, nfactors=5, 
                              rotate = "varimax", scores = T)

factor.Hitters.5 # print results 

# - RC1,2,etc. are the "Varimax-rotated" Principal Component loadings

# - h2 shows the "communalities" for each variable. Communality is
#       the sum of squared factor loadings for that variable and
#       represents the proportion of variance of that variable that
#       is accounted for by the factors extracted. Communalities 
#       close to 1 are desirable.

# - u2 shows the "uniqueness" of that variable = 1 - Communality

# - com shows the "component complexity" for that variable -- some
#      complexity index (Hoffman's) -- higher when loadings are more
#      evenly spread; lower when loadings are more "simply" weighted
#      on one or few variables (more desirable)

factor.Hitters.5$values 
# Best solutions when eigen values become <1, 3 in this case

# Display factor scores for each observation
factor.Hitters.5$scores
# Shows the factor value for each observation

# Display factor weights to construct the factor scores
factor.Hitters.5$weights 

# These are like standardized regression coefficients that show the effect of each variable on each factor.

# Draw a scree plot for the factors

scree(cor.Hitters, factors = T) 

# Now Let's try a 3-factor solution, which the Scree plot and eigen value<1 suggested as the best solution.

# First without Varimax rotation

principal(x.Hitters, nfactors = 3, scores = T)

# Now let's rotate and save the results

factor.Hitters.3 <- principal(x.Hitters, nfactors = 3, 
                              rotate = "varimax", scores = T)
factor.Hitters.3 # print results 

# Now let's sort the variables from highest to lowest factor
# loading to make it easier to decide which variables to group

factor.Hitters.3.sorted <- fa.sort(factor.Hitters.3) # print results 
factor.Hitters.3.sorted


## Reliability Analysis and Cronbach's Alpha

# This statistic is essentially a group correlation, but it works with multiple variables and it make an adjustmet to the statistic based on how many variables are included, because more variables increase the reliability.

# To compupte the Cronbach-Alpha statistic, we first need to group the variables that we want to analyze. Based on our rotated factor loadings, we found that CAtBat, CRuns, CHits, CRBI, CWalks and CHRun loaded together. Let's call this group "HitEffectiveness":

library(dplyr) # Contains the select() function below
library(psych) # Contains the alpha() function below

HitEffectiveness <- select(Hitters, CAtBat, CRuns, CHits, CRBI, CWalks, CHmRun)

names(HitEffectiveness)
head(HitEffectiveness)

alpha(HitEffectiveness)

# Key Output Items:

# - raw_alpha -- Alpha based on covariances (only useful if all 
#   items in the group are measured in the exact same scale)

# - std.alpha -- Alpha based on correlations -- this is the Cronbach
#   Alpha statistic you need to pay attention to; it is like a group
#   correlation -- the closer to 1 the better; anything above 0.7 is
#   considered highly reliable

# - Other various reliability stats -- less popular

# - "Reliability if an item is dropped" shows whether the Alpha would
#   improve or worsen if you were to remove that item from the group.
#   If Alpha is higher when a variable is dropped, you should drop it.

# Note: sometimes some variables have a reversed scale so that they have a strong but negative correlation (e.g., how much you like this class; how much you dislike the material). In that case, you need to use the "keys" attribute to reverse the necessary scale, otherwise the average Alpha correlation statistic will be incorrect. In the example above, there are 6 variables and they all correlate positively, so there is no need for the "keys" attribute. But if you wanted to reverse, ssay the second variable the alpha command would look like this:

# alpha(HitEffectiveness, keys = c(1,-1,1,1,1,1))


## Grouping Variables Into Factors

# You can group variables using Principal Components Regression, in which the PC's are orthogonal (i.e., un-correlated) with each other and thus ideal as truly "independent variables". But the idea behind Factor Analysis with survey data is to find groupings of highly correlated survey items that can help interpret survey results. 

# Thus, the standard parctice with survey data is to use Factor Analysis with Varimax rotation to find groups of variables with high factor loadings and then aggregate (e.g., average) them for analysis. Once aggregated you then use the aggregated factors rather than the original survey items. But these aggregated factors must have some business meaning

# In the example above you could compute an average variable:

Hitters$HitEffectiveness <- (CAtBat + CRuns + CHits + CRBI + CWalks +CHmRun) / 6
Hitters$HitEffectiveness


### Other Transformations


## Box-Cox

# The Box-Cox transformation is actually a family of various transformations (see lecture slides). It tries several values of a parameter called "lambda" and selects the one that best approximates a normal distribution for the outcome variable. The coefficients in ox-Cox transformed models are difficult to interpret, but a Box-Cox transformation my improve predictive accuracy. The "boxcox(){MASS}" function tests Box-Cox transformations

library(MASS) # Contains the boxcox() function and Boston data set
attach(Boston) # We will work with this data

lm.fit <- lm(medv ~ lstat, data = Boston) # Try a linear model first
summary(lm.fit) # Take note of the R squared

boxcox(lm.fit) # Now let's try Box-Cox transformations

# Notice that the best lambda yielding the highest log-likelihood is near 0.

# Let's amplify the graph from lambda = -0.05 to 0.10 in 0.01 increments:

boxcox(lm.fit, lambda = seq(-0.05, 0.10, 0.01))

# Best lambda is around 0.04, let's model it. Let's transform the outcome variable accordingly:

medv.box=medv^0.04 # Let's create the Box-Cox transformed variable

lm.fit.box.cox <- lm(medv.box ~ lstat, data = Boston)
summary(lm.fit.box.cox) # Notice the improvement in the R-Squared

# Let's split the plot window to see both regression lines side by side

par(mfrow=c(1,2)) # Divide the output into a 2x2 frame

plot(lstat, medv) # Plot the linear model first
abline(lm.fit,lwd = 3, col = "red") # And its regression line

plot(lstat, medv.box) # Now plot the Box-Cox transformed model
abline(lm.fit.box.cox,lwd = 3,col = "red") # Notice the improvement in fit


## Rank Transformation 

attach(Boston)
head(Boston)

# To add a new column with ranks in the Boston data frame:

Boston$CrimeRankLoHi <- rank(Boston$crim) # Lowest to highest
Boston$AgeRankLoHi <- rank(Boston$age) # Lowest to highest
Boston$LStatRankLoHi <- rank(Boston$lstat) # Lowest to highest
Boston$medvLoHi <- rank(Boston$medv) # Lowest to highest

# The highest median house value can be thought of as being ranked #1 in value. That is, values and their respective ranks often move in opposite directions. Let's create a rank variable that assigns a rank of 1 to the highest median value, 2 to the second highest, and so on:

Boston$medvHiLo <- rank(-Boston$medv) # Highest to Lowest
head(Boston)

lm.fit <- lm(medv ~ crim + age + lstat, data = Boston)
lm.fit.rank <- lm(medvHiLo ~ crim + age + lstat, data = Boston)

lm.fit.rank.all <- lm(medvHiLo ~ CrimeRankLoHi + AgeRankLoHi + LStatRankLoHi, 
                      data=Boston)

summary(lm.fit)
summary(lm.fit.rank)
summary(lm.fit.rank.all)


### Other Topics ###


# We will not cover these topics in this class, although I may give you a quick overview if time permits. But I provide the R code for these methods for you to explore if you wish. This code is mostly from the ISLR book and the topics are well explained in the book.


## Interaction Terms - Continuous x Continuous

# These interaction effects are tricky to interpret. Proceed with caution.

# When multiplying two continuous variables to create an interaction term the resulting model will suffer from a number of issues:

# 1. High multicollinearity -- if the two main effects are 
#    somewhat correlated, their product will be highly 
#    correlated with both

# 2. Scale invariance -- when you change the scale of a 
#    variable in OLS (e.g., inches to feet) the p-value 
#    of its coefficient does not change and the value of 
#    the coefficient will change proportionally (e.g., 
#    the effect of increasing 1 foot is 12 times the 
#    effect of increasing 1 inch)

# 3. Effect interpretation -- the interaction effect 
#    represents the effect change of one variable when 
#    the other one is 0, and vice versa 
#    (y = b1x1 + b2x2 + bintx1x2). But this is not useful if 
#    0 is not a possible value for one of the variables 
#    (e.g., number of rooms in a house; weight of a car).

# Centering the dependent variable and both variables involved in the interaction term solves the problems above:

# 1. With uncentered variables, if both variables have 
#    a positive value then their product will be positive, 
#    thus generating collinearity. But if the variables 
#    are centered, then their values may be positive or 
#    negative, so their product will be less correlated

# 2. With centered variables, it can be shown that the 
#    main effects will remain scale invariant.

# 3. Because centered variables take the value of 0 at 
#    their respective means, the resulting interaction 
#    term represents the effect of one variable when the 
#    other one is at its mean, which is more meaningful 
#    for interpretation. 

# Once the variables are centered, they are modeled in a similar manner to binary variables. Let's illustrate this:

library(MASS) # To access the Boston data set
attach(Boston)

# This is the incorrect way and may lead to wrong conclusions when we don't center variables

inter.fit.uncenter <- lm(medv ~ lstat * age, data = Boston)
summary(inter.fit.uncenter)

# Now let's center the response variable and the two variables in the continuous x continuous interaction term:

Boston$medv.c <- scale(Boston$medv, center = T, scale = F)
Boston$lstat.c <- scale(Boston$lstat, center = T, scale = F)
Boston$age.c <- scale(Boston$age, center = T, scale = F)

inter.fit.center <- lm(medv.c ~ lstat.c * age.c -1, data = Boston)

# Note: The -1 fits a regression without an intercept. Since we have centered all variables, the regression line should go through the origin. Forcing the regression through the origin is not necessary if there are other uncentered variables in the model

summary(inter.fit.center)


## Smoothing Spline

# A quick spline Summary:

# A piecewise spline fits a different model in each segment delimited by knots, but the lines of two contiguous segments don't connect at the knot

# A spline forces these segment models to connect at the knots using basis functions or other means. But the model may have high variance at the head and tail ends of the curve. Also, the slope of the model at the right and left side of the knots will be different.

# A natural spline minimizes the first problem by forcing the first and last segments to a straight line

# A smoothing spline addresses the second problem by forcing the curve to have the same slope at the left and right and all around a each knot

# First, let's get the data to predict wage based on age

library(ISLR) # Contains the Wage data
attach(Wage)

agelimits <- range(age)
age.seq <- seq(from = agelimits[1], to = agelimits[2])

# This model specifies an arbitrary df = 16

fit.smooth1 <- smooth.spline(age, wage, df = 16) 
fit.smooth1

 # See how the model find the lambda and CV corresponding to df = 16

# This model lets cross-validation find the best lambda

fit.smooth2 <- smooth.spline(age, wage, cv=T) 

fit.smooth2$lambda # The best lambda
fit.smooth2$df # And the corresponding degrees of freedom

# Let's plot the results

plot(age, wage, xlim = agelimits, cex = 0.5, col = "darkgrey")

lines(fit.smooth1,col="red",lwd=2)
lines(fit.smooth2,col="blue",lwd=2)
title("Smoothing Spline")

legend("topright",legend=c("16 DF","6.8 DF"),
       col=c("red","blue"),lty=1,lwd=2,cex=.8)


## Local Regression

library(ISLR) # Contains the Wage data
attach(Wage)

agelimits <- range(age)
age.seq <- seq(from = agelimits[1], to = agelimits[2])

# Try on your own -- We are not focusing on local regression in this class, but here are some R commands if you wish to model one

# loess(){stats} fits local regressions

fit.local <- loess(wage ~ age, span = 0.2, data = Wage) 
fit.local.2 <- loess(wage ~ age, span = 0.5, data = Wage)

plot(age, wage, xlim = agelimits, cex = 0.5, col="darkgrey")
title("Local Regression")

lines(age.seq, predict(fit.local, data.frame(age = age.seq)),
      col="red",lwd=2)

lines(age.seq, predict(fit.local.2, data.frame(age = age.seq)),
      col = "blue", lwd = 2)

legend("topright", legend = c("Span = 0.2","Span = 0.5"),
       col = c("red", "blue"), lty = 1,lwd = 2, cex = 0.8)


## Generalized Additive Models (GAMs)

library(ISLR) # Contains the Wage data
attach(Wage)

agelimits <- range(age)
age.seq <- seq(from = agelimits[1], to = agelimits[2])

# In OLS regression, if the predictors are truly independent with 0 correlation, then it makes no difference to fit several individual simple linear regression models, one for each variable, than fitting a multivariate model with all the variables. The coefficients in both modeling options should be identical. 

# Also, the sum of R squares of the individual single models will equal the R square of the multivariate model. However, when there is some correlation among the predictors, this does not hold true any more. Reduced models will be more biased due to corrlation, and larger models will begin to experience multicollinearity, high variance and other dimensionality problems.

# The idea behind GAMs is that, since we can add individual single models into a multivariate full model, you could add instead different transformations (i.e., basis functions) for each variable. Some variables can be modeled as polynomials, some as splines, others as natural cubic splines, and so on.

# Again, GAM models are almost impossible to interpret, but they can help improve predictive accuracy

# One way to do this is to simply use the lm() function with a different basis function for each variable or not, as needed

gam1 <- lm(wage ~ ns(year,4) + ns(age,5) + education, data = Wage)

# But the {gam} library provides more options

library(gam)

# We use the s() function for smoothing splines: Year with 4 df's and age with 5 df's We leave education as is (it is a qualitative variable)

gam.m3 <- gam(wage ~ s(year, 4) + s(age, 5) + education, data = Wage) 

par(mfrow=c(1,3))
plot(gam.m3, se = T,col = "blue") # Plot each variable in gam.m3

# We can also plot non-gam objects (such as gam1 which is an lm object) with plot.gam

plot.gam(gam1, se = T, col = "red")

# We can fit a number of models gam.m1=gam(wage~s(age,5)+education,data=Wage)

gam.m2 <- gam(wage ~ year + s(age,5) + education, data = Wage)

summary(gam.m2) # Take a look at the summary
summary(gam.m3) # Take a look at the summary

# Now let's make some predictions

preds <- predict(gam.m2,newdata=Wage) # Predicting on the full training set

# To include local regression in GAM use the lo() function Notice that we use both, the s() function on year and lo() on age

gam.lo <- gam(wage ~ s(year, df = 4) + lo(age, span = 0.7) + education, data = Wage)

plot.gam(gam.lo, se = T, col = "green")

gam.lo.i <- gam(wage ~ lo(year, age, span = 0.5) + education, data = Wage)

library(akima)
plot(gam.lo.i)


## GAM and Logistic Regression

# Try on your own

# We can easily fit a logistic model using the I() function in the dependent variable

gam.lr <- gam(I(wage > 250) ~ year + s(age, df = 5) + education,
           family = binomial, data = Wage)

summary(gam.lr)

par(mfrow = c(1,3))
plot(gam.lr, se = T, col = "green")
table(education, I(wage > 250))

gam.lr.s <- gam(I(wage > 250) ~ year + s(age, df = 5) + education,
                family = binomial, data = Wage,
                subset = (education != "1. < HS Grad"))

plot(gam.lr.s, se = T, col = "blue")


## The Bootstrap

# Try on your own -- not covered in class, but I provide the code for you to explore if you wish

require(boot) # Contains the boot() function below

# This is a function to compute alpha, per 5.7

alpha.fn <- function(data,index){
  X = data$X[index]
  Y = data$Y[index]
  return( (var(Y) - cov(X,Y)) / (var(X) + var(Y) - 2 * cov(X,Y)) )
}

# Computes the alpha for all 100 observations in Portfolio

alpha.fn(Portfolio, 1:100) 

set.seed(1)

# This creates 100 observations of size 100 with replacement

alpha.fn(Portfolio, sample(100, 100, replace = T)) 

# The procedure above is the long way. Below is the fast way using the boot() function

# This produces 1000 bootstrap estimates for alpha

boot(Portfolio, alpha.fn, R = 1000) 

# Estimating the Accuracy of a Linear Regression Model with the bootstrap method

# Here we use the bootstrap approach to evaluate the variability of regression coefficients. First we create a function to run regressions on the Auto data set and return the coefficients

boot.fn = function(fndata, index)
  return(coef(lm(mpg ~ horsepower, data = fndata, subset = index)))

# We then apply this regression to all 1 to 392 observations (a single regression)

boot.fn(Auto, 1:392) 

# The steps below are the long way

set.seed(1)

boot.fn(Auto, sample(392, 392, replace=T))

# This is the fast way with bootstrapping to compute standard errors on 1000 bootstraps

boot(Auto, boot.fn, 1000) 

# This provides the coefficients for a single regression model

summary(lm(mpg ~ horsepower, data = Auto))$coef 

# Note both methods, plain regression and bootstrap give coefficients and standard errors, but the regression method is parametric and the boostrap is not. Now, let's run a bootstrap regression on the quadratic model 

boot.fn = function(data, index)
  coefficients(lm(mpg ~ horsepower + I(horsepower^2), data = data, subset=index))

set.seed(1)

boot(Auto, boot.fn, 1000)
summary(lm(mpg ~ horsepower + I(horsepower^2), data = Auto))$coef

# Note that because the quadratic model fits the data better, the bootstrap and the plain regression produce very similar results


## Support Vector Classifier

library(e1071)

# Let's generate some observations in two classes, X and Y

set.seed(1)

x <- matrix(rnorm(20 * 2), ncol = 2)
y <- c(rep(-1, 10), rep(1, 10))

x[y == 1,] = x[y == 1,] + 1
plot(x, col=(3 - y)) 

# Note that the randomly generated data is not separable by a straight line

# We need to encode the response as a factor variable

dat <- data.frame(x = x, y = as.factor(y)) 

# We now use the svm() function to fit a support vector classifier

svmfit <- svm(y ~ ., data = dat, kernel = "linear", cost = 10, scale=F)

# the kernel="linear" argument is used to fit a support vector classifier the scale=FALSE tells SVM NOT to stadardize the variables. In some cases we may want to standardize the data and use scale=TRUE

# Let's plot the SVC fit

plot(svmfit, dat)

# svmfit is the fitted model output and the input data

# Note: the jagged line is really a straight line

# Also, note that the support verctors as noted as crosses -- to find out which ones they are:

svmfit$index

# To get some basic information on the model

summary(svmfit) 

# e.g., 7 support vectors, cost=10, 2 classes, 4 support vectors in 1 and 3 in the other

# Let's try a different cost, e.g., 0.1

svmfit <- svm(y ~ ., data = dat, kernel = "linear", cost = 0.1, scale = F)

plot(svmfit, dat)
svmfit$index

summary(svmfit)

# Let's do cross validation with the tune() function available in the "e1071" library

set.seed(1)

# This is how we can test various cost values

tune.out <- tune(svm, y ~ ., data = dat, kernel = "linear",
              ranges = list(cost = c(0.001, 0.01, 0.1, 1,5,10,100)))

# This will display the cross validation errors for each model

summary(tune.out) 

# Note: the cross validation was done with 10-fold

# Best performance is with cost=0.1, but you can list the best model with these 2 commands:

bestmod <- tune.out$best.model
summary(bestmod)

# We can use the predict() function to predict the class label of test observations

xtest <- matrix(rnorm(20 * 2), ncol = 2)
ytest <- sample(c(-1, 1), 20, rep = T)

xtest[ytest == 1,] = xtest[ytest == 1,] + 1
testdat <- data.frame(x = xtest, y = as.factor(ytest))

# Using the best model to predict with the test data

ypred <- predict(bestmod, testdat) 
table(predict = ypred, truth = testdat$y)

# Let's try to predict with different cost values

svmfi <- svm(y ~ ., data = dat, kernel = "linear", cost = 0.01, scale = F)

ypred <- predict(svmfit, testdat)
table(predict = ypred, truth = testdat$y)

# Now let's try data that is separable by a straight line

x[y == 1,] = x[y == 1,] + 0.5
plot(x, col = (y + 5) / 2, pch = 19)

dat <- data.frame(x = x,y = as.factor(y))

svmfit <- svm(y ~ ., data = dat, kernel = "linear", cost = 1e5)

summary(svmfit)
plot(svmfit, dat)

svmfit <- svm(y ~ ., data = dat, kernel = "linear", cost = 1)

summary(svmfit)
plot(svmfit,dat)


## Support Vector Machine

# We proceed just like with support vector classifier, but using kernel="polynomial" or kernel="radial" depending on the desired fit method.

# Let's generate some random data

set.seed(1)

x <- matrix(rnorm(200 * 2), ncol = 2)
x[1:100,] = x[1:100,] + 2
x[101:150,] = x[101:150,] - 2

y <- c(rep(1, 150), rep(2, 50))

dat <- data.frame(x = x, y = as.factor(y))
plot(x, col = y)

# Obviously, the separating boundary is not linear

# Separate 1/2 of the data for the training set

train <- sample(200, 100)
svmfit <- svm(y ~ ., data = dat[train,], kernel = "radial", gamma = 1, cost = 1)

plot(svmfit, dat[train,])
summary(svmfit)

# Given the large number of training errors in the fitted model. So, let's try a larger cost (at the expense of a more irregular decision boundary).

svmfit <- svm(y ~ ., data = dat[train,], kernel = "radial", gamma = 1, cost = 1e5)

plot(svmfit, dat[train,])

# Let's inspect the cross validation errors for a few costs and gamma values

set.seed(1)

tune.out <- tune(svm, y ~ ., data = dat[train,], kernel = "radial", 
                             ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                             gamma=c(0.5,1,2,3,4)))

summary(tune.out)

# Best model is with cost=1 and gamma=2

# Now let's predict on the test set (i.e., -train)

table(true = dat[-train,"y"], 
      pred = predict(tune.out$best.model, newx = dat[-train,]))


## Support Vector Machine with Multiple Classes

# svm() will use one vs. one approach for multiple classes. Let's generate some data

set.seed(1)

x <- rbind(x, matrix(rnorm(50 * 2), ncol = 2))
y <- c(y, rep(0, 50))

x[y == 0,2] = x[y == 0,2] + 2
dat <- data.frame(x = x, y = as.factor(y))

par(mfrow = c(1, 1))
plot(x, col = (y + 1))

svmfit <- svm(y ~ ., data = dat, kernel = "radial", cost = 10, gamma = 1)
plot(svmfit, dat)

# Application to Gene Expression Data

library(ISLR)

# the Khan data set has xtrain, xtest, ytrain and ytest data sets already prepared

names(Khan) 
dim(Khan$xtrain)
dim(Khan$xtest)
length(Khan$ytrain)
length(Khan$ytest)
table(Khan$ytrain)
table(Khan$ytest)

dat <- data.frame(x = Khan$xtrain, y = as.factor(Khan$ytrain))

# Let's try a linear kernel

out <- svm(y ~ ., data = dat, kernel = "linear", cost = 10)

summary(out)
table(out$fitted, dat$y)

# Note that there are no training errors (e.g., data is separable by a straight line)

# NOw let's try it in the test set

dat.te <- data.frame(x = Khan$xtest, y = as.factor(Khan$ytest))

pred.te <- predict(out, newdata = dat.te)

table(pred.te, dat.te$y)

# Now there are 2 test set errors


## K-Nearest Neighbors (KNN)

# Complete all KNN code on your own

library(class) # needed for the knn() function

# KNN does estimation and prediction in one step, as opposed to Logistic, LDA and QDA. It requires 3 matrices and a scalar k: 

# 1. Training predictors;
# 2. Testing predictors; 
# 3. Training class vector (the y's)
# 4. Value of K (how many near neighbors to use)

# 1. Column bind the two training predictor variables

train.X <- cbind(Lag1, Lag2)[train,]

# 2. Column bind the two test predictor variables

test.X <- cbind(Lag1, Lag2)[!train,] 

# 3. Vector with response values ("Up" or "Down") in the training set

train.Direction <- Direction[train] 
train.Direction # Check it out

# 4. k -- a smaller K is more overfitting; a larger K 
#   provides a smoother classifier boundary

set.seed(1) # To get repeatable results

# knn() function with its 3 matrix parameters, plus k

knn.pred <- knn(train.X, test.X, train.Direction, k = 1) 

table(knn.pred, Direction.2005) # Confusion matrix
mean(knn.pred == Direction.2005) # Accuracy rate
(83 + 43) / 252 # Same result
mean(knn.pred != Direction.2005) # Error rate

# Let's try 3 nearest neighbors

knn.pred <- knn(train.X, test.X, train.Direction, k = 3)
table(knn.pred, Direction.2005)
mean(knn.pred == Direction.2005) # Better

# Application to Caravan Insurance Data

dim(Caravan)
attach(Caravan)
head(Caravan)
summary(Purchase)
348 / 5822

# Note on standardization -- when the scale of items may be an issue, it is customary to standardize variables, by subtracting the mean and dividing by the standard deviation, so that the mean of the stanardized variable is now 0 and the standard deviation is 1 -- scale is no longer an issue.

# scale() standardizes data (exclude col 86, which is qualitative)

standardized.X <- scale(Caravan[, -86]) 

var(Caravan[,1]) # Check the variance of the first column
var(Caravan[,2]) # Check the variance of the first column
var(standardized.X[,1]) # Variance of the standardized variable is 1
mean(standardized.X[,1]) # Mean is zero (or very, very small)
var(standardized.X[,2]) # Same thing

# Let's fit and predict in one pass. Also, let's use the first 1000 observations as the test set

test <- 1:1000 
train.X <- standardized.X[-test,] # Matrix (1)
test.X <- standardized.X[test,]  # Matrix (2)
train.Y <- Purchase[-test] # Matrix (3)
test.Y <- Purchase[test]
set.seed(1)

knn.pred <- knn(train.X, test.X, train.Y, k = 1)

mean(test.Y != knn.pred) # Error rate
mean(test.Y != "No") # Number of people who buy insurance

table(knn.pred, test.Y)
10 / (68 + 10) # Accuracy of Yes predictions only

knn.pred <- knn(train.X, test.X, train.Y, k = 3)
table(knn.pred, test.Y)
5 / 26

knn.pred <- knn(train.X, test.X, train.Y, k=5)
table(knn.pred, test.Y)
4 / (11 + 4) # Accuracy of Yes predictions only

# Let's use logistic regression with the same data to compare results

glm.fit <- glm(Purchase ~ ., data = Caravan, family = binomial, subset = -test)
glm.probs <- predict(glm.fit, Caravan[test,], type = "response")
glm.pred <- rep("No", 1000) # Set all responses to "No"

# Use the classifier probability>0.5 set to "Yes"

glm.pred[glm.probs > 0.5] = "Yes" 
# We are wrong about predicting "Yes" EVERY TIME

table(glm.pred, test.Y) 
glm.pred <- rep("No", 1000)

# Let's solve this problem by lowering the classifier threshold
glm.pred[glm.probs > 0.25] = "Yes" 
table(glm.pred, test.Y)
11 / (22 + 11) # Accuracy rate is 33.3% which is better


## ROC Curves

install.packages("ROCR") # Needed for ROC curves
library(ROCR)

# Quick function to plot and ROC curve for a given vector

rocplot <- function(pred, truth, ...){
  predob <- prediction(pred, truth)
  perf <- performance(predob, "tpr", "fpr")
  plot(perf,...)}

# Note: use decision.values=T to obtain fitted values. Positive fitted values then the observations are assigned to one class. 

# Negative fitted values are assigned to the other class

svmfit.opt <- svm(y ~ ., data = dat[train,], kernel = "radial",
                         gamma = 2, cost = 1, decision.values = T)

# Now, the predict() function will output the fitted values

fitted <- attributes(predict(svmfit.opt, dat[train,],
                             decision.values = T))$decision.values

par(mfrow=c(1,2))

rocplot(fitted, dat[train, "y"], main = "Training Data")

# Let's increase the gamma value to 50

svmfit.flex <- svm(y ~ ., data = dat[train,], kernel = "radial",
                          gamma = 50, cost = 1, decision.values = T)

fitted <- attributes(predict(svmfit.flex, dat[train,],
                             decision.values = T))$decision.values

rocplot(fitted,dat[train,"y"], add = T, col = "red")
# This gives a better model (hugging the corner more closely)

# Let's now try the ROC curves on the test data

fitted <- attributes(predict(svmfit.opt, dat[-train,],
                             decision.values = T))$decision.values

rocplot(fitted, dat[-train, "y"], main = "Test Data")

fitted <- attributes(predict(svmfit.flex, dat[-train,],
                             decision.values = T))$decision.values

rocplot(fitted, dat[-train, "y"], add = T, col = "red")

# Gamma=2 (svmfit.opt) does better with the test data


## Boosted Trees

library(gbm) # Generalized Boosted Regression Models 
library(MASS) # Contains the Boston data set

# Like Bagging and Random Forest, Boosting models several trees and aggregates the result. Unlike Bagging and Random Forest, Boosting does not fit several random trees, but it fits an initial tree and then fit another one to explain the residuals (errors), then again, etc. 

# Bagging and Random Forest are considered "fast" learning methods because the best model is generated in the first few samples and subsequent trees may or may not improve the MSE, whereas Boosting is considered to be a "slow" learning method because every new tree builds upon and improves the prior tree. 

# The tuning parameter "lambda" (works just like shrinkage in Ridge and LASSO) controls the speed of learning.

# Aside: to understand this concept, imagine that you run an OLS regression with certain variables and you get some fairly large residuals (i.e., errors). You can then build another OLS regression model to explain (i.e., predict) those residuals. This new regression will explain some of the error variance, but will also yield new errors (smaller than the first ones, becasue some of the variance in the errors is already explained with the second model). 

# Then you can fit a third regression model to explain the new residuals, and so on. You can then aggregate all the regression models, which on the aggregate, will have small residuals. Boosting applies this concept when generating trees.

set.seed(1) # To get replicable results

# Let's fit a Boosting model on the Boston data. Use:

# - distribution="gaussian" (i.e., normal distribution) 
#   for regression models.

# - distribution="bernoulli" for classification models

# Let's fit a model with 5000 trees, limiting the depth of each tree to 4 and using all available predictors

boost.boston <- gbm(medv ~ ., data = Boston, 
                              distribution = "gaussian", 
                              n.trees = 5000, interaction.depth = 4)

summary(boost.boston) # Provides relative influence stats and plot

# Note again, that lstat and rm are the most important variables

# Let's plot their partial dependencies to see how predicted price houses vary with lstat and rm

par(mfrow = c(1,2))
plot(boost.boston, i = "rm")
plot(boost.boston, i = "lstat")

# Let's now do Cross-Validation predictions with the test data

train <- sample(1:nrow(Boston.n), 0.7 * nrow(Boston.n)) # Train index

Boston.train <- Boston.n[train,]
Boston.test <- Boston.n[-train,] 

boost.boston <- gbm(medv ~ ., data = Boston.train, 
                              distribution = "gaussian", n.trees = 5000, 
                              interaction.depth = 4)

yhat.boost.pred <- predict(boost.boston, newdata = Boston.test, n.trees = 5000)

mean( (yhat.boost.pred - Boston.test$medv) ^2) # Computing the means squared error

# Shrinkage -- Boosting has a similar "shrinkage" effect on variables just like Ridge and LASSO regression. The shrinkage applies over each tree model, including the first one. A small lambda shrinks he prior tree model more, thus making it less important for the final aggregated model (i.e., slow learning). Large lambdas give more weight to the initial trees, thus learning fast.

# To vary the shrinkage factor (lambda) use the shrinkage attribute (the default is 0.01).

boost.boston <- gbm(medv ~ ., data = Boston.train, 
                              distribution = "gaussian", n.trees = 5000, 
                              interaction.depth = 4, shrinkage = 0.2, verbose = F)

# Now let's do predictions with the test data

yhat.boost.pred <- predict(boost.boston, newdata = Boston.test, n.trees = 5000)
mean( (yhat.boost.pred - Boston.test$medv) ^2) 
# Not much different


## Principal Components for Descriptive Analytics

# Note: we have covered Principal Components for two applications in this class -- Principal Components Regression and Factor Analysis. But Principal Components is a popular statistical method that simply rotates the covariance matrix of a data set and it is used in a number of statistical applications. This section discusses the use of Principal Components for Descriptive Analytics. This is not the focus of this course but the R code is provide below for your enjoyment.

# Also note that the libraries and functions described in this section are different than the ones covered earlier. There are many paths to Rome.

# Let's look at the USArrests data in the base R pacakge
?USArrests

# Rows in this data set are named by states in alphabetical order

states <- row.names(USArrests)

states
names(USArrests) # These are the variables collected for each state

# Get the mean for each column (using 1 instead of 2 to get means by row)

apply(USArrests, 2, mean) 
apply(USArrests, 2, var) # Get the variance for each column

# Note that means and variances differ widely. This means that we need to scale variables before doing PCA. Otherwise the Assault variable would dominate the analysis.

# The prcomp() function does principal components in the base R package

pr.out <- prcomp(USArrests, scale = T) 

# Notice that we are scaling the variables to have standard deviation of 1. Also note: by default, prcomp() centers variables to have mean of 0

names(pr.out) # Check out the output variables for prcomp()
class(pr.out) # It shows that pr.out is a prcomp() object
summary(pr.out) # Shows various statistics of the principal components

pr.out$sdev # Standard deviations of the principal components

# Squaring the standard deviations gives us the variance explained or "eigenvalues"

pr.out.eigen  <-  pr.out$sdev^2 
pr.out.eigen # Display the eigen values

# Note: one important reason to display eigen values is because it is used by the "Kaiser" criteria to decide how many principal componenents are good enoug. The Kaiser criteria is one of the most popular methods for this. Simply, the average eigen values equal 1 (because variables were scaled). The Kaiser criteria says that you should retain all the principal components with eigenvalues > 1 (i.e., above average variance). In this example, only PC1 meets this criteria.

pr.out$center # Variable means before centering

# Standard deviation (i.e., square root of variance) before scaling

pr.out$scale 
pr.out$rotation # Shows all the rotated principal components

# Display the scree plot

screeplot(pr.out, main = "Scree Plot", xlab = "Components") 

# Display the scree plot as a line diagram

screeplot(pr.out, type = "line", main = "Scree Plot") 

# Shows the data and the principal components in one diagram

biplot(pr.out, scale = 0)

# x is a matrix that contains the calculation of the principal component for each data point

pr.out$x 

# ASIDE -- Not in the textbook -- Varimax Rotation

# Note: principal components are often further rotated to "tweak" and cleanup the components a bit more. The rotation makes large factor loadings larger and small factor loadings smaller, such that the most correlated variables are easier to identify because they have larger factor loadings when rotated. The most popular rotation method is the one developed by Kaiser and is called "Varimax". Think of this as further rotating the already rotated principal components just a tad. This is how to rotate the principal components

pr.out.varimax <- varimax(pr.out$rotation) 

# Now compare the two sets of principal components and note the the rotated components does a better job at grouping variables. This is a most popular application in survey analysis methods

pr.out.varimax # Varimax rotated components
pr.out$rotation # Original principal components

# Interesting fact: you can rotate the principal components 180 degrees and get the same results

dim(pr.out$x)
biplot(pr.out, scale = 0)

pr.out$rotation <- -pr.out$rotation # Flip the signs
pr.out$x <- -pr.out$x

# The direction changes, but not the line orientation

biplot(pr.out, scale = 0) 

# To compute the proportion of variance explained by each principal component, enter:

pve <- pr.out.eigen / sum(pr.out.eigen) 
pve # Take a look

# Notice that the first principal component explains 62% of the variance in the data, not bad.

# To plot the proportion of variance explained:

plot(pve, xlab = "Principal Component", 
          ylab = "Proportion of Variance Explained", 
          ylim = c(0, 1), type = 'b')

# The "cumulative" proportion of variance is useful because it shows how much variance is explained by the first x components. This can be obtained with the cumsum() function
cumsum(pve)

# Notice that the first 2 principal components explain 86.7% of the variance

# Now let's plot this

plot(cumsum(pve), xlab = "Principal Component", 
                  ylab = "Cumulative Proportion of Variance Explained", 
                  ylim = c(0, 1), type = 'b')

# Notice that the first 2 principal components


## K-Means Clustering

set.seed(2)

x <- matrix(rnorm(50 * 2), ncol = 2)

x[1:25, 1] = x[1:25, 1]+3
x[1:25, 2] = x[1:25, 2]-4

km.out=kmeans(x, 2, nstart = 20)
km.out$cluster

plot(x, col = (km.out$cluster + 1), 
     main = "K-Means Clustering Results with K=2", 
     xlab = "", ylab = "", pch = 20, cex = 2)

set.seed(4)
km.out <- kmeans(x, 3, nstart = 20)
km.out

plot(x, col = (km.out$cluster + 1), 
     main = "K-Means Clustering Results with K=3", 
     xlab = "", ylab = "", pch = 20, cex = 2)

set.seed(3)

km.out <- kmeans(x, 3, nstart = 1)
km.out$tot.withinss

km.out <- kmeans(x, 3, nstart = 20)
km.out$tot.withinss


## Hierarchical Clustering

hc.complete <- hclust(dist(x), method = "complete")
hc.average <- hclust(dist(x), method = "average")
hc.single <- hclust(dist(x), method = "single")

par(mfrow=c(1,3))

plot(hc.complete, main = "Complete Linkag", xlab = "", sub = "", cex = 0.9)
plot(hc.average, main = "Average Linkage", xlab = "", sub = "", cex = 0.9)
plot(hc.single, main = "Single Linkage", xlab = "", sub = "", cex = 0.9)

cutree(hc.complete, 2)
cutree(hc.average, 2)
cutree(hc.single, 2)
cutree(hc.single, 4)

xsc <- scale(x)

plot(hclust(dist(xsc), method = "complete"), 
     main = "Hierarchical Clustering with Scaled Features")

x <- matrix(rnorm(30 * 3), ncol = 3)
dd <- as.dist(1 - cor(t(x)))

plot(hclust(dd, method = "complete"), 
     main = "Complete Linkage with Correlation-Based Distance",
     xlab = "", sub = "")

# NCI60 Data Example -- see textbook for explanation

# The NCI60 data

library(ISLR)
 
nci.labs <- NCI60$labs
nci.data <- NCI60$data

dim(nci.data)
nci.labs[1:4]
table(nci.labs)

# Principal components analysis on the NCI60 Data

pr.out <- prcomp(nci.data, scale = T)

Cols <- function(vec){
  cols = rainbow(length(unique(vec)))
  return(cols[as.numeric(as.factor(vec))])
}

par(mfrow=c(1,2))

plot(pr.out$x[, 1:2], col = Cols(nci.labs), pch = 19, xlab = "Z1", ylab = "Z2")
plot(pr.out$x[, c(1,3)], col = Cols(nci.labs), pch = 19, xlab = "Z1", ylab = "Z3")
summary(pr.out)

plot(pr.out)
pve <- 100 * pr.out$sdev^2 / sum(pr.out$sdev^2)

par(mfrow=c(1,2))

plot(pve,  type = "o", ylab = "PVE", 
     xlab = "Principal Component", col = "blue")

plot(cumsum(pve), type = "o", ylab = "Cumulative PVE", 
                  xlab = "Principal Component", col = "brown3")

# Clustering the Observations of the NCI60 Data -- see textbook for explanation

sd.data <- scale(nci.data)

par(mfrow=c(1,3))

data.dist <- dist(sd.data)

plot(hclust(data.dist), labels = nci.labs, main="Complete Linkage", 
                        xlab = "", sub = "", ylab = "")

plot(hclust(data.dist, method = "average"), labels = nci.labs, 
                       main = "Average Linkage", 
                       xlab = "", sub = "", ylab = "")

plot(hclust(data.dist, method = "single"), labels = nci.labs,  
                       main = "Single Linkage", 
                       xlab = "", sub = "", ylab = "")

hc.out <- hclust(dist(sd.data))
hc.cluster <- cutree(hc.out,4)

table(hc.clusters, nci.labs)

par(mfrow=c(1,1))

plot(hc.out, labels = nci.labs)
abline(h = 139, col = "red")
hc.out

set.seed(2)

km.out <- kmeans(sd.data, 4, nstart = 20)
km.clusters <- km.out$cluster

table(km.clusters, hc.clusters)

hc.out <- hclust(dist(pr.out$x[, 1:5]))

plot(hc.out, labels = nci.labs, 
             main = "Hier. Clust. on First Five Score Vectors")

table(cutree(hc.out, 4), nci.labs)