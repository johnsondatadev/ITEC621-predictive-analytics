################################
# ITEC 621 Data Pre-Processing #
################################

# Filename: ITEC621_DataPreProcessing.R
# Prepared by J. Alberto Espinosa
# Last updated on 1/30/2022

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: the material covered in this script ASSUMES that you have gone through the self-paced course KSB-999 R Overview for Business Analytics and are thoroughly familiar with the concepts illustrated in the RWorkshopScripts.R script. If you have not completed this workshop and script exercises, please stop here and do that first.


#######################################################
#                       INDEX                         #
#######################################################

## Transformation 1: Categorical To Dummy Variable Predictors
## Transformation 2: Polynomials

## Transformation 3: Log Models

# Inspecting for Normality
# Log-Transformed Models
# Logit-Transformed Models (Binary Y)
# Log-Transformed Models (Count Y) 

## Transformation 4: Centering Data 

## Transformation 5: Standardizing Data

# Standardized regression

## Transformation 6: Lagging data

# Testing for Serial Correlation
# Multivariate Time Series Models

#######################################################
#                  END OF INDEX                       #
#######################################################

# Experts state that about 80% of an analytics project goes into data work that needs to be done before analysis can be done with that data. This is generally referred to as data "pre-processing". There are many reasons to do pre-processing and various types of pre-processing that can be done. But these can be generally classified into two broad categories: (1) data quality; and (2) data transformations. 

# Data quality pre-processing involves things like dealing with things like: missing data; incorrect or inconsistent data; inappropriate data formats or spelling/punctuation issues, etc. A lot of the data quality work in data pre-processing is often done with the analytical tools, as they all have data evaluations and manipulation tools. But the most effective ways of working with data is the realm of "database" and "big data" fields, and it is a key skill in the "data science" profession.

# We will cover "data transformations" in this section. These have to do with data manipulations that are done to: 

# (1) comply with a particular model assumption (e.g., lagged models in forecasting; weighted regression models to correct for heteroscedasticity, etc.); 

# (2) improve the predictive accuracy of a model (e.g., logs, Box-Cox, etc.); and 

# (3) convert categorical and other non-quantitative data into quantitative data that can be used for statistical analysis 
# (e.g., categorical to dummy variables, word counts, etc.)

# While it is often necessary or desirable to transform predictor variables, many regression models are robust when it comes to predictor variables (e.g., predictor variables often don't need to be normally distributed). In contrast, dependent variables must be transformed to the correct types of values for the particular model. For, example, if the dependent variable is binary or categorical, you must transform the outcome variable and use a logistic, discriminant or other classification models. Good news: R makes this very easy.


## Transformation 1: Categorical To Dummy Variable Predictors

# Qualitative variables (e.g., states, categories, etc.) are not quantitative and they cannot be used as is for statistical analysis. Qualitative data has to be converted into some form of quantitative data. If the dependent variable is qualitative then you need to employ classification methods like logistic regression. When the independent variables are qualitative you can use the data with some conversions. R makes this very easy.

# The most common way to deal with qualitative variables is to create dummy variables. For example, if you have 50 states and you want to analyze whether the state where the house is located has an effect on the house value, you can create 50 dummy variables, one for each state. For example, the variable MD will take a value of 1 if the state is Maryland and 0 otherwise. 

# Each state dummy variable would be constructed in a similar manner. You would then leave one of these variables out of the model which will be the "baseline" or "reference" variable (e.g., MD). You would then include in the model the remaining 49 state dummy variables. The intercept coefficient of this model gives you the effect of the baseline or reference variable (e.g., the median housing value prediction for the state of MD). Each of the other 49 state coefficients will give you the price difference for that state, relative to the reference variable (i.e., MD).

# There are two ways to do this: you create the dummy variables yourself or you let R do it for you (with less control). Let's use the Carseats data set:

library(ISLR) # Contains the "Carseats" data set
options(scipen = 4)

names(Carseats) # Briefly inspect the variable names
?Carseats # See the description for ShelveLoc

# In the model below, qualitative variables will be converted into multiple dummy variables. The qualitative variable ShelveLoc has 3 values: Bad, Medium and Good. If we include ShelveLoc in the model R will convert this into only 2 variables: ShelveLocGood and ShelveLocMedium. R automatically names these variables with the original variable name with the respective qualitative value appended to the variable name. R also drops one of the three dummy variables, automatically, which is the first one alphabetically, or ShelveLocBad in this case.  

# The "Dummy Variable Trap"

# If you have a group of dummy variables, which are mutually exclusive, then one of these variables is fully dependent on the others. For example if all the 49 state dummy variables have a value of 1, then the MD variable must be 1 (i.e., if it is not one of the 49 included states, it must be the excluded states). If you were to include all 50 variables, any one of these variables will be fully dependent on the other 49. This totally violates the OLS assumption of variable independence and the OLS is literally unsolvable because you would have perfect multi-collinearity. This problem is commonly referred to as the "Dummy Variable Trap". Older statistical software would simply hang or give you an error (the standard errors are infinite). But modern statistical software detect this linear dependency and automatically exclude one of the variables to avoid the dummy variable trap.

# Let's first explore the factor variable ShelveLoc

class(Carseats$ShelveLoc) # Factor
nlevels(Carseats$ShelveLoc) # Number of levels or categories

levels(Carseats$ShelveLoc) # Unique levels or categories

# Note, levels are ordered internally. By default the order is done alphabetically, so Bad is level 1, Good is 2 and Medium is 3

# When modeling categorical or factor variables, it is important to have enough data points in each category, more than 10 or 15 data points per category is adequate

summary(Carseats$ShelveLoc) # Check out the quantities

lm.fit <- lm(Sales ~ ., data = Carseats)

summary(lm.fit) # ShelveLocBad is the default reference category, i.e., level 1

# To see the categorical to dummy variable conversion:

contrasts(Carseats$ShelveLoc) # Shows how the dummy variables were coded

# Notice that the qualitative variable ShelveLoc was automatically converted into dummy variables. Also notice that ShelveLocBad was excluded from the model and only ShelveLocMedium and ShelveLocGood were included. 

# The intercept represents the average sales in thousands of units when ALL predictor variables are 0, and of course, for ShalveLocBad because all included dummy variables are at 0. 

# ShelveLocMedium shows that, on average and holding everything else constant, sales increase by 1,613 units when car seats are in a medium shelf location, compared to a bad shelf location. Similarly, on average and holding everything else constant, sales increase by 4,033 units when car seats are in a goods shelf location, compared to a bad shelf location.

# It is pretty obvious from this contrast table that "Bad" is the baseline variable omitted.

# You can inspect the coefficients visually too:

library(coefplot)
coefplot(lm.fit)

# Re-Leveling

# If you want to change the baseline or reference level you can use the relevel() function to select the reference level variable to exclude. The value you enter in the `ref =` parameter has to match an existing level in the data (case sensitive). For example, in the above model "Bad" is the ShelveLoc reverence level chosen by R by default because it is the first one alphabetically. If you want to use "Good" as the reference level instead, do the following:

Carseats$ShelveLoc <- relevel(Carseats$ShelveLoc, ref = "Good")
summary(lm(Sales ~ ., data = Carseats)) # Check it out

# Noticed what changed:

# The effect of ShelveLocBad is -4.033, which is the same coefficient we got for ShelveLocGood before releveling, but with a negative sign. This makes perfect sense, Bad locations sell 4,033 fewer units than Good locations, which is the same as saying that Good locations sell 4,033 more units than bad locations.

# The coefficient for ShelveLocMedium has changed. This also makes sense because in the prior model, the reference level was against Bad shelf locations, whereas in the releveled model, the reference level is against Good shelf locations.

# Absolutely nothing else changed. We just changed the references for the parallel regression lines for the various levels of ShelveLoc, but nothing else changed.


## Transformation 2: Polynomials

# We will cover polynomial models in more depth later on, but we discuss polynomials here in the context of variable transformations.

library(MASS) # Contains the Boston housing data set
options(scipen = 4)

# Let's start with a linear model

lm.fit1 <- lm(medv ~ lstat, data = Boston)
summary(lm.fit1)

# Then, let's include a quadratic term

# Note the use of the I() function, which is the identity or "as is" function. ^2 in R has other interpretations, so I(lstat^2) is needed to tell R to treat the ^2 as a square

lm.fit2 <- lm(medv ~ lstat + I(lstat^2), data = Boston)
summary(lm.fit2)

# Cubic Polynomial

lm.fit3 <- lm(medv ~ lstat + I(lstat^2) + I(lstat^3), data=Boston)
summary(lm.fit3)

# Note that the R-squared for the quadratic model is larger than the one for the linear model. However, to evaluate if the difference is significant we need an ANOVA test (or F-test, which is similar)

anova(lm.fit1, lm.fit2, lm.fit3) # Tests if quadratic model is superior

# The F-statistic is quite high and the p-value is <0.001, so, yes the quadratic model is statistically superior and provides additional explanatory power. And the cubic model is superior to the quadratic model.

par(mfrow = c(2, 2)) # Lets split the screen into 2 x 2 frame
plot(lm.fit2) # Display diagnostic plots
par(mfrow = c(1, 1)) # Reset the plot display

# Polynomial Terms -- the "poly()" function

# You can use the poly() function to model higher polynomials (more on this later). For example, this will add quadratic, cube, etc. terms up to the 5th power. But to get the same result than plain polynomials (e.g., I(lstat^5)), you need to use the parameter `raw = T`

lm.fit5 <- lm(medv ~ poly(lstat, 5, raw = T), data = Boston) 
summary(lm.fit5) # See the higher R-Squared


## Transformation 3: Log Models

# Dependent variables are often logged if they have skewed distributions. A log transformation is also necessary when the outcome variable contains counts (e.g., number of applicants, number of store visitors, etc.) -- see Transformation 5 below on count data.

# Inspecting for Normality

# Independent variables with a skewed distributio don't need to be log-transformed if the sample size is large (degrees of freedom > 50. But with smaller samples the sample means can no longer assumed to be normal, so log transformation may be necessary.

library(ggplot2) # Contains the "diamonds" data set
options(scipen="4") # To change the decimals display

# Inspect for normality with histograms and QQ-Plots

hist(diamonds$price, main = "Diamond Price", xlab = "Price")

qqnorm(diamonds$price, main = "Diamond Price")
qqline(diamonds$price)

shapiro.test(diamonds$price)
shapiro.test(diamonds[sample(nrow(diamonds), 5000),]$price)

# Note: a QQ-Plot plots the actual data quartiles against the quartiles of normally distributed data, so if the data is normal, the plot will be a straight line. The qqline function draws a reference line that goes throught he 1st and 3rd quartile of the data. The more the data departs from the line the less normal it is.

# Now let's look at the logged data

hist(log(diamonds$price), 
     main = "Diamond (Log) Price", 
     xlab = "Log Price")

qqnorm(log(diamonds$price), main="Diamond (Log) Price")
qqline(log(diamonds$price))

# Much better normal distribution, at least in the center of the data


# Log-Transformed Models

summary(lm(price ~ carat, data = diamonds))
summary(lm(log(price) ~ carat, data = diamonds))
summary(lm(price ~ log(carat), data = diamonds))
summary(lm(log(price) ~ log(carat), data = diamonds))

# Let's look at housing data and model a log linear regression. First, let's attach the Boston data set to avoid having to use the Boston$ prefix

attach(Boston)

hist(medv) # Slightly skewed to the left
qqnorm(medv); qqline(medv) # Produces a Q-Q Plot, along with a Q-Q line

# If the data is normally distributed, the dots fall along the line You can see how the data deviates from the line at the far right

# Let's start with a linear model with one predictor, rm

summary(lm(medv ~ rm, data = Boston))

# Now let's check the logged data

hist(log(medv)) # More normally distributed
qqnorm(log(medv)); qqline(log(medv)) # A bit better

summary(lm(medv ~ log(rm), data = Boston)) 

# The coefficient for log(rm) is 54.05; This means that a 1% increase in rooms increases medv by 54.05/100 or 0.5405 units

# Now let's log-transform the dependent variable only

summary(lm(log(medv) ~ rm, data = Boston)) 

# The coefficient for rm is 0.36; This means that a 1 unit increase in room leads to a 100*0.36 or 36% increase in medv

# Now let's try a log-log (elasticity) model

summary(lm(log(medv) ~ log(rm), data = Boston)) 

# The coefficient for log(rm) is 2.22. This means that a 1% increase in rooms leads to a 2.22% increase in medv


# Logit-Transformed Models (Binary Y)

# Logit models are the most popular choice when the response variable is binary, i.e., 1,0; success, failure; approve, disapprove; etc.

# The general procedure when Y is binary is to:

# 1. Transform the response variable 1 using the Logit function
# 2. Leave the predictors as is
# 3. Model the regression with the glm() function, 
#    rather than the lm() function, which can be 
#    used when Y is not normally distributed.
# 4. Interpret the logistic regression coefficients
#    and fit statistics with caution. See below.

# We use this same example later in the classification models section. This data set was compiled by the ISLR book authors and it contains data on heart failures.

browseURL("http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.info.txt")

# chd is the binary response variable: 1 = heart disease; 0 = no heart disease

# Let's read the data (you can use instead the .csv we saved above if you prefer)

heart <- read.table("Heart.csv", sep=",", head=T)
head(heart) # take a look

# Logistic model predicting coronary heart desease

heart.fit = glm(chd ~ ., 
                family = binomial(link = "logit"), 
                data=heart)

summary(heart.fit)

# Looks like tobacco, ldl, family history, type a and age are the strongest predictors of coronary heart desease (chd). Interestingly Once you controll for these factors obesity and alcohol are not significant predictors. Take a closer look:

# You can inspect the coefficients visually too:

require(coefplot)
coefplot(heart.fit)

# Transforming coefficients

log.odds <- coef(heart.fit) # To get just the coefficients
log.odds # Check it out

# To convert Log-Odds to multiplicative change in odds

odds <- exp(coef(heart.fit)) 
odds # Check it out

prob <- odds / (1 + odds) # TO convert odds to probabilities
prob # Check it out

round(cbind(log.odds, odds, prob), digits = 3)


# Log-Transformed Models (Count Y)

# OLS is often used, but should not be used to predict count data (e.g., number of student applications, number of customers coming to the store, etc.) because the outcome variable is: 

# (1) discrete (no decimals);
# (2) non-negative; and 
# (3) truncated at 0. 

# The correct way is to use the "glm()" function with a "Poisson" distribution and the outcome variable needs to log transformed (i.e., the link function is link="log": 

College <- read.table("College.csv", header = T, sep = ",")

lm.fit <- lm(Apps ~ Outstate + PhD + S.F.Ratio, data = College)
summary(lm.fit)

# Incorrectly modeled above as OLS. Try to interpret the meaning of a negative intercept. The correct way is with:

glm.fit.count <- glm(Apps ~ Outstate + PhD + S.F.Ratio, 
                     family = poisson(link = "log"), 
                     data = College)

summary(glm.fit.count)

# The family = poisson(link = "log") attribute tells glm() that the response variable follows a Poisson distribution. The "link" function tells glm() to log-transform the outcome variable. Because y is logged (and the x's are not), the coefficients represent the percent increase in applications, when the x goes up by 1 unit.

# You can inspect the coefficients visually too:

require(coefplot)
coefplot(glm.fit.count)


## Transformation 4: Centering Data

library(MASS) # Contains the Boston housing data set
options(scipen=4)

head(Boston) # First, take a quick look at the data

# We can center the data with the scale() function. As we will see later on, the scale() function can also be used to standardize data -- "center=TRUE" will center the variable and "scale=TRUE" will standardize the variable

?scale()

# This will center the entire data frame, provided that all variables are numeric

Boston.centered <- data.frame(scale(Boston, center = T, scale = F))

# Note that we also used the data.frame() function because the scale() function converts the data into a matrix and some functions like the linear model lm() function only work with data frames

head(Boston.centered) # Check it out

lm.uncentered <- lm(medv ~ lstat + age, data = Boston)
summary(lm.uncentered)

# Note: centered regression models fit a regression line through the origin. However, it always gives an intercept, which is very close to zero. Take a look:

summary(lm(medv ~ lstat + age, data = Boston.centered))

# So, when fitting centered models, it is better to force the regression line through the origin. Including -1 in the model formula removes the intercept term, that is, it forces the regression line through the origin:

lm.centered <- lm(medv ~ lstat + age -1, data = Boston.centered)
summary(lm.centered)

# Notice that the uncentered and centered regressions above produce identical results because centering does not change the coefficients, except for the intercept.

# You can inspect the coefficients visually too:

require(coefplot)
coefplot(lm.centered)

# So, why would you center your data, if you are going to get identical results? If all you have is a linear model, there is no need to center the data. However, if you have polynomials or interaction models, certain methods will require that you center the data.

# For example, centering is necessary when including interaction terms involving 2 continuous variables:

summary(lm(medv ~ lstat * age, data = Boston)) # With uncentered data
summary(lm(medv ~ lstat * age -1, data = Boston.centered)) # With centered data

# Notice in the regressions above that the results are no longer the same between the centered and original data once you include continuous x continuous interaction terms. The centered data is the correct way to model continuous interaction terms.

# Note: if you run the commands below you will alter the Boston data frame in your computer memory. This is OK, but some functions like predict() may not run properly with this extra data. If this happens, simply click on the broom icon above to clear objects from your workspace and then re-attach the original Boston data set.

# This will center specific columns

Boston$medv.c <- scale(Boston$medv, center = T, scale = F)
Boston$lstat.c <- scale(Boston$lstat, center = T, scale = F)
Boston$age.c <- scale(Boston$age, center = T, scale = F)
head(Boston) # Check it out

# Interpreting interaction effect with continuous variables is not easy. Centering variables in interaction terms makes this interpretation easier. Interaction between 2 continuous variables means that the effect of one variable depends on the value of the other variable. The main effects in the above model show the effect of that variables, when the other variable is at its mean.


## Transformation 5: Standardizing Data

# You can standardize data with the same scale() funtion but using scale=TRUE. You may or may not center the data, depending on your modeling needs. But generally, the data is both, centered and scaled, so that all variables have a mean of 0 and a standard deviation of 1.

head(Boston) # First, take a quick look at the data
options(scipen = "4")

# This will standardizing all the data:

Boston.standardized <- data.frame(scale(Boston, center = T, scale = T))
head(Boston.standardized) # Check it out

# Compare results

summary(lm(medv ~ lstat + age, data = Boston)) # Unstandardized

# Coefficients indicate how many units Y changes when X changes by 1

summary(lm(medv ~ lstat + age -1, data = Boston.standardized)) # Standardized

# Coefficients indicate how many standard deviations Y changes when X changes by 1 standard deviation

# As with centering, the commands below will alter the Boston data 
# frame in memory.

# These instructions will standardize (center and scale) the respective predictors and save the results in new variables suffixed .std. 

Boston$medv.std = scale(Boston$medv, center = T, scale = T)
Boston$lstat.std = scale(Boston$lstat, center = T, scale = T)
Boston$age.std = scale(Boston$age, center = T, scale = T)
head(Boston) # Check it out

# Standardized regression

# You can run the regular linear model function and then extract standardized coefficients with the {lm.beta} package, which will yield "identical" results than runing a regression with all variables standardized.

install.packages("lm.beta") # If not installed already
library(lm.beta)

head(Boston) # First, take a quick look at the data

# Run a regression model as usual and store it in an object

lm.fit <- lm(medv ~ lstat + age, data = Boston)

# This object lm.fit does not have standardized coefficients:
summary(lm.fit)

# But the lm.beta() function calculates them:

lm.standardized <- lm.beta(lm.fit)
summary(lm.standardized) # Check it out

# You can inspect the coefficients visually too:

require(coefplot)
coefplot(lm.standardized)


## Transformation 6: Lagging data

# Let's work with a housing starts data file

options(scipen = "4")

HousingStarts <- read.csv("HousingStarts.csv", 
                          header = T,
                          na.strings = "?")

HousingStarts <- na.omit(HousingStarts) # Removes NA's - important when lagging

# Month -- data period
# T -- Month converted into a numeric ordinal sorted variable
# KUnits -- houses started x 1000
# S.P -- S&P index
# Q1, Q2, Q3, Q4 -- dummy variables for each quarter

head(HousingStarts)

# Let's run a regression to predict KUnits

lm.KUnits <- lm(KUnits ~ T + S.P + Q2 + Q3 + Q4, 
                         data = HousingStarts)

plot(HousingStarts$T, lm.KUnits$residuals)
abline(0,0, col="red") # Red line with intercept and slope equal to 0

summary(lm.KUnits)

# You first need to sort the data by the ordinal variable (e.g., time). In our case, the data was already sorted in Excel


# Testing for Serial Correlation

# Let's do a Durbin-Watson (DW) test for serial correlation. This package has a DW test function for lm() objects:

library(lmtest) # Activate the package

dwtest(lm.KUnits) # Run the DW test
# DW = 0.308 << 2 --> High positive serial correlation

# Let's try using 4 lags of the dependent variable as predictors

attach(HousingStarts)


# Multivariate Time Series Models

# The {DataCombine} package has functions to work with data

library(DataCombine) # Contains the slide() function

# The slide{DataCombine} function can be used to create lag variables. In this case we are creating 4 lagged variables. Note that we need to save the lagged data in a data frame, which could be the same or a new data frame. In this case we are saving the data to the same data frame HousingStarts. Also, use negative "slideBy" values to create lag variables and positive "slideBy" values to create lead variables.

HousingStarts = slide(HousingStarts, 
                      Var = "KUnits", 
                      NewVar = "KUnits.L1", 
                      slideBy = -1)

HousingStarts = slide(HousingStarts, 
                      Var = "KUnits", 
                      NewVar = "KUnits.L2", 
                      slideBy = -2)

HousingStarts = slide(HousingStarts, 
                      Var = "KUnits", 
                      NewVar = "KUnits.L3", 
                      slideBy = -3)

HousingStarts = slide(HousingStarts, 
                      Var = "KUnits", 
                      NewVar = "KUnits.L4", 
                      slideBy = -4)

HousingStarts = slide(HousingStarts, 
                      Var = "KUnits", 
                      NewVar = "KUnits.L12", 
                      slideBy = -12)

HousingStarts[1:20, ] # List first 20 observations

# IMPORTANT Technical Note: the slide() function works on data frames. Some time, data sets come in other formats or as more complex data frames and the slide() function complains with errors. The typical error is if you slide a column with 400 values by one, the resulting column will only have 399 values. But in a data frame, the first value is assigned an NA, so it works fine and you get 400 values. If you get one of these mismatch errors, a solution that often works is to re-create the frame like this: MyData <- as.data.frame(MyData). This command will create a simple data frame that works well with slide().

# Now let's try the regression model with all 4 lagged variables

lm.KUnits.all = lm(KUnits ~ T + S.P + Q2 + Q3 + Q4 + 
                            KUnits.L1 + KUnits.L2 + KUnits.L3 + KUnits.L4, 
                            data = HousingStarts)

summary(lm.KUnits.all) # Check out the results

# Notice that KUnits.L3 and KUnits.L4 are significant predictors of KUnits

dwtest(lm.KUnits.all) # Run the DW test
# DW = 2.388 and p-value is not significant, which is acceptable -- no serial correlation

# You can inspect the coefficients visually too:

require(coefplot)
coefplot(lm.KUnits.all)

# Let's try another model with only lags for 1 and 12 periods

lm.KUnits.1.12 <- lm(KUnits ~ T + S.P + Q2 + Q3 + Q4 +
                              KUnits.L1 + KUnits.L12, 
                              data = HousingStarts)

summary(lm.KUnits.1.12) # Check out the results

# Test visually

plot(HousingStarts$T[13:length(HousingStarts$T)], 
     lm.KUnits.1.12$residuals)

abline(0,0, col="red") # Visually, serial correlation appears to be corrected too

# Test statistically

dwtest(lm.KUnits.1.12) # Serial correlation was corrected
