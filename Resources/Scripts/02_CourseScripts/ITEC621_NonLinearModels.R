##############################
# ITEC 621 Non-Linear Models #
##############################

# Filename: ITEC621_NonLinearModels.R
# Prepared by J. Alberto Espinosa
# Last updated on 3/20/2022

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: the material covered in this script ASSUMES that you have gone through the self-paced course KSB-999 R Overview for Business Analytics and are thoroughly familiar with the concepts illustrated in the RWorkshopScripts.R script. If you have not completed this workshop and script exercises, please stop here and do that first.


#######################################################
#                       INDEX                         #
#######################################################

## Interaction Terms - Binary x Continuous

## Polynomial Regression
# Alternatively approach with the I() function

## Polynomial Logistic Regression
## Step Functions
## Piecewise Linear Regression
## Linear Spline Regression
## Polynomial and Cubic Splines
## Natural Cubic Splines

# Comparing the 3 models with Cross-Validation

#######################################################
#                  END OF INDEX                       #
#######################################################


## Interaction Terms - Binary x Continuous

# Interaction effects should not be interpreted by thembselves, but in conjunction with their respective main effects before an interaction enhances or offsets a main effect.

# Binary x Continuous interactions are easy to interpret because the interaction effect specifies how much the main effect of the continuous variable changes when the binary variable changes in value from 0 to 1.

# Let's compare a linear with an interaction model to predict car gas mileage

library(ISLR) # Contains the Auto data set
options(scipen = 4)

#Let's convert the origin variable into a new variable called "foreign" = 1 if European or Japanese and 0 if American.

data(Auto)
Auto$foreign <- ifelse(Auto$origin == 1, 0, 1)

# Linear model without interaction

fit.linear <- lm(mpg ~ horsepower + weight + foreign + year, 
                 data = Auto)

summary(fit.linear)

# Let's add an (binary x continuous) interaction term foreign x weight

fit.inter <- lm(mpg ~ horsepower + weight + foreign + year +
                      foreign * weight, 
                data = Auto)

summary(fit.inter)

# Let's add another (binary x continuous) interaction term foreign x year

fit.inter.2 <- lm(mpg ~ horsepower + weight + foreign + year + 
                        foreign * weight + 
                        foreign * year, 
                  data = Auto)

summary(fit.inter.2)

anova(fit.linear, fit.inter, fit.inter.2) # Comparing all 3 models

# Interpretation: 

# 1. The linear model indicates that foreign cars have better gas mileage than domestic cars, other things being equal. When we add a foreign*weight interaction term the interaction is negative and significant. This means that the negative effect of weight on gas mileage is stronger for foreign cars that for domestic cars. Or, conversely, the positive effect of being a foreign car is diminished for heavier cars.

# 2. Interestingly, when we add an interaction term foreigh*year, it not only becomes significant, but now the main effect of foreign is negative. This means that foreign cars are less fuel efficient, but this effect is offset rapidly with newer cars.

# 3. The ANOVA tests shows that the 2-interaction model is better than the 1 interaction model, which in turn is better than the linear model.

# R makes this easier for you because if you only include the interaction term, R knows that the respective main effects also need to be included because interaction effects are meaningless by themselves. Note how R includes the main effects even if we don't explicitly include them:

fit.inter.2 <- lm(mpg ~ horsepower + foreign*weight + 
                        foreign * year, 
                  data = Auto)

summary(fit.inter.2)

# If you would like to exclude the main effects for some strange reason, you can do this using the ":" rather than the "*" operator:

fit.inter.not <- lm(mpg ~ horsepower + 
                          foreign:weight + 
                          foreign:year, 
                    data = Auto)

summary(fit.inter.not)

# Try to interpret these results. It's not really possible.

# You can use the ":" to specify interaction terms when you have explicitly modeled the main effects:

# A recap on how to interpret binary x continuous interaction effects:

# 1. If the interaction effect is not significant, there 
#    are no interaction effects

# 2. If the interaction effect is significant:

#    2.1 If the sign of the interaction effect is in the 
#        same direction as the main effect, then the other 
#        variable "enhances" the main effect.

#    2.2 If the sign of the interaction effect is in the 
#        opposite direction than the main effect, then the 
#        other variable in the interaction term "offsets" or 
#        "diminishes" the main effect.

#    3.1 If the continuous variable does not have a significant main
#        effect, then it's effect is not significant when the binary
#        variable value is 0, but it is significant when it is 1.

# For example, in the model below lstat has a negative effect on median home values and the interaction effect with age is significant. This means that age and lstat offset each other. In other words, lstat has a negative effect on median home values, but this negative effect is diminished with the age of the housing.


## Polynomial Regression

# Before we start with polynomials, let's take a look at the Wage data

library(ISLR) # Contains the Wage data set
head(Wage)

# Caution: note that there is a variable also called "wage" but with a lower case w, which contains the wage for each person

plot(Wage$age, Wage$wage, 
     xlab = "Age", 
     ylab ="Wage", 
     cex = .5, 
     col = "darkgrey")

# The lowess function performs a "locally weighted smoothing", which is like dividing the data into segments and fitting mini-regressions in each, and then use the fitted values to drass a "smooth" line on the plot:

lines(lowess(Wage$age, Wage$wage), 
      col = "red") # Trend line

# You can change the smoothing on the line with the parameter f, which is the proportion of data points in each segment. The default is f = 2/3. Try a very small value (e.g., 0.001) for little smoothing (i.e., a jagged line) and a large one (e.g. 0.90) for a lot of smoothing (1 give a straight line).

lines(lowess(Wage$age, Wage$wage, f = 0.001), 
      col = "blue") # Jagged trend line

# A casual inspection of the plot suggests that the relationship between wage and age is not linear

Wage$wage # Take a look at all the wages
Wage$age # and all the ages

# Let's start simple, with a linear model

fit.poly1 <- lm(wage ~ age, data=Wage)
summary(fit.poly1)
# Good fit; age is significant; very low R-Square

# Let's try a squared model

fit.poly2 <- lm(wage ~ poly(age, 2), data = Wage)
summary(fit.poly2)
# Good fit; significant squared term; R-Square improve just a tad

# Let's fit a polynomial up to the power of 4

fit.poly4 <- lm(wage ~ poly(age,4), data = Wage)
summary(fit.poly4)

anova(fit.poly1, fit.poly2, fit.poly4)

# Good fit; significant squared and cube term; 4th poly coefficient is not significant; R-Square improve just a tad. Higher polynomial model has stronger predictive power.

# The higher the poly power; the higher the R-square; the better the fit; but you start running into dimensionality problems and overfitting and the model becomes more difficult to interpret.

# Important Note: the poly() function does not give coefficients for x, x-square, x-cube, etc., but for an "orthogonal polynomial" of these variables. This sounds complicated, but these are simply the principal components of the polynomial variables. This is done by the poly() function because including various powers of a variable will create high multi-collinearity. With these orthogonal polynomials, all 4 variables have 0 correlation, no multi-collinearity, therefore yielding more stable models. However, the interpretation of orthogonal polynomials is tricky because it is hard to relate them to the original variable x. But polynomial are hard to interpret anyway, orthogonal or not.

# If you would like to fit a model using the raw variables x, x-squared, x-cube, etc., without converting these variables into an orthogonal polynomial, just use the attribute raw=TRUE or raw=T

fit.poly4.raw <- lm(wage ~ poly(age, 4, raw = T), data = Wage)
summary(fit.poly4.raw)

# Alternatively approach with the I() function

fit.poly4.I <- lm(wage ~ age + I(age^2) + I(age^3) + I(age^4),
                  data = Wage)

summary(fit.poly4.I)
# Notice that the results are identical to the previous poly model


# Predictions with Polynomials

# Rather than using all data points, let's get a list of the unique age values in the data and make predictions for these values. We start by computing the range of the data, that is the lower and upper values of the age variable range using range()

agelimits <- range(Wage$age)
agelimits # Take alook -- the ages range between 18 and 80

agelimits[1]
agelimits[2]

# agelimits is a vector with 2 values, so agelims[1]=18 and agelims[2]=80, so the command above creates a sequence of ages from 18 to 80 increments of 10.

# Now lets create a vector containing a sequence of the ages in the data using the seq() function and store it in an object named age.seq:

age.seq <- seq(from = 18, to = 80)
age.seq # Take a look -- it's a sequence with all ages in the data

# The above calculation works, but if the data change and we have ages less than 18 or more than 80, then the age.seq vector would be incomplete. So, it is better to write a formula that will pick up the actual extreme ages in the data:

age.seq <- seq(from = agelimits[1], to = agelimits[2])
age.seq # Take a look -- same results

# Now let's run the predict() function for all the ages in the data, and include standard errors in the output

preds <- predict(fit.poly4, newdata = list(age = age.seq), se = T)

# The command above seems convoluted, but we are using new data to make predictions. Since our only prediction is a variable named "age", we need to use the command list(age=age.grid) to create a variable called "age" and assign to this variable each of the unique age values contained in age.grid. Notice that age.grid and list(age=age.grid) yield the same values:

age.seq
list(age = age.seq)

# but notice that the list has the $age name designation, which the fit.poly.raw linear model expects

# Now, let's take a look at the predicted wages and standard errors for these predictions

preds 

# Let'do some plotting -- NOTE: you can only plot Y against a single X predictor

# Let's split the plot frame and do some visualization and move the margins to make room for a title

# mar() controls the number of lines at the bottom, left, top right
# oma() control the outer margins -- play with these values

par(mfrow = c(1, 1), mar = c(4.5, 4.5, 1, 1), oma = c(0, 0, 2, 0)) 

# Plot the dots first

plot(Wage$age, 
     Wage$wage, 
     xlab = "Age", 
     ylab ="Wage", 
     xlim = agelimits, 
     cex = 0.5, 
     col = "darkgrey") # Do the plot first

title("Degree-4 Polynomial", outer = T) # Add a title

# Now draw a line with the ages and predicted wage values

lines(age.seq, preds$fit, lwd = 2, col = "blue")

# Now draw confidence intervals at +/- 2 standard errors

lines(age.seq, preds$fit - 2 * preds$se.fit, lwd = 1, col = "red") 
lines(age.seq, preds$fit + 2 * preds$se.fit, lwd = 1, col ="red") 

# One important observation: whether you use the raw x, x-squared, x-cube, etc. or orthogonal polynomials, the predicted values will be identical. Take a look

preds2 <- predict(fit.poly4, newdata = list(age = age.seq), se = T)
# preds uses raw polynomials and preds2 uses orthogonal polynomials

max(abs(preds$fit - preds2$fit)) # Notice how small are the differences

# If you would like to find the best polynomial, you can test how much predictive power you add every time you add a higher polynomial term, using the anova() function, as explained above for testing reduced vs. extended sets:

fit.1 <- lm(wage ~ age, data = Wage) # Just x
fit.2 <- lm(wage ~ poly(age, 2), data = Wage) # Add x-squared
fit.3 <- lm(wage ~ poly(age, 3), data = Wage) # etc.
fit.4 <- lm(wage ~ poly(age, 4), data = Wage)
fit.5 <- lm(wage ~ poly(age, 5), data = Wage)

# Now let's do an F-Test with ANOVA
anova(fit.1, fit.2, fit.3, fit.4, fit.5)

# These results support model 3, which has x-cube -- beyond that the power does not increase significantly

summary(fit.3) # Note that all coefficients are significant

# Interestingly, when we add higher power coefficients the model breaksdown

summary(fit.5) 

# Important note: we used raw variables, but the book example uses orthogonal polynomials.This means that the coefficients will be different, but the ANOVA F-Tests should yield similar results

# You can try adding other variables and testing them with F-tests, for example:

fit.1 <- lm(wage ~ education + age, data = Wage)
fit.2 <- lm(wage ~ education + poly(age, 2), data = Wage)
fit.3 <- lm(wage ~ education + poly(age, 3), data = Wage)

anova(fit.1, fit.2, fit.3)


## Polynomial Logistic Regression

# IMPORTANT: Try on your own after the classification models lecture

# Let's try the same method with Logistic regression with the dependent variable as the probability that Wage is greater than 250K using orthogonal polynomial

library(ISLR) # Contains the Wage data
attach(Wage)
head(Wage)

fit.logit <- glm( I(wage > 250) ~ poly(age, 4),
                  data = Wage, 
                  family = binomial(link = "logit"))

summary(fit.logit) # Take a look

# Before we do predictions, let's re-create the age.grid data. No need to re-create it if you have not shut down R Studio since you created it

agelimits <- range(age)
age.seq <- seq(from = agelimits[1], to = agelimits[2])

# Now let's do predictions on the same age range we used above

preds <- predict(fit.logit, newdata = list(age = age.seq), se = T)

# The preds vector has one predicted value for each age.grid value

preds # Take a look at the log-odds of Wage>250K predictions

preds.odds <- exp(preds$fit) # Odds predictions of Wage>250K
preds.odds # Take a look

preds.probs <- preds.odds / ( 1 + preds.odds) # Probability predictions
preds.probs # Take a look

# Now let's compute the confidence interval as predicted value +/- 2 standard errors, using the log odds (you can try odds and probs).

se.bands <- cbind(preds$fit - 2 * preds$se.fit, 
                  preds$fit + 2 * preds$se.fit)
se.bands # Take a look

# And also let's do the same for odds and probabilities:

se.bands.odds <- exp(se.bands)
se.bands.odds # Take a look

se.bands.probs <- se.bands.odds / (1 + se.bands.odds)
se.bands.probs # Take a look

# Note: if we use type="response" with the predict() function, it causes the predicted values to be the probabilities; otherwise the default is log-odds.

preds.probs.2 <- predict(fit.logit, 
                         newdata = list(age = age.seq), 
                         type = "response", 
                         se = T)

preds.probs.2 # Take a look

# Now let's do the plotting

plot(age, 
     I(wage > 250), 
     xlim = agelimits, 
     ylim = c(0, 1))

lines(age.seq, preds.probs,lwd = 2, col = "blue")

# matlines (matrix lines) works like lines but plots all columns of a matrix, in this case it plots both columns of se.bands.probs against age

matlines(age.seq, se.bands.probs, lwd = 1, col = "red", lty = 3)

# We can amplify up the plot by changing the ylim to 0,0.2 and dividing the I by 5 just so that the 1's show up in the 0,0.2 range

plot(age, 
     I(wage > 250) / 5, 
     xlim = agelimits, 
     ylim = c(0, 0.2))

lines(age.seq, preds.probs,lwd = 2, col = "blue")
matlines(age.seq, se.bands.probs, lwd = 1, col = "red",lty = 3)

# Notice the "wagging the tail" issue, typical of polynomials


## Step Functions

library(ISLR) # Contains the Wage data
attach(Wage)
head(Wage)

# As illustrated in the plot above, polynomials tend to over-fit the data and yield high-variance models. "Wagging the tail" is a common issue with polynomials in which the standard error band widens at the tail-end of the data. Why do you think that is? Well, if there is error in x, the errors get amplified with powers of x, particularly as x gets larger.

# Step functions alleviate this problem to some extent. A step function fits a bunch of horizontal lines in different "segments" or "partitions" of the data. The points where the partitions begin and end are called "cut points". To fit a Step function, we can use the cut() function to arbitrarily divide the age data into, say 4 partitions.

# I repeat these steps for convenience, to yield age data points for plotting and prediction

agelimits <- range(Wage$age) 
age.seq <- seq(from = agelimits[1], to = agelimits[2])

# Now let's fit a step model. First let's take a look at the cut() funcion, which divides a variable into segments. In this example, we divide the age variable into 4 segments. Take a look:

table(cut(age, 4)) # Shows age ranges for each segment and number of observations in the segment

fit.step <- lm(wage ~ cut(age, 4), data=Wage) # Fit the step function

summary(fit.step) # Take a look (coefficients are relative to intercept)

# Let's do some predictions with the age.grid data

preds <- predict(fit.step, 
                 newdata = list(age = age.seq), 
                 se = T)

preds # Take a look

# Let's inspect the results visually

plot(age, 
     wage, 
     xlim = agelimits, 
     cex = 0.5, 
     col = "darkgrey") # Do the plot first

title("Step Regression of Wage on Age", outer = T) # Add a title

# Let's draw a line with the ages and predicted wage values

lines(age.seq, preds$fit, lwd = 2, col = "blue") 

# And also the confidence interval line with age; 
# Note: you can use this method or the matlines() method above

lines(age.seq, preds$fit - 2*preds$se.fit, lwd = 1, col = "red") 
lines(age.seq, preds$fit + 2*preds$se.fit, lwd = 1, col = "red") 
# Notice that the "tail wagging" issue has been reduced

# The problem now is that we have a bunch of horizontal lines, which don't capture trends. It seems like sloping the lines can give us a better fit, which is what Piecewise Linear Regression does.


## Piecewise Linear Regression 

# This code is not in the textbook

library(ISLR) # Contains the Wage data set
attach(Wage)
head(Wage)

# Re-generate the age sequence for convenience

agelimits <- range(Wage$age) 
age.seq <- seq(from = agelimits[1], to = agelimits[2])

# To fit a sequence of regression lines sequentially along the data, we first need to figure out the "knots" or "cut points", where the regression slope changes. Take a look at the data first:

plot(Wage$age, Wage$wage, xlim = agelimits, cex = 0.5, col = "darkgrey")

# Visually, it appears that the data changes directions at ages 25, 40 and 60. So, let's build a stepwise linear regression model with 3 knots yielding 4 different but connected regression lines (i.e., segments) to the right and left of each knot. To do this, we can create dummy variables that are 0 to the left of each knot and 1 to the right. We can then add the interaction term of this dummy variable with age.

fit.piecewise <- lm(wage ~ age + 
                           I((age - 25) * (age > 25)) + 
                           I((age - 40) * (age > 40)) + 
                           I((age - 60) * (age > 60)), 
                      data = Wage)

# Note that the I() function creates the respective dummy variables and interaction terms. Also note that we use (age-25) rather than age in the first interaction term, which is equivalent to moving the Y axis to the knot at age=25 and then multiply it by (age>25), which will yield 0 for age<=25 and age for age>25 we then apply the I() function to the whole operation.

summary(fit.piecewise) # Take a look at the results

# Interpretation: the intercept is at -42.47. The regression slope starts with 5.38K more wage dollars for every additional year of age, up to age 25. At that point the slope changes to 5.38-3.50=1.88K more wage per year of age, upt to age 40. Then, it changes to 1.88-1.98 = -0.10K less wage dollars for every additional year of age (i.e., almost flat), up to age 60. After that it declines at the rate of -0.10-1.40 = -1.50K fewer dollars for each additional year of age.

# Let's look at this visually. First, let's create a vector with age values of interest -- 0, the 3 knots and 80 (the largest age in the set)

age.knots <- c(0, 25, 40, 65, 80)

# Now let's predict wages for these 5 values, with standard errors

preds <- predict(fit.piecewise, 
                 list(age = age.knots), 
                 se = T)

# Note also that we assign a variable name "age" to the age.knot list because that's the predictor variable name stored in the fit.piecewise object. 

# Repeat the plot here for convenience

plot(Wage$age, 
     Wage$wage, 
     xlim = agelimits, 
     cex = 0.5, 
     col = "darkgrey")

# Now let's draw blue lines between the knots, which will show the piecewise linear model line by connecting the dots between knots

lines(age.knots, preds$fit, col = "blue", lwd = 2)

# Lower and upper confidence interval lines

lines(age.knots, 
      preds$fit - 2 * preds$se.fit, 
      col = "red", 
      lwd = 1)

lines(age.knots, 
      preds$fit + 
      2 * preds$se.fit, 
      col = "red", 
      lwd = 1)

# A little "tail wagging, but not as much as with the polynomial"


## Linear Spline Regression

# The piecewise linear model above is very interpretable, and tend to has better fit than polynomials and step functions because we can set as many knots as we want and therefore control the slopes in the various resulting segments. But if we are not so interested in interpretation, and are more interested in making predictions, regression splines are an alternative option. Spline regressions are easier to formulate and we can not only fit linear splines, but also polynomial splines tjat connect knots with curves, rather than lines. Naturally, interpretation becomes almost impossible as we include polynomial splines, but the prediciton can be more accurate some times because the regression line can follow the data more precisely (but be careful with over-fitting). Let's take a look.

library(ISLR) # Needed to access the Wage data set
library(splines) # Needed to fit spline models

# We use the bs(){splines} basis spline function, which builds polynomial splines at the specified cut points. "degree=1" specifies a linear spline, whereas higher degrees specify polynomials

fit.x1 <- lm(wage ~ bs(age, 
                       knots = c(25, 40, 60), 
                       degree = 1), 
             data = Wage)

# This model fits wage as a linear function of age, so it will return 4 coefficients -- 1 for age and then 1 for each of the three segments after each knot. That is, if d=1 (degree 1) and K=3 (knots) there will be K+1=4 coefficients (one for each segment). 

summary(fit.x1) # Take a look

# You should see 4 different regression lines changing at the knots, almost identical to the piecewise linear model above.

# Important note about interpreteation: the intercept is not the prediction for Age=0, but for the lowest age in the data range. The knot coefficients show the difference between the intercept and the predictions at each knot. To find the prediction at a given knot you need to add its coefficient to the intercept.

# Re-generate the age sequence for convenience

agelimits <- range(Wage$age) 
age.seq <- seq(from = agelimits[1], to = agelimits[2])

# Let's make predictions for all ages in the data range

pred.x1 <- predict(fit.x1, 
                   list(age = age.seq), 
                   se = T)

pred.x1 # Take a look at the predicted wages at the knots

plot(Wage$age, Wage$wage, col = "gray")
lines(age.seq, pred.x1$fit, col = "blue", lwd = 2)

# Confidence intervals

lines(age.seq, 
      pred.x1$fit + 2 * pred.x1$se.fit, 
      col = "red", 
      lwd = 1)

lines(age.seq, 
      pred.x1$fit - 2 * pred.x1$se.fit, 
      col = "red", 
      lwd = 1)

# Another way to locate the knots

# In the model above we arbitrarily set the knots at ages 25, 40 and 60. However if we would like to split the spline segments and knots evenly we can use the "df" (degrees of freedom) parameter, rather than the "knots" parameter. If df=4, this means that there are 4 spline segments with 3 knots in between. Take a look:

dim(bs(age, df = 4, degree = 1)) # Creates 4 partitions with 3 knots

# Check out the knot locations

attr(bs(age, df = 4, degree = 1), "knots") # Knots are evenly spaced
attr(bs(age, df = 6), "knots") # Default degree is 3 or cubic

# Notice that this also provides 3 knots because 3 degrees of freedom are taken by the 3 powers of age

# Let's fit a linear spline with df=4

fit.x1.df <- lm(wage ~ bs(age, df = 4, degree = 1), data = Wage)
summary(fit.x1.df)

# Re-generate the age sequence for convenience

agelimits <- range(Wage$age) 
age.seq <- seq(from = agelimits[1], to = agelimits[2])

# Let's make predictions

pred.x1.df <- predict(fit.x1.df, 
                      newdata = list(age = age.seq), 
                      se = T)

plot(Wage$age, 
     Wage$wage, 
     col = "gray")

lines(age.seq, 
      pred.x1.df$fit, 
      col = "blue", 
      lwd = 2)

# Confidence intervals

lines(age.seq, 
      pred.x1.df$fit - 2 * pred.x1.df$se.fit, 
      col = "red", 
      lwd = 1)

lines(age.seq, 
      pred.x1.df$fit + 2 * pred.x1.df$se.fit, 
      col = "red", 
      lwd = 1)

# You should see a similar piecewise linear model but with 4 knots evenly spaced.


## Polynomial and Cubic Splines

library(ISLR) # Needed to acces the Wage data set
library(splines) # Needed to fit spline models

# For some reason "cubic" splines are very popular. One of the reasons is that a polynomial of power 3 provides enough curves, inflection points, peaks and valleys to fit most data patterns. Beyond the power of 3 the spline becomes too "wiggly", over-fitting, and almost impossible to interpret. This is why the B-Spline function bs() uses a polynomial degree default of 3, or a "cubic spline" but, naturally, you can change this with the "degree" attribute.The next model fits a cubic spline, i.e., degree=3

fit.x3 <- lm(wage ~ bs(age, knots = c(25, 40, 60)), 
             data = Wage) 

# Technical Note: the bs() function generates a B-Spline Basis function which is a convenient and simple way to model polynomial splines. The most popular polynomial spline is the cubic spline, so I refer to cubic splines in this explanation, but the principle applies to any polynomial. The idea is to fit a separate cubic regression in each of the spline segments. But formulating separate cubic regressions is too complex and uses too many degrees of freedom. 

# The B-Spline approach simplifies this by estimating a general cubic model for all the data (i.e., all segments combined), and then estimating a cubic term for each segment, which specifies how much the coefficient of the cubic term changes in each segment. For convenience, this segment correction is only applied on the cubic term (or highest polynomial term), thus simplifying the model. For example, the model above will yield 7 coefficients (or 1+d+k coefficients for a B-spline polynomial model of degree d with k knots):

# 1. The first coefficient is the regression intercept

# 2. The next 3 coefficients (1-3) are for x, x^2 and x^3 (or d coefficients for x to x^d in a d polynomial)

# 3. The next 3 coefficients (4-6) provide the difference in predicted values at each of the knots, resulting from the additional cubic term added to each segment after the first segment.

summary(fit.x3) # Take a look

# This is difficult to interpret, but the good news is that if you are using cubic splines is because your analysis goal is predictive accuracy, not interpretation.

# Re-generate the age sequence for convenience

agelimits <- range(Wage$age) 
age.seq <- seq(from = agelimits[1], to = agelimits[2])

# Now let's calculate the predicted values and plot them:

pred.x3 <- predict(fit.x3, 
                   newdata = list(age = age.seq), 
                   se = T) # Or simply

pred.x3 <- predict(fit.x3, 
                   list(age = age.seq), 
                   se = T)

# Note: we are using the age.grid values (which contain all unique ages in the data) rather than age.knots because we are no longer plotting straight lines between knots (the spline is curvilinear)

plot(Wage$age, 
     Wage$wage, 
     col = "gray")

lines(age.seq, 
      pred.x3$fit, 
      col = "blue", 
      lwd = 2)

# Notice that we have a curvilinear model, but it really contains 4 separate curves joined at the knots. 

# Confidence intervals, this time with "dashed" lines

lines(age.seq, 
      pred.x3$fit + 2*preds$se, 
      col = "red", 
      lty = "dashed")

lines(age.seq, 
      pred.x3$fit - 2*preds$se, 
      col = "red", 
      lty = "dashed")


## Natural Cubic Splines

library(ISLR) # Needed to acces the Wage data set
library(splines) # Needed to fit spline models

# Polynomial splines are notorious for having high variance at the beginning and tail ends of the data range, as exemplified in the plot above. A "Natural Spline" makes a simple correction to reduce this problem by forcing the first and last segments to a straight line, rather than a curve.

# The ns() function produces natural cubic splines (so there is no "degree" attribute with this function)

fit.ns <- lm(wage ~ ns(age, 
                       df = 4), 
             data = Wage) # Even spaced knots, or

fit.ns <- lm(wage ~ ns(age, 
                       knots = c(25, 40, 60)),
             data = Wage) # Specific knots

summary(fit.ns)

# Re-generate the age sequence for convenience

agelimits <- range(Wage$age) 
age.seq <- seq(from = agelimits[1], to = agelimits[2])

# Let's make predictions

pred.ns <- predict(fit.ns, 
                   list(age = age.seq), 
                   se = T) # Predicting

# Let's first superimpose a natural spline line in the current plot, so that you can compare the B-Spline and the N-Spline. They should be identical, except for the first and last segments in which the N-Spline fits a linear model

lines(age.seq, 
      pred.ns$fit, 
      col = "green", 
      lwd = 2)

# Now let's graph a clean plot for the natural spline

plot(Wage$age, 
     Wage$wage, 
     xlim = agelimits, 
     cex = 0.5, 
     col = "darkgrey")

title("Natural Cubic Spline")

lines(age.seq, 
      pred.ns$fit, 
      col = "blue", 
      lwd = 2)

# Confidence intervals

lines(age.seq,
      pred.ns$fit - 2*preds.ns$se, 
      col = "red",
      lwd = 1, 
      lty = "dashed")

lines(age.seq,
      pred.ns$fit + 2*preds.ns$se, 
      col = "red",
      lwd = 1, 
      lty = "dashed")

# Plotting all 3 spline models above in one graph

plot(Wage$age, 
     Wage$wage, 
     xlim = agelimits, 
     cex = 0.5, 
     col = "darkgrey")

lines(age.seq,
      pred.x1$fit, 
      col = "black" , 
      lwd = 2)

lines(age.seq,
      pred.x3$fit, 
      col = "blue" , 
      lwd = 2)

lines(age.seq,
      pred.ns$fit, 
      col = "green", 
      lwd = 2)

# All 3 lines are pretty close


# Comparing the 3 models with Cross-Validation

train <- sample(1:nrow(Wage), 0.7 * nrow(Wage))
test <- (!train)

test <- Wage[-train, ]

train
test

# Linear Spline

fit.x1.train <- lm(wage ~ bs(age, knots = c(25, 40, 60), 
                             degree = 1),
                   data = Wage[train,])

fit.x1.test.mse <- 
  mean((Wage$wage - predict(fit.x1.train, Wage))[-train] ^ 2) 

# Cubic B-Spline

fit.x3.bs.train <- lm(wage ~ bs(age, knots = c(25, 40, 60)), 
                      data = Wage[train, ])

fit.x3.bs.test.mse <- 
  mean((Wage$wage - predict(fit.x3.bs.train, Wage))[-train] ^ 2 ) 

# Cubic N-Spline

fit.x3.ns.train <- lm(wage ~ ns(age, knots = c(25, 40, 60)), 
                      data = Wage[train, ])

fit.x3.ns.test.mse <- 
  mean( (Wage$wage - predict(fit.x3.ns.train, Wage) )[-train]^2) 

mse.all <- c("MSE Linear Spline" = fit.x1.test.mse, 
             "MSE Cubic B-Spline" = fit.x3.bs.test.mse, 
             "MSE Cubic N-Spline" = fit.x3.ns.test.mse)

mse.all
