########################################
#       ITEC 621 Stats Refresher       #
########################################

# Filename: ITEC621_Stats.R
# Prepared by J. Alberto Espinosa
# Last updated on 1/17/2022

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: the material covered in this script ASSUMES that you have gone through the self-paced course KSB-999 R Overview for Business Analytics and are thoroughly familiar with the concepts illustrated in the RWorkshopScripts.R script. If you have not completed this workshop and script exercises, please stop here and do that first.

#######################################################
#                       INDEX                         #
#######################################################

## Descriptive Analytics

## Correlation
# rcorr(){Hmisc} Function
# corrplot() for Visual Correlation

## ANOVA
# aov(){stats} Function
# anova(){stats} Function

## Chi-Square Test of Independence

#######################################################
#                  END OF INDEX                       #
#######################################################


## Descriptive Analytics

# All predictive analytics work should start with very thorough descriptive analytics first, to become familiarized with the data at hand. The most basic descriptive analytics often performed before start building predictive models include:

# Visual inspection of the data, including things like scatter plots, histograms, QQ Plots, etc. Please refer to the ITEC621_RIntro.R script.

# Descriptive Statistics, including things like means, medians, standard deviations, minimum/maximum values, outliers, etc. Please refer to the ITEC621_RIntro.R script.

# Covariance Analysis: covariance is an important and foundational statistical concept to help understand if two variable co-vary in one direction or another, or not. But it is not practical when variables have dissimilar scales because results will change if you change the scale of a variable (e.g., from inches to feet). 

# Correlation Analysis (quantitative x quantitative):  Is the covariance of two variables, divided by the standard deviation of each of the two variables. Because we divide by the variance of each variable, all issues of scale go away and the correlation values are bound between -1 (perfectly negatively correlated) to 0 (uncorrelated or independent) to +1 (perfectly positively correlated). Because correlation is based on differences in values and variance, you can only compute correlation when both variables involved are quantitative. If one of the variables (or both) is (are) binary you can still compute the correlation statistic, but it is not as useful as ANOVA.

# ANOVA or Analysis of Variance (quantitative x categorical). If one of the variables is quantitative and the other is categorical or even binary, it is more useful to evaluate if they co-vary with an ANOVA test. ANOVA computes the mean for the quantitative variable for each of the categories of the other (categorical) variable and evalutes if these means vary significantly across the categories. It is called Analysis of "Variance" and not analysis of "Means" because the means are evaluated to see whether they vary more within each category (not significant or independent) than across categories (significant co-variation). I illustrate this more clearly below.

# Chi-Square Test of Independence (categorical x categorical). If you want to understand the co-variation between two categorical variables, correlation and ANOVA won't help. To evaluate this, the typical approach is to cross tabulate all the categories of one variable in rows and all the categories of the other variables as columns, with the cross-tabulated counts in the respective cells. If the two variables are independent, the proportion of counts between any two cells in a given column (or row) will be similar to the proportion of counts for the respective row (or column) totals. But if one variable has a significant influence on the other, the cell proportions will be very different than the row (or column proportions). I illustrate this more clearly below


## Correlation

#  (Quantitative x Quantitative)

# It is important to develop a good sense for which variables covary or not with others. Generally speaking, when building predictive models, the goal is to have predictors that are highly correlated with the outcome (i.e., dependent) variable, but are not correlated with each other (i.e., independent variables). Thus, correlation matrices provide very useful information when developing predictive models.

# Let's look at the diamonds data in the ggplot package:

require(ggplot2) # Contains the diamonds dataset
data(diamonds) # Load the dataset into the work environment
head(diamonds) # Take a look at the first 6 rows

# The cor(){stats} function provides quick correlation data. It requires a matrix as an input, so data frames need to be first converted into a matrix. For example, let's bind 3 variable vectors into a data frame:

MyDiamonds.dat <- data.frame(diamonds$price,diamonds$carat, diamonds$depth)
# Now let's convert the data frame into a matrix

MyDiamonds.mat <- as.matrix(MyDiamonds.dat)
head(MyDiamonds.mat)

# Alternatively, you can just specify the column numbers this way, all rows, columns 7, 1 and 5

MyDiamond.mat <- as.matrix(data.frame(diamonds[, c(7,1,5)]))
head(MyDiamond.mat)

# Now you can compute the correlation matrix for these 3 variables

cor(MyDiamonds.mat, use="complete.obs") # Discard rows with incomplete data
cor(MyDiamonds.mat, use="pairwise.complete.obs") # Discard pairs without data
# The results are the same in this case

# rcorr(){Hmisc} Function

# Unfortunately the cor() function only gives correlations, not p-values. For p-values, the rcorr() function in the "Hmisc" package does the job It returns the number of observations plus 2 matrices -> one with correlations and one with the respective p-values

library(Hmisc) # Note the H is upper case
# Note: rcorr() also requires a matrix as input
rcorr(MyDiamonds.mat, type="pearson")

# Another example with the mtcars{database} data set
rcorr(as.matrix(mtcars)) # The default is Pearson correlation

# You can display correlations (above the diagonal), distributions (diagonal) and scatterplots (below the diagonal) all together with GGally and ggpairs():

require(GGally) # Package with useful graphics displays

# WARNING -- the following takes a LONG TIME
ggpairs(MyDiamonds.dat) # It can be a bit slow

# WARNING -- the ggpairs() can take a VERY LONG time to run
ggpairs(diamonds) # Works with categorical data too

# The pairs() function from the base {graphics} package works well too. It works with both, matrices and data frames

pairs(MyDiamonds.dat) # Summary plot of all pairs of variables
pairs(diamonds) # With the full data set (takes a VERY LONG time)

# Example with the mtcars data:

head(mtcars)
rcorr(as.matrix(mtcars))
ggpairs(mtcars) # Takes a bit of time
pairs(mtcars) # A bit quicker, but not as nice

# corrplot() for Visual Correlation

library(corrplot) # Library for correlation plots
mtCorr <- cor(mtcars) # First, store the correlation object
corrplot(mtCorr, method = "number", order="hclust") # Show correlation
corrplot(mtCorr, method = "circle", order="hclust") # Then plot it
corrplot(mtCorr, method = "ellipse", order="hclust") # Slanted left/right for +/- 

# Order variables clustered by correlation values
# and omit the diagonal
corrplot(mtCorr, method="number", order="hclust", diag=F, title="MT Cars Correlation Matrix")
?corrplot # See all the methods

# Changing the annoying scientific notation (the "scipen" keyword stands for "scientific notation penalty"). 

options(scipen="4") # To limit displays in scientific notation
# i.e., if a value has more than 4 zeros after the decimal point it will be printed in scientific notation. 


## ANOVA

#  (Quantitative x Categorical)

# Analysis of Variance (ANOVA) is a test that compares the means of 2 or more groups or categories. The means of 2 groups may be different, but ANOVA tells us if this difference is significant. Generally speaking, ANOVA compares the within-category variance for both groups against the between-category variance across both groups. If the between-group variance is significantly larger than the between-group variance, then the difference in means between the two groups is significant. Otherwise it is not. 

# ANOVA is used any time you need to compare means between or across groups. For example, an OLS regression reports an R-Squared (i.e., explained variance) and a p-value for the entire regression model. The p-value is based on an ANOVA test, which analyzes whether the variance in the errors or residuals around the regression line are significantly smaller that the variance of the dependent variable values, relative to its overall mean. If the p-value or ANOVA test is significant, we say that the regression model has a significant explanatory power, relative to the plain mean of provides significantly more explanation than just the mean and its variance (i.e., the "null" model, with no predictors).

# There are 2 popular functions to do anova tests: aov() and anova(), both in {stats}

# The "aov()" function is useful to compare the means of a continuous variable (e.g., price) across various categories (e.g., clarity, color). In a nutshell, it tests whether the variance across groups is larger than within groups. The larger the F statistic, the more confidence we have that the variance across groups is significant

library(ggplot2) # Contains the diamonds dataset

options(scipen = "4") # To limit displays in scientific notation

aov(price ~ clarity, data = diamonds) # Run the ANOVA on a single factor

summary(aov(price ~ clarity, data = diamonds)) # More detailed results


# aov(){stats} Function

# Alternatively, store the ANOVA results in an object
MyAOV <- aov(price ~ clarity, data = diamonds)
summary(MyAOV) # Show the ANOVA object result summary

# You can do ANOVA for a continuous variable across more than one categorical variable

summary(aov(price ~ clarity + color, data = diamonds)) # ANOVA on 2 factors
summary(aov(price ~ clarity + color + cut, data = diamonds)) # On 3 factors

# You can also visualize the differences

boxplot(price ~ clarity, data = diamonds)
boxplot(price ~ color, data = diamonds)
boxplot(price ~ cut, data = diamonds)

# Replicating ISLR textbook example with the credit Default dataset

library(ISLR) # Contains the Default dataset
# Need to balance the default == "No"
def.yes <- subset(Default, default=="Yes")
def.no <- subset(Default, default=="No")
def.no <- def.no[sample(nrow(def.no), 333),] # Match size with defaults
def.sub <- rbind(def.yes, def.no) # Bind the defaults and no defaults
plot(income ~ balance, data=def.sub, col=ifelse(default=="No","lightblue","orange")) # Plot
boxplot(balance ~ default, data=def.sub, col=c("lightblue","orange"))
boxplot(income ~ default, data=def.sub, col=c("lightblue","orange"))

summary(aov(balance ~ default, data=def.sub))
summary(aov(income ~ default, data=def.sub))

# anova(){stats} Function

# The "anova()" function is useful when comparing nested linear models (i.e., one model is the subset of another):

library(ggplot2) # Contains the diamonds dataset

lm.null <- lm(price~1, data=diamonds) # Null model
lm.small <- lm(price~carat, data=diamonds) # Small model
lm.large <- lm(price~carat+clarity, data=diamonds) # Large model

anova(lm.null, lm.small, lm.large) # Compare 3 nested models

# p-value is significant -> lm.large has more predictive power than lm.small, and lm.small has more predictive power than lm.null. In other words, carats improve the predictive power of the null model, but adding "clarity" as a predictor improves the explanatory power of the small.


## Chi-Square Test of Independence

# Categorical x Categorical

library(ggplot2) # Contains the diamonds dataset
head(diamonds) # Take a look at the data
attach(diamonds) # For convenience to avoid using diamonds$

# For example, say that we want to see if diamond "cut" and "color" co-vary. In other words, we want to know if diamond cut and color are independent (i.e. one does not affect the value of the other) or if they are dependent (i.e., the value of one variable influences the value of another)

# The first step is to cross tabulate both variables

cross.table <- table (cut, color) # Store results in a cross table
cross.table # Check it out

rowSums(cross.table) # Check the row totals
colSums(cross.table) # Check the column totals

table.tot <- sum(rowSums(cross.table)) # Compute table totals
table.tot # Check the total for the whole table

sum(colSums(cross.table)) # Same result

# If the row and column variables are totally independent the proportion of rowSums would be identical to the proportion of any two cells on the same two rows of the table. Similarly, the proportion colSums would be identical to the proportion of any two cells on the same two columns. 

# In fact, you can create a table of expected values as follows: Exp.Cell(i,j) = rowSum(i) * colSum(j) / table.tot. If the observed (actual) values in the table are close to the expected values, the two variables are independent. If the observed values are very different than the expected values, the two variables are dependent (i.e., they co-vary)

# The Chi-Square Test of Independence does this test for you. The null hypothesis is that the two variables are independent. A significant p-value rejects the null hypothesis suggesting that the variables are dependent.

chiSq.diamonds <- chisq.test(cross.table)

chiSq.diamonds$observed # Check the observed values, or simply"
cross.table # Same thing

print(chiSq.diamonds$expected, digits=2) # Check the expected values

chiSq.diamonds # Check out Chi Square test results

# The p-value is significant -> cut and color are dependent. Let's take a look 

# Indeed, they appear to be different
