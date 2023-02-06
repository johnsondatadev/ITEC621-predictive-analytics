library(tidyverse)

hyp <- function(a, b){
  return(round(sqrt(a^2 + b^2), digits = 2))
}

a <- 7
b <- 10

print(paste("The hypotenuse of a triangle with sides", a, "and", b, "is", hyp(a, b)))

# 2
Pizza <- read.table("../../../Dataset/PizzaCal.csv", header = T, row.names = 1, sep = ",")

head(Pizza)

class(Pizza)

class(Pizza$cal)

class(Pizza$fat)

class(Pizza$brand)

Pizza.mat <- as.matrix(Pizza[, 3:9]) 

class(Pizza.mat)

head(Pizza.mat)

# 3 - Descriptive Statistics
summary(Pizza)

library(psych)
describe(Pizza)[3:9, 1:9]


## 4 - Correlation Analysis

### 4.1
Pizza.cor <- cor(Pizza.mat)

library(corrplot)
corrplot(Pizza.cor, method = "number", order = "hclust")
corrplot(Pizza.cor, method = "ellipse", order = "hclust")


### 4.2
# Two desirable predictors for cal would be `***mois***` and `***fat***`.
# As observed from the correlation plot, `mois` has a correlation of ***-0.76*** indicating a moderately strong _negative relationship_ with `cal` while `fat` has a correlation of ***0.76*** also, indicating a _positive relationship_ with `cal`. From the ellipse correlation plot, the darker the blue or red, the more correlated it is. The red indicates a negative correlation while the blue indicates a positive correlation. From the plot, it was observed that the darkest blue is `fat` while the darkest red is `mois`.

## 5 - Descriptive Analytics: Normality
par(mfrow = c(1, 2))

hist(Pizza$cal, main = "Calories Histogram", xlab = "Calories")

qqnorm(Pizza$cal, main = "Calories QQ Plot")
qqline(Pizza$cal)

hist(Pizza$fat, main = "Fat Histogram", xlab = "Fat")

qqnorm(Pizza$fat, main = "Fat QQ Plot")
qqline(Pizza$fat)

par(mfrow = c(1, 1))

# Calories appears to be normally distributed. There is a noticeable trace of bell shape on the histogram and the qq plot also justifies this. Most of the data are on the line of the qq plot with only a few number of deviation.

# Fat does not appear to be normally distributed. The histogram does not reveal a bell shape. Furthermore, the qq plot shows that just a few observations are on the line while most of the observations are off the line.


## 6.0 : Descriptive Analytics - Boxplots and ANOVA
### 6.1
par(mfrow = c(1, 2))

boxplot(fat~brand, data = Pizza)

boxplot(cal~brand, data = Pizza)

par(mfrow = c(1, 1))


### 6.2
aov.fat <- aov(fat ~ brand, data = Pizza) # ANOVA - fat by brand
aov.cal <- aov(cal ~ brand, data = Pizza) # ANOVA - cal by brand

# Get the summaries of the ANOVA table
summary(aov.fat)

# Specify a space in between the summaries.
cat("\n") # The function cat concatenates and prints strings and "\n" is the code for a new line.

summary(aov.cal)

### 6.3


## 7.0 : Simple Linear Regression Model

### 7.1

fit.simple <- lm(cal ~ fat, data = Pizza)
summary(fit.simple)

# The effect of fat is positive and significant (p-value < 0.0001) indicating that on average, an additional 1 gram per slice of fat is estimated to increase the calories by 5.28 calories per slice.


## 8.0 : Linear Regression Model with a Binary Predictor

### 8.1 
fit.dummy <- lm(cal ~ import + fat, data = Pizza)
summary(fit.dummy)


### 8.2


## 9.0 : Multivariate Linear Model
fit.full <- lm(cal ~ import + fat + carb + mois, data = Pizza)
summary(fit.full)


## 10: Residual Plots and Model Evaluation
plot(fit.full, which = 2)


anova(fit.simple, fit.dummy, fit.full)
