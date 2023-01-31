library(ggplot2)
library(MASS)
library(lmtest)
data("diamonds")

attach(diamonds)

head(diamonds)

cross.table <- table(cut, color)
# class(cross.table)

rowSums(cross.table)
colSums(cross.table)
sum(colSums(cross.table))
chiSq.diamonds <- chisq.test(cross.table)
chiSq.diamonds$observed

print(chiSq.diamonds$expected, digits = 2)


# Breush-Pagan Test for Heteroscedasticity
lm.ols <- lm(medv~., data = Boston)
bptest(lm.ols, data = Boston)
plot(lm.ols, which = 1)


ggpairs(data.frame(
  "Salary" = Salaries$salary,
  "Gender" = Salaries$sex,
  "Years Since PhD" = Salaries$yrs.since.phd,
  upper = list(combo = 'box')
))
