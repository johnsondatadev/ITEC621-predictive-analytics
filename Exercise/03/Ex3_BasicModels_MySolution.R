install.packages("tidyverse")
library(tidyverse)
library(ISLR)
options(scipen = 4)

?Hitters
head(Hitters, 10)
glimpse(Hitters)

Hitters <- na.omit(Hitters)

fit.ols <- lm(Salary ~ AtBat + Hits + Walks + PutOuts + Assists + HmRun, data = Hitters )

summary(fit.ols)

plot(fit.ols, which = 1)

library(lmtest)
bptest(fit.ols) # p = 0.01699 indicating heteroskedasticity

fitted.ols <- fitted(lm.test)
abs.res <- abs(residuals(fit.ols))

cbind(fitted.ols, abs.res)[1:10, ]

lm.abs.res <- lm(abs.res ~ fitted.ols)
fitted(lm.abs.res)[1:10]

plot(fitted.ols, abs.res)
abline(lm.abs.res, col = "red")

wts <- 1 / fitted(lm.abs.res) ^ 2
wts[1:10]

wls.fit <- lm(Salary ~ AtBat + Hits + Walks + 
                PutOuts + Assists + HmRun, 
              data = Hitters,
              weights = wts)

summary(wls.fit)

fit.wglm <- glm(Salary ~ AtBat + Hits + Walks + 
                  PutOuts + Assists + HmRun,
                data = Hitters,
                weights = wts,
)
summary(fit.wglm)

myopia <- read.table("../../Dataset/myopia.csv", header = T, row.names = 1, sep = ",")
head(myopia)

myopia.logit <- glm(myopic ~ age + female + sports.hrs + read.hrs + mommy + dadmy, family = "binomial"(link = "logit"), data = myopia)

summary(myopia.logit)

log.odds <- coef(myopia.logit)

odds <- exp(log.odds)

print(cbind("Log-Odds" = log.odds, "Odds" = odds), digits = 2)

library(tree)

fit.tree.salary <- tree(Salary ~ AtBat + Hits + Walks + 
                          PutOuts + Assists + HmRun, 
                        data = Hitters)

plot(fit.tree.salary)
text(fit.tree.salary, pretty = 0)
title("Baseball Salaries Regression Tree")

class(myopia$myopic)

myopia$myopic.f <- as.factor(myopia$myopic)

class(myopia$myopic.f)

fit.tree.myopia <- tree(myopic.f ~ age + female + sports.hrs + read.hrs + mommy + dadmy, data = myopia)

plot(fit.tree.myopia)
text(fit.tree.myopia, pretty = 0)
title("Myopia Classification Tree")

