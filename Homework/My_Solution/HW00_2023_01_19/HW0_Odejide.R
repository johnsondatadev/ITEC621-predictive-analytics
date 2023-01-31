library(tidyverse)


Area <- function(length, breadth){
  area <- length * breadth
  area
}

# Using the function
length <- 6
breadth <- 4
print(paste("The area of a rectangle of sides ", length, 
            "x", breadth, " is ", Area(length, breadth)))

# Question 2
for(i in 1:10){
  print(paste("The area of a rectangle of sides ", i, 
              "x", (i*2), " is ", Area(i, i*2)))
}
  
# Question 3
credit <- read.csv("Dataset/Credit.csv")
head(credit)
head(credit, 5)[, 1:5]

# Question 4
fit.rating <- lm(Rating ~ ., data = credit)
summary(fit.rating)

lm.rating <- lm(Rating ~ Income, data = credit)

plot(credit$Rating ~ credit$Income, xlab = "Income", ylab = "Credit Rating")
abline(lm.rating)

fit.rating.sig_preds <- lm(Rating ~ Income + Limit + Cards + Married + Balance, data = credit)
plot(fit.rating.sig_preds, which = 1)

summary(fit.rating.sig_preds)

glimpse(credit)

class(credit$Gender)

class(credit$Income)
class(credit$Cards)


qqnorm(credit$Rating)
qqline(credit$Rating)
