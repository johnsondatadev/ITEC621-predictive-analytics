#### ITEC 621 R Packages ####

# Prepared by J. Alberto Espinosa
# Last updated 1/1/2019

# To view vignettes available for various  pacakges:
browseVignettes()

# To view the vignette associated with a particular package you installed, enter:

browseVignettes("car") # Vignette for the package "car"

# Needed for R Markdown
install.packages("rmarkdown")

# Used at various parts in the course

install.packages("AppliedPredictiveModeling") # Package for that textbook
install.packages("car") # Companion to Applied Regression
install.packages("caret") # Classification and Regression Training
install.packages("coefplot") # Regression Coefficient Plots
install.packages("corrplot") # Graphical Display of Correlation Matrix
install.packages("ctv") # Task Viewing for Multiple Package Installations
install.packages("DataCombine") # Tools for Combining and Cleaning Data Sets
install.packages("dygraphs") # Interactive visualizations
install.packages("dplyr") # Data manipulartion
install.packages("expss") # Tables with labels, e.g., cross-tabs
install.packages("GGally") # Extends the ggplot packaage for graphics
install.packages("gbm") # Generalized Boosted Regression Models
install.packages("ggplot2") # Sophisticated Grammar of Graphics
install.packages("glmnet") # Ridge, LASSO and other GLM regressions
install.packages("HH") # Heiberger & Holland Various Statistical Tasks
install.packages("Hmisc") # Miscellaneous Data Analysis Functions
install.packages("ISLR") # Companion and data sets for ISLR textbook
install.packages("lattice") # Data visualization functions
install.packages("lm.beta") # Standardized regression coefficients
install.packages("lmtest") # Breusch-Pagan heteroskedasticity test and others
install.packages("leaps") # Search for best regression subsets
install.packages("MASS") # Modern Applied Statistics (and data sets) with S 
install.packages("neuralnet") # To train, plot and test neural networks
install.packages("olsrr") # Tools for OLS regression analysis
install.packages("perturb") # Collinearity statistics
install.packages("pls") # PLS and PCR regressions
install.packages("pROC") # ROC curves
install.packages("psych") # Psychology statistical proceures
install.packages("randomForest") # Bagged, Random Forest and other trees
install.packages("rjson") # For reading .json (Javascript Object Notation) files
install.packages("ROCR") # ROC curves
install.packages("tidyverse") # Set of packages for data work
install.packages("tree") # Classification and regression trees
install.packages("VGAM") # For multinomial logistic and other models
install.packages("brnn") # To train, plot and test Bayesian Regularized Neural Networks, required for {caret} 
install.packages("reticulate") # To run Python code inside R Studio
install.packages("RColorBrewer") # R color brewer
install.packages("wordcloud") # To render text word clouds

# Not covered in class, but covered in the ISLR book

install.packages("gam")
install.packages("akima")
install.packages("e1071")
install.packages("stargazer") # To format multiple regression models output

# Other packages, not used in ITEC 621, but useful

install.packages("BaylorEdPsych") # Package with useful data sets
install.packages("lavaan") # Package for structural equation models (SEM)
install.packages("MVN") # For plotting histograms
install.packages("plspm") # PLS for Path Modeling with SEM
install.packages("semPlot") # Plot SEM diagrams

# Packages for text analytics (not used in ITEC 621)

install.packages("irlba")
install.packages("quanteda")
install.packages("SnowballC")
install.packages("tm")

# Packages with datasets

install.packages("nycflights13") # 2013 flights originating in NYC

# Packages for reading data

install.packages("data.table")
install.packages("XML")
install.packages("xlsx")
install.packages("xlsxjars")
install.packages("rJava")

# Packages required for Social Network Analysis (SNA)

# Many of the libraries illustrated in SNA follow methods described in the book "A User Guide to Network Analysis in R" (UGNA), by Douglas A. Luke, Springer:

browseURL("https://www.springer.com/gp/book/9783319238821")

# The code illustrated in this book, along with some datasets and packages are available in Luke's github site:

install.packages("devtools") # Needed to install the github package
library(devtools) # Need to load this library for github installation next
install_github("DougLuke/UserNetR") # Textbook data sets

# Note: two important packages for SNA are {network} and {sna}, but these will be installed automatically when you install {statnet} below because of the dependencies among these packages.

install.packages("statnet", dependencies=TRUE) # For statistical modeling of network data
install.packages("igraph") # A comprehensive SNA library
install.packages("influenceR") # A library to quantify structural importance
install.packages("intergraph") # A library to work with network data objects
install.packages("igraphdata") # {igraph} data sets
install.packages("RColorBrewer") # A library to manipulate color palettes
install.packages("networkD3") # A package for interactive network graphs
install.packages("rgl") # For 3D network graphs
install.packages("ergm") # For ERGM regressions (will only cover lightly)
install.packages("amap") # For hierarchical clustering and k-means

# The {sand} package contains useful data sets and libraries used in the book "Statistical Analysis of Network Data" by Kolaczyk and Csárdi (the creator of {igraph}). If you are interested in using {igraph} proficiently I highly recommend their book:

browseURL("https://link.springer.com/book/10.1007/978-1-4939-0983-4")

install.packages("sand")


# While it is a good idea to install all the packages above so that you have them ready to go when you need them, you should only load the respective libraries when need them because they use up memory.

# These libraries contain useful data sets. To browse the data sets you first need to load the libraries and then use the data() function:

library(AppliedPredictiveModeling)
library(car)
library(caret)
library(DataCombine)
library(GGally)
library(ggplot2)
library(glmnet)
library(HH)
library(ISLR)
library(lattice)
library(lm.beta)
library(lmtest)
library(MASS)
library(perturb)
library(psych)
library(VGAM)
library(neuralnet)

# These libraries don't have data or the data are not interesting, but if you want to load them at once, here they are:

library(rmarkdown)
library(coefplot)
library(corrplot)
library(ctv)
library(expss)
library(gbm)
library(Hmisc)
library(leaps)
library(MASS)
library(perturb)
library(pls)
library(ROCR) 
library(pROC) 
library(tree)
library(stargazer)

# To view all available data sets in loaded libraries:
data()

# To view the data sets in one package, e.g., ISLR, type:
data(package="ISLR")

# To view all available data sets in all available packages:
data(package = .packages(all.available = TRUE))

# To inspect a data set, e.g., Boston, in a loaded library (e.g., ISLR) type:
?Boston

# To view a data set, load it and view it:
data(Boston)
View(Boston) # Cap V

# A few sites with data:

browseURL("https://catalog.data.gov/dataset")
browseURL("https://vincentarelbundock.github.io/Rdatasets/datasets.html")
browseURL("https://www.data.gov/")
browseURL("https://datasetsearch.research.google.com/")


