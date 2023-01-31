###########################
# ITEC R Workshop Scripts #
###########################

# Prepared by J. Alberto Espinosa
# Rev. 2.0
# Last updated on 12/1/2020

# This script is the intellectual property of Prof. J. Alberto Espinosa. It was prepared for an American University class at the Kogod School of Business. It's use is intended specifically for class work for the Kogod analytics students as a complement to class lectures and lab practice. Nobody, except the author of this script can use it for commercial purposes or share the script with individuals outside of the Kogod School of Business community.

# Quick Note: to run a script command place your cursor in that line and press Ctrl-Enter

## How to use the Index below

# The INDEX below is to help you navigate this script. To jump to a particular section, highlight the corresponding text in the index below, then hit Ctrl-F (the search/find shortcut) and you will see the text you selected in the search box at the top left of this screen. You can then press "Enter" or click "Next". Since I used the same titles in the Index and in the script below, the cursor will jump to that section.


#######################################################
#                       INDEX                         #
#######################################################

### Introduction

## Installing R and R Studio
## General Information About this Script
## General Information About R and R Studio
## R Overview
## R Packages
## R Studio Overview
## R Studio Packages
## Create a Working Directory and Project
## 3 Ways to Run R Commands
## Getting Help
## Comments
## Tutorials
## Books
## Useful Web Sites

### Object Orientation

## Object-Oriented Programming Fundamentals
## Objects, Classes, Methods, Properties and Inheritance
## R Objects
## Working with R Objects
## R Operators
## The R Assignment Operator
## Exploring R Objects

### Packages, Libraries and Views

## Packages
## Installing and Using Packages
## Libraries
## Vignettes
## Views
## Datasets

### R Markdown

## Open and follow RWorkshopRMarkdown.Rmd

### Working with Data

## Reading Data Into R
## Saving R Data to a File
## Getting Data from R Packages

### Variables and Data Types

## Variables
## Data Types
## Numeric Data
## Character Data
## Date Data
## Logical Data

### Data Structures

## Complex Data Structures
## Vectors
## Factor (Categorical) Variables
## Data Frames
## Matrices
## Lists

### Key Data Issues

## Data Re-Shaping
## Indexing 
## Sub-Sampling for Machine Learning & Cross Validation
## Missing Data

### Programming Fundamentals

## Built-In Functions
## User-Defined Functions
## The apply() Family of Functions
## Conditional Statements (if/else) 
## For and While Loops
## Error Handling and Debugging

### Visual Analytics

## Simple Graphics
## Analyzing Normality
## Exporting Graphics
## The {ggplot2} Library

### Working with Text

## Pasting and Concatenating
## Extracting Text
## Text Expressions

### Basic Descriptive Analytics

## Descriptive Statistics
## Covariance and Correlation
## Analysis of Variance (ANOVA)
## Chi-Square Test of Independence

### Simple Predictive Models

## Linear Regression
## Logistic Regression
## Decision Trees

### Introduction to Shiny

#######################################################
#                  END OF INDEX                       #
#######################################################


### Introduction


## Installing R and R Studio

# The R installation programs are stored in several servers called "mirrors" because the original R installation program is stored in a small server, but many sites copy and "mirror" the installation software, which makes it faster and more efficient. Simply, go to the CRAN site and select any mirror site. It is usually preferred to select a mirror site from a reputable institution in close geographic proximity to you. Select a mirror site and install the latest version of R:

browseURL("https://cran.r-project.org/mirrors.html")

# After installing R, install the free version of R Studio from this site:

browseURL("https://rstudio.com/products/rstudio/download/")


## General Information About this Script

# The material contained in this script is original for the most part, but in a few areas I use R code provided in two very useful books:

# - R for Everyone -- excellent book to learn R from scratch on your own. It not only has a lot of useful R tips, but also several statistical illustrations:

browseURL("https://www.amazon.com/dp/013454692X/")

# In addition, there are many web sites with excellent material to learn R proficiently. This is one of the best by Roger D. Peng from Johns Hopkins:

browseURL("https://bookdown.org/rdpeng/rprogdatascience/")

# You can also find some R programming and debugging tips in the Tech Notes area of Blackboard or Canvas, in a PDF document: ITEC621_TechNotes_R_CodingTips&Principles.pdf


## General Information About R and R Studio

# R is an open source software language specifically designed for statistical analysis and analytics. it is a dialect of an earlier software language called S. S was created in the Bell Labs by John Chambers and his team as an object oriented language to do data analysis, statistical modeling, simulation and graphics. R is a dialect of S. In other words, R is an improved version of S with very flexible and powerful capabilities to model problems and analyze data. R is different than S, but it was conceptualized as an improvement to S, so most S code will run fine in R. Not all R code will run in S. The name R comes from their two creators Ross Ihaka and Robert Gentleman (what a great name) from the University of Auckland, New Zealand.

# R is a free, "open source", "object-oriented" software programming language written specifically for data analysis. Not all open source software is free, but most are. Open source means that there is a community of volunteer developers with open access to the program source code, who develop and maintain the software. That is, it is not owned by a company. This is good, because most open source software is free, or available via subscription to its community. It can be problematic because there is no commercial entity responsible for providing support and managing the product, so you need to lean on the R community for help, tutorials, etc.

# The R community material is available at the CRAN site.

# Note: CRAN is the "Comprehensive R Archive Network", which has tons or R resources. It is located at: 
browseURL("http://cran.r-project.org/") 

# Also, this is a (Google) search engine custom taylored to R: 

browseURL("http://rseek.org")

# To search and use open source R code created by others, sign up for GitHub, then search for R software code - you can copy all you want, it's open source!!

browseURL("https://github.com/")


## R Overview

# A comment in the R code starts with a # which tells R that it is not an actual R command, but just some documentation text. As you will see in this script, I include tons of # comments to help you follow my scripts and methods. I encourage you to comment your work heavily. Sometimes we get inspired and write interesting software code, but it is almost a certainty that you will not remember why you did what you did 6 months or a year for now. Commenting heavily will payoff big time later.

# R is very popular for data science, machine learning and analytics for a number of good reasons. Being free is one of them. But the true power is in the R packages. There is an active community of R developers who are constantly writing packages. These packages are available in the CRAN repository and you can install them free of charge. A package will contain one or more libraries and/or data set. Each library will contain one or many useful function to do all kinds of quantitative and visual analytics. Some packages are well-written and well-behaved, others not so much. So stick to the well-known tested packages. There are literally thousands of R packages readily available for download, installation and use, with just about any possible statistical routine you can imagine. An R package must be installed once and should be updated every so often. A library must be loaded into the R work environment before it can be used. The library will be active and available to use until you close down R. When you re-start R, you need to re-load the library if you want to use it again -> Install once -> update when needes -> load each time you want to use it.

# For further information on R, see Lecture Slides. Also see:

browseURL("https://www.r-project.org/about.html")
browseURL("https://en.wikipedia.org/wiki/R_%28programming_language%29")

# For frequently asked questions on R, see:

browseURL("https://cran.r-project.org/doc/FAQ/R-FAQ.html")

# To get started with R, you can use this documentation from CRAN

browseURL("https://cran.r-project.org/doc/contrib/Paradis-rdebuts_en.pdf")
browseURL("https://cran.r-project.org/doc/manuals/R-intro.pdf")

# You can install R by downloading it from a "mirror" site. The R installation files are posted in several similar sites. You can pick any mirror site, but it is recommended to pick one that is in close proximity to your location. In my experience, it is best to pick a mirror site that is more reliable. The best mirror site in the USA (Berkeley) seems to work well.

browseURL("https://www.r-project.org/")


## R Packages

# The power of R rests on the thousands of packages already written for R, which are publicly available at CRAN. We discuss how to install packages and load their respective libraries a bit later. To view all R packages, see:

browseURL("https://cran.r-project.org/web/packages/available_packages_by_name.html")

# A package can contains one or many libraries and a library can contain one or many functions. When we install a package (e.g., {tree}), it gets stored in your hard drive. But in order to use the package you need to load the desired library (e.g., {tree}). Once the library is active in your R environment, you can use any of its functions (e.g., tree()). In this example, the package, the library and the function all have the same name, but in some cases the names are different, especially when a package contains multiple libraries and when libraries contain multiple functions.


## R Studio Overview

# You can run R and write code and develop R scripts directly in the R console. However, like with most software languages, it is best to use an "Integrated Development Environment" (IDE). An IDE gives you a nice environment and tools to make it easier to develop programs. Without question the IDE of choice for R is R Studio, which is also open source and free. Chances are, you are using R Studio right now.

# R Studio provides 4 different windows:

# 1. Top-Left: Script window -- this is where you write your R scripts

# 2. Bottom-Left: R Console -- this is where you can type R commands directly and where R displays results and messages when you run commands. If you were not using R Studio, you would only see this R Console.

# 3. Top-Right: Environment window -- this is where you can see any R objects created or opened (e.g., variables, data tables, etc.). Notice that there is also a History tab where you can review all the R commands you have run since you started R. This is a great tool when you are trying various commands in the R Console and you want to re-trace your steps or copy-paste a batch of commands from the History to a script.

# 4. Bottom-Right: Model explorer -- this is where you can view the files in the working directory, plots generated by R commands, packages available and installed, help displays, etc. 


## R Studio Packages

# R Studio has a number of very useful packages you can load to make your development environment more productive and have access to more functions and datasets. These include:

# - R Markdown: a package to create documents with marked up text, R code, R output, graphs, etc. 

# - knitr: a companion to R Markdown, which knits your markdown files into HTML, Word or PDF documents.

# - shiny: a package to build interactive R applications that run on the web.

# - ggplot2: one of the most popular and powerful packages to visualize data

# - tidyverse: a collection of R packages for data science

# - dplyr: powerful library to manipulate data

# - TensorFlow: library for deep learning models

# For a complete description of available R Studio Packages, see:

browseURL("https://www.rstudio.com/products/rpackages/")

# In addition, there are many R Studio open source packages written by third party developers

browseURL("https://rstudio.com/about/rstudio-open-source-packages/")


## Create a Working Directory and Project

# Before you start working in R, create a folder in your computer where you will keep all your R work for the class. For example, C:/RWorkshop/R.

# I STRONGLY suggest that you select a SINGLE FOLDER in  your computer for this class and that you keep all your scripts and data files for this class in that folder. Once you have a working folder in your Windows or Mac machine, you need to point your R environment to that folder so that R can find the scripts and datasets you need. Again, for now, I recommend having a single folder for all R scripts and datasets for this class. Once you become comfortable with R, you can organize your material some other way.

# You can do this 2 ways:

# 1. Set your R working directory pointing to your work folder, for example: 

setwd("C:/RWorkshop/R") # This command will only work if this folder exists

# If you have several working folders for different projects, you can use setwd() to switch to another environment. Also, to find out in which working directory you are presently, use this:

getwd() # To display the current working directory

# 2. Setup a project for each of your classes and for each R project you have

# This is BY FAR the best way to manage your working folders. When you create a Project, you will see it displayed at the top-right corner of R studio next to the R. If it says "none", STOP, and create your project folder before you continue. This will make it easier to keep the files for one project in one location, and also easier to switch back and forth from one project to another. More importantly, you can save your work environment when exit your R session and all the variables and objects you created before will be saved too.

# To create a new Project Environment, from the menu, select:

# File -> New Project

# Alternatively, and more easily, click on the pick list in the upper right corner of R Studio, and select New Project. It's that little down arrow at the far right.

# If you have already created a folder in your computer for this class, select "Existing Directory"(ff not, select "New Directory" to create the directory and the project in one scoop). Once you enter the working directory for this class it will show up in the project tool bar at the upper-right upper corner in the R Studio screen, along with any other projects you may have created. 


## 3 Ways to Run R Commands

# VERY IMPORTANT: BEWARE, R is Case Sensitive !!! lm() is different than LM(). Be sure to type upper or lower case as required.  You must type all commands and library names EXACTLY as they are spelled.

# You can run R commands in a number of ways, including: 

# 1. From the R Console below, just type the command, e.g., type in the console:

x <- 3 + 4 # then
x # to view the results

# 2. From the script, go to any script line and press Ctrl+Enter (PC) or Cmd+Enter (Mac). Try it:

lm.fit <- lm(mpg ~ cyl + disp + hp + wt, data=mtcars)

# Or, highlight a group of commands and press Ctrl+Enter (PC) or Cmd+Enter (Mac). Try it:

lm.fit <- lm(mpg ~ cyl + disp + hp + wt, data=mtcars)
summary(lm.fit)

# 3. Go to the History tab in the upper right window

# When working with R, it is better to keep all scripts and data in one directory. You can have different directories for different projects. Change your working directory to that directory with the "setwd" command (my files for this course are in the C:/AU/Courses/ITEC621/R directory:


## Getting Help

# In RStudio, the help system is located in the "Help" tab

# You can request help for a keyword with ??"<keyword>", e.g.:

??"regression"

# Use ? to get help for a given command, e.g.: 

?lm() # Help on the lm() function to build linear models

# If you need help on a particular function contained in a specific library, you need to load that library first, and then request help. For example, to get help on the ggplot() function you need to first load the {ggplot2} library:

library(ggplot2)
help(ggplot)

# Once you load a library, you can get help with specific data sets and functions in that library:

? diamonds # A data set with diamond data
? ggplot()

# One excellent resource to search anything in R is to go to the rseek.org web site, which is a search engine optimized for R help.

browseURL("http://rseek.org")


## Comments

# The # sign is use to write non-executable comments. Don't rely on your memory and document your scripts heavily. You'll thank me for it later. You can comment like I'm doing in this text, or you can simply add a sign after an R command to document that command. Everthing in the command line after the # sign will be ignored during the command execution.

# This can be used to suppress the execution of an R command without deleting the command (i.e., commented out). For example:

# library(ISLR)

# There are many great resources for learning R. There are two very good  video lectures in Lynda.com (access from the AU Portal:


## Tutorials

# AU Portal -> Technology -> LinkedIn Learning web-based training (requires AU Portal login):

# Up and Running with R: 

browseURL("https://www.linkedin.com/learning/learning-r/")
  
# R Statistics Essential Training:  

browseURL("https://www.linkedin.com/learning/r-statistics-essential-training/")
  
# Creating Interactive Presentations with R and Shinny:
  
browseURL("https://www.linkedin.com/learning/creating-interactive-presentations-with-shiny-and-r/")
  
# Many more:  

browseURL("https://www.linkedin.com/learning/")

# Another great way to learn R is with the Datacamp tutorials:
browseURL("https://www.datacamp.com/")

# For example, once you are comfortable with R you can easily teach Python yourself with a DataCamp class:

browseURL("https://www.datacamp.com/onboarding/learn?technology=python")


## Books

# A fair amount of the material in this section comes from: 
# "R for Everyone", by Jared P. Lander, Addison Wesley

browseURL("http://www.jaredlander.com/r-for-everyone/")
# This is a great starting book to learn R

# This is another good book on R: "R Cookbook"

browseURL("http://cdn.oreillystatic.com/oreilly/booksamplers/9780596809157_sampler.pdf")


## Useful Web Sites

browseURL("http://www.statmethods.net/")
browseURL("https://support.rstudio.com/hc/en-us")
browseURL("https://bookdown.org/rdpeng/rprogdatascience/")


### Object Orientation


# What is "object-Oriented" software? 

# Most modern software languages are object-oriented. All it means is that EVERYTHING you do in R (and any other object-oriented language) gets stored in containers called "objects". Object-oriented  software is a paradigm in which a program is developed to manipulate behavior and data of digital objects, which are representations of real objects in the physical world. 


## Objects, Classes, Methods, Properties and Inheritance

# Objects

# An object has two things: 

# (1) properties -- i.e., data; and/or 
# (2) behaviors -- i.e., programs or functions that manipulate object data 

# When you create an object, you can then access it to run its programs/functions and read or manipulate its data. Objects can be used to create other objects using a property of object-orientation called "inheritance", such that the data and programs from one object are inherited by another.

# Classes

# A Class is a like a container of objects. All objects belong to a particular class. For example:

lm.fit <- lm(mpg ~ cyl + disp + hp + wt, data=mtcars) 

# Notice that the command above does not diaplay anything. It simply creates an object of class "lm", which we chose to name "lm.fit". We will discuss how to display the contents of this object shortly.

# lm() is a function that fits linear models (regressions) and is available in the {stats} package (note: it is customary to enclose package names in curly brackets), which loads automatically when you start R. It will fit a linear model to predict vehicles mileage or mpg (miles per gallon) using cylinders, displacement, horsepower and weight as predictors. The model is fit using the mtcars dataset available in the {datasets} package, which also loads automatically when you start R. 

# The results of this regression are stored in an "lm" object (i.e., the function that created it), which we have chosen to name "lm.fit". We could have used any name for the object. 

# Once we create an object, you can display the class it belongs to:

class(lm.fit) # It is an "lm" object

# If we type a long model formula that we may use several times, it is probably easier to create a "formula" object, which in this case I am naming "lm.formula"

lm.formula <- mpg ~ cyl + disp + hp + wt # Create the formula object
lm.formula # Display it's content

class(lm.formula) # Show it's class. It is class "formula"

# We could fit the model above like this instead:

lm.fit <- lm(lm.formula, data=mtcars) 

# The lm.fit object above, and all objects in general, have "properties" and "methods". Properties are the data contained in the objects. Properties in objects have names, so that you can access their contents. Methods are programs or functions written inside the object, which can be used to manipulate the properties (i.e., data) in the object.

# For example "lm()" is a function or "method". But internally, it is an class that contains methods to create lm objects. Similarly, the object lm.fit we created above has a lot of useful data (i.e., properties).


## R Objects

# Anything you create in R is stored in an object. For example, you can use the plot() function to view useful graphs with lm.fit data:

plot(lm.fit, which=1) # plot() yields 4 residual graphs, this is the first one

# If you wish to use the plot for any other purpose, you can store it in an object:

my.plot <- plot(lm.fit, which=1) # Store it in an object named my.plot
my.plot # use it later

# This is very useful because in many analytics packages, we create an object, which we then feed into a different function to do something else.


## Working with R Objects

# Inheritance: is one of the most important aspects of object behavior. You can use a class (e.g., lm) to create objects (e.g., lm.fit). When you do, the object "inherits all the properties and methods of the class from which it was created. That is, all object of a class will behave identically and will contain the same data attributes (i.e., their names and types, but the actual data values will be different for each object). Objects can be used to create other objects too. The objects created will also inherit all the properties and methods from the object that created them.

# In R Studio, the content of the workspace is shown in the Environment tab (top-right). Its contents can also be displayed in the console with the ls() and objects() commands 

ls()
objects()

# To delete an object, we pass it as parameter to the rm() function (for remove) by writing it in the brackets of the command: 

lm.nothing <- lm(mpg ~ 1, data=mtcars) # Create an object
ls() # Verify it was created
rm(lm.nothing) # Now remove it
ls() # Verify it was removed


## R Operators

# There are many operators used in R to manipulate data and assign content to objects, including:

# Arithmetic - for computations: + add, - subtract, * multiply, /divide, ^ or ** exponentiation

4 + 3 - 1

# Logical - to make comparisons: < less than, <= less than or equal, > greater than, >= greater than or equal, == exactly equal to, != not equal to

3 == 4 # Is 3 equal to 4?
4 != 3 # Is 3 not equal (different than) 4?
3 > 4 # Is 3 greater than 4?
4 > 3 # Is 4 greater than 3?
4 >= 4 # Is 4 greater than or equal to 4? 

# When you want to evaluate if something is equal to something, R requires that you use the == operator. The single = operator is used for assignments and other things, so == eliminates any confusion:

ifelse(3 == 4, "Right", "Wrong") # Wrong
ifelse(3 == 3, "Right", "Wrong") # Right
ifelse(3 = 3, "Right", "Wrong") # Error message, can't use a single =


## The R Assignment Operator

# The assignment is a most important operator in R. It is used to assign content to objects and to store function results in objects. This content could be data from another objects, the result of an R function, etc. 

# You can use the <-, ->, and = operators to make these assignments.

# For example:

x <- 3 + 4 # Look at the value of x in the Environment window
x # Check it out the results

x = 3 + 4 # The same assignment with the = operator
x # Check it out

3 + 4 -> x # The same assignment in the opposite direction
x # Check it out

# Assign lm() results to lm.object

lm.object <- lm(mpg ~ cyl + disp + hp + wt, data=mtcars) 
summary(lm.object) # Check it out

lm.object = lm(mpg ~ cyl + disp + hp + wt, data=mtcars)  # Same thing
summary(lm.object) # Check it out

lm(mpg ~ cyl + disp + hp + wt, data=mtcars) -> lm.object # Same thing
summary(lm.object) # Check it out

# Please note that x <- 3 + 4 is DIFFERENT than x < - 3 + 4. The first entry assigns 3+4 to the variable object x. In the second entry we are evaluating if the object x is smaller than -3+4. One space shift can make a huge difference. So, please ensure that when you use the assignment operator <- there is no space between the < and the -


## Exploring R Objects

# We can explore the data inside an object in many ways, depending on the level of detail desired:

# 1. Object Name: simply type the name of the object in a script line or the command window, which will give you some of the properties in the respective objects:

lm.formula # A little information on the contents of lm.formula
lm.fit # A little information on the contents of lm.fit

# 2. Object class()

class(lm.formula) # Show it's class. It is class "formula"
class(lm.fit) # Show it's class. It is class "formula"

# 2. summary(): If we need more information than this, we can use the summary() function. The summary() function is a convenient way to view a summary of important data stored in an object. The summary() output will look different depending on which object you are displaying. For example: 

summary(lm.formula) # Some summary information on the formula object
summary(lm.fit) # Typical results in a regression model output

# If you want to store the summary() results for later use simply assign it to an object:

lm.summary <- summary(lm.fit)
class(lm.summary)
lm.summary

# Polymorphism: Formulas in R are "polymorphic". This simply means that the formula will yield different results for objects of different classes. Fore example, the summaries for the mtcars data set and the lm.fit object yield very different results:

summary(mtcars) # Summary statistics for the data set
summary(lm.fit) # Summary results for the linear regression model

? summary # Generic help for summary() function
class(lm.fit) # lm.fit is an "lm" object
? summary.lm # Specific helf for summary with "lm" objects

# 3. str(): If you want to see the entire contents of an object, you can use the str() function, which displays the entire structure of an object. str() is invaluable when you need to inspect an object in depth.

str(lm.formula) # All the data contained in lm.formula
str(lm.fit) # All the data contained in lm.fit

# Notice in the display in the str(lm.fit) results above that there are a few values prefixed with $. To access specific data elements from an object we can use the $ symbol. The $ symbol is used to point to variables (i.e., named attributes) or columns in data objects.

# For example, these show the regression model's coefficients (i.e., Effects)

lm.fit$coefficients

# This displays the model's residuals (i.e., regression errors):

lm.fit$residuals

# This displays the model's predicted (i.e., fitted) values:

lm.fit$fitted.values

# ASIDE: you can use the options() function to change how R reports results to some extent. The "scipen" attribute is useful to convert scientific notation (e.g., 1.8e-14) to actual numbers (e.g., the number with 14 zeros after the decimal point). Try it

options(scipen="20") # scipen = Scientify Notation Penalty
summary(lm.fit) # Check it out

# The scipen (i.e., scientific penalty) value tells R to display the value in scientific if there are more than 20 zeros after the decimal point. Now try

options(scipen="4")
summary(lm.fit) # See the difference


### Packages, Libraries and Views


## Packages

# You can write lots of very powerful R Scripts, which is great, but the real power of R is in the thousands of R packages that others have already written, which are public and free.

# The {base} and {stat} Packages

# R comes with the pre-installed packages {base} and {stat} with lots of useful libraries. These packages are installed automatically with R and they are also loaded automatically in the R environment when you start R. So there is no need to install them or load them. You can learn more about the {base} package at:

? base # General description
library(help="base") # Functions
browseURL("https://stat.ethz.ch/R-manual/R-devel/library/base/html/00Index.html")

# You can learn more about the {stat} package at:

? stats
library(help="stats") # Functions
browseURL("https://stat.ethz.ch/R-manual/R-devel/library/stats/html/00Index.html")

# Packages can contain datasets. The {datasets} package is also installed automatically with R and it loads automatically when you start R and it contain several useful datasets students can use to practice. See:

? datasets
library(help="datasets") # Functions
browseURL("https://stat.ethz.ch/R-manual/R-patched/library/datasets/html/00Index.html")

# All other packages need to be installed before you can use them. You can find all available packages in the CRAN web site:

browseURL("https://cran.r-project.org/web/packages/available_packages_by_name.html")


## Installing and Using Packages

# Using a package is a 2 step process: 

# 1. ONE-TIME STEP -- Package Installation: You first need to install the package. You only need to do this once and the package will remain installed until you uninstall it. This is similar to any software installation in your computer (e.g., before you can use Excel, you need to install it in your computer). From time to time, you may want to check to see if a package was updated. To update all packages enter update.packages().

# To install a package type "install.packages("packagename"). Note that the package name needs to be enclosed in quotes.

# To install all the packages needed for this course, open the script "RWorkshopPackages.R" and execute all the install.packages() commands in the section labeled:

# Used at various parts in the course

# You can install all the other packages in the script. You may need them at one point or another.

# To update a specific package type update.packages("packagename"). 

# To remove an installed package type remove.packages("packagename")

# To list all installed packages enter:

installed.packages()

# To view just the package names (column 1)

installed.packages()[,1]

# Then get help on that package
help(package="base")

# 2. WHEN YOU START R -- To use a library contained in a package, you need to "load" the library into your R environment. Most libraries have the same name as the package, but some packages have more than one library. Once you load a library, it will remain open until you quit your R session. So, you only need to load the library once in your R session, but you will need to re-load it if you quit and re-start R. More on this in the ## Libraries section

# Also, note that some packages have dependencies (i.e., a given package needs other packages to work). In this case, all the dependent packages also get installed automatically.

# Review the help documentation for these functions:

# You can also install packages by clicking on the Packages tab in the model viewer, clicking Install, and then following the steps. Similarly, you can also activate packages by checking the corresponding box in the Packages tab.


## Libraries

# Many packages contain only one library, which usually has the same name as the package. But there are also many packages that contain more than one library or they contain a library with a different name than the package.

# As I mentioned above, libraries need to be loaded into your R environment before you can use them. Once you load a library, it will remain loaded until you exit R. Once you re-start R you will need to load the library again if you need to use it. 

# To load a library enter library(libraryname) or require(libraryname). Note that, in contrast to install.packages() you should NOT use quotes in either of these two functions. These two functions are identical. There is only one minor difference between the two, but it only has programming implications that won't affect what we do in this course.

# For example, once you install the {ggplot2} package, you can load it to start using it. You can use any of these two commands to load a library into your work environment:
  
library(ggplot2) # or
require(ggplot2)

# Once you have loaded the package, you can get some (limited) help on the library:

? ggplot2
help(package="ggplot2")

# YOu can also see the datasets included in the library

data(package="ggplot2")

# As I mentioned above, the {base}, {stat} and {datasets} are loaded automatically every time you start R. Any other library needs to be loaded explicitly.

# To list available libraries installed in your computer enter:

library()

# Packages get unloaded automatically when you exit R. But you may sometimes need to deactivate or unload packages. You can do this by unchecking the corresponding box in the Packages pane or entering this command (this does not uninstall the package, it just unloads it from the work environment):

detach("packages:PackageName")


## Vignettes -- Important Documentation Supplements

# You will quickly learn that the R documentation on packages and functions can be cryptic and often incomplete. This documentation is written by package developers and they often include just the minimal information requiried to understand the package contents. More complete documentation and code examples are usually provided in additional documents called "vignettes", which are supplementary and optional documentation for packages and functions. Some packages have vignettes and some do not. To find out the vignettes in your installed packages"

browseVignettes()

# To view the vignette associated with a particular package you installed, enter:

browseVignettes("car") # Vignette for the package "car"


## Views

# Views are collections of packages by function or type of analysis. You can view the various packages that are contained in a view at

browseURL("http://cran.r-project.org/web/views/")

# Once you find a view and wish to install the packages it contains, installing the view automatically installs all the packages it contains, if not already installed. 

# In order to install views, you load the CRAN Task View "ctv" library:

library("ctv")

# Then, for example, if you want to install the "Graphics" view, use:

install.views("Graphics") # which will install all these packages:
browseURL("https://cran.r-project.org/web/views/Graphics.html")


## Datasets

# Packages can also contain datasets. As I mentioned above, the {datasets} package is automatically available when you load R.

# To view the data sets available in all the libraries loaded in your R environment enter:

data()

# To view the data sets in a particularly active package enter:

data(package="ISLR") # Example for the ISLR package

# to list the datasets in all packages available, enter:

data(package = .packages(all.available = TRUE))


### R Markdown

# R Markdown is an R Studio package that allows you to output the results of your script to a Word, PDF, HTML, Shinny or PowerPoint documents. There are many other formats available for the output. You can control the output to include or suppress text, formatting, R code, R output results, etc. R Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. See:

browseURL("http://rmarkdown.rstudio.com/")

# Before you can use R Markdown you need to install and load the "rmarkdown" package, if you haven't done this already

# You need to install.packages("rmarkdown") if you haven't yet

library(rmarkdown)

# Once R Markdown is installed, you can create R Markdown files with File -> New File -> R Markdown -> Document. Enter the document title and author and select HTML (R Markdown can create HTML, Word and PDF files; we will be working with HTML files only)


## Open and follow RWorkshopRMarkdown.Rmd

# Open the script and continue this part of the lecture in that R Markdown file.


### Working with Data


## Reading Data Into R

# There are different ways to read data into R. Data comes in a variety of formats, so you need to use the appropriate function to read a data table. Also, you may need to specify some parameters to indicate if you want to read things like column headers, row names, etc.

# The most common file format is the .csv (comma separated value) file, which has one line with the data for the first row of the table, with each of the respective column values separated by a comma. You can open any .csv file in Notepad to see how the file is stored internally. But you can also open .csv files in Excel to view the data as a table, which is a very useful thing to do when reading data to ensure that the data was read properly.

# As the name .csv implies, the data is separated by commas. These commas are called "delimiters", but actually, csv files may be delimited with semi-colons, tabs, fixed blank spaces, or any other character. However most files are delimited with commas (sep=",").

# .csv files are very popular for data files because just about any program can export/import a proprietary data file into/from a .csv file. 

# In many cases, the names of the columns, or headers, are stored in the first line of the .csv file. If the .csv file has a header you need to indicate it (header=T)

# In some cases, the names of the rows are stored in the first column the .csv file. If these names are just name and not data that you need for the analysis, you need to indicate it too (row.names=1).

# The read.table() and read.csv() are popular functions to read .csv data files into R datasets. These 2 functions are very similar, but they have different defaults:

# read.table() 

# This function uses a blank as the default separator (sep=""), so you need to indicate sep="," when the delimiter is a comma (most often). Also, the header default is no header (header=F), so you need to change this to header=T if your .csv file has column headers. Finally, text data is read as "text strings". If you need to read text data as "factors" (i.e., categories, more on this later), you need to specify stringsAsFactors=T.

# read.csv()

# This function uses a comma  as the default separator (sep=","), which you can change as needed. Also, the header default is true (header=T), so you need to change this to header=F if your .csv file does not have column headers. Finally, text data is read as "factors" (more on this later). Factors are categories, which can be used in most models. If you need to read text data as plain text, you need to specify stringsAsFactors=F.

# Let's work with the South Africa heart disease data set, Heart.csv, provided by Prof. Tibshirani from Stanford University. Take a look at the documentation of the file:

browseURL("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.info.txt")

# A .csv file is a text file, so let's inspect Heart.csv with a text editor like Notepad.

# Notice how the data has a header row and is delimited by commas. Notice the text "Present" and "Absent", which are two categories denoting the presence or absence of family history of heart disease. These categories are what we call "factors" because they are not plain text strings, but are "categories" that get repeated in various lines. There are only 2 categories in the "famhist" column. 

# Now open the same file in Excel so that you can view it as a table.

# Now, let's read the data into our R environment using a different methods:

heart <- read.table("Heart.csv", header=T, sep=",")
summary(heart) # Note that famhist was read as a text string (characters)

heart <- read.table("Heart.csv", header=T, sep=",", stringsAsFactors=T)
summary(heart) # Note that famhist was read as factors, with 2 levels 

heartcsv <- read.csv("Heart.csv") # Didn't need to specify the header or sep
summary(heartcsv) # Note that famhist was read as a text string (characters)

heartcsv <- read.csv("Heart.csv", stringsAsFactors=T )
summary(heartcsv)

# If the "csv" file is not located in the working directory, you need to specify the directory using forward slashes "/" (don't use the backslashes "\"). Ex:

mydata <- read.table("c:/au/courses/RWorkshop/R/Heart.csv", header=T, sep=",")

# You can read more about how to read .csv data file here:
browseURL("http://uc-r.github.io/import_text_files")

# VERY IMPORTANT: 

# Different versions of R in different computers and operating systems (i.e., Windows or Mac) may work differently when reading external data, especially .CSV files. I strongly recommend that you develop familiarity with one of the functions above. I use read.table() most of the time but some students are more comfortable with read.csv(). I also strongly recommend to open the data file in Excel before you import it to inspect visually the data and column headers, and then open the data set once imported into R, so that you become familiar with the actual data set structure and contents in your R environment. 

# Reading other data types

# Also, there are a number of packages that let you read from and write to various file types (e.g., Excel, SPSS, SAS, etc.). My experience with these packages has been that these file formats change over time and what works one day may not work another day. In contrast, .CSV files work all the time. 

# If you need to write programs to exchange data with these files, you can search for the various packages available for this purpose, such as {readr}, {readxl} for Excel, {jsonlite} for JSON data files, {XML} for XML data, etc.

# You can read more about the various packages to work with data at:

browseURL("https://www.computerworld.com/article/2921176/great-r-packages-for-data-import-wrangling-visualization.html")


## Saving R Data to a File

# In many instances you will read a data file into an R dataset object and do some manipulations. These manipulations will modify the dataset you are working with and if you save your environment when you exit R (you can enable or disable this in the Tools -> Global Options menu), the changes will be there when you re-open R. If you don't save your R environment when you exit or if you clear your environment by clicking on the little broom icon on the upper right, then all data manipulations will be gone and the next time you load the dataset it will be exactly like the source dataset. If you want to save your data changes to a file permanently, you need to do it explicitly. There are several ways to do this:

# Saving a data frame to a "csv" file

# The write.table() function works exactly like read.table(), except that it saves an R data frame to a .CSV file. For example:

write.table(mydata, "mydata.csv", sep=",") # For CSV files

# The "sep=" parameter specifies the delimiter between values, with "," being the most common and one the Excel will read. Or,

write.table(mydata, "mydata.csv", sep=",", header=T) # writes column headers in first row


## Getting Data from r Packages

# Many R packages come with data sets. Most of these data sets are dated and not very useful, but they are great to test R scripts and practice. To access a data set in a package, all you have to do is load the respective library (assuming that the corresponding package is already installed).

# To list the data sets in all active (loaded) libraries in your R environment enter:

data()

# This function will list all data sets in the {datasets} package that loads automatically with R, plus data sets from any other package you may have loaded. For example,

library(carData)

# You should now be able to see at least all data sets in the {datasets} and {carData} packages

# To list the datasets contained in a particular package, for example {carData} enter:

data(package="carData")

# to list the datasets in all packages available, enter:

data(package = .packages(all.available = TRUE))

# To use a data set in a loaded package, you don't need to do anything else, just use it. It will be in your R environment ready to be used.

# For example, the mtcars data set in the {datasets} package is always availeble. You can inspect it like this:

? mtcars # Documentation on the mtcars data set

head(mtcars) # First 6 rows (i.e., head) of the data set
head(mtcars, n=10) # Change the default to 10 rows

tail(mtcars) # Last 6 rows (i.e., taile) of the data set
tail(mtcars, n=10) # Change the default to 10 rows

# You can also double-click on the data set in the Environment tab on the upper-right window to view the full table. 

# You can also view and edit the full table with the fix() function (be careful not to change the data unless you want to):

fix(mtcars)

# While you can read multiple data sources into R (such as SPSS, SAS and Excel), I find it a lot easier to manipulate data sets in Excel and then save the datasheet as a "comma separated values" or "CSV" file. For example, many of the packages to read Excel data into R require current versions of Java or certain versions of Excel, whereas CSV files work universally.

# The which argument specifies which table to read if there are more than 1 Specify header=TRUE if the table has a header


### Variables and Data Types


## Variables

# R, like all statistical software, uses many different data/variable types and classes. Understanding how to work with different data types and classes is key to understanding how to build and interpret models in R.

# Think of a variable as a "named" data container. For example x <- 2 is like having a container named x and it contains the number 2. x <- x + 1 takes the value stored in x, adds 1 and stores it back in the same container x. 

# A variable is of a certain "type" and a certain "class". These are two different things.

# The variable's type refers to the type of data contained in the variable (e.g., character) will contain data of the same type.

# A variable's class refers to the type of object of the variable. A "scalar" variable is a variable that contains one value at a time. A scalar variable type is closely related to its class. Later we will discuss "complex data structures" (e.g., tables) in which the variable's type is less relevant and very different than the variable's class.

# Scalar Variables

x = 2.3
class(x) # The class is numeric (i.e., a numeric object)
typeof(x) # The type is more specific -- double (i.e., double precision w/decimals)

# Variable Names

# Variable names start with a letter, but you can use any character after that, including a period (.). In other software languages a period (.) in the name has a meaning, but in R is just part of the name.

#  Periods are often use to make variable names more readable. For example, productunitprice is very hard to read, so we would often name the variable product.unit.price to make it easier to visualize the different words in a variable name.

# VERY VERY IMPORTANT: R is case sensitive. A variable named price is different than a variable named Price. These would be two different variables. Errors in the case of a variable name are the MOST COMMON syntax errors in R. When nothing works, check the case spelling of your variable and object names. 

# Upper and lower case are often used to improve readability. For example, productunitprice is very hard to read, so we would often name the variable ProductUnitPrice to make it easier to visualize the different words in a variable name.


## Data Types

# Scalar variables are of a given type, such as: numeric, character, factor, date and logical. In some software languages you need to explicitly declare the type of a variable before you can use it, but not in R. The variable type is implicitly (automatically) declared when a value is stored in the variable. Certain data types require delimiters, such as "double quotes" for character (i.e., text) data. It is often useful to check or display a particular variable data type:

x <- 3.2
typeof(x)

y <- "R Workshop"
typeof(y)


## Numeric Data

x <- 2.4 # Declares x as numeric amd stprs 2.4 in it
x # Check it out
typeof(x) # Double precision number
class(x) # Numeric object

y=as.integer(2) # Integer (no decimals) 
y # Check it out
typeof(y) # Integer variable
class(y) # Integer object

# We can always evaluate the variable type

is.numeric(x) # Check if x is numeric -> TRUE
is.numeric(y) # Check if y is numeric -> TRUE
is.integer(x) # Check if x is an integer -> FALSE
is.integer(y) # Check if y is an integer -> TRUE

# Operations on numeric data

x <- 2 # Assigns the value of 2 to a variable named x
x # Check it out
x <- x + 1 # Adds 1 to the value of x and stores the result back in x
x # Check it out


## Character Data

# Note: character data, like variable names, are case sensitive

x="Alberto"
x # Check it out

typeof(x) # Character variable
class(x) # Character object

# We can always evaluate the variable type

is.numeric(x) # Check if x is numeric -> FALSE
is.character(x) # Check if x is numeric -> TRUE

# There are certaing things we can do with character that we can't do with numeric data and the other way around

x <- x + 1 # Error, you can't add numbers to character data

nchar(x) # Count the number of characters in the value stored in x
nchar("Alberto") # Count the number of characters in the literal text "Alberto"

# I discuss working with text data in more depth later, but there are a few basic things you may want to know about useful data functions:

x = "Alberto"
y = "Espinosa"
name = paste(x,y) # Concatenates x and y with a blank space in between
name # Check it out
name = paste(y,x, sep=", ") # Use , and blank instead of a blank separator
name # Check it out


## Date Data

x=as.Date("2016-10-20") # Converts a text string into a date value
# Note that the word "Date" is capitalized in the function above
x # Check it out

# Also note the date format is YYYY/MM/DD, which can be changed as follows:
x=as.Date("10/20/2016", "%m/%d/%Y") # Use upper case %Y for 4-digit years
x # Check it out

x=as.Date("10/20/16", "%m/%d/%Y") # Reads the year as 16, not as 2016
x # Check it out

x=as.Date("10/20/16", "%m/%d/%y") # Reads 16 as 2016 with lower case %y
x # Check it out

typeof(x) # Dates are stored internally as numbers
class(x) # But a date variable contains a date object

# Note: you can use other date formats with: %a abbreviated weekday; %A weekday; %b abbreviated month; %B month.

# Also, you can use the as.POSIXct() function to read date and time:

x=as.POSIXct("2016-10-20 17:30") # Note that a space is needed after :
x # Check it out

# You can subtract dates and add days to a date: 
born <- as.Date("2000/2/12")
born # Check it out

today <- Sys.Date() # Today's date
today # Check it out

aWeekAgo <- today - 7 # Go back in time
aWeekAgo

aWeekFromNow <- today + 7 # Into the future
aWeekFromNow

MyAgeInDays <- as.numeric(today - born) # Days between today and born dates
MyAgeInDays

MyAge=floor(MyAgeInDays / 365) # The floor() function rounds down decimals
MyAge


## Logical Data

# Logical variables can be either TRUE or FALSE. Numerically, TRUE is identical to 1 and FALSE to 0. TRUE and FALSE values are upper case and can be abbreviated at T and F.  Logical variables and values are important to evaluate conditions

TodayIsMyBirthday <- F # Can use F instead of FALSE
typeof(TodayIsMyBirthday) # Logical variable type
class(TodayIsMyBirthday) # Logical data object

# You can evaluate variables this way

is.logical(TodayIsMyBirthday) # Evaluate if the variable is logical
2 == 3 # Evaluate if two values are equal (double == is required) -> FALSE
2 != 3 # Evaluate if two values are different -> TRUE
2 < 3 # Evaluate if 2 is smaller than 3 -> TRUE
2 <= 3 # Evaluate if 2 is smaller than or equal to 3 -> TRUE


### Data Structures


## Complex Data Structures

# R is very rich on data structures. The scalar variables and data types discussed above are simple data structures that are used for variables containing a single value. But there are more complex data structures in R (like with other software languages) like vectors, matrices, data frames and lists. These complex structures give R a lot of power for data manipulation. Understanding how these data structures work is key to unleashing the power of R for data analysis.


## Vectors

# A vector is simply a list of values, but all values must be "of the same type" (e.g., character, numeric, date, etc.)

#  R is said to be a "vectorized" language, meaning that many values are stored in vectors and that R has many convenient features to manipulate data contained in these vectors.

# Vector Construction - The Popular c() Function

# The c() function is used to create vectors. "c" stands for "concatenate", "create", "combine" or "construct" (take your pick) values into a vector. Vectors are convenient ways to store groups of values of the same type (e.g., coefficients, residuals, predictions, etc.)

x <- c(1, 2, 3, 2)
x # Notice that there are 4 values and all are numeric (of the same type)
typeof(x) # Double precision numeric data type for all its elements
class(x) # Numeric object

# You can't have values of different types (e.g., text with data). If you try to mix data types in a vector R will "coerce" (i.e., change) the data to be of a given type that works for all elements. For example, if you try to create a vector with a number and a character, R corrects all values to character:

z <- c(1,"Alberto")
z # Notice that "1" was converted to character

# We can extract individual values from a vector with an [index] in squared brackets

x <- c(1,2,3,2,6,3,5)
x # Check it out
x[4] # 4th element of the vector

# A negative index removes an element from the vector
x <- x[-4] 
x # The 4th vector element got removed

# One of the nice things about R vectors is that we can manipulate all the values in a vector with a simple command. For example:

y <- x*3 # Multiplies every value of the x vector by 3 and stores them in y
y # Check it out

# It works the same way with text vectors

x <- c("ITEC 610", "ITEC 620", "ITEC 621")
x # Try it
x[3] # Check out the third element

# You can also give names to each vector element. This is important when you need to access vector values later. You can do name vector elements in 2 ways:

# Directly, when the vector is created:

x <- c(FirstName="Alberto", LastName="Espinosa", Title="Professor")
x # Check it out

# Or after, using the names() function

x <- c("Alberto", "Espinosa", "Professor")
x # No names
names(x) <- c("FirstName", "LastName", "Title") # Names for the x vector elements
x # Check out the names


## Factors

# Factor or Categorical variables are one of the most important variable types in R. This is so because text is difficult to process quantitatively in without some transformation. 

# A Factor variable is similar to character or text data, but it is used for categories that repeat many times in the data (e.g., "Freshman", "Sophomore", etc.). A nice feature of Factor variables, as opposed to plain character data, is that R assings an internal reference level value to each category in the factor variable, which allow you to do quantitative analysis. 

# Understanding factor variables is key in understanding regression models with categorical data and classification models (e.g., logistic regression) which predict the likely classification of an observation.

# For example, take the location types of real estate properties stored in a vector variable name LocType. You may have thousands of values in a given variable, one for each property in the data set. But each of these values belong to one of only three categories: Rural, Urban and Suburban. Then, it is better to store the data as factors (i.e., categories), rather than plain characters. 

# Take this character vector:

x <- c("Rural","Urban","Suburban","Urban","Urban","Suburban")
x # Notice that Urban and suburban are repeated, as you would expect
typeof(x) # Character type, not good for quantitative analysis
class(x) # Same thing

y <- as.factor(x) # Now, Convert x into a factor variable
y # Check out the 3 factors extracted out of x
typeof(y) # No longer a character, but an integer, one for each category
class(y) # The vector is now a factor variable object

# We can do a few new things with factor data

levels(y) # Display the unique categories in y
as.numeric(y) # Check the numeric level assigned to each factor value
nlevels(y) # Number of levels in y

# Factor levels are assigned alphabetically by R. In our example, Rural is level 1, Suburban is level 2 and Urban is level 3. You can easily change this:

levels(y) # Rural is the first level (alphabetically)
y <- relevel(y, ref="Urban") # Make Urban the first (i.e., reference) level
levels(y)

# Note: the [1] preceding the levels indicates that the first element is level 1, the next one is 2, etc. Also, we refer to the first level in a factor variable as the "reference" level because some statistical routines using factor variables use the first level for comparison to the other levels.


## Data Frames

# Data frames are fundamental in R and they are key to understanding how to access and manipulate data in R. The easiest way to understand a data frame is to think of it as an Excel sheet with various columns each with a column name (or think of it as a database table). Because we tend to manage a lot of real life data in tables, data frames are a very convenient data structure.

# Another way to think of a data frame is as table composed of a collection of (vertical) vectors. That is, columns can contain different data types. But the data in one column has to be of one type (i.e., a vector). 

# Creating Data Frames

# You can either read a .CSV file into a data frame

heart <- read.table("Heart.csv", header=T, sep=",")
head(heart)
class(heart)

# Or you can creating from column vectors

colleagues <- c("Espinosa", "Cakici", "Simon", "Armour")
years <- c(17, 25, 88, 105)
my.team <- data.frame(colleagues, years) # Function to create data frames
my.team # Check it out
class(my.team) # Check its class

# We can name the columns

colnames(my.team) <- c("Name", "Age")
my.team # Check it out


## Matrices

# A matrix is identical to a data frame in most respects (i.e., a table with values in rows and columns), except that ALL values in the entire matrix must be of the same type. Most often matrices contain only quantitative values, which can be easily manipulated with matrix algebra. 

# Matrices are very useful when you want to apply functions that apply to one type of data and not another. For example, if you want to compute correlations between all variables, you need to exclude all non-numeric variables because correlation does not apply to character data. Similarly, if you want to do some text operations, they will not work with numeric data. You can extract all numeric data into a matrix and all the character data into another matrix and work with the matrix.

# Since a vector contains data of the same type, you can think of a matrix as a collection of vectors of the same type. A matrix has 2 dimensions: number of rows and columns. For example, to create a blank matrix with 10 elements arranged into 5 rows (and 2 columns, of course):

x.mat=matrix(c(1,2,3,4,5,6,7,8,9,10), nrow=5) # 5x2 matrix
x.mat # Check it out

# Or you can use a sequence 1:10 instead

x.mat=matrix(1:10, nrow=5) # 5x2 matrix
x.mat # Check it out

# You could also accomplish the same thing with:

x.mat=matrix(1:10, ncol=2)
x.mat # Check it out

# Now  you can name the columns and rows of matrices:

colnames(x.mat)=c("Ref No.", "Sales")
rownames(x.mat)=c("John", "Judy", "Sally", "Moe", "Maria")
x.mat

# The [row, column] brackets in matrices works exactly like data frames


## Lists

# A list is similar to a vector but in contrast to vectors, lists can contain data of DIFFERENT types. Vectors can be used to create columns in data frames (i.e. tables), but lists cannot. However, because a list element is not limited to a scalar value, but can be another list or a data frame, lists are very useful to create very complex data structures for specialized applications. For example, lists are very popular in social network analytics because a social network not only has rows (i.e., members) and columns (i.e., attributes), but also has relationships between members and these relationships can also have attributes. You really cannot hold a complete data set for a social network without using a list structure.

# A list can be "unnamed" -- useful for quick calculations

y <- list("Alberto", "Professor", 16)
y # Check it out -- the data types are preserved

# Or it can be "named" -- useful for complex objects

y <- list(name="Alberto", title="Professor", age=16)
y # Check it out -- the data types are preserved

# Notice that the list has 3 numbered and named elements. The name of each element is preceded by a $ sign, just like with data frames. You can access list elements with an index or a name:

y[2] # Check it out
y["title"] # Check it out

# Also notice that each of the 3 elements in the list above has just [1] as a sub-index, indicating that each of the list elements contains a single value.

# But a list can be more complex because its elements can be anything, that is, single values, vectors, other lists. For example, here is a list of lists (notice that we have name lists inside of a list):

friends <- list(MyName="Alberto", MyAge=15, 
                Charlie=list(age=20, major="Analytics"), 
                John=list(age=30, job="Programmer"), 
                Dan=list(age=40, profession="Lawyer"), 
                Others=c("Joe", "Moe", "Doe"))

# You can extract distinct elements from a list with the $ or [[]]

friends # Check it out
str(friends) # Inspect the list structure
friends$John # Let's get all the data for John
friends$John$job # Let's get John's job

# You can extract one element of the list with indices:

friends["John"] # Works with named lists only
friends[4] # We can use a numbered index to get the 4th list element

# Notice above that each element is preceded by the element name (i.e., $John), so age is listed with its full name $john$age. This is convenient to access the various elements of a list operation

# Double brackets work the same way, but this doesn't return the list element names, which sometimes is necessary for computations.

friends[["John"]] # Double [[ ]] brackets return the results as a list of vectors
friends[[4]] # We can use a numbered index to get the 4th list element

# Notice above that the name $John does not appear anywhere


### Key Data Issues


## Data Re-Shaping

# Data re-shaping is about converting data from one type of data structure to another. The ability to re-shape data is central to R because some data may be available in one particular format (e.g., data frames) but a particular R function we want to use my only work with another format (e.g., matrix). I discuss a few key data re-shaping transformations.


# Creating Data Frames from Vectors

# Let's create a vector with 8 course ID's

courses <- c("ITEC 610", "ITEC 616", "ITEC 620", "ITEC 621", "ITEC 660", "ITEC 670", "KSB 620", "KSB 621")
courses # Check it out

# Now let's create another vector with the pre-requisites for these 8 courses

prereqs <- c("None", "None", "ITEC 610", "ITEC 620", "ITEC 610", "KSB 065", "ITEC 620", "ITEC 621")
prereqs # Check it out

# Now let's create another vector with the instructors for these courses

instructors <- c("Cakici", "Carmel", "Simon", "Espinosa", "Rinehart", "Armour", "Oroojeni", "Cogburn")
instructors # Check it out

# Now let's create a data frame with courses, pre-requisites and instructors

course.data <- data.frame(courses, prereqs, instructors) # Create a data frames
course.data # Check it out
class(course.data) # Check its class


# Creating Vectors from Data Frames

# For this purpose, we use the $ sign. We use $ after the data frame name to denote the specific column we want to access. For example, a column named "LastName" in a data frame named "Employees" can be accessed using this name: "Employees$LastName". 
# For example, to extract the courses from the data frame example above:

course.data$courses # Check it out

# Naturally, we can store the results of this extraction in another vector:

my.professors <- course.data$instructors
my.professors # Check it out


# Creating Matrices from Vectors

# You can bind vectors into matrix rows or into columns

# Binding Vectors Into Columns

courses.vector <- c("ITEC 610", "ITEC 620", "ITEC 621", "ITEC 670")
instructors.vector <- c("Cakici", "Simon", "Espinosa", "Armour")
course.mat <- cbind(courses.vector, instructors.vector)
course.mat
class(course.mat)

# Or you can create the vectors and matrix all at once

course.mat <- cbind(c("ITEC 610", "ITEC 620", "ITEC 621", "ITEC 670"), 
                    c("Cakici", "Simon", "Espinosa", "Armour"))

# Let's name rows and columns as before

colnames(course.mat) <- c("Course", "Instructor")
rownames(course.mat) <- c("Req 1", "Req 2", "Req 3", "req 4")
course.mat # Check it out
class(course.mat)


# Binding Vectors into Matrix Rows

# But, say that you have a bunch of unnamed vectors and you want to bind them either as rows or columns. You can do that in this way:

c("ITEC 610", "Cakici") # First row in a vector
c("ITEC 620", "Simon") # Second row in a vector
c("ITEC 621", "Espinosa") # Thirst row in a vector
c("ITEC 670", "Armour") # Fourth row in a vector

# We can now bind all these rows into a matrix with the rbind() function

course.mat <- rbind(c("ITEC 610", "Cakici"), 
                    c("ITEC 620", "Simon"), 
                    c("ITEC 621", "Espinosa"), 
                    c("ITEC 670", "Armour"))

course.mat # Check it out
class(course.mat) # Matrix

# Let's give names to the rows and columns as before

colnames(course.mat) <- c("Course", "Instructor")
rownames(course.mat) <- c("Req 1", "Req 2", "Req 3", "req 4")
course.mat # Check it out
class(course.mat) # A matrix


# Creating Matrices from Data Frames

# Because matrices only take one type of data for the entire matrix, it is important that we first extract the columns we need for the conversion, which need to be of the same type.

heart <- read.table("Heart.csv", header=T, sep=",") # Read the data
class(heart) # A data frame
head(heart) # Take a look

# Let's create a data frame with just a few numeric columns

heart.num <- heart[ , c("typea", "obesity", "alcohol")]
heart.num # Check it out
class(heart.num) # A data frame

heart.mat <- as.matrix(heart.num) # Convert data frame to matrix
heart.mat # Check it out
class(heart.mat) # A matrix


# Creating Data Frames from Matrices

# This is very simple because all matrices can be converted to data frames

heart.data <- as.data.frame(heart.mat) # Convert matrix to data frame
heart.data # Check it out
class(heart.data) # A data frame


## Indexing

# A data value in a data frame can be located using an index enclosed in squared brackets (like with vectors), containing the row and column numbers.It is very important to understand the use of indices in R in data frames and other data structures. For example, if you have a table that was read into R as a data frame with the name "Employees", containing one record for each employee and one column for every attribute we are collecting about employees you can access:

course.data[2, 3] # Data frame element in 2nd row and 3rd column
course.data[1, ] # First row (all columns) (i.e., leave the column blank)
course.data[ , 1] # First column (all rows) (i.e., leave the row blank)
course.data[2:5, 2:3] # Rows 2 to 5 and Columns 2 to 3

# Another popular way to index, which we will use below for sub-sampling is to create a vector with values corresponding to specific rows we want to extract from a data frame. For example:

row.index = c(1, 3, 5) # Index to extract rows 1, 3 and 5
course.data[row.index, ] # Get the 3 respective rows

# IMPORTANT: a thorough understanding of data frame and vector indices is key to understanding sub-sampling, cross-validation and machine learning. Try to understand this well now and I will come back to this later.

# Let's practice with indices. First create a vector that will have the row indices for a data frame we will create shortly called MyDataFrame We will call this vector MyIndex (be careful with the capitalization)

course.data[4, c("courses","instructors")] # List selected columns for row 4
course.data[ , c("courses","prereqs")] # List ALL rows from selected columns
course.data[2:4 , c("courses","prereqs")] # List rows 2 to 4 from selected columns

# While using indices to access values in data frames is very powerful and useful, we can also assign names to rows and columns and use those as reference. Since we used named vectors to create the data frame example above, we did not need to define names for the data frame columns. Each column assumed the name of its corresponding vector.


## Sub-Sampling for Machine Learning & Cross Validation

# Learning about Machine Learning and Cross-Validation is beyond the scope of this course. You will learn this in ITEC-320, ITEC-620 and ITEC-621. But sampling and sub-sampling are at the heart of most statistical, cross-validation and machine learning methods. I provide a high-level overview of these concepts here.

# Machine Learning (ML)

# ML is a branch of artificial intelligence, which is about developing models that can learn from data, without having to be re-programmed as the data changes. In essence, you could think of any predictive model, like regression analysis, as machine learning. But the most interesting advances in machine learning are based on continually testing the accuracy of our models. Cross-validation is the de-facto preferred method to evaluate model accuracy

# Cross-Validation (CV)

# Models are sometimes over fitting. All this means is that it is relatively easy to build models that fit the available data very well. For example, in the Hitters data set above, there are 263 observations with baseball player salaries. If we build a tree model to find which player attributes predict salaries better, we can split the data into 2 tree branches with low and high salaries. We can then predict that every player in the left branch makes the average salary of all players in that branch, and the same thing with the right branch. We can then split those two branches further into 2 more branches, etc. The more we split, the more accurate or predictions become (i.e., closer to the actual data). Eventually we will end up with branches with single data points, which cannot be split further. These are called "terminal" nodes. At this point, the tree predicts perfectly, because the prediction is the same value as the salary for that observation.

# If the above were true, we would model trees with terminal nodes every time. But how about a data set with a billion records. Would you grow a tree up to a billion branches? You could, mathematically. The problem is that we are evaluating the ability of our tree model to predict salaries using the same data we used to build the model. But the interesting questions is, would this model predict accurately with new data?

# An overfitting model is one in which the model accuracy is over estimated when tested with the same data used to build the model. These models tend to under perform when tested with new data.

# To overcome this problem, CV employs a very simple principle: split the data into two different sub-samples, build the model with one sub-sample and test it with the other sub-sample. The sub-sample used to build the model is called the "train" sub-sample (i.e., data we use to train the model), and the one we use to evaluate the accuracy of the model is called the "test" sub-sample.

# The typical cycle to build accurate models using ML with CV testing is:

# 1.Split a data set into train and test subsets; 
# 2.Fit (i.e. train) the model using the "train" subset; 
# 3.Test the model accuracy with the (different) "test" subset;
# 4.Select the model with the best CV accuracy

# Re-Sampling

# The above CV cycle generally works well, but it is possible to have one lucky or unlucky sub-sampling, so it is customary to repeat the 4 steps above multiple times and then average the results. This is called re-Sampling. You can do this yourself, but there are well established CV re-sampling methods that are often incorporated in R model estimation functions. We won't cover these methods here, but here is a quick overview:

# Leave-One-Out (LOOCV) -- Use all observations except one for training, then use the observation left out for testing. Then leave out a different observations. Then continues to do the same until each observation has been left out once. Then average results/

# K-Fold (KFCV) -- LOOCV is computationally intensive. KF simplifies this by dividing the data set into K folds. For example, 10-Fold CV (10FCV) divides the data into 10 equal folds. It then trains the model with 9 of the folds and tests the accuracy with the remaining fold. It then leaves a different fold out and trains the model with the remaining 9 folds. Then continues to do the same until all folds have been left out once.

# Bootstrapping -- This is a popular method in which a sample of N observations is extracted from a data set with N observations. Wouldn't this included all the data points? Yes, except that the random sampling in bootstrapping is done "with replacement". That is, some observations are sampled more than once and few are left out. If we draw several bootstrapped samples, the duplicated data points and the ones left out will be randomly mixed up. In bootstrapping CV, the model is trained with the bootstrapped sample and the testing is done with the points left out. This process is repeated as many times as desired, but typically beyond 100 bootstraps there is little improvement in performance.

# Sub-Sampling

# You will learn the methods above in ITEC-320, ITEC-620 and ITEC-621. In this course, I will just provide an overview of how to split the data into train and test sub-samples. Let's work with the Boston data set from the {MASS} package, which contains data on median housing prices in Boston neighborhoods.

library(MASS) # Load library to access the Boston housing data set
nrow(Boston) # The data set contains 506 observations.

# When you split the data into train and test subsets, you need to decide what percentage of the observations to use in the train and test sub-sets. Traditional splits are 50-50, 70-30, 80-20 and 90-10. Which one is better? If you have a large data set, 50-50 may work better. If you have a small data set and don't want to lose too much statistical power, 90-10 may be better. You can always try all of these and see which one gives you better results. 

# For this illustration, let's split the data into a 70% training subset, 30% test subset. 

# The typical way to do this is to first create a vector we will name "index" to serve as an index to select random rows from the data set.  

# Before we do the sub-sampling, we need to define the random sampling "seed". If you open any statistics book you will see random tables in some appendix. The way to draw a random sample is to pick any number in that table (called the seed) and then continue drawing numbers from that point. If we want to re-test our models with repeatable resuls, we would use the same seed next time. If we want to randomize things, we can use different seed values each time.

# For this example, we just pick a seed of 1

set.seed(1)

# The number of rows in the Boston data set can be obtained with the nrow() function:

nrow(Boston) # 506 records

# This means that the row numbers in this data set will be 1, 2, 3, etc., 506. We need to select 70% of these randomly. Let's create a vector that will contain 70% of randome values between 1 and 506, which will correspond to the row numbers of the train data sub-set.

train.index <- sample(1:nrow(Boston), 0.7*nrow(Boston))

# The command above takes a sequence of numbers from 1 to nrow(Boston), that is 506, and extracts a sub-sample of them into another vector named "index", with 70% of the 506 values, randomly selected.

train.index # Check it out
length(train.index) # 354 (70%) of the 506 sequential numbers in the index

# We can now use this index vector of random values to split the data set into train and test subsamples: 

# For the train subset, named Boston.train, we select all observations with a row number matching the numbers in the index vector and all columns (i.e., blank after the comma)

Boston.train <- Boston[train.index, ]
nrow(Boston.train) # 354 observations, that is 70% of the Boston dataset

# For the test subset, we simply remove the index values from the sub-sample index

Boston.test <- Boston[-train.index, ]
nrow(Boston.test) # 152 observations or 30% of the Boston data set

# You don't need to know this at this time, but to complete this exercise, let me illustrate how this works in ML. This would be a traditional statistical regression model using ALL the data to predict housing values using the crime rate, number of rooms, average house age and taxes in a Boston neighborhood

lm.fit <- lm(medv ~ crim + rm + age + tax, data=Boston)
summary(lm.fit) # Check it out

# This is the same model, but estimated with just 70% of the data in the train sub-set. Notice that all we are doing is changing the data set.

lm.fit.train <- lm(medv ~ crim + rm + age + tax, data=Boston.train)
summary(lm.fit.train) # Check it out

# The results change some when the model is estimated with the train subset, because it is a different data sample. You could re-sample if you wish and would get somewhat different results. If the results don't change much, the model is said to have "low variance" and is an indication that the model is stable. If the results change a lot, the model is said to have "high variance" and is an indication that the model is not very stable.

# You can now make predictions using the trained model to predict outcomes with the test subset:

predict(lm.fit.train, Boston.test)

# What follows is beyond the scope of this course, but it is provided here for informational purposes.

# We can now do a number of things with these results. For example, we can compute the difference between the actual values in the test subsample and the predictions and compute an error for each observation. We can also compute a mean of these errors to see how accurate our model is. A typical accuracy test statistic is to square the errors (to make them all positive) and compute an average of these squared errors, or Mean Squared Error (MSE). This is also called the "deviance", which you want it to be as small as possible.


## Missing Data

# Missing data is a very complex issue in analytics and needs to be treated carefully. The temptation is to throw away rows of data if any attribute has a missing value. This is OK when there are a few missing values, but otherwise we are throwing away too much data. Also, throwing away data may bias our analysis. For example, if we are trying to predict an illness, but those with the illness tend to not report it for privacy reasons, our data will be unbalanced with more participants without the illness.

# Missing data deserves a substantial portion of a data wrangling or data engineering course, which is beyond the scope of this course. For an excellent overview of the various techniques to deal with missing data, see:

browseURL("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3668100/")

# For the purposes of this course, we are just scratching the surface. If we assume that the missing data is relatively random (as opposed to systematic) and there is not a lot of missing data, or if removing missing data still leaves us with sufficient observations for statistical power in our analysis, we can simply omit all records that contain missing data.

# Letts look at the Hitters dataset of baseball player data in the {ISLR} package

library(ISLR) # Contains the Hitters data set

nrow(Hitters) # Let's check -> 322 rows in the Hitters data set
is.na(Hitters) # Show which data is missing (TRUE)

# Let's count the rows without omitted data

nrow(na.omit(Hitters)) # Only 263 rows with full data
100*nrow(na.omit(Hitters))/nrow(Hitters) # As a percentage

# There are more than 80% of rows with full data in this dataset. Assuming that the missing data is random, we can just delete these observations. However, if we believe that highly played salaries don't have their salaries listed, then it would be problematic to just delete these players from the analysis and our results would be biased towards lower paid players.

# Let's go ahead and delete missing data records from the Hitters data set. We can do this in two ways:

# 1. Create a new data set with a new name, to preserve the original data set

Hitters.No.NA <- na.omit(Hitters) 
nrow(Hitters.No.NA) # 263 observations only

# 2. Replace the original data set. Since the original dataset can be re-loaded or re-read any time, this is often more practical

Hitters <- na.omit(Hitters) 
nrow(Hitters) # 263 observations only


### Programming Fundamentals


## Built-In Functions

# Functions

# Functions are programs that can get invoked any time to carry out a pre-programmed computation. Functions may require zero, one or more than one "parameter". The parameters instruct the function how to apply the function. Here is an example of a function without parameters:

ls() # Lists all active objects in your R environment

# In contrast, for example, the linear model lm() function fits a linear (regression) model and requires 2 parameters. The first required parameter is the linear model formula and the second parameter is the data set to be used to estimate the model:

lm.fit <- lm(mpg ~ cyl + disp + hp + wt, data=mtcars)

# Polymorphic Functions

# Many functions are "polymorphic", meaning that they provide different results depending on the object passed as a parameter. Notice the difference between these two summary() outputs:

summary(mtcars) # Summary statistics of the mtcars data set
summary(lm.fit) # Summary results in the lm.fit linear model object

# The plot() function is also polymorphic and will plot different things, depending on the parameters you pass. For example, 

plot(mtcars) # Yields scatter plots for all variables in the data set
plot(lm.fit) # Yields 4 error plots for lm.fit. Need to hit return 4 times in the console to see all 4 plots

# By default, plot() with lm() objects renders all 4 residual plots, but if you only want the first one, you can pass an optional paramerer for this:

plot(lm.fit, which=1) # renders the first residual plot only

# Help with polymorphic functions

# If you request help with a polymorphic function you will get some general help that may not be what you are looking for. Take for example:

? plot() # You will get some general documentation, nothing specific
? summary() # Same thing

# If you need help with a polymorphic function with respect to a particular object, you can request help using this format ? function.objectType, which will give you specific help for that function, when applied to the specified object type. For example,

? plot.lm # Help with the plot() function when used with lm() objects
? summary.lm # Help with the summary() function when used with lm() objects


# Built-In Functions

# There are two types of functions: built-in, which are available for you to use; and user-defined, which are the ones you write yourself. Any time you load a library and use its functions, you are using "built-in" functions. All the examples above are build in functions, because yo did not have to write them to use them. They are pre-programmed in packages.

# R has many open source packages ready for you to use. As discussed earlier, packages contain libraries and libraries contain functions. These functions have been written by others, but are available for you to use once you load the respective libraries.

# An important aspect of using built-in functions effectively is to review documentation for the function, so that you understand how to use it, what parameters you need to pass to the function, what optional parameters it has, what are their default values, and what are the expected results from the function.

# For example, the {stat} package loads automatically when you start R and makes many popular statistical functions available to you. You can view these functions here (this works for any package):

library(help = "stats")

# For example, you can use the mean() function to compute the mean of the values in a vector

x=c(2, 3, 6, 8, 12, 1, 5) # Simple vector
mean(x) # Mean function
median(x) # Median function
var(x) # Variance function


## User-Defined Functions

# User-defined functions involve three actions: 

# (1) creating/defining the function
# (2) Loading the function in your R work environment, and 
# (3) using the function as needed

# With a built-in function, the function is already created/defined. You just need to load its library and then use the function. But you can create any function with the "function()" function (pardon the redundancy), then load into your work environment, and then invoke it when needed.

# (1) creating/defining the function

# A function is defined with function() {. The curly bracket is the delimiter of the function, meaning that anything after the opening curly bracket is part of the function until you end the function definition with the closing curly bracket }.

# Function Name: a function needs to have a name, so that you can call it when needed.

# Function Parameters: again, you can define a function without parameters (e.g., to pop a warning message) or with one or more parameters. When a parameter is passed to a function, the function will do something with that parameter.

# Return Values: a function may or may not return values. For example, if you want the function to do some data manipulation, but you don't need to see the results, then you don't need to return anything. But if you are computing a value that will be needed by your script (e.g., a multiplication of variables) then you need to use the "return" keyword to specify what the function returns.

# (2) Loading the function in your R work environment 

# To invoke or load a function into your R environment you just need to to highlight and execute all the commands (in between { and } bracktes) associated with the function. Once you run the function it will remain in memory until you unload it explicitly or until you exit R. 

# Let's create a simple function without parameters to display the text "This is my R World!!" We will call this function "MyWorld()"

# We first need to write the function. We can then run it by highlighting all the commands in between the from function() to the closing curly bracket. In this example, we are using the function "function()" to define the function, which we are naming "MyWorld".

MyWorld <- function() # No parameters
{print("This is my R World!!")}

# We can change where we place the curly brackets for readability. The above function can be written as:

MyWorld <- function() {
  print("This is my R World!!")
  } 

# or

MyWorld <- function() 
{
  print("This is my R World!!")
  } # or

# Highlight all the commands in the function and press Ctrl-Enter to run them together. You will then notice that the function is loaded in your R Environment tab. You can then run the rnn the function as needed

# (3) using the function as needed

MyWorld()

# Example

# Now let's write a function with one parameter, named AnyonesWorld. This function takes the parameter named "WhoAreYou" and stores your entry in a variable with the same name, so that you can use it in the function. It will then concatenate the text "This is " with the name you entered into WhoAreYou and then end with the text "'s R World". By default, the paste() function adds a blank space, so we use sep="" to have no spaces between text strings.

# Highlight all the lines in the function below and hit Ctrl-Enter to run the function. You should see the AnyonesWorld function object poppling in your R environment.

AnyonesWorld <- function(WhoAreYou) # "WhoAreYou" is the parameter
{
  print(paste("This is ", WhoAreYou, "'s R World", sep=""))
  }  

# Now execute the function but enter who's world it is (change the name)

AnyonesWorld("Alberto") # The parameter WhoAreYou will be replaced with "Alberto"

AnyonesWorld(readline(prompt="Enter Your Name: ")) # Try any name you wish

# Returning Values

# Functions are useful when you need to do complex calculations and return the results. For example to write a function that will return the squared value of a number x, we make x the parameter and then specify what we want to return, i.e., it's squared value

SquareMe <- function(x) # Whatever we enter in SquareMe(x) will be squared
{
  return(x^2)
  }

# Run the function definition above and then see how it works:

SquareMe(4)

# More than one parameter

MultiplyUs <- function(x,y) # This function has 2 parameters
{
  return(x*y)
  }

MultiplyUs(3,4) # Check it out

# Note: the functions above are simple illustrations in which the function commands fit in one line. Typically, functions span many lines. For example, let's create a function: that takes a value, then adds 2 and then squares this sum

# x is a parameter we are passing to the function:

SquareMePlus2 <- function(x) 
  {
  y <- (x^2)+2
  return(y)
  } 

SquareMePlus2(4) # i.e., (4^2)+2 = 18


## The apply() Family of Functions

# The apply() function is useful when you want to apply a function over values in rows or columns in a matrix. As you know, all elements in the matrix must be of the same type. If you use apply() on a data frame the data are first converted to a matrix. This function takes 3 parameters:

# 1. The matrix object you are working with
# 2. The value 1 to apply the function over rows, and 2 to apply it on columns
# 3. The function we want to apply

# For example, let's create a matrix:

my.mat <- matrix(c(4,2,6,8,9,10,20,1,3,12,11,17), nrow=4) # 4x3 matrix
my.mat # Check it out

apply(my.mat, 1, sum) # Get the sum for each row
rowSums(my.mat) # Does the same thing

apply(my.mat, 1, mean) # Get the mean for each row
rowMeans(my.mat) # Does the same thing

apply(my.mat, 2, sum) # Get the sum for each column
colSums(my.mat) # Does the same thing

apply(my.mat, 2, mean) # Get the mean for each column
colMeans(my.mat) # Does the same thing

# lapply() works similarly to apply(), but operates over lists and returns the results as a list. For example, the list below has a 3x3 matrix named A and a vector named B

my.list <- list(A=matrix(c(1,1,1,2,2,2,3,3,3), nrow=3), B=c(4,4,4,4))
my.list # Check it out

lapply(my.list, sum) # The sum is applied over each list element, returned as a list

# sapply() works like lapply() over lists, but returns the results as a vector

sapply(my.list, sum) # The sum is applied over each list element, returned as a vector


## Conditional Statements (if/else) 

# If/else statements are one of the most widely used program control logic commands

# Most R scripts you will be writing for this course will simply have a "stack" of commands that will execute sequentially. You can execute command stacks in full by highlighting the respective lines and then Ctrl-Enter or one line at a time. 

# However, there will be times when you want to execute some commands, only if some condition is met. This is called "conditional logic" in software programming. In such cases, you will need to understand how to control the program execution sequence (i.e., logic). There are several types of program controls, but the most common ones are: if/else's and loops. I will discuss loops in the next segment.

# if/else sequence

# 1. Enter if followed by a blank space and the if test you want to evaluate enclosed within parentheses

# 2. Next, type in the commands you want to execute if the if condition is true

# 3. If you don't specify an "else" condion, you are done

# 4. An "else" condition specifies the commands to execute if the if condition is not true

# 5. The "else" statement MUST be in the same line as the first closing bracket }

# 6. Then enclose the "else" instructions within curly brackets

# Example, you can change the value of MyValue below to see how the if control works

MyValue=2

if (MyValue > 10) 
  {print("Your number is large")} else
  {print("Your number is small")}

# For quick evaluations you can use the "ifelse()" function, just like an if function in Excel

MyValue=5
ifelse(MyValue > 10, "Large Number", "Small Number")

# You can use this in vectors too:

MyValues <- c(5, 10, 15, 20)
ifelse(MyValues > 10, "Large Number", "Small Number")

# This is a more complete illustration, which gets several input values for a loan application, does a number of calculations and then makes a decline/approve decision recommendation. 

# Input data:

CarPrice <- 30000
DownPayment <- 2000
LoanYears <- 5
AnnualInterest <- 0.04 # i.e., 4%
AnnualIncome <- 60000
MonthlyObligations <- 2300

# Calculations

LoanAmount <- CarPrice - DownPayment
LoanMonths <- LoanYears * 12
MonthlyInterest <- AnnualInterest / 12
MonthlyPmt <- LoanAmount*MonthlyInterest / (1-(1/(1+MonthlyInterest)^LoanMonths))
MonthlyIncome <- AnnualIncome / 12
DisposableIncome <- MonthlyIncome - MonthlyObligations
PmtToDisposableRatio <- MonthlyPmt / DisposableIncome

# Displaying results

print(sprintf("Your monthly payment is %s", MonthlyPmt))
print(sprintf("Your disposable income is %s", DisposableIncome))
print(sprintf("Your monthly payment to disposable income ratio is %s",
              PmtToDisposableRatio))

if (PmtToDisposableRatio > 0.2) # i.e., loan payment is more than 20% if income
  {print("Your loan application has been declined")} else
  {print("Your loan application has been approved")}

# It is often helpful to break the lines for readability. This works the same:

if (PmtToDisposableRatio > 0.2) # i.e., loan payment is more than 20% if income
  {
    print("Your loan application has been declined")} else
  {
    print("Your loan application has been approved")}

# Note: The script commands are commingled with the results in the console. This is difficult to avoid in R or R Studio, but it is easily resolved with R Markdown. Refer to the R Markdown lecture.


## For and While Loops

# Loops are another powerful and popular form of program logic control. A loop allows you to perform a group of commands several times until some condition is met to terminate the loop. Typically, there is an index or value that changes in each loop. For example, we may want to perform a bunch of calculations on the first 10 rows of a table.

# There are various types of loops, but the most common are "for" and "while" loops. " 

# In contrast "while" loops will continue to loop "while" a certain condition is met. I illustrate both below.

# CAUTION: Improperly written loops are one of the most common sources of software malfunction. In particular, loops that do not specify the "for" or "while" condition correctly may cause a program to go into an infinite loop that never terminates. Did you ever wonder why a program spins and spins sometimes? Most likely, it was an infinite loop somewhere in the software code.

# "For" Loops

# For loops cause a number of commands to execute several times with a loop "for" each of the values specified in the loop scope. When the list of values in the scope ends, the loop ends too.

# There is usually a loop variable, like i, that controls the loop. In the example below, in the first loop cycle, i will take the value of 1. In the second loop it will take the value of 2, etc. In the last loop, i will take a value of 10 and this will be the last loop cycle. The loop will end after that.

for (i in 1:10) {
  print(paste("The number is", i)) # The paste function concatenates strings
}

# If you want i to increment by more than 1 you can use the seq() function, in the example below i takes values in the sequence from 1 to 10, but in increments of 2.

seq(1, 10, 2) # Yields a vector with a 1 and values incremeneted by 2

for (i in seq(1, 10, 2)) {
  print(paste("The number is", i)) # The paste function concatenates strings
}

# While Loops

# While loops are a bit more complicated because you need to tell the loop when to stop and, if you don't do this correctly, you may end up with an endless loop. But they can be very useful when the loop scope is very large and you want to terminate the loop as soon as a certain condition is encountered.

# In the example below, the loop runs while the condition stated in the while statement is true. The control variable, i in this case, usually needs an initial value that will be changed in each loop pass and then evaluated to see if the loop needs to end.

# Note: the loop below will continue to run as long as i <= 10. So we MUST change i in each loop pass or we will end up with an endless loop.

i = 1 # This is like a counter, which we initialize to 1

while (i <= 10) { # This loop will stop when this condition is false
  print(paste("The number is", i)) 
  # Now let's increment i by 1 to control the loop
  i <- i + 1 
}


## Error Handling and Debugging

# This is an excellent video lecture by Roger D. Peng from Johns Hopkings with a good overview about R errors and debugging:

browseURL("https://www.youtube.com/watch?v=LHQxbRInyyc&feature=youtu.be")

# I also provided a link to Prof. Peng's site above, but this is his chapter on debugging:

browseURL("https://bookdown.org/rdpeng/rprogdatascience/debugging.html#somethings-wrong")

# There are two types of error messages in R. "Warnings" are presented as messages, but they are NOT errors. Warnings are things you need to pay attention to or inspect. R is said to be "chatty" and you may see many orange warnings. Generally, these are harmless. This will give you a warning:

log(-1) # The warning says that a NaN (not a number) value was obtained

# Real "errors" are displayed in color red and usually have an error message. These need to be debugged and corrected. This will give you an error:

lm(x ~ y+z) 

# x, y and z in the lm() function above are not defined anywhere. Typically, we need to have a "data=" parameter where we can locate x, y and z. Reading the error message carefuly can help identify the source of the problem. In this case, the eval() message is saying that the function could not be evaluated, and then the error says that y could not be found. You would either need to have a vector named y in your environment or a column named y in a data set specified in the "data=" parameter.

# There are two types of errors. One type (e.g., a syntax error) stops the program execution of the program and gives you an error message. These errors are obvious because the program stops running. It may take you a while to find the issue, but at least you know there is an error.

# The second type of error is the worst kind, which is the silent error. If your computations are incorrect or if you used the wrong data set or variable, but there is no syntax error, the program will just continue to follow your instructions and give you incorrect results. These types of errors can go unnoticed for a while and give you misleading results. Over time, you need to develop skills to eyeball and evaluate your results to figure if your program is working properly. 

# If you get an error message or if you suspect that you may have a silent error, here are some debugging tips:

# 1. DO NOT try to fix your error by changing your code randomly to see what works. You will not learn anything this way. It is better to read carefully the R error message in the console and try to figure out what is wrong.

# 2. Fixing errors can be an excellent learning experience. Hopefully, if you know what caused an error you will not make the same error again. But trying random fixes won't solve or teach you anything.

# 3. Check the upper/lower case on your object and variable names. Using the wrong case is one of the most common syntax errors.

# 4. If you get an error in a given script line and you review that line many times and you are convinced that the line is written correctly, chances are that you made an error in some prior command line, which caused your present command line not to work properly. You may have opened the wrong file, copied the wrong data, imported values from the wrong variable, etc. Your error can sometimes be many lines before where you got the error. If you suspect that the error may be in a prior, not the current command line, take a close look at the objects you are working with in the present line and go back and inspect every command line in your script that either created or manipulated those objects. This is the most common error that students make and cannot figure out.


### Visual Analytics

## Simple Graphics

# Graphs are powerful visual tools to analyze data. Almost all quantitative analytics process begins with some kind of visual exploration of the data. Graphs are also good to evaluate results (e.g. regression residual plots, normality, etc.). There is no better way to gain familiarity with your data set than rendering some graphs.

# Plots 

# Plots are one of the most commonly used graphs. plot() is a polymorphic function, so it will behave differently, depending on what you are doing.

# Let's work with the diamonds data set in the {ggplot2} package

library(ggplot2)
? diamonds # Explore the documentation of this data set.

# Scatterplot

# A scatterplot helps visualize the trend relationship between diamond carats (x axis) and diamond prices (y axis)

plot(diamonds$carat, diamonds$price)

# If you fit a linear model to predict diamond prices based on carats

attach(diamonds)
plot(carat, price)

lm.fit <- lm(price ~ carat, data = diamonds)
summary(lm.fit) # Check the results
plot(lm.fit, which = 1) # Display the first of four residual plots

# Note: you may not know much about regression residual plots at this time, but just for context, this residual plot doesn't look very good. A good model should yield a relatively horizontal trend line (in red) at 0. And the residuals (i.e., errors) should be evenly distributed across the line (i.e., a cloud of data points with no pattern). Clearly, the trend is not horizontal and there is a downward pattern in the residual points.

# Boxplots

# Boxplots can be used to inspect a single variable. The box plot shows a box with a thick line around the median value of the variable. The gray box shows where 2nd and 3rd quartiles of the data (25% below and above the median). The two horizontal lines outside the box are called the "whiskers" and they delimit where about 75% of the data resides. Data points outside the whiskers are extreme values and may actually be outliers.

boxplot(diamonds$carat)

# The boxplot above shows that the median carat value is less than 1 carat, about 50% of the diamonds are somewhere between 0.4 and 1.1 carats. 75% of the data is somewhere between 0.3 and 3 carats. And, there are a few larger diamonds with 2 or more carats.

# Boxplots are also useful to contrast a quantitative variable (e.g., diamond price) across various values of a categorical variabel (e.g., diamond clarity)

boxplot(diamonds$price ~ diamonds$clarity)

# The boxplot above shows that the SI2 clarity commands the highest median price, but with so many extreme values in prices across all clarity categories, it is hard to tell.


## Analyzing Normality

# Many models require some variables or regression residuals to be normally distributed. There are two primary ways to inspect for normality visually:

# Histograms

hist(diamonds$carat) # Carats are skewed to the right, not normally distributed
hist(diamonds$price) # Same thing

# Let's add some labels

hist(diamonds$carat, main="Carat Histogram", xlab="Carats") # w/labels
hist(diamonds$price, main="Price Histogram", xlab="Price")

# Right skewed distributions can sometimes be corrected by taking a log of the variables

# More normal, but not totally
hist(log(diamonds$carat), main="Carat Histogram", xlab="Log(Carats)") 

# Same thing
hist(log(diamonds$price), main="Price Histogram", xlab="Log(Price)") 

# QQ Plots

# QQ plots are perhaps the best visual tool to evaluate normality. A QQ plot requires 2 graphs. One shows the data quartiles (one Q) and where the quartiles should fall for perfectly normally distributed data (the other Q) (per the Gaussian formula). Since the quartiles and the theoretical quartiles are the same, this graph is a straight line called the QQ Line. The other graph shows the actual data. If the actual data is closely aligned with the (theoretically normal) QQ Line, then the data is relatively normally distributed. 

# Wagging the tail -- it is not uncommon to depart from normality at both ends of the QQ Plot. This is sometimes referred to a "wagging the tail". This is not a problem if about 75% of the data in the middle part of the line is closely aligned with the QQ Line

# Technical note: graphs can be "primary" graphs or "layer" graphs. A primary graph creates a new graph (erasing the prior graph, if any). Layer graphs add elements or layers to primary graph (e.g., labels, colors, lines, etc.).

# Let's draw the QQ Plots and QQ Lines for diamond carats and prices 

qqnorm(diamonds$carat) # Actual QQ Plot data
qqline(diamonds$carat) # Theoretical QQ Line, not much alignment

qqnorm(diamonds$price) # Same for price
qqline(diamonds$price) # Same

# Let's log the data

qqnorm(log(diamonds$carat)) 
qqline(log(diamonds$carat)) # Closer to normally distributed, some tail wagging

qqnorm(log(diamonds$price))
qqline(log(diamonds$price)) # Same


## Exporting Graphics

# If you need to export a visual graph to a JPG or PDF file, you can do that easily with a 3-step process:

# 1. Start the JPG file and give it a file name
jpeg("MyGraph.jpg")

# 2. Render the graph. You won't see it because it if piped to the JPG file
qqnorm(diamonds$price) # Same for price
qqline(diamonds$price) # Same

# 3. Terminate the graph production
dev.off() # Close the graph

shell.exec("MyGraph.jpg") # You can now open and view the JPG file


## The {ggplot2} Library

# There are several popular and powerful graphic R packages. But none as good and versatile as {ggplot2}. 

library(ggplot2) # Load the library

# Note, there is another package called {lattice}, which installs and loads automatically when you install or load {ggplot2}

# This package is based on the popular "Grammar of Graphics", which is a well conceptualized syntax to render graphs and add layers. You can learn more about the "Grammar of Graphics" from Wilkinson (2005):

browseURL("https://www.amazon.com/dp/0387987746/") # Grammar of Graphics

# Any one seriously interested in constructing professional and very attractive graphs is encouraged to become familiar with the grammar of graphics and the excellent book by the developer of the {ggplot2} package:

browseURL("https://www.amazon.com/dp/0387981403")

# This book states that the Grammar of Graphics tells us that a statistical graphic is a mapping from data to aesthetic attributes (colour, shape, size) of geometric objects (points, lines, bars). A graph may also contain statistical transformations of the data. Faceting can be used to generate the same plot for different subsets of the data set. It is the combination of these independent components that make up a graphic.

# Scatterplot

# Price by Carats
ggplot(diamonds, aes(x=carat, y=price)) + geom_point()

# Add a Smooth Trend Line
ggplot(diamonds, aes(x=carat, y=price)) + geom_point() + geom_smooth()

# Histograms: the geometric object is geom_histogram and the aesthetic is x=carat:

ggplot(data=diamonds) + geom_histogram(aes(x = carat))

# We can save graphs in objects:

g <- ggplot(diamonds, aes(x=carat, y=price)) # Saved, but not displayed yet

# We can then add layerss to the graph:

g + geom_point()

# Then change the properties if you wish, e.g., add colors

g + geom_point(aes(color = clarity))
g + geom_point(aes(color = clarity)) + facet_wrap(~ clarity) # Faceted by clarity
g + geom_point(aes(color = clarity)) + facet_grid(cut ~ clarity) # Faceted in grids


## Color Names

# Colors are abundant and rich in R. Here are a few resources to identify colors name for your graphs:

browseURL("http://www.stat.columbia.edu/~tzheng/files/Rcolor.pdf")
browseURL("https://www.datanovia.com/en/blog/awesome-list-of-657-r-color-names/")
browseURL("http://sape.inf.usi.ch/quick-reference/ggplot2/colour")


### Working with Text

# R is very powerful for manipulating text. To seriously learn text mining and text analytics you should consider taking a text analytics course, either ITEC 724 or ITEC 321.

## Pasting and Concatenating

# You can Concatenate text with the "paste()" function

# The default separator in the paste() functin is a blank space

MyText <- paste("My", "name", "is", "Alberto") # To store it in a variable
MyText # Note that sub-strings are separated by a blank space by default

# You can change the separator with the "sep=" attribute

paste("My", "Name", "is", "Alberto", sep="-") # to change default separator to "-"
paste("My", "Name", "is", "Alberto", sep="") # to eliminate the blank space

# You can concatenate literal text enclosed in quotes with text contained in variables without quotes:

MyName <- "Alberto" # To create a variable that contains text
MyCourse <- "R Workshop" # And another variable
paste("My", "name", "is", MyName, "and I teach", MyCourse) # To embed variables

# sprintf() does the same, but it places variable contents in each %s

sprintf("My Name is %s and I teach %s", MyName, MyCourse)

# In contrast to "paste()", the "c()" function concatenates into a vector

MyText <- c("My", "Name", "is", "Alberto") # Creates a vector with 4 elements
MyText

# Which you can then collapse into a single text string, if needed

paste(MyText, collapse="") # Concatenate the vector into a string with no spaces
paste(MyText, collapse=" ") # Or with spaces


## Extracting Text

# A useful function to search sub-strings that match some some string within a text variable or object is "grep()". The name grep comes from Unix, which is a command utility to match expressions. grep stands for (g)lobal search for (r)egular (e)xpressions and (p)rint matching. Go figure !!. The R function grep() is in R's {base} package, so it is always available and has a similar purpose. grepl() is a variation of this function. I discuss both below:

MyBioText <- "Prof. Espinosa is a Full Professor of Information Technology and Analytics at Kogod School of Business, American University. He holds Ph.D. and Master of Science degrees in Information Systems from the Tepper School of Business at Carnegie Mellon University, a Masters degree in Business Administration from Texas Tech University; and a Mechanical Engineering degree from Pontificia Universidad Catolica, Peru. He is the architect of Kogod's MS Analytics degree, both campus and online. He is also the curriculum architect for all information systems and technology undergraduate specializations."

substr(MyBioText, start=7, stop=14) # Extract characters 7 to 14

# Or you can store it in a text variable

MyText <- substr(MyBioText, start=7, stop=14)
MyText # Same thing

# Replace "Espinosa" with "Alberto" in MyBioText

gsub("Espinosa", "Alberto", MyBioText) 

# Preparing for data mining

# When doing text mining or analytics, it is often necessary to store the individual words in a text corpus in a vector. Once the text is vectorized this way, there are many analytical methods you can apply, like comparing text documents and searching for predominant words.

# To split the text into "list" elements every time there is a blank space, you can use the strsplit() function

MyBioList <- strsplit(MyBioText, split = " ") 
MyBioList # Check it out
class(MyBioList) # A List

# Extract the text from the list as a vector: lists are more cumbersome to work with, so it is sometimes convenient to extract the word vector inside the list

MyBioVector <- unlist(MyBioList)
MyBioVector  # Check it out
class(MyBioVector) # A character vector
length(MyBioVector) # Count the number of words

# You can now apply any vector functions to find and manipulate characters


## Text Expressions

# Word Cloud Generator

library(wordcloud) # Word Cloud package
library(RColorBrewer) # To work with color palettes

# This function generates a word cloud from text in my bio. You can change random.order to T if you don't want the most frequent text at the center. You can change min.freq if you only want words that repeat a number of times. You can change max.words if your text is very large. You can play with the color brewer palette.

wordcloud(MyBioText, random.order=F, min.freq=1, 
          max.words=200, colors=brewer.pal(8, "Dark2"))

# Need to zoom the graph to see it better

# You can change the appearance of the word cloud by changing the minimum frequence to count (1 instance in this case), maximum words to be plotted (200 in this case) and the color scheme.

# Basic Text Operations

MyBioText.Lower <- tolower(MyBioText) # Converts text to lower case
MyBioText.Lower

MyBioText.Upper <- toupper(MyBioText) # Converts text to upper case
MyBioText.Upper

nchar(MyBioText) # Number of characters


### Basic Descriptive Analytics

## Descriptive Statistics

# All analytics projects begin with some investigation of the data at hand. Descriptive statistics are statistics aimed at summarizing the data at hand. They are mostly about measuring the central tendency of variables (e.g., means, medians, etc.) and variability or dispersion of the data around the center (e.g., standard deviation, variance, minimums, maximums, etc.). For character and categorical data, descriptive statistics also includes frequency counts.

# Let's use the diamonds data set in the {ggplot2} package for illustrations:

library(ggplot2) # Contains the "diamonds" data set
? diamonds # Quicly inspesct the data variables

# You can display a quick summary of the data:
  
summary(diamonds)

# You can use the aggregate() function to summarize descriptive statistic by categorical varaibles

# Let's get diamond price means by cut

aggregate(price ~ cut, diamonds, mean) # Price mean by cut
aggregate(price ~ cut, diamonds, sum) # Price sum by cut
aggregate(price ~ cut, diamonds, var) # Price variance by cut
aggregate(price ~ cut, diamonds, sd) # Price standard deviation by cut

# To group by more than one attribute use the + operator
aggregate(price ~ cut + color, diamonds, mean)

# To aggregate more than one column, use cbind()
aggregate(cbind(price, carat) ~ cut, diamonds, mean) 

# The functions below work on vectors or data frame columns

mean(diamonds$price) # Sample mean
median(diamonds$price) # Sample median
max(diamonds$price) # Maximum value in the sample
min(diamonds$price) # Minimum value in the sample
sd(diamonds$price) # Sample standard deviation
var(diamonds$price) # Sample variance

# The {psych} package specializes in statistical functionts for psychology analysis and has a convenient describe() function to summarize statistics for the whole data set:

library(psych) 
describe(mtcars) # Descriptive statistics


## Covariance and Correlation

# Covariance

# Covariance measures how two numeric variables vary together. Because co-variance can only be computed for quantitative or numeric variables, some covariance functions require that we input the data as a matrix, rather than a data frame. Every covariance function works differently.

# Large positive covariances indicate that the two variables tend to move in the same direction (i.e., when one is postitive the other one is too). Large negative convariances indicate that the two variables tend to move in opposite direction (when one is positive the other one is negative). Covariances around zero mean that there is no co-variability pattern between the two variables.

cov(mtcars) # Covariance matrix for mtcars data set

# A few facts about covariance: (1) The covariance of two variables are affected by the scale of the variables. We we change one variable, say from miles to kilometers, the covariance results will change because; (2) The covariance of one variable with itself is its variance.

# Correlation

# Because covariance varies with the scale of the variables, co-variability is often analyzed with correlation instead. The correlation between two variables is simply their covariance divided by the standard deviations of each of the variables. Essentially, this eliminates the issue of scale because correlation values can only be between -1 and +1. 

# Like covariance, correlation measures how two numeric variables vary together. Because correlation can only be computed for quantitative or numeric variables, some correlation functions require that we input the data as a matrix, rather than a data frame. Every correlation function works differently.

# A correlation of +1 indicates perfect positive correlation (i.e., one variable is mathematically related to the other). That is, both variables co-vary together perfectly. Correlation valuse between 0.5 and 1 indicate high co-variability between the variables. Similarly, -1 indicates that both variables vary in exact opposite directions. Correlations between -0.5 and -1 indicate strong negative correlation. Correlations around zero indicate that the variables are somewhat "independent", that is, there is no co-variability.

# The correlation of a variable with itself is always 1.

cor(mtcars) # Correlation matrix for mtcars data set

# The cor() function reports correlation scores, but not p-values. p-values are indicators of the statistical significance of the correlation value. A general rule is that p-values should be 0.05 or less (i.e., less than 5% probability that the observed correlation is by chance). The {Hmisc} package has a rcorr() function that renders the p-values. But it only takes matrices as an input, which are easy to conver to:

library(Hmisc) # Note the H is upper case
rcorr(as.matrix(mtcars)) # Convert to matrix and extract correlations

# Notice that the function above yields 2 matrices, one with correlation values and one with p-values. Large correlation values tend to have smaller p-values. That is, the larger the (absolute value of the) correlation, the smaller the p-value (i.e., the more significant the correlation).

# For example. The correlation between mpg and carb is -0.55 and it's p-value is 0.0011, so they have a negative significant correlation. But mpg and qsec have a positive correlation of only 0.42, with a p-value of 0.0171. So the correlation is positive, but below 0.5 and its p value is greater than 0.05, so this correlation is not significant.

# Another correlation function

# The corrplot() function in the {corrplot} library yields nicely formatted and visually appealing correlation statistics.

library(corrplot) # Library for correlation plots
mtcarsCorr <- cor(mtcars) # First, store create a correlation object with cor()

# Then feed the correlation object into the corrplot() function to display correlations in various formats

corrplot(mtcarsCorr, method="number", order="hclust")
# Try method = “circle" or "ellipse" for other formats


## Analysis of Variance (ANOVA)

# Correlation can only compare pairs of numerical variables. If one variable is numerical and the other is categorical, there is no correlation. But the co-variability of a numerical variable across categories of another variable can be evaluated with ANOVA.

# Essentially, the ANOVA test compares the means of 2 or more groups or categories. The means of 2 groups may be different, but ANOVA tells us if this difference is significant. If the difference is not significant, then there is little co-variability between the two variables. That is, the two variables are independent. If the difference is significant, then the mean of the numeric variable changes across the categories of the categorical variable. That is, they co-vary or are dependent.

my.aov <- aov(price ~ clarity, data=diamonds) # ANOVA of price by clarity
my.aov # Check out the basic results

summary(my.aov) # Check out more detailed results
# The p-value is 2e-16, which 2 with 16 zeros to the left, very significant
# Conclusion: clarity affects diamond prices


## Chi-Square Test of Independence

# Now, what if both variables are categorical. For example, let's say that you want to investigate if diamonds cut and color influence each other or are independent. Well, we can't use correlation or ANOVA because both variables are categorical. In such cases, we can conduct a chi-square test of independence.

# Essentially, what we do is cross-tabulate how many instances of one category are associated with instances of the other category. We can then use this counts so see if one category influences the other one. 

# Let's cross tabulate diamond cut with color

cross.table <- table(diamonds$cut, diamonds$color) # Store results in a cross table
cross.table # Check which cuts are associate with which diamond colors

chiSq.diamonds <- chisq.test(cross.table) # Conduct the chi-square test on the table
chiSq.diamonds # Check out Chi Square test results

# The chi-square is significant because the p-value is really small. This means that cut and color are related. If the test were not-significant, we would conclude that they are independent.


### Simple Predictive Models


## Linear Regression Models

# A linear regression model evaluates the effect of one or more predictors on a quantitative outcome

# For example, let's use the Boston data set in the {MASS} package to evaluate the effect of per capita crime rate (crim), average number of rooms per house (rm) and average distance to Boston employment centers (dis) on the median value of Boston neighborhood houses (medv).

library(MASS) # Contains the Boston data set
? Boston # Quickly explore the various variables

lm.fit <- lm(medv ~ crim + rm + dis, data=Boston) # Fit the model, store the results in lm.fit
summary(lm.fit) # Display regression output summary

# Results: crime has a negative significant effect on median house values (based on p-values, not the 3 stars). The average number of rooms has a positive significant effect. Distance to employment centers has a positive effect, but this effect is not significant.


## Logistic Regression

# Logistic regression is similar to a linear regression model, but instead of evaluating the effects on a quantitative outcome, we evaluate the likelihood that an outcome is in one category or another. When there are 2 possible outcomes (e.g., approve vs. decline, default loan vs. no default) the model is called "binomial logistic". When there are more than 2 possible outcomes (e.g., "freshman", "sophomore", "junior", "senior") the mode is called "multinomial logistic". I illustrate one binomial regression model.

# For example, let's read an external .CSV South Africa heart disease data set from Professor Trevor Hastie from Stamford:

browseURL("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.info.txt")

# We want to predict the likelihood of patients having coronary heart disease (chd)

# I have already extracted the data file Heart.csv. Let's bring it into an R data frame:

heart <- read.table("Heart.csv", sep=",", head=T)

# You may want to open the heart data set from the Environment tab and do a quick inspection

# Now let's fit the binomial logistic model

heart.fit <- glm(chd ~ tobacco + alcohol + ldl + age, 
                 data = heart, 
                 family = binomial(link = "logit"))

# A number of observations about this logistic or logit model:

# 1. We use the glm() function, not the lm() function. The lm() function is appropriate when we are trying to predict quantitative outcomes. But logistic regression predicts "likelihood" of categorical outcomes, in this case it is the likelihood that a patient has coronary heart desease (chd = 1) or not (chd = 0). So we use a "generalized linear model" (GLM), glm(), which has more general application than lm().

# 2. Because glm() can be used for various types of models, we need a third parameter to specify the type of outcome we are predicting and the type of model we want to use:

#    family = binomial(link = "logit") specifies that our outcome of 
#             interest is binomial and the model we want to fit is "logit".

summary(heart.fit) # Take a look

# Results: you will learn to interpret these results properly in othe analytics courses, but for now, we can say that the results suggest that tobacco, ldl and age increase the likelihood of having heart disease. we know alcohol is generally not good, but it does not have a significant effect on the likelihood of having heart disease.


## Decision Trees

# Decision trees are easy to interpret and fit. You can use them to predict quantitative (i.e., regression trees) or categorical (i.e., classification trees) outcomes. 

# How decision trees work:

# In the example below we use a tree model to predict a quantitative outcome, baseball player salaries (Salary), using years in the major leagues (Years), runs batted in (RBI), number of hits (Hits) and number of walks (Walks) as predictors. On a quantitative regression tree, the tree function splits the data into 2 sections, called branches or nodes. The tree function finds the predictor that does the best job at explaining salary differences and at which value to split the data. 

# Once the tree does the first split, we can use the average salary in each branch to predict player salaries in that branch. Then we split the tree again. The further we split the tree, the more precise our salary predictions are because the average salary is computed over a smaller group of players.    

# Let's try it:

library(ISLR) # Contains the Hitters data set with baseball player data
library(tree) # Contains the tree() function

? Hitters # Inspect all variables in the data set

fit.tree <- tree(Salary ~ Years + RBI + Hits + Walks, Hitters) # Fit the tree
fit.tree # Check it out
summary(fit.tree) # Inspect the summary results

plot(fit.tree) # Plot the tree (has no labels)
text(fit.tree) # Add text with data and labels

# How far do we grow the tree? Well, we can keep splitting the tree until each branch has a single data point. These last branches are called "terminal nodes". The trees get more accurate the larger they grow, but very large trees are "overfitting", which means that they don't predict well with new data. We can decide how large to grow the tree with cross-validation (see discussion above), but that is beyond the scope of this class.

# For this exercise, we will control the tree size to render a visually clear tree. If you have a lot of data your tree may end up being too large to read on the screen. You can use a parameter called "mindev" to control the size of the tree. The default mindev value when you don't say anyting if 0.01. What this means is that the tree() function will stop growing the tree when the prediction error is 1% of the prediction error without a tree. The smaller the mindev value, the larger the tree. Conversely, the larger the mindev value the smaller the tree. Check it out:

fit.tree.small <- tree(Salary ~ Years + RBI + Hits + Walks, Hitters, mindev = 0.1)
plot(fit.tree.small) 
text(fit.tree.small) # Very small tree

fit.tree.large <- tree(Salary ~ Years + RBI + Hits + Walks, Hitters, mindev = 0.005)
plot(fit.tree.large) 
text(fit.tree.large) # Very large tree


### Introduction to Shiny


# Like R Markdown, Shiny is another R Studio product, which allows you to create interactive web applications. We will not cover Shiny in this course. But you are more than welcome to explore Shiny on your own and submit your work using Shiny. See:

browseURL("https://www.rstudio.com/products/shiny/") # From R Studio web site
browseURL("https://www.youtube.com/watch?v=c1sEvi_qrY4") # From R Studio web site

