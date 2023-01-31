##########################
# ITEC Introduction to R #
##########################

# Filename: ITEC_RIntro.R
# Prepared by J. Alberto Espinosa
# Last updated on 1/10/2022


# READ THIS BEFORE YOU START

# This script was prepared for an American University class ITEC 621 Predictive Analytics. It's use is intended specifically for class work for the MS Analytics program and as a complement to class lectures and lab practice.

# IMPORTANT: the material covered in this script ASSUMES that you have gone through the self-paced course KSB-999 R Overview for Business Analytics (posted in the LMS for online students) and are thoroughly familiar with the concepts illustrated in the RWorkshopScripts.R script. If you have not completed this workshop and script exercises, please stop here and do that first.

# Complete these steps before you start:

# Campus Program

# 1. Log onto Canvas

# 2. Self-enroll into KSB-999 at:
browseURL("https://american.instructure.com/enroll/9DJDYC")

# 3. Download all the R scripts and data sets provided to a working directory in your computer.

# 4. Open the ITEC621_Packages.R scripts and install all packages listed. IMPORTANT: The ITEC621_Packages.R script gets updated periodically. Please use the latest script posted on Canvas for ITEC 621.

# 5. Go to the Course Content area and go through all lectures and sections. There are almost 7 hours of video lectures to follow along with the provided R script. If you are already familiar with R, you can try to follow the R script on your own.

# 6. Complete the self-graded homework HW0 in KSB-999 before starting ITEC 621. Complete the homework without looking at the posted solution. Once you are done, then compare your answers to the solution and make any necessary corrections. IMPORTAN T: HW0 in KSB-999 is similar in nature (but somewhat different) than HW1 in ITEC 621. If you do a consciencious job with HW0, HW1 should be pretty straight forward.

#######################################################
#                       INDEX                         #
#######################################################

## How to use this Index: 

# Highlight the index topic you want to jump to. Then press Ctrl-F and the highlighted text will appear in the search box in the upper left corner in R Studio. Then press Enter or click Next to jump to the corresponding topic in the script.

#### 0. COURSE INTRODUCTION

## 0.1 Working Directory and Project Setup
## 0.2 Scripts for this Class
## 0.3 ISLR textbook, R scripts and datasets

#### 1. Introduction ####

### 1.1 R Overview
### 1.2 R Packages
### 1.3 R Studio Overview

#### 2. R for Analytics ####

### 2.1 General Information About this Script
### 2.2 General Information on R and R Studio

## Two Ways to Run R Commands
## Getting Help
## Vignettes
## Comments
## Tutorials
## Books
## Useful Web Sites

### 2.3 Packages, Libraries and Views

## Installing Packages
## Installing Views

### 2.4 Reading Data Into R

## Excel Files
## MS Access files
## SPSS Files
## Other statistics data files
## Saving Data from R to a file

### 2.6 Working with R Objects

### 2.7 Variables, Data Types, Objects, Classes and Data Structures

## Overview
## Simple Data Variable & Data Types
## Numeric Data
## Character Data
## Date Data
## Logical Data
## Factors

## Complex Data Structures
## Vectors
## Data Frames
## Matrices
## Lists
## Data Reshaping
## Subsampling

### 2.8 Working with Text
### 2.9 Functions
### 2.10 Program Control: If'S 
### 2.11 Program Control: Loops

### 2.12 Simple Statistics with R

## Working with Data
## Simple Graphics
## ggplot2
## Capturing Graphics in PDF
## Simple Statistics
## Regression
## Capturing Graphics in PDF

### 2.13 R Markdown
### 2.14 Shiny 

#######################################################
#                  END OF INDEX                       #
#######################################################


#### 0. COURSE INTRODUCTION


## 0.1 Working Directory and Project Setup

# It is important that you tell R what your work directory is. I STRONGLY recommend creating one working folder (e.g., c:\AU\Courses\ITEC621\R) in your computer for all this work in this class and keep all your scripts and data files in that directory. R works better when all the scripts and data files for a project are in a single directory and this will make your life much easier.

# IMPORTANT: BY FAR, THE MOST COMMON SOURCE OF PROBLEMS using R Studio is the IMPROPER SETUP of the working folders and project environment. Please ensure that you follow this instructions. It will most definitely save you time later.

# Once you create a working folder for this class in your computer, you can define your working directory as follows:

setwd("C:/AU/Courses/ITEC621/R")

# As you work with many other projects, you will probably have several working directories and you may be switching directories from time to time. TO see, which is the current working directory in your R Studio environment enter:

getwd()


# Project Directory Setup

# While setting a working directory is useful, it is even better to set a project directory. When you create a project in R Studio and then open it, your working directory will be set automatically. In addition, all the data files and variables you were working with in your last session will open too. This is a very convenient feature of R Studio. In my opinion, setting a project is a MUST when working with R and it makes it very easy to switch from one project to another. It will really make your R Studio work very efficient.

# To create a project for this class, select:

# File -> New Project -> Existing Directory. 

# Then browse and navigate to your working folder and click on Create Project. That's it. Now you can open or switch projects from the scrollable list in the upper right corner of R Studio.


## 0.2 Scripts and datasets for this Class

# Copy all of the following scripts provided to your work folder

# ITEC_RIntro.R -- (This) Script used for this overview
# ITEC621_Packages.R -- Installation of all packages for ITEC 621 and more
# ITEC621_<Topic>.R -- Collection of scripts by Topic for ITEC 621; not needed for the intro
# ITEC621_Goodies.R -- Additional scripts for your reference; not needed for the intro
# ITEC_RMarkdown.Rmd -- Overview of R Markdown

# Also, please copy all the posted CSV data files to your work directory


## 0.3 ISLR textbook, R scripts and datasets

# ISLR stands for "Introduction to Statistical Learning with R", which is an excellent book on statistical learning, with its own package with data samples, some of which we use in the R training. Please install this package so that you get access to the included data samples. For further information on the ISLR book, see:

browseURL("http://www-bcf.usc.edu/~gareth/ISL/index.html")

# The companion R code included with this book is in the ISLR library:

library(ISLR)

# All data sets used by the book authors are contained in the ISLR package. We will use several of these data sets for this class, plus some others.

# Auto: Gas mileage, horsepower, and other information for cars. 
# Caravan: Information about individuals offered caravan insurance. 
# Carseats: Information about car seat sales in 400 stores. 
# College: Demographic characteristics, tuition, and more for USA colleges. 
# Default: Customer default records for a credit card company. 
# Hitters: Records and salaries for baseball players. 
# Khan: Gene expression measurements for four cancer types. 
# NCI60: Gene expression measurements for 64 cancer cell lines. 
# OJ: Sales information for Citrus Hill and Minute Maid orange juice. 
# Portfolio: Past values of financial assets, for use in portfolio allocation. 
# market: Daily percentage returns for S& P 500 over a 5-year period. 
# SArrests: Crime statistics per 100,000 residents in 50 states of USA. 
# Wage: Income survey data for males in central Atlantic region of USA. 
# Weekly: 1,089 weekly stock market returns for 21 years.

# You can view all data sets in the ISLR package with:

data(package="ISLR") # To list the data sets in the ISLR package

# Or visit:
browseURL("http://www-bcf.usc.edu/~gareth/ISL/data.html")

# To read a data set from this (or other) web sites, for example use:

MyData <- read.table("http://faculty.marshall.usc.edu/gareth-james/ISL/Credit.csv", sep=",",head=T, row.names=1)

# csv denotes a "comma-separated" file
# sep="," denotes that the data elements are separated by a comma (some files are separated by semi-colons, tabs, etc.)
# head=T or head=TRUE mean that the first row of the table contains column names
# row.names=1 means that the first column does not contain data, but row names to identify the observations.

# Or more generally:
MyData <- read.table("web-URL/DataFileName.data", sep=",",head=T,row.names=1)

# To list the data sets in all active (loaded) libraries enter:
data()

# To list the datasets contained in a particular package enter:
data(package="ISLR")

# to list the datasets in all packages available, enter:
data(package = .packages(all.available = TRUE))

# I have also setup a web site to facilitate access to the ISLR lectures, both in PDF form and in videos

browseURL("http://fs2.american.edu/alberto/www/analytics/ISLRLectures.html")


#### 1. R Introduction ####


### 1.1 R Overview

# Notes:

# 1. A comment in the R code starts with a # which tells R that it is not an actual R command, but just some documentation text.

# 2. To run an R command, either type it below in the R console, or select the line in the script and press Ctrl-Enter.

# R is a dialect of S. S was created in the Bell Labs by John Chambers and his team as an object oriented language to do data analysis, statistical modeling, simulation and graphics. 

# R is a dialect of S. In other words, R is an improved version of R with very flexible and powerful capabilities to model problems and analyze data. R is different than S, but it was conceptualized as an improvement to S, so most S code will run fine in R.

# The name R comes from their two creators Ross Ihaka and Robert Gentleman (what a great name) from the University of Auckland, New Zealand.

# R is an "open source", "object-oriented" software programming language written specifically for data analysis:

# What is "open source" software? It is software developed by a community of volunteers, who also maintain and update the software and all related packages and datasets. Not all open source software is free of charge, but R is free. The R community material is available at the CRAN site.

# What is CRAN? It is the "Comprehensive R Archive Network" and it serves as the central repository of R software, documentation and other R resources.

browseURL("https://cran.r-project.org/")

# What are "R packages" and "libraries"? The power of R resides in R's packages. A package is a bundle of functions, routines and datasets developed by members of the R community and available for free. There are literally thousands of R packages readily available for download, installation and use, with just about any possible statistical routine you can imagine. An R package may contain one or more libraries that group various R functions. An R package must be installed once and should be updated every so often. A library must be loaded into the R work environment before it can be used. The library will be active and available to use until you close down R. When you re-start R, you need to re-load the library if you want to use it again -> Install once -> update when needes -> load each time you want to use it.

# What is "object-oriented"? Most modern software languages are object-oriented. All it means is that EVERYTHING you do in R gets stored in a container called an object. An object has 2 things: properties and methods. Properties are the data contained in the object; Methods are routines and functions that can be used to work with the object. They are all conveniently stored inside the object. 

# For example, in:

lm.fit <- lm(mpg~cyl+disp+hp+wt, data=mtcars) 

# lm() is a linear model (regression) function available in the {stats} package, which loads automatically when you start R. It will fit a predictive model to predict mpg (miles per gallon) using cylinders, displacement, horsepower and weight as predictors. The model is fit using the mtcars dataset available in the {dataset} package, which also loads automatically when you start R. The results of this regression are stored in an lm() object (i.e., the function that created it), which we have chosen to name lm.fit. We could have used any name for the object. 

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

# R Studio has lots of packages and resources to help you work with R. Here are a couple of useful R Studio resources:

browseURL("https://www.rstudio.com/resources/") # R Studio Resources
browseURL("https://www.rstudio.com/resources/cheatsheets/") # Useful cheat sheets


### 1.2 R Packages

# The power of R rests on the thousands of packages already written for R, which are publicly available at CRAN. We discuss how to install packages and load their respective libraries a bit later. To view all R packages, see:

browseURL("https://cran.r-project.org/web/packages/available_packages_by_name.html")


### 1.3 R Studio Overview

# You can run R and write code and develop R scripts directly in the R console. However, like with most software languages, it is best to use an "Integrated Development Environment" (IDE). An IDE givew you a nice environment and tools to make it easier to develop programs. Without question the IDE of choice for R is R Studio, which is also open source and free. Chances are, you are using R Studio right now.

# R Studio provides 4 different windows:

# 1. Top-Left: Script window -- this is where you write, save and open your R scripts

# 2. Bottom-Left: R Console -- this is where you can type R commands directly and where R displays results and messages when you run commands. If you were not using R Studio, you would only see this R Console.

# 3. Top-Right: 3 tabs: (1) Environment window -- this is where you can see any R objects created or opened (e.g., variables, data tables, etc.). Notice that there is also a History tab where you can review all the R commands you have run since you started R. This is a great tool when you are trying various commands in the R Console and you want to re-trace your steps or copy-paste a batch of commands from the History to a script; (2) History -- R Studio keeps track of all the commands you issue from the console. You can highlight and group of commands from the history and paste them into the active script; (3) Connections -- we will not use this tab in this class, but this section helps you setup and access connections to data sources.

# 4. Bottom-Right: Model explorer -- this is where you can view: files in the working directory; plots generated by R commands; packages available and installed; help displays, etc. 

# Special R Studio Packages: most R packages are designed to manipulate and/or analyze data. But R Studio has a number of very useful packages you can load to make your development environment more productive. These include:

# - R Markdown: a package to create documents with marked up text, R code, R output, graphs, etc. 

# - knitr: a companion to R Markdown, which knits your markdown files into HTML, Word or PDF documents.

# - Shiny: a package to build interactive R applications that run on the web.

# - ggplot2: one of the most popular and powerful packages to visualize data

# - dplyr: powerful library to manipulate data

# - tidyverse: is not a package, but a collection of other packages designed for data science work. Installing {tidiverse} will install packages like {ggplot2}, [dplyr], etc. There is a companion book by master trainers, Wickham and Grolemund

browseURL("https://r4ds.had.co.nz/")

# For a complete description of available R Studio Packages, see:

browseURL("https://www.rstudio.com/products/rpackages/")


#### 2. R FOR ANALYTICS ####


### 2.1 General Information About this Script

# The material contained in this script is original by the author, but in a few parts I use R code provided in two very useful books:

# - R for Everyone -- excellent book to learn R from scratch on your own. It not only has a lot of useful R tips, but also several statistical illustrations:

browseURL("https://www.amazon.com/dp/013454692X/")

# - Introduction to Statistical Learning (ISLR) by James, Witten, Hastie and Tibshirani -- an excellent book for statistical learning and predictive modeling, which include R code and sample data sets:

browseURL("https://www.amazon.com/dp/1461471370/")

# Note: the authors offer their book free of charge in PDF format:

browseURL("http://www-bcf.usc.edu/~gareth/ISL/")


### 2.2 General Information on R and R Studio

# Note: CRAN is the Comprehensive R Archive Network, which has tons or R resources. It is located at: 
browseURL("http://cran.r-project.org/") 

# Note, you can also use the shell.exec() to execute external commands, including opeining web sites:

browseUR("http://cran.r-project.org/") 

# But this may not work for Mac users, so if you cannot run the shell.exec() function on a Mac, use browseURL()

# Also, this is a search engine custom taylored to R: 

browseURL("http://rseek.org")

# To search and use open source R code created by others, signup for GitHub then search for R software code - you can copy all you want, it's open source!!

browseURL("https://github.com/")


## Two Ways to Run R Commands

# 1. From the R Console below, just type the command, e.g., type in the console:

?setwd() # which will display help for the setwd() function

# 2. From the script, go to any script line and press Ctrl+Enter (PC) or Cmd+Enter (Mac). Try it:

?setwd()

# When working with R, it is better to keep all scripts and data in one directory. You can have different directories for different projects. Change your working directory to that directory with the "setwd" command (my files for this course are in the C:/AU/Courses/ITEC621/R directory:

setwd("C:/AU/Courses/R Workshop/R")
getwd() # To display the current working directory


## Getting Help

# In RStudio, the help system is located in the "Help" tab

# You can request help for a keyword with ??"<keyword>", e.g.:

??"regression"

# Use ? to get help for a given command, e.g.: 

?lm() # Help on the lm() function to build linear models

# If you need help on a particular function contained in a specific library, you need to load that library first, and then request help. For example, to get help on the ggplot() function you need to first load the {ggplot2} library:

library(ggplot2)
help(ggplot)

# One excellent resource to search anything in R is to go to the rseek.org web site, which is a search engine optimized for R help.

browseURL("http://rseek.org")


## Vignettes -- Important Documentation Supplements

# You will quickly learn that the R documentation on packages and functions can be cryptic and often incomplete. This documentation is written by package developers and they often include just the minimal information requiried to understand the package contents. More complete documentation and code examples are usually provided in additional documents called "vignettes", which are supplementary and optional documentation for packages and functions. Some packages have vignettes and some do not. To find out the vignettes in your installed packages"

browseVignettes()

# To view the vignette associated with a particular package you installed, enter:

browseVignettes("car") # Vignette for the package "car"


## VERY IMPORTANT: R is Case Sensitive !!!
                                                              
# Be sure to type upper or lower case as required  For example GetWD() or GETWD() won't work!! You must type all commands and library names exactly as they are spelled -- e.g., getwd() will work!!


## Comments

# The # sign is use to write non-executable comments. Don't rely on your memory and document your scripts heavily. You'll thank me for it later. You can comment like I'm doing in this text, or you can simply add a sign after an R command to document that command. Everthing in the command line after the # sign will be ignored during the command execution.

# This can be used to suppress the execution of an R command without deleting the command (i.e., commented out). For example:

# library(ISLR)

# There are many great resources for learning R. There are two very good  video lectures in Lynda.com (access from the AU Portal:


## Tutorials

# Datacamp -- an excellent source of high quality tutorials
browseURL("https://www.datacamp.com/")

# Datacamp R course:
browseURL("https://www.datacamp.com/onboarding/learn?technology=r")

# Datacamp Python course:
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


### 2.3 Packages, Libraries and Views


## Installing Packages

# Packages

# You can write lots of very powerful R Scripts, which is great, but the real power of R is in the thousands of R packages that others have already written, which are public and free.

# In the R documentation, it is customary to refer to packages with curly brackets, e.g., {base}. Functions or libraries contained in these packages are usually specified like this: lm(){stats}

# Base Package

# R comes with a pre-installed {base} package with lots of useful libraries
help(package="base")

# All other packages need to be installed and activated. 

# For available functions in the {base} package, see:
browseURL("https://stat.ethz.ch/R-manual/R-devel/library/base/html/00Index.html")

# Similarly, R comes with several included data sets with the base installation, which are contained in the {dataset} package

data(package="datasets")
library(help="datasets")

# Using a package is a 2 step process: 

# 1. You first need to install the package. You only need to do this once and the package will remain installed until you uninstall it. 

# From time to time, you may want to check to see if a package was updated. To update all packages you can use the update.packages() function.

?update.packages 

# To update a specific package type update.packages("packagename"). 

# To view installed packages enter installed.packages()

# 2. To use a library contained in a package, you need to load the library. It will remain open until you quit your R session. To load a library enter library(libraryname) or require(libraryname).

# Please note that some packages are installed automatically with the standard installation of R. There is no need to further install these packages. Also, some packages have dependencies (i.e., a given package needs other packages to work). In this case, all the dependent packages also get installed automatically.

# Review the help documentation for these functions:

?install.packages()
?remove.packages
?update.packages
?installed.packages()
?library()
?require()

# You can also install packages by clicking on the Packages tab in the model viewer, clicking Install, and then following the steps. Similarly, you can also activate packages by checking the corresponding box in the Packages tab.

# You rarely need to uninstall packages, but if you wish to you can do it by unchecking the corresponding box in the Packages pane, or using the command:

remove.packages()

# You may sometimes need to deactivate or unload packages. You can do this by unchecking the corresponding box in the Packages pane or entering this command (this does not uninstall the package, it just unloads it from the work environment):

detach("packages:PackageName")

# Masking

# Important note about detaching packages. When you load one package, its functions may mask functions from other previously loaded packages, if they have the same function name. Most of the thime, masking is not an issue, but ocassionally there may be two functions with the same name in memor from different packages and one may mask the other. For example, the summarize() function is available in these two packages:

require(plyr)
require(Hmisc)

# Each does something different. Check it out:

?summarize

# If you want to use summarize() from the {plyr} library, you can do this 2 ways: (1) detach the {Hmisc} library; or (2) indicate with double columns which function you wish to invoke, 

# e.g., plyr::summarize().

# To list all installed packages enter:
installed.packages()

# To view just the package names (column 1)
installed.packages()[,1]

# Then get help on that package
help(package="base")

# To view all available packages go to:
browseURL("https://cran.r-project.org/web/packages/available_packages_by_name.html")

# To list available libraries enter:
library()

# To get documentation and a list of functions in a given package enter:
library(help=PackageName)

# To view the data sets available in all active packages enter:
data()

# To view the data sets in a particularly active package enter:
data(package="ISLR") # Example for the ISLR package

# To load a specific data set in to memory
data(dataSetName) 

# to list the datasets in all packages available, enter:
data(package = .packages(all.available = TRUE))


## Installing Views

# Views are collections of packages by function or type of analysis. You can view the various packages that are contained in a view at

browseURL("http://cran.r-project.org/web/views/")

# Once you find a view and wish to install the packages it contains, installing the view automatically installs all the packages it contains, if not already installed. In oreder to install views, you first need to install and load the CRAN Task View "ctv" package:

install.packages("ctv")
library("ctv")

# Then, for example, if you want to install the "Graphics" view, use:

install.views("Graphics") # which will install all these packages:
browseURL("https://cran.r-project.org/web/views/Graphics.html")


### 2.4 Reading Data Into R


# It is recommended to place all your data sources in your Project working folder.

# Many data sets, and most of the ones we will use in this class are contained in packages, so the data becomes available after you load the respective libraries. To view all data sets that either come with R or are avilable in loaded libraries enter:

data()

# To view the first 6 rows of a data set and the respective column name for example "mtcars" enter:

head(mtcars) # Car and gas consumption data
tail(mtcars) # Shows the last 6 rows in the table

# You can also double-click on the data set in the environment viewer on the right to view the full table. You can also view the full table with the fix() function, which also allows you to edit the data:

fix(mtcars) # Check it out

# While you can read multiple data sources into R (such as SPSS, SAS and Excel), I find it a lot easier to manipulate data sets in Excel and then save the datasheet as a "comma separated values" or "CSV" file. For example, many of the packages to read Excel data into R require current versions of Java or certain versions of Excel, whereas CSV files work universally.

# If you want to read a "csv" file from your working directory, enter:

heart <- read.table("Heart.csv", header=T, row.names=1, sep=",")

# For, example, this command above reads the Heart.csv data file (which must be in your Project Environment) into an R data frame named "heart". The "header=T" attribute specifies that the first row in the "csv" file contains the column or variable names, which is the recommended way to do it. row.names=1 specifies that the first column does not have data, but row names, which is useful if you need to identify observations (think of this as a primary key). The sep="," attribute specifies that the values are separated by commas. 

# If the "csv" file is not located in the working directory, you need to specify the directory using forward slashes "/" (don't use the backslashes "\"). Ex:

mydata <- read.table("c:/au/courses/RWorkshop/R/Auto.csv", header=T, sep=",")

# or simply 
mydata <- read.table("Auto.csv", header=T, sep=",")

head(mydata) # To see the column headers and first few records
names(mydata) #To see the column (i.e., variable) names in the table


## Excel Files (TRY ON YOUR OWN)

# There are many packages to read Excel files, but I've had trouble with all of them at one point or another. The easiest thing is to save your original data source as an xlsx file and do all your data manipulation in Excel and when you are ready, save a copy as a CSV file for your analysis.Several experts recommend this too.

# Again, it is much easier to save your Excel file as a "csv" file, but if you feel adventurous, here are some packages to read Excel files:

install.packages("xlsx")
install.packages("xlsxjars")
install.packages("rJava")
library(xlsx)
require(xlsx)
mydata <- read.xlsx("c:/myexcel.xlsx", 1)

# read in the worksheet named mysheet
mydata <- read.xlsx("c:/myexcel.xlsx", sheetName = "mysheet") 

# Also try this (ON YOUR OWN):

library(gdata)                   # load gdata package 
help(read.xls)                   # documentation 
mydata = read.xls("mydata.xls")  # read from first sheet


## MS Access files (TRY ON YOUR OWN): 

# The best thing is to manage your data directly in MS Access. Then, open any data table you may need, or run an SQL query to produce the data you need for your analysis. Then with the data table or query results open, select the External Data tab in Access. Then, in the Export section of the ribbon select "text file" and follow the instructions. You can save the file as a .txt or .csv file. It makes no difference. Just remember the full file name.


## SPSS Files (TRY ON YOUR OWN):

# You will first need to load the library "foreign" (comes with R). The foreign library has functions to read other data formats:

library(foreign) 

# Then read the file
mydata <- read.spss(file = "MyDataFile.sav", to.data.frame= TRUE, reencode = TRUE)


## Other statistics data files:

# You should also be able to read other statistics data files with the foreign library issuing commands similar to read.table, but instead use: 
# read.spss for SPSS; read.dta for Stata; read.ssd for SAS; read.octave for Octave; read.mtp for Minitab and read.systat for Systat

# Personally, my preference is to convert data to the general format .csv which is much easier to manipulate (in Excel or R) and read. Many data import routines don't work as adverstised, so .csv is a safe option

# Data From Web Sites (TRY ON YOUR OWN)

require(XML) # You need this library
URLToRead <- "http:// etc" # This is the URL for the HTML data
MyData <- readHTMLTable(URLToRead, which=1, header=FALSE, stringsAsFactors=FALSE)

# The which argument specifies which table to read if there are more than 1 Specify header=TRUE if the table has a header


## Saving Data from R to a file

# Saving a data frame to a "csv" file

write.table(mydata, "mydata.csv", sep=",") # For CSV files.
# The "sep" option specifies the delimiter between values, with "," being the most common and one the Excel will read

# Saving to an Excel file

library(xlsx)
write.xlsx(mydata, "mydata.xlsx") 

# Saving to an SPSS or SAS file

library(foreign)
write.foreign(mydata, "mydata.txt", "mydata.sps",   package="SPSS") # SPSS 
write.foreign(mydata, "mydata.txt", "mydata.sas",   package="SAS") # SAS
write.dta(mydata, "mydata.dta") # Stata


### 2.6 Working with R Objects

# R is an "object-oriented" softwre language, therefore, it treats everything as objects. An object has two things: 

# (1) data; and 
# (2) programs or functions, encapsulated toghether. 

# When you create an object, you can then access it to run its programs/functions and read or manipulate its data. Objects can be used to create other objects using a property of object-orientation called "inheritance", such that the data and programs from one object are inherited by another.

# For example, we can run a regression model to predict miles per gallons using the mtcars data set:

lm(mpg~cyl+disp+hp+wt, data=mtcars)

# Or, we can store the model formula in a formula object and then use this object throughout the script. This is convenient when you need to specify a very long and complex formulat and use it in several parts of your script:

lm.formula <- mpg~cyl+disp+hp+wt
lm.formula # Check it out -- it's just a text string with the formula, which we can now use
lm(lm.formula, data=mtcars)

# The lm() function above creates an lm() object, but it isn't stored anywhere. To store it for later reference, assign it to an object with the <- assignment operator:

lm.object <- lm(lm.formula, data=mtcars)

# Two important functions that are used pervasively to retrieve important data from objects are:

# summary() -- displays key data stored in an object. What gets displayed changes from one object type to another

# str() -- shows the "structure" of the object, i.e., just about all the variable name and sample data stored in the object

summary(lm.object)
str(lm.object)

# Alternatively, you can get the summary() and str() without creating objects, but this is not so useful:

summary(lm(mpg~cyl+disp+hp+wt,data=mtcars))
str(lm(mpg~cyl+disp+hp+wt,data=mtcars))

# If you want to store the summary() for later use simply assign it to an object:

lm.summary <- summary(lm.object)
lm.summary

# Notice in the display in the str() results that there are a few values prefixed with $. To access specific data elements from an object we can use the $ symbol. The $ symbol is used to point to variables or columns in data objects.

# For example, these show the regression coefficients
lm.summary$coefficients # or
summary(lm.object)$coefficients

# This shows the residuals:
lm.summary$residuals # or
summary(lm.object)$residuals

# Note: you can use the options() function to change how R reports results to some extent. The "scipen" attribute is useful to convert scientific notation (e.g., 1.8e-14) to actual numbers (e.g., the number with 14 zeros after the decimal point). Try it

options(scipen="20") # scipen = Scientify Notation Penalty
summary(lm.object) # Check it out

# The scipen (i.e., scientific penalty) value tells R to display the value in scientific if there are more than 20 zeros after
# the decimal point. Now try

options(scipen="4")
summary(lm.object) # See the difference

# Programs/functions that do something with objects have round brackets after them, e.g., mean(), length(), summary()

# The formula mpg~cyl+disp+hp+wt is a typical model formula using R syntax. This function notation is used in various R commands.

# The symbols <- represent an "assignment" in R. The lm.fit formula above runs the lm() linear model function and assigns the result to an object named lm.fit. 

# In many cases the "=" operator can be used interchangeably with "<-". Many serious R developers don't like to use the "=" sign because it has a special meaning in some other operations (e.g., testing for equality), but I use it all the time for convenience. See:

# This works just like with "<-"
lm.object = lm(mpg~cyl+disp+hp+wt, data=mtcars) 

# For example:
x <- 3+4 # Look at the value of x in the Environment window
x # Check it out
# The value resulting from computing 3+4 has been assigned to the variable x

# These 2 assignment commands do the exact same thing:
x = 3+4
x # Check it out
3+4 -> x
x # Check it out

# If x does not exist, it is created in the working memory of R, which is called the workspace. To see the content of x, it must be called (i.e., executed):

x 
str(x) # To see what else is contained in the object x

# Please note that x <- 3+4 is DIFFERENT than x < -3+4. The first entry assigns 3+4 to the variable object x. In the second entry we are evaluating if the object x is smaller than -3+4. One space shift can make a huge difference. So, please ensure that when you use the assignment operator <- there is no space between the < and the -

# Exploring objects

# In R Studio, the content of the workspace is shown in the Environment tab (top-right). Its contents can also be displayed in the console with the ls() and objects() commands 

ls()
objects()

# To delete an object, we pass it as parameter to the rm() function (for remove) by writing it in the brackets of the command: 

rm(x)


### 2.7 Variables, Data Types, Objects, Classes and Data Structures


## Overview

# R, like all statistical software, uses many different data/variable types and classes. Understanding how to work with different data types and classes is key to understanding how to build and interpret models in R.

# A variable of a certain type (e.g., character) will contain data of the same type.
# A variable class and type are two different things. For example:

x = 2.3
class(x) # The class is numeric
typeof(x) # The type is more specific -- double (i.e., with decimals)

# The best way to think of a variable is as a container. They can contain just about anything, e.g., a single value, a vector, an array of values a graph, results from an analysis or any R object:

# What's the meaning of the dot (.)?

# Note: a "." in a variable name is simply part of the name and nothing else; e.g., x, lm.fit, my.data are valid variable names

## Simple Data Variable & Data Types

# These contain single values of a given type, e.g., numeric, character, factor, date, and logical. You don't need to declare a variable's type. The variable type is implicitly (automatically) declared when a value is stored in the variable. Certain data types require delimiters, such as double quotes " for character (i.e., text) data. It is often useful to check or display a particular variable data type:

## Numeric Data

x=2.4 # Automatically declares x as numeric
y=as.integer(2) # Declares y as an integer 
x 
y

class(x) # Check it out
typeof(x)
class(y) 
typeof(y)

is.numeric(x) # Check if x is numeric
is.numeric(y) # Check if y is numeric
is.integer(x) # Check if x is an integer
is.integer(y) # Check if y is an integer

## Character Data

# Note: characters are case sensitive

x="Alberto"
x
class(x) # Check it out
typeof(x)
is.numeric(x) # Check that x is NOT numeric
is.character(x) # Check that x is character
nchar(x) # Number of characters in the value of x
nchar("Alberto") # Number of characters in the literal text "Alberto"

## Date Data

x=as.Date("2016-10-20") # Converts a text string into a date value
# Note that the word "Date" is capitalized
x # Check it out

# Also note the date format is YYYY/MM/DD, which can be changed as follows:
x=as.Date("10/20/2016", "%m/%d/%Y") # Use upper case %Y for 4-digit years
x # Check it out
x=as.Date("10/20/16", "%m/%d/%y") # Use lower case %y for 2-digit years
x # Check it out
class(x) # Check it out
typeof(x) # Note that dates are stored internally as numbers

# Note: you can use other date formats with: %a abbreviated weekday; %A weekday; %b abbreviated month; %B month.

# Also, you can use the as.POSIXct() function to read date and time:

x=as.POSIXct("2016-10-20 17:30") # Note that a space is needed after :
x # Check it out

# You can subtract dates and add days to a date: 
born=as.Date("2000/2/12")
born # Check it out

today=Sys.Date()
today # Check it out

aWeekAgo=today-7
aWeekAgo

aWeekFromNow=today+7
aWeekFromNow

ageInDays=as.numeric(today-born) # Need to convert to numeric
ageInDays

age=floor(ageInDays/360) # The floor() function rounds down decimals
age

## Logical Data

# Logical variables can be either TRUE or FALSE. Numerically, TRUE is identical to 1 and FALSE to 0. Logical variables and values are important to evaluate conditions

TodayIsMyBirthday=FALSE # Can use F instead of FALSE
class(TodayIsMyBirthday)
typeof(TodayIsMyBirthday)
is.logical(TodayIsMyBirthday)
2==3 # The double == is used to evaluate if two values are equal
2!=3 # Evaluate if 2 is not (!) equal to 3
2<3 # Evaluate if 2 is smaller than 3


## Factors

# Text is difficult to process quantitatively without some transformation. 

# A factor is a special type of text data, which is like a category, but unlike free text, factors have a fixed number of unique values that repeat throughout the data.

# Understanding factor variables is key in understanding regression models with categorical data and classification models (e.g., logistic regression) which predict the likely classification of an observation.

# For example, house location types. Say, if there are 3 types of house locations: Rural, Urban and Suburban, we can create a factor for this data that finds the unique factors (i.e., text values) to categorize houses by location type. The factor conversion also assigns a number to each category, so that you can process things quantitatively. Take this text string vector:

x <- c("Rural","Urban","Suburban","Urban","Urban","Suburban")
x # Notice that Urban and suburban are repeate, as you would expect
y <- as.factor(x) # Now, Convert x into a factor variable
y # Check out the 3 factors extracted out of x
levels(y) # Display the unique categories in the data
as.numeric(y) # Check the unique number assigned to each factor


## Complex Data Structures

# R is very rich on data structures. The simple variables and data types discussed above are simple data structures. But there are more complex data structures like vectors, matrices, data frames and lists, which give R a lot of power for data manipulation. Understanding how these data structures work is key to unleashing the power of R for data analysis.


## Vectors

# A vector is simply a list of values, but all values must be of the same type (e.g., character, numeric, date, etc.)

#  R is said to be a "vectorized" language, meaning that many values are stored in vectors and that R has many convenient features to manipulate data contained in these vectors.

# The Popular c() Function

# The "c" function is used to create vectors. The "c" means to "concatenate", "create" or "combine" values into a vector. Vectors are convenient ways to store groups of values of the same type (e.g., coefficients, residuals, predictions, etc.)

x <- c(1,2,3,2)
x # Notice that all values are numeric
class(x)
typeof(x)

# R tries coerces data to be of a given type when values are incompatible. For example, if you try to create a vector with a number and a character, R corrects this:

z <- c(1,"al")
z # Notice that "1" is converted to character to be compatible with "al"

# Note that we already had a variable called x. The command avobe replaces the prior value of 
# We can extract individual values from a vector with an [index]

x <- c(1,2,3,2,6,3,5)
x # Check it out
x[4] # 4th element of the vector

# A negative index removes an element from the vector
x <- x[-4] 
x # The 4th vector element got removed

# One of the nice things about R vectors is that we can manipulate all the values in a vector with a simple command. For example:

y <- x*3 # Multiplies every value of the x vector by 3
y # Check it out

# Vectors can also hold text values (need to enclose the text in quotes)

x <- c ("ITEC 610", "ITEC 620", "ITEC 621")
x # Try it
x[3] # Try it

# You can also give names to each vector element, in 2 ways

# Directly:
x <- c(fname="Alberto", lname="Espinosa", title="Professor")
x # Check it out

# Or with the names() function
x <- c("Alberto", "Espinosa", "Professor")
x
names(x) <- c("fname", "lname", "title")
x


## Data Frames

# Data frames are fundamental to understanding how to access and manipulate data in R. The easiest way to understand a data frame is to think of it as an Excel sheet with various columns each with a column name (or think of it as a database table).

# Another way to think of it is as table composed of vectors. That is, columns can contain different data types. But the data in one column has to be of one type (i.e., a vector)

# $ is NOT money in a data frame, but it is how you extract a vector column from a data frame table. For example, a column named LastName in a data frame called Employees can be accessed using this name: "Employees$LastName" (note: in MS Access we would access this column with Employees.LastName) 

# Also, every row and column in a data frame is "indexed". It is very important to understand the use of indices in R for data manipulation. For example, you can access:

# Employees[1,] -- First row (all columns) 
# Employees[,1] -- First column (all rows)
# Employees[2,3] -- Element in 2nd row and 3rd column
# Employees[3:10,2:4] -- Rows 3 through 10 from columns 2 throgh 4

# IMPORTANT: a thorough understanding of data frame and vector indices is key to understanding sub-sampling, cross-validation and machine learning. Try to understand this well.

# Let's practice with indices. First create a vector that will have the row indices for a data frame we will create shortly called MyDataFrame We will call this vector MyIndex (be careful with the capitalization)

MyIndex <- 1:8 # Creates the MyIndex vector with values 1 to 8
MyIndex # Check it out

# Now let's create a vector with 8 course numbers

courses <- c("ITEC 610", "ITEC 616", "ITEC 620", "ITEC 621", "ITEC 660", "ITEC 670", "KSB 620", "KSB 621")
courses # Check it out

# Now let's create a vector with the pre-requisites for these 8 courses
prereqs <- c("None", "None", "ITEC 610", "ITEC 620", "ITEC 610", "KSB 065", "ITEC 620", "ITEC 621")
prereqs # Check it out

# Now let's create the data frame
MyDataFrame <- data.frame(MyIndex, courses, prereqs)
MyDataFrame # Check it out

# Use the "$" sign after the data frame to reference a single column
MyDataFrame$courses

# Lets look at just part of the data frame
head(MyDataFrame) # Display the first few rows
tail(MyDataFrame) # Display the last few rows
MyDataFrame[2,c("courses","prereqs")] # To list selected columns for row 2
MyDataFrame[,c("courses","prereqs")] # To list selected columns for all rows


## Matrices

# A matrix is identical to a data frame in most respects (i.e., a table with values), except that all values must be of the same type in the entire matrix. Most often matrices contain only quantitative values, which can be easily manipulated with matrix algebra. 

# Matrices are important in R because some statistical routines, like some correlation functions" only work with quantitative matrices. This is how you create a matrix:

x.mat=matrix(1:10, nrow=5)
# This command creates a matrix with 10 elements organized into 5 rows (i.e., the matrix has 2 columns, so it is a 5x2 matrix). 
x.mat # Check it out

# You could also accomplish the same thing with:
x.mat=matrix(1:10, ncol=2)
x.mat # Check it out

# You can name the columns and rows of matrices:
colnames(x.mat)=c("Ref No.", "Sales")
rownames(x.mat)=c("John", "Judy", "Sally", "Moe", "Maria")
x.mat


## Lists

# A list is similar to a vector but it can contain data of different types. Vectors can be used to create columns in data frames (i.e. tables), lists cannot

y <- list(name="Alberto", title="Professor", age=16)
y # Check it out -- the data types are preserved

# A list can be quite complex because its elements can be anything, that is, single values, vectors, other lists. For example (notice that we have name lists inside of a list):

friends <- list(MyName="Alberto", MyAge=15, Charlie=list(age=20, major="Analytics"), John=list(age=30, job="Programmer"), Dan=list(age=40, profession="Lawyer"), Others=c("Joe", "Moe", "Doe"))

# You can extract distinct elements from a list with the $ or [[]]

friends # Check it out
str(friends) # Inspect it
friends$John # Let's get all the data for John
friends$John$job # Let's get John's job
friends[["John"]]


## Data Reshaping

# If you have various columns (i.e., vectors) of data of the same type and you would like to combine them into a matrix you can use the "cbind()" function. Notice below how all the values are automatically converted to text because MyIndex values cannot be numbers -- all elements in a matrix must be of the same type.

My.Matrix <- cbind(MyIndex, courses, prereqs)
My.Matrix
class(My.Matrix)

# Or, if you would like to combine the vectors into a data.frame, you can use the data.frame(). Notice how all the values can now be of different types

My.DataFrame <- data.frame(MyIndex, courses, prereqs)
My.DataFrame
class(My.DataFrame)


## Subsampling

# The conceept of subsampling is "CENTRAL" TO CROSS-VALIDATION AND MACHINE LEARNING.

# Note: we will discuss this in more depth in Predictive Analytics, but here is some R code to get you thinking about machine learning

library(MASS) # Contains the Boston housing data set
nrow(Boston) # This function counts the total rows in the Boston dataset = 506

set.seed(1) # This command sets the first number of the random number generator. You can use any number instead of 1. Use the same seed number over and over to get the same random sample each time (i.e., repeatable results), or use different seed numbers each time if you want the samples to change.

# Let's genterate a subset of the Boston data set containing a subsample of 70% of the observations selected randomly. In Machine Learning, we would use this 70% to "train" the model, which we would then "test" with the remaining 30%. We will cover this in more depth in ITEC 621, but let's see how to draw the 70% train subsample:

# First, let's create a vector named "train" to serve as an "index" to select random rows from the data set. The next command takes the sequence of numbers from 1 to nrow(Boston) (i.e., 1,2,3,,,,506) and creates a random sample vector with 70% of these values. This vector will be used later to select 70% of the rows or observations from the Boston data set. 

train <- sample(1:nrow(Boston), 0.7*nrow(Boston))

# Again, these are NOT observatiosn, but simply a bunch of numbers drawn randomly between 1 and the number of observations in the data set. We will use these numbers as "indices" to define the training and test data sets.

train # Check it out
length(train) # 354 (70%) observations in the train data set

# You can now use the index [train,] to select the 70% training observations from the data set. Let's draw the subset and store it in a new data frame object named Boston.train:

Boston.train <- Boston[train,]

# Note that we used a "," in [train,] above. This is a subtle thing, but we must use a "," when selecting a subset from a data frame. The "," followed by nothing tells R to select all the columns

Boston.train # Check it out
nrow(train.subset)


### 2.8 Working with Text

# We will not be doing text analytics in this course, but R is very powerful for manipulating text, some of which is illustrated next. 

# You can Concatenate text with the "paste()" function

# The default separator in the paste() functin is a blank space
MyText <- paste("My", "Name", "is", "Alberto") # To store it in a variable
MyText # Note that sub-strings are separated by a blank space by default

# You can change the separator with the sep= attribute
paste("My", "Name", "is", "Alberto", sep="-") # to change default separator to "-"
paste("My", "Name", "is", "Alberto", sep="") # to eliminate the blank space

# You can concatenate literal text enclosed in quotes with text contained in variables without quotes:

MyName ="Alberto" # To create a variable that contains text
MyCourse = "ITEC 621" # And another variable
paste("My", "Name", "is", MyName, "and I teach", MyCourse) # To embed variables

# sprintf() does the same, but it places variable contents in each %s

sprintf("My Name is %s and I teach %s", MyName, MyCourse)
        
# In contrast to "paste()", the "c()" function concatenates into a vector

MyText <- c("My", "Name", "is", "Alberto") # Creates a vector with 4 elements
MyText

# Which you can then collapse into a single text string

paste(MyText, collapse="") # To concatenate the verctor into a string with no spaces
paste(MyText, collapse=" ") # Or with spaces

# Extracting Data from Documents

# You can get HTML data from documents and the web with the {XML} package

library(XML) # Activate the package when you need it
?xml # Check it out

browseURL("http://www.loc.gov/rr/print/list/057_chron.html")

URL.Location <- "http://www.loc.gov/rr/print/list/057_chron.html"

# The readHTMLTable() function can be used to get data from HTML tables

?readHTMLTable # Check the documentation

presidents <- readHTMLTable(URL.Location, which=3, as.data.frame=TRUE, skip.rows=1, header=TRUE, stringsAsFactors=FALSE)

# Note: the "which" parameter specifies which table in the html file, if more than one; "stringsAsFactors" converts strings into factor variables, say FALSE for now

names(presidents)
head(presidents)

# TO retain only some rows, 1-64 for example:

presidents <- presidents[1:64,]
tail(presidents)


### 2.9 Functions

# Functions can be either "built-in" (available in the base package or other installed packages); pre-programmed in packages, or "user-defined" (written in your R script)

# Built-In Functions -- there are thousands of these, e.g.,

x=c(2,3,6)
# One example of a simple built-in function in the {base} package
mean(x) 

# Pre-programmed functions in packages -- each package has its own programmed functions, which are at the core of what we do with R packages. For example, the lm() function we used above is a function that fits a linear regression model, which is avaliable in the {stats} package, which loads automatically when you start R.

# User-Defined Functions

# These involve two actions: 

# (1) creating/defining the function, and then 
# (2) invoking (i.e., using) the function when needed

# You can create any function with the "function()" function (pardon the redundancy). The steps that the function executes are enclosed within curly brackets {}. To run a function you have to highlight and execute all the commands (in between { and }) associated with the function, which will load the function commands into memory. Once you do this you can use the function any time until you shut down R.

# Some functions require parameters, others don't. A parameter is a value in the function that requires that we pass some value(s) to the function, which the function then uses to calculate something. For example:

ls() # A built-in function that lists all active objects and does not require parameters
mean(x) # A function that requires a vector x as a parameter

# Let's create a simple function that displays "This is my R World!!" We will call this function "MyWorld()" and will not require any parameters -- i.e., there will be nothing inside the "()"

# We first need to create the function. The function commands need to be written within curly brackets "{}"
# Highlight the two lines that follow and run them with Ctrl-Enter

MyWorld <- function() # No parameters/arguments
{print("This is my R World!!")}

# Then run the function as needed
MyWorld()

# Once a function is created and executed, it is available for use until you terminate the R session

# Now let's write a function with parameters/arguments:

AnyonesWorld <- function(who) # "who" is the parameter
{print(paste("This is ", who, "'s R World", sep=""))}  

# Now execute the function but enter who's world it is (change the name)

AnyonesWorld("Alberto") # Requires that we enter an argument
AnyonesWorld("Joe") # Try any name you wish
AnyonesWorld("Sally")

# Functions are useful when you need to do complex calculations and return the results. For example to write a function that will return the squared value of a number x, we make x the parameter and then specify what we want to return, i.e., it's squared value

SquareMe <- function(x) # Whatever we enter in SquareMe(x) will be squared
{return(x^2)}

# Run the function definition above and then see how it works:
SquareMe(4)

# Note: the function above is a simple illustration in which the function fits in one line. Typically, functions span many lines, in which case it is customary, for readability, to put the open and closing curly brackets in separate lines, For example, let's create a function: that takes a value, then adds 2 and then squares this sum

# x is a parameter we are passing to the function:

SquareMePlus2 <- function(x) 
  {
  y=(x^2)+2
  return(y)
  } 

SquareMePlus2(4) # i.e., (4^2)+2 = 18


### 2.10 Program Control: If'S 

# Most R scripts for this course will simply have a "stack" of commands that will execute sequentially. You can execute command stacks in full by highlighting the respective lines and then Ctrl-Enter or one line at a time. However, there will be times when you want to execute some lines only if some condition is met, or there may be times when you want to execute some lines multiple times in a loop. This is called "conditional logic" in software programming. In such cases, you will need to understand how to control the program execution sequence (i.e., logic). There are several types of program controls, but the most common ones are: if/else's and loops.

# if/else 

# Example, change the value of MyValue to see how the if control works

# Notes: 
# (1) the "if" condition must be inside the curly brackets
# (2) the "else" condition too; IMPORTANT: the "else" statement must be in the same line as the first } or the command will fail

MyValue=2
if (MyValue>10) 
  {print("Your number is large")} else
  {print("Your number is small")}

# The "ifelse()" function works like if in Excel

MyValue=5
ifelse(MyValue > 10, "Large Number", "Small Number")

MyValue <- c(5, 10, 15, 20) # Works with vectors too
ifelse(MyValue > 10, "Large Number", "Small Number")

# This is a more complete example that gets several input values for a loan application, does a number of calculations and then makes a decline/approve decision recommendation. 

# Input data:

CarPrice = 30000
DownPayment = 2000
LoanYears = 5
AnnualInterest = 0.04 # i.e., 4%
AnnualIncome = 60000
MonthlyObligations = 2300

# Calculations

LoanAmount = CarPrice-DownPayment
LoanMonths = LoanYears*12
MonthlyInterest = AnnualInterest/12
MonthlyPmt = LoanAmount*MonthlyInterest/(1-(1/(1+MonthlyInterest)^LoanMonths))
MonthlyIncome = AnnualIncome/12
DisposableIncome = MonthlyIncome-MonthlyObligations
PmtToDisposableRatio = MonthlyPmt/DisposableIncome

# Displaying results

print(sprintf("Your monthly payment is %s", MonthlyPmt))
print(sprintf("Your disposable income is %s", DisposableIncome))
print(sprintf("Your monthly payment to disposable income ratio is %s", PmtToDisposableRatio))

if (PmtToDisposableRatio>0.2) # i.e., loan payment is more than 20% if income
  {print("Your loan application has been declined")} else
  {print("Your loan application has been approved")}

# Note: The script commands are comingled with the results in the console. This is difficult to avoid in R or R Studio, but it is easily resolved with R Markdown, which we will see shortly

# It is often helpful to break the lines for readability. This works the same:

if (PmtToDisposableRatio>0.2) # i.e., loan payment is more than 20% if income
  {print("Your loan application has been declined")
  } else
  {print("Your loan application has been approved")}


### 2.11 Program Control: Loops

# Loops allow you to perform a number of commands several time until some condition is met to terminate the loop. Typically, there is an index or value that changes in each loop. For example, we may want to perform a bunch of calculations on the first 10 rows of a table.

# There are various types of loops, but the most common are "for" and "while" loops. "For" loops performs a loop "for" each of the values specified. When the values end the loop ends. In contrast "while" loops will continue to loop "while" a certain condition is met. I illustrate both below.

# CAUTION: Improperly written loops are one of the most common sources of software malfunction. In particular, loops that do not specify the "for" or "while" condition correctly may cause a program to go into an infinite loop that never terminates. You ever wonder why a program spins and spins some times. Most likely, it is an infinite loop somewhere in the code.

# Example of a "for" loop

# In the first loop, i will take the value of 1. In the second loop it will take the value of 2, etc. In the last loop, i will take a value of 10 and this will be the last loop.

for (i in 1:10) {
  print(paste("The number is", i)) # The paste function concatenates strings
}

# If you want i to increment by more than 1 you can use the seq() function, in the example below i takes values in the sequence from 1 to 10, but in increments of 2:

for (i in seq(1,10,2)) {
  print(paste("The number is", i)) # The paste function concatenates strings
}

# Example of a "while" loop -- the loop runs while the condition is true While loops generally require initializing a value that will be checked in the while condition, in this case the value is i

i = 1 # This is like a counter, which we initialize to 1
while (i <= 10) { 
  # The paste function concatenates strings
  print(paste("The number is", i)) 
  # we need to increment i in each loop or you will have an endless loop
  i <- i+1 
}

# Note: omitting the counter increment command "i <- i+1" would be an example of a software error that would cause an "infinite loop". Can you see why?


### 2.12 Simple Statistics with R


## Working with Data

# Let's first open a data file
require(ggplot2) # Contains the "diamonds" data set
# This data set contains attributes for several diamonds
data(diamonds) 

# Note: certain R commands and models require that a data set be active in memory, which we accomplish with the data() function. Also, note that the diamonds data table is contained in the ggplot2 package.

head(diamonds)

# Let's get diamond price means by cut
aggregate(price~cut, diamonds, mean)
# Works like SQL command: SELECT Avg(Price) FROM diamonds GROUP BY cut

# To group by more than one attribute use the + operator
aggregate(price~cut+color, diamonds, mean)

# To aggregate more than one column, use cbind()
aggregate(cbind(price, carat)~cut, diamonds, mean) 

# Note the cbind function binds columns together. To bind rows use rbind


## Simple Graphics

# R Base Package:

# Plots
boxplot(diamonds$carat) # Boxplot of a single variable
plot(diamonds$carat, diamonds$price) # Scatterplot of 2 variables

# Histograms
hist(diamonds$carat) # Histogram of 1 variable

# Let's add some labels
hist(diamonds$carat, main="Carat Histogram", xlab="Carats") # w/labels
hist(diamonds$price, main="Diamond Price", xlab="Price")

# Note: graphics commands can be "high-level" or "low level". A high-level command creates a new graph (usually erasing the prior graph, if any). Low level graphic commands add elements or layers to high-level graphs, and don't erase the prior graphs.

# For example, qqplots are popular for detecting non-normality in the data -- if the dots do not align with the straight line the data deviates from normality:

qqnorm(diamonds$price) # A high-level graph of the qqplot
qqline(diamonds$price) # A low-level graph command to add a qqline

# The data is not very normal. Maybe loging the data will help:
hist(log(diamonds$price), main="Diamond Log(Price)", xlab="Price")
qqnorm(log(diamonds$price))
qqline(log(diamonds$price))
# Yes it helps align the dots to the line

## ggplot2

# Graphics packages -- ggplot2 and lattice are popular ones:

# Note, the ggplot2 package installation will automatically install the lattice package

library(ggplot2) # Activate when needed

# Please note that ggplot2 has a unique syntax. This syntax derives from a well established "Grammar of Graphics" by Wilkinson (2005) and the "Layered Grammar of Graphics" (Wickham 2010)

# Here is an excellent book on how to use ggplot written by the package author:
browseURL("https://www.amazon.com/dp/0387981403")

# This book states that "the grammar [of graphics] tells us that a statistical graphic is a mapping from data to aesthetic attributes (colour, shape, size) of geometric objects (points, lines, bars). The plot may also contain statistical transformations of the data and is drawn on a specific coordinate system. Facetting can be used to generate the same plot for different subsets of the dataset. It is the combination of these independent components that make up a graphic.

# Histograms -- the geometric object is geom_histogram and the aesthetic is x=carat:
ggplot(data=diamonds) + geom_histogram(aes(x = carat))

# We can save graphs in objects:

# Basic definition, no graph yet:
g <- ggplot(diamonds, aes(x=carat, y=price)) 

# Then add properties, e.g., type of graph, and display it later:
g + geom_point()

# Then change the properties if you wish, e.g., add color
g + geom_point(aes(color = color))
g + geom_point(aes(color = color)) + facet_wrap(~color) # Faceted by color
g + geom_point(aes(color = color)) + facet_grid(cut~color) # Faceted in grids


## Simple Statistics

# Set the seed number first. Random number generators as based on long tables that contain random numbers. The seed is the first number in the random table to use. If you only need one random sample, the seed number is not important -- just pick any seed number. But if you will be re-sampling several times, use the same seed to draw the same sample each time (i.e., when you want repeatable results), or use different seeds to draw different samples each time.

set.seed(1)

# Generating a random sample of 50 observations from a normal distribution

x=rnorm(50) 
x # Check out the sample

# Generating samples and sub-samples

obs = 1000 # Suppose you have 1000 observations

# To select 100 random numbers from 1 to obs (i.e., 1,000)

train <- sample(1:obs, 100)
train

# To select a percentage (e.g., 75%) of random  numbers from 1 to obs

train=sample(1:obs, 0.75*obs) 
train

# Random sample of 100 numbers with replacement
X <- sample(1:1000, size=100, replace=TRUE)

# replace=true means that values in the sample can repeat
X # Check it out

# Bootstrapping: is a statistical sampling method that can be used in many statistical procedures. We will not cover bootstrapping methods in this R tutorial, but bootstrapping is about re-sampling wiht replacement. Suppose you have 100 observations. If you get a sample of 100 with replacement, some of these 100 values will be repeated. If you re-sample another 100 values with replacement, you will get a different sample because the repeated values are likely to be different ones. For example:

boot.x <- sample(1:100, size=100, replace=TRUE)
boot.x

# Other basic statistics in R

x <- sample(1:1000, size=100, replace=TRUE)
mean(x) # Sample mean
median(x) # Sample median
max(x) # Maximum value in the sample
min(x) # Minimum value in the sample
sd(x) # Sample standard deviation
var(x) # Sample variance
summary(x) # Sample summary statistics
hist(x) # Histogram of sample values
qqnorm(x) # QQ Plot of x
qqline(x) # QQ Line of x
summary(mtcars) # Summary statistics on a dataset
cor(mtcars) # Correlation matrix for mtcars dataset

library(psych) # Has useful statistical functions
describe(mtcars) # Descriptive statistics

library(ggplot2)  # Contains the diamonds dataset
describe(diamonds) # Descriptive statistics

# See what's in the object
descriptive.diamonds <- describe(diamonds)
str(descriptive.diamonds)

# Now extract what you wish
descriptive.diamonds$n
descriptive.diamonds$mean
descriptive.diamonds$sd

# Create a data frame with just what you need
data.frame("N"=descriptive.diamonds$n, "Mean"=descriptive.diamonds$mean, "Std.Dev."=descriptive.diamonds$sd)

# About correlations, there are many tools to visualize correlations, for example:

library(corrplot) # Library for correlation plots
mtCorr <- cor(mtcars) # Store the correlation object
corrplot(mtCorr, method = "circle") # Then plot it
corrplot(mtCorr, method = "ellipse") # Slanted left/right for +/- 
corrplot(mtCorr, method = "number") # Show correlation

# To order variables clustered (grouped) by correlation values and omit the diagonal
corrplot(mtCorr, method = "number", order = "hclust", diag = FALSE, title = "MT Cars Correlation Matrix")
?corrplot() # See all the corrplot() methods


## Regression

# Regression formulas: there are 2 ways to specify regression models. Which one you use will depend on the libraries and functions you are using. But the most common way to specify a regression formula is: y~x1+x2+x3+etc., where y is the outcome variable and the x's are the predictors. For example:

lm.formula <- mpg~cyl+disp+hp+wt

# In the case above, we have created a "formula object" named lm.formula. This is NOT the regression model, but just a formula. This is very useful if the formula is long and complex and you plan to use it in many models, this way you don't need to be re-tryping it. Check it out:

lm.formula # Show the object just created
class(lm.formula) # Show the object class

# To run the actual regression model we need to use the linear model lm() function. We can do this in sevral ways

lm(mpg~cyl+disp+hp+wt, data=mtcars) # Re-typing the formula
# Notice that we need to specify the dataset to fit the model

# Or using the fomula object created above:
lm(lm.formula, data=mtcars) 

# The two methods above are OK for quick models, but it is better to store the regression results in an object, so that you can access its methods and properties, either with:

lm.model <- lm(mpg~cyl+disp+hp+wt, data=mtcars) # Or
lm.model <- lm(lm.formula, data=mtcars)

# Now that we have the regression results stored in an object, let's extract information from it:

lm.model # Quick display of the model results
summary(lm.model) # More complete regression output
str(lm.model) # This function shows the entire object structure

lm.model$coefficients # Extracting just the coefficients
lm.model$fitted.values # Extracting predicted values
lm.model$residuals # Extracting errors

# The lm object contains 4 useful graphs:

par(mfrow=c(2,2)) # Divide the output into a 2 rows x 2 cols frame
plot(lm.model) # Graph all 4 at once


### 2.13 R Markdown

# R Markdown is a companion program to R Studio, which allows you to create HTML, PDF and Word documents on the fly, which cand include R code and R output. R Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. See:

browseURL("http://rmarkdown.rstudio.com/")
  
# Before you can use R Markdown you need to install and load the "rmarkdown" package

install.packages("rmarkdown") # If not installed already
require(rmarkdown)

# Once installed, you can create R Markdown files with File -> New File -> R Markdown -> Document. Enter the document title and author and select HTML (R Markdown can create HTML, Word and PDF files; we will be working with HTML files only)

# NOTE: You will be required to turn in all your homework and project work as an HTML file generated by R Markdown. This is a great way to create reports directly from your R analysis.

# ADVICE: You should prepare your work in an R script, because it is easier to work with and debug your R code, and run selected command lines as you test your code. R Markdown is not very helpful to write,  test and debug R code because, unlike R, you need to run the entire markdown script, and cannot run portions of the script. In addition, it does not have the nice color coding that R scripts have. Once you are sure your R code is working fine in your R script, copy/paste it to an R Markdown file and place it between a starting line that has

# ```{R}
# The R code goes here
# ```

# Note: the quotes above are not regular quotes ' but "angled" quotes ` which are at the top left of the keyboard.

# For more details on using R Markdown see 
browseURL("http://rmarkdown.rstudio.com")

# Some quick help with R Markdown:
browseURL("http://rmarkdown.rstudio.com/authoring_basics.html")

# A quick cheatsheet for R Markdown, see:
browseURL("https://www.rstudio.com/wp-content/uploads/2015/02/rmarkdown-cheatsheet.pdf")

# R Markdown files are stored in files with an extension .Rmd Open the OECD_RMarkdown.Rmd file and continue this example there


### 2.14 Shiny 

# Like R Markdown, Shiny is another R Studio product, which allows you to create interactive web applications. We will not cover Shiny in this course and you will not be required to submit your work in Shiny. But you are more than welcome to explore Shiny on your own and submit your work using Shiny. See:

browseURL("https://www.rstudio.com/products/shiny/")
