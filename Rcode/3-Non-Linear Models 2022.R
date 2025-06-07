####################
# PSC 8185: ML4SS
# Session 3: Non-Linear Models
# January 31, 2022
# Author: Iris Malone
####################
rm(list=ls(all=TRUE))

###############################
## NON-LINEAR MODELS
##############################

#Qualitative Data
library(foreign)
pewdata = read.spss("pew_Mar20/ATP W63.5.sav", to.data.frame=TRUE)

#Question: What is your opinion of Russia?
#Categorical Responses:
summary(pewdata$Q5_c_W63.5)


## CREATE ORDERED DV 
pewdata$supportrussia = ifelse(pewdata$Q5_c_W63.5 == "Refused", NA, as.character(pewdata$Q5_c_W63.5))
pewdata$pid = ifelse(pewdata$F_PARTY_FINAL == "Republican", "Republican", NA)
pewdata$pid[pewdata$F_PARTY_FINAL == "Democrat"] = "Democrat"
pewdata$pid[pewdata$F_PARTY_FINAL == "Independent"] = "Independent"
table(pewdata$pid)
table(pewdata$F_CREGION)
table(pewdata$supportrussia)
#Ordered Outcome

# Partition Data into Training and Test Set
smp_size = floor(0.8* nrow(pewdata))

## set the seed to make your partition reproductible
set.seed(1234)
train_ind = sample(seq_len(nrow(pewdata)), size = smp_size)

train = pewdata[train_ind, ]
test = pewdata[-train_ind, ]


################
## ORDERED LOGIT (ORDERD OUTCOME)
################
#Recommend this source: https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/

require(MASS)
## fit ordered logit model and store results 'm'
m = polr(factor(supportrussia) ~ factor(pid) + factor(F_CREGION), data=train, Hess=TRUE)

## view a summary of the model
summary(m)

ci = confint.default(m) # CIs assuming normality
#if confint span zero, not stat signif

## odds ratios
exp(coef(m))

## OR and CI
exp(cbind(OR = coef(m), ci))
#if exponentiate confint span 1, often not stat signif

#Interpretation:
#PID
#For Republicans, the odds of being very favorable towards Russia 
# (i.e., somewhat favorable or very favorable versus very/somewhat unfavorable) 
# is 0.4 times that of Democrats (e.g. Republicans more likely to support Russia)

#For those living in  the Midwest,
#the odds of being very favorable towards Russia are not 
#statistically greater than those living in the Northeast (confint overlap)

###################
# MULTINOMIAL LOGIT (UNORDERED OUTCOME)
###################

#Unordered Outcome
#Recommend this source: https://stats.idre.ucla.edu/r/dae/multinomial-logistic-regression/
#Multinomial Logit

#Question: Which country is the world's leading military power? US, China, or Russia
#Categorical Responses:
summary(pewdata$Q15_W63.5)

require(nnet) #multinom logit in neural net package

m = multinom(factor(Q15_W63.5) ~ factor(pid) + factor(F_CREGION), data=train)
summary(m)


#Republicans is associated with 0.38 decrease in the log odds 
#of seeing China as the main military power over the US compared to Democrats

## extract the coefficients from the model and exponentiate
exp(coef(m))

#relative risk of seeing China as the main military power compared to US is 0.68x higher 
#for Republican compared to Democrats 
#(or Republicans 32\% less likely to see China as main military power compared to US than Democrats)

################
## POISSON MODEL
################
#Recommend this source: https://stats.idre.ucla.edu/r/dae/poisson-regression/
require(pscl)

#funny dataset on deaths by horse kicks. 
# standard in teaching poisson models for some reason.
data(prussian)
?prussian

#see distribution of poisson count 
#frequency of deaths by horse kicks
hist(prussian$y)
dim(prussian)

# Partition Data into Training and Test Set
smp_size = floor(0.9* nrow(prussian))

## set the seed to make your partition reproductible
set.seed(1234)
train_ind = sample(seq_len(nrow(prussian)), size = smp_size)

train = prussian[train_ind, ]
test = prussian[-train_ind, ]

#estimate poisson model by specifying family = "poisson"
#recall glm for logit, but specifies family = "binomial" (Link Function!)

m1 = glm(y ~ corp, family=poisson,data=train)
summary(m1)
#expected difference in log count of deaths by horse kick for corp II is 0.51 less than for corp G
exp(coef(m1))
#incident rate in horse kick deaths for corp II is 0.6x higher than for corp G 
#(incident rate is 40% less)

## calculate and store predicted values
test$yhat = predict(m1, type="response", newdata=test)
head(test)

#Corp designation confusion matrix
table(actual=I(test$y>=1), pred=I(test$yhat>=1))

## plot predicted counts
#require(ggplot2)
#ggplot(test, aes(x = corp, y = y)) +  geom_point(aes(y = yhat), alpha=.5) +  geom_line(size = 1, color="red") +  labs(x = "Corps", y = "Expected number of deaths")

###############################
## NON-LINEAR MODELS 
###############################

################################
## POLYNOMIAL REGRESSION
################################

#use polynomial regression example from ISLR text

library(ISLR)

#want to estimate how horsepower/engine strength affects gas mileage efficiency
data(Auto)
?Auto #dataset on gas mileage efficiency given engine strength

##correlation between mpg and horsepower looks quadratic or exponential (look at pairs(Auto))

##as a baseline, gen a linear regression
m1 = lm(mpg ~ horsepower, data = Auto)
summary(m1)
#recall: horsepower is a unit of power equal to 550 foot-pounds per second (745.7 watts).
#results show 1-unit increase in horsepower reduces mpg by 0.16

##examine a polynomial regression with cubic for weight 
m2 = lm(mpg ~ horsepower + I(horsepower^2) + I(horsepower^3), data = Auto)
summary(m2) 

#results show increasing flexibility that increases in horsepower don't always reduce mpg

#the equivalent way of doing this is
##examine a polynomial regression with three cubic for weight 
m2alt = lm(mpg ~ poly(horsepower, 3, raw=T), data = Auto)
summary(m2alt) #see that it generates similar results

#results show increasing flexibility that increases in horsepower don't always reduce mpg

#for polynomial regressionm, need togenerate a grid of values for horsepower for predictions
attach(Auto)
hplims = range(horsepower) #define range of values
hplims #horsepower ranges from 46 to 230

horsepower.grid = seq(from =hplims[1], to = hplims[2]) #create grid of values
head(horsepower.grid)
preds = predict(m2alt, newdata = list(horsepower = horsepower.grid), se = TRUE)

names(preds)
head(preds)

se.bands = cbind(preds$fit + 2*preds$se.fit, preds$fit - 2*preds$se.fit)
par(mfrow = c(1, 1), mar = c(4.5, 4.5, 1, 1), oma = c(0, 0, 4, 0))
plot(horsepower, mpg, xlim = hplims, cex = .5, col = "darkgrey")
title("Degree-3 Polynomial", outer = T)
lines(horsepower.grid, preds$fit, lwd = 2, col = "blue")
matlines(horsepower.grid, se.bands, lwd = 1, col = "blue", lty = 3)


###use anova to determine best polynomial fit
fit1 = lm(mpg ~ horsepower, data = Auto)
fit2 = lm(mpg ~ poly(horsepower, 2), data = Auto)
fit3 = lm(mpg ~ poly(horsepower, 3), data = Auto)
fit4 = lm(mpg ~ poly(horsepower, 4), data = Auto)
fit5 = lm(mpg ~ poly(horsepower, 5), data = Auto)
anova(fit1, fit2, fit3, fit4, fit5)

#results suggest quadratic (d=2) or quintic (d=5) perform better than base model. 

anova(fit2, fit5)
#results suggest d=5 performs best so would be optimal
  
#can apply polynomial approach to binary classification problems
#next consider a categorical difference between mpg
#for example national fuel efficiency standards above 15 mpg
#estimate logit mpg based on horsepower
fit = glm(I(mpg > 15) ~ poly(horsepower, 4), data = Auto, family = binomial)
summary(fit)
#find different effects based on degree flexibility

preds = predict(fit, newdata = list(horsepower = horsepower.grid), se = T)
pfit = exp(preds$fit)/(1 + exp(preds$fit))
se.bands.logit = cbind(preds$fit + 2*preds$se.fit, preds$fit-2*preds$se.fit)
se.bands = exp(se.bands.logit)/ (1 + exp(se.bands.logit))

#generate a plot of the data
plot(horsepower, I(mpg > 15), xlim = hplims, type = "n", ylim = c(0, 1))
points(jitter(horsepower), I((mpg > 15)/5), cex = .5, pch = "l", col = "darkgrey")
lines(horsepower.grid, pfit, lwd = 2, col = "blue")
matlines(horsepower.grid, se.bands, lwd = 1, col = "blue", lty = 3)

  
################################
## STEP FUNCTION
################################
#plot a step function
summary(Auto$horsepower)
#range data 46-230
hist(Auto$horsepower)

#cut will take the range of the data and cut it into 4 bins covering equal range (e.g. 46-units)
table(cut(Auto$horsepower, 4))
#can also cut the data (equal bins, particular quantiles, etc)
?cut

fit = lm(mpg ~ cut(horsepower, 4), data = Auto)
coef(summary(fit))
#results tell us how 1-unit increase of horsepower affects mpg for X in that bin
#note coef size varies based on bins

  
  
################################
## SPLINES
################################
#Next we look at both cubic and natural splines to see if they better help fit the data. 
#The plots again demonstrate a non-linear relationship between horsepower and mpg. 
#We see that adding knots helps control for fewer observations around the boundary constraints. 


library(splines)

#################
##cubic splines
#################

# We could also use the df option in the bs() function to use knots at uniform quantiles
#this produces a spline with six basis functions (K+3) and seven dof
# This will set the knots at the 25th, 50th, and 75th percentiles of the horsepower data
fit = lm(mpg~bs(horsepower, df=6), data=Auto)
summary(fit)

table(cut(Auto$horsepower, 4)) #could cut at quantiles
#pre-specify knots (randomly) at 92, 138, and 184 horsepower 
fit = lm(mpg ~ bs(horsepower, knots = c(92, 138, 184)))

summary(fit)
#coef results will tell you estimate per basis function, but not really interpretable


#mainly focused on overall model performance

pred = predict(fit, newdata = list(horsepower = horsepower.grid), se = T)

#The black line is the model estimate with splines; 
plot(horsepower, mpg, col = "gray")
lines(horsepower.grid, pred$fit, lwd = 2,col="black")
lines(horsepower.grid, pred$fit+2*pred$se, lty = "dashed")
lines(horsepower.grid, pred$fit - 2*pred$se, lty = "dashed")

#################
#natural cubic splines
#################

fit2 = lm(mpg ~ ns(horsepower, df = 4), data = Auto)
pred2 = predict(fit2, newdata = list(horsepower= horsepower.grid), se = T)

#compare results of natural cubic spline with cubic spline
# (note linearity on ends)
lines(horsepower.grid, pred2$fit, col = "red", lwd = 2)


#################
##local regression
#################


#Finally, local regression considers the percent of observations or span 
#to control the flexibility of the non-linear fit. 
#As the span increases, the model becomes more linear as it tries to map to the global data.

#model local regression using different spans of the data
fit = loess(mpg ~ horsepower, span= .2, data =Auto)
fit2 = loess(mpg ~ horsepower, span =.5, data = Auto)


plot(horsepower, mpg, xlim = hplims, cex = .5, col = "darkgrey")
title("Local Regression")

lines(horsepower.grid, predict(fit, data.frame(horsepower = horsepower.grid)), col = "red", lwd = 2 )
lines(horsepower.grid, predict(fit2, data.frame(horsepower = horsepower.grid)), col = "blue", lwd = 2 )
legend("topright", legend = c("Span = .2", "Span = .5"), col = c("red", "blue"), lty = 1, lwd = 2, cex = .8)
#note how model becomes more rigid (lower variance) as span increases

  
################################
## GENERALIZED ADDITIVE MODELS
################################

#GAM allow for separate non-linear functions of each variable so 
#we consider the addition of other possible covariates here to see if 
#that affects the non-linear relationship.


#fit a generalized additive model using natural splines and multiple predictors
library(gam)
#we'll use a combination of different functional forms to model 3 different parameters:

##ns is natural splines, s is splines, lo is local regression
#horsepower: natural spline with seven basis functions (K+3) 
#cylinders: natural spline with six basis functions
#weight: natural spline

gam = lm(mpg ~ ns(horsepower, 4) + ns(cylinders, 3) + ns(weight, 3), data = Auto)
summary(gam)
#see flexibility different models

#use gam which will provide more interpretable form of each parameter distribution
gam3 = gam(mpg ~ s(horsepower, 4) + s(cylinders, 3) + s(weight, 3), data = Auto)
summary(gam3)
#s stands for smoothing function and is a set of smoothing splines set to each functoin
#choose cubic splines 

#The parametric part refers to the linear effect of the covariate involved  in the smooth. 
#The non-parametric part refers to nonlinearity beyond the linear/parametric part.

#main takeaway:
#If the nonlinear / nonparametric part is significant it suggests that 
#a linear effect of that covariate is not supported by the data. 
#if it's nonsignif (eg. weight) then might get better model treating it as linear

par(mfrow=c(1, 3))
#see functional form for each different parameter - note difference!
plot(gam3, se = TRUE, col = "blue")
#in these plots weight looks rather linear so we may perform ANOVA to determine which model is best: 
#model that excludes weight, model that uses linear function of weight, or spline function of weight

#compare gam model to alt models
gam1 = gam(mpg~ s(horsepower, 4) + s(cylinders, 3), data = Auto)
gam2 = gam(mpg ~ weight + s(horsepower, 4) + s(cylinders, 3), data =Auto)

#
anova(gam1, gam2, gam3)
#find gam2 model superior - don't need to model weight as overly flexible

#Adding Different Functions - Local Regression in GAM
#may also use localized linear regression
gam.lo = gam(mpg ~ s(weight, df = 4) + lo(horsepower, span = .7) + s(cylinders, 3), data = Auto)
plot(gam.lo, se = TRUE, col = "green")

#can do the same procedure with binary
#logistic regression result in 2-dimensional

gam.lr = gam(I(mpg > 27) ~ s(horsepower, df = 5) + s(cylinders, 3) + weight, data = Auto, family = binomial)
par(mfrow=c(1, 3))
plot(gam.lr, se = T, col = "green")

  
#The results above show compelling evidence that a GAM with a linear function of weight 
#is better than a GAM that doesn't include weight at all. 
#There is no evidence that a non-linear function of weight is needed so we prefer M2. 
#Overall, we see that horsepower is still non-linear but that other covariates like weight 
#do not require their own fitting function because it is essentially linear.
  



