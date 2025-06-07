####################
# PSC 8185: ML4SS
# Session 5: Variable Selection
# February 14, 2022
# Author: Iris Malone
####################
rm(list=ls(all=TRUE))
singhway = read.csv("singhway.csv")

head(singhway)

#Outcome Variables for NW Program:

#Did state explore NW acquisition?
table(singhway$explore)

#Did state start to pursue NW acquisition?
table(singhway$pursue)

#Did state acquire NW?
table(singhway$acquire)

#so rare. much imbalance.

# Key IVs
head(singhway)
dim(singhway)
require(dplyr)
#select just relevant columns
singhway = select(singhway, c(7, 9:19))

#not going to deal with missing data for now - would have to probably impute
singhway = na.omit(singhway)

dim(singhway)
#10 IVs for 10-25 outcomes. lots of potential for overfitting
require(car)
round(cor(singhway, use="complete.obs"),3)

#can also look at pairwise correlations although this takes a second to load
pairs(singhway, pch = 19, lower.panel = NULL)

#### BASE MODEL (TABLE 2/4)
# Technological determinants: gdp/cap, gdp/cap^2, and industrial capacity index
# External determinants: rivalry, dispute involvement, number 
# Internal determinants: democracy, democratization, dem neighborhood, econ openness, liberalization

#to motivate var selection, let's start off by looking at initial model
m = glm(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + 
          allies + democracy + democratization + percentdems + 
          openness + liberalization, data=singhway, family="binomial" )
summary(m)

#base results imply GDP/cap, industrial capacity, rivalry, disputes, 
#allies, percent dems, and econ openness affect pursuit 
#(essentially everything BUT democracy, democratization, and econ liberalization)


############
#Motivation: We may want to know whether these results generalize? 
############

# Partition Data into Training and Test Set

## set the seed to make your partition reproducible
set.seed(0214)
smp_size = floor(0.75* nrow(singhway))
train_ind = sample(seq_len(nrow(singhway)), size = smp_size)

train = singhway[train_ind, ]
test = singhway[-train_ind, ]

#Let's get predictive/baseline model performance estimates

#correct for class imbalance
require(ROSE)
train.over = ovun.sample(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + 
                           allies + democracy + democratization + percentdems + 
                           openness + liberalization,
                         data=train, method = "over", seed = 214)$data
#estimate model
base.model = glm(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + 
                  allies + democracy + democratization + percentdems + 
                  openness + liberalization, data=train.over, family="binomial")
summary(base.model)

#get predictions
require(caret)
singhwaypred = predict(base.model, type="response", newdata=test)
hist(singhwaypred) 
#estimate performance
base.test.mse = mean((test$pursue - singhwaypred)^2 )
base.test.mse 
base.binarypred = ifelse(singhwaypred > 0.5, 1, 0)
base_cm = confusionMatrix(as.factor(base.binarypred), as.factor(test$pursue),  positive="1")
base_cm
#sensitivity (tp rate) = 0.8
#specificity (tn rate) = 0.86

##################
## WHEN YOU HAVE CLASS IMBALANCE...
##################
# Best to do feature selection before correcting for class imbalance
# If you try to correct, then you'll have info leak 
#(samples will be correlated) which will 
# artificially boost relative importance of indicators in sample

#############################
## BEST SUBSET SELECTION
#############################
library(leaps)

subset_full = regsubsets(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + allies + 
                           democracy + democratization + percentdems + openness + liberalization, 
                         data = train, 
                         nbest=1, #minimum number of variables in each model
                         nvmax = 11) #nvmax to allow all variables in subset

#the summary tells us which variables are included in which model
#model 1 - dispute only
#model 2 - industry + dispute
#model 11 - all predictors
summary(subset_full)

full_summary = summary(subset_full)

#see parameters we can pull up to compare fits
names(full_summary)
#which tells us what variables go into each model
#we can look at a range of assessment metrics: rsq, rss, adjr2, cp (aic), and bic

## Look at Cp (AIC)
## optimal model will have min AIC/Cp
## As we are looking for the Min, we can use the which.min function and color it red.
plot(full_summary$cp, xlab = "Number of Variables", ylab = "Cp")
points(which.min(full_summary$cp), full_summary$cp[which.min(full_summary$cp)], 
       pch = 20, col = "red")

#results suggest best model has 2 (!) variables 

#can plot and see which variables are in each subset model
# Areas that are colored black indicate the variable is present in the model 
# at the corresponding Cp level, 
# while white areas communicate an absence of the variable.
plot(subset_full, scale = "Cp")

# results show that smallest Cp only has industry1 and disputes

#pull up best model coef(model name, whichmin(full_summary$cp))
#these are regression estimates
coef(subset_full, 2)

#we can double-check that the model just estimates ols regressions
m= lm(pursue ~ industry1 + disputes,  data = train)
summary(m)

#can also look at alternative model types
plot(full_summary$adjr2, xlab = "Number of Variables", ylab = "adjr2")
which.max(full_summary$adjr2) #this suggests model 5 is best

plot(full_summary$bic, xlab = "Number of Variables", ylab = "bic")
which.min(full_summary$bic) #finds model 1  is best


#################
## MODEL ASSESS
################

#correct for class imbalance
train.over = ovun.sample(pursue ~ industry1 + disputes,
                         data=train, method = "over", seed = 0214)$data
#estimate model
bss.model = glm(pursue ~ industry1 + disputes, data=train.over, family="binomial")
summary(bss.model)

#get predictions
singhwaypred = predict(bss.model, type="response", newdata=test)

#estimate performance
bss.test.mse = mean((test$pursue - singhwaypred)^2 )
bss.test.mse 
bss.binarypred = ifelse(singhwaypred > 0.5, 1, 0)
bss_cm = confusionMatrix(as.factor(bss.binarypred), as.factor(test$pursue),  positive="1")
bss_cm
#sensitivity (TP) rates are better using much simpler model!

#slight difference in test MSE
base.test.mse
bss.test.mse

base_cm$byClass['Sensitivity']
bss_cm$byClass['Sensitivity']

base_cm$byClass['Specificity']
bss_cm$byClass['Specificity']


#############################
## FORWARD SELECTION
#############################

#try iterative forward selection
forward_step = regsubsets(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + allies + 
                            democracy + democratization + percentdems + openness + liberalization, 
                          data = train, 
                          nbest=1,
                          nvmax = 11, 
                          method="forward") #specify forward selection
summary(forward_step)
forward_summary = summary(forward_step)

#use AIC to identify best model
plot(forward_summary$cp, xlab = "Number of Variables", ylab = "Cp")
points(which.min(forward_summary$cp), forward_summary$cp[which.min(forward_summary$cp)], 
       pch = 20, col = "red")

#note same conclusion 
plot(forward_step, scale = "Cp")

#We can also start to get fancier 
#and use cross-validation techniques to make sure we pick optimal model


#Choosing among models using validation set
val.errors = rep(NA, 11)
testpursue = test$pursue
test = model.matrix(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + allies + 
                      democracy + democratization + percentdems + openness + liberalization, 
                    data=train)

for (i in 1:11) { #length ivars 
  
  coefi = coef(forward_step, id = i) #for each model i
  pred = test[, names(coefi)] %*% coefi #make predictions on test set
  val.errors[i] = mean((testpursue - pred)^2) #estimate test error
}

#this suggests validation error can actually increase as the model gets bigger
plot(sqrt(val.errors), ylab = "Root MSE", pch = 19, type = "b")
legend("bottomright", legend = c("Validation"), col = c("black"), 
       pch = 19)

#create predict function to automate
predict.regsubsets = function(object,newdata,id,...){
  form = as.formula(object$call[[2]])
  mat = model.matrix(form,newdata)
  coefi = coef(object,id=id)
  xvars = names(coefi)
  mat[,xvars]%*%coefi
}

#Choosing Among Models Using Cross-Validation


set.seed(0214)
#number of folds
k = 10
folds = sample(rep(1:k, length = nrow(train)), replace=T)
folds

cv.errors = matrix(NA, k, 11)

for (j in 1:k) { #number of cv folds
  
  #estimate training model on imbalanced data
  best.fit = regsubsets(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + allies + 
                          democracy + democratization + percentdems + openness + liberalization,
                        data = train[folds != j, ], nvmax = 11, 
                        method = "forward")
  
  #identify best features
  best_summary = summary(best.fit)
  print(which.min(best_summary$cp))
  
  #apply 10-fold cv to best fit model
  for (i in 1:11) { #number of potential predictors in model
    pred = predict.regsubsets(best.fit, train[folds == j, ], id = i)
    cv.errors[j, i] = mean((train$pursue[folds == j] - pred)^2)
  }
}

head(cv.errors) #matrix of CV errors for different folds/j-variable

#apply function to average over columns of matrix to obtain vector 
#showing us cv error for j-variable model
rmse.cv = sqrt(apply(cv.errors, 2, mean))
rmse.cv

#10-fold cv finds 2-variable model best
plot(rmse.cv, col = "blue", pch = 19, type = "b")
legend("bottomright", legend = c("10-Fold CV MSE"), col = c("blue"), 
       pch = 19)

which.min(rmse.cv) #best optimal model is 2

coef(best.fit, 2)


#############################
## BACKWARD SELECTION
#############################
backward_step = regsubsets(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + allies + 
                             democracy + democratization + percentdems + openness + liberalization, 
                           data = train, 
                           nbest=1,
                           nvmax = 11, 
                           method="backward") #specify forward selection
summary(backward_step)

backward_summary = summary(backward_step)

plot(backward_summary$cp, xlab = "Number of Variables", ylab = "Cp")
points(which.min(backward_summary$cp), backward_summary$cp[which.min(backward_summary$cp)], 
       pch = 20, col = "red")

#this model reaches same conclusion
plot(backward_step, scale = "Cp")

#create predict function to automate
predict.regsubsets = function(object,newdata,id,...){
  form = as.formula(object$call[[2]])
  mat = model.matrix(form,newdata)
  coefi = coef(object,id=id)
  xvars = names(coefi)
  mat[,xvars]%*%coefi
}
#Choosing Among Models Using Cross-Validation
set.seed(0214)
k = 10
folds = sample(rep(1:k, length = nrow(train)), replace=T)
folds

cv.errors = matrix(NA, k, 11)

for (j in 1:k) {
  best.fit = regsubsets(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + allies + 
                          democracy + democratization + percentdems + openness + liberalization,
                        data = train[folds != j,], nvmax = 11, 
                        method = "backward")
  for (i in 1:11) {
    pred = predict.regsubsets(best.fit, train[folds == j, ], id = i)
    cv.errors[j, i] = mean((train$pursue[folds == j] - pred)^2)
  }
}

head(cv.errors) #matrix of CV errors for different folds/j-variable

#apply function to average over columns of matrix to obtain vector 
#showing us cv error for j-variable model
rmse.cv = sqrt(apply(cv.errors, 2, mean))

#10-fold cv finds 1-variable model best
plot(rmse.cv, col = "blue", pch = 19, type = "b")
which.min(rmse.cv)
legend("bottomright", legend = c("10-Fold CV MSE"), col = c("blue"), 
       pch = 19)

#pull up optimal model
coef(backward_step, 1)

###################
#from here - you would:
#correct class imbalance and set-up 10-fold cv
#re-run on specified model
#estimate test MSE

#to do, first create a control function for training
#to do, first set our cross-validation seeds for reproducibility
set.seed(1234)
#k is the number of folds for cv, e.g. k=10 for 10-fold cv
#repeated cv is n_resampling (method="repeatedcv")
#length = (k-folds*nresampling) + 1
cvseeds = vector(mode = "list", length = 11) #length 11 because it is k+1
for(i in 1:11) cvseeds[[i]] <- sample.int(1000, 51) #number of tuning parameter combinations

## For the last model:
cvseeds[[11]] <- sample.int(1000, 1) 
#https://stackoverflow.com/questions/32099991/seed-object-for-reproducible-results-in-parallel-operation-in-caret

require(caret)
ctrl = trainControl(method = "cv",
                    number = 10,
                    summaryFunction = twoClassSummary,
                    classProbs = TRUE, 
                    allowParallel=T, seeds=cvseeds)

train = singhway[train_ind, ]

test = singhway[-train_ind, ]

set.seed(1234)
#have to rename levels
train$pursue = as.factor(train$pursue)
test$pursue = as.factor(test$pursue)
levels(train$pursue) = c("none", "pursue")
levels(test$pursue) = c("none", "pursue")

set.seed(1234)

train_control = trainControl(method="cv", #method
                             number=10, #number of folds
                             savePredictions=T, 
                             classProbs=T,                 # should class probabilities be returned
                             summaryFunction=twoClassSummary,
                             sampling="down", seeds=cvseeds)
model = train(pursue ~ industry1 + disputes, data = train, 
              method = "glm", 
              family="binomial", 
              trControl=train_control)
names(model)
table(model$pred[,1], model$pred[,2])
simplepred = predict(model, type="prob", newdata=test) #produces raw prob simplepred[,2] 
head(simplepred)
head(simplepred[,2]) 

simplepred = predict(model, type="raw", newdata=test) #automatically classifies based on simplepred[,2] 
head(simplepred)
simple_cm = confusionMatrix(as.factor(simplepred), as.factor(test$pursue),  positive="pursue")

#confusionMatrix(model$pred[,1], model$pred[,2], mode="prec_recall", positive="pursue")

base_cm$byClass

simple_cm$byClass #simpler model has better TP/Recall than full model

####################
## SUMMARY
####################
#Methods generally find 2 variables optimal (industry and disputes)

#overturns a lot of Singh and Way



#############################
## RIDGE REGRESSION
#############################
#major drawback is that glmnet cannot handle NAs

library(glmnet)
?glmnet

#imbalanced data
#note - like KNN, we create a matrix of ONLY explanatory variables
set.seed(1234)
x = model.matrix(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + 
                   allies + democracy + democratization + percentdems + 
                   openness + liberalization-1, data = train) 

y = as.numeric(train$pursue)

#Alpha (shrinkage penalty = 0)
#recall if shrinkage penalty high, should return OLS

#when alpha = 0 we get coefs of
ridge_model = glmnet(x, y, alpha = 0)

#this shows that as lambda gets larger, the coef for each variable shrinks to zero
plot(ridge_model, xvar = "lambda", label = TRUE)

#identify optimal lambda
#note: if you have collinear variables, you always have to identify lambda first. 
#caret doesn't really adjust for this well so you are better off manually input

#Penalty type (alpha=0 is ridge)
cv.lambda = cv.glmnet(x=x, y=y, 
                      alpha = 0,
                      lambda=exp(seq(-5,8,.1)))  #don't need to specify the lambda range
#but you can do it anyways

plot(cv.lambda)                                 #MSE for several lambdas


#now get the coefs with 
#the lambda found above
lmin =cv.lambda$lambda.min

lmin #optimal lambda is 0.03
ridge.model = glmnet(x=x, y=y,
                     alpha = 0, 
                     lambda = lmin)

ridge.model$beta


# Using caret to perform ridge and CV
require(caret)
set.seed(1234)
#k is the number of folds for cv, e.g. k=10 for 10-fold cv
#repeated cv is n_resampling (method="repeatedcv")
#length = (k-folds*nresampling) + 1
cvseeds = vector(mode = "list", length = 11) #length 11 because it is k+1
for(i in 1:11) cvseeds[[i]] <- sample.int(1000, 51) #number of tuning parameter combinations

## For the last model:
cvseeds[[11]] <- sample.int(1000, 1) 
#https://stackoverflow.com/questions/32099991/seed-object-for-reproducible-results-in-parallel-operation-in-caret

train_control = trainControl(method="cv", #method
                             number=10, #number of folds
                             savePredictions=T, 
                             classProbs=T,                 # should class probabilities be returned
                             summaryFunction=twoClassSummary,  # results summary function
                             sampling="up", seeds=cvseeds) #correct for class imbalance

trainX = model.matrix(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + 
                        allies + democracy + democratization + percentdems + 
                        openness + liberalization-1, data = train) 

#need to transform since it's a binary variable
trainY  = as.factor(train$pursue)
levels(trainY) = c("nopursue", "pursue")

#use alpha = 0 to get ridge
set.seed(0214)
test_class_cv_model = train(trainX, trainY, 
                            method = "glmnet", #to specify lasso or ridge
                            family="binomial",
                            metric="ROC",
                            trControl = train_control, 
                            #new paramater is tune grid which is same number of values
                            tuneGrid = expand.grid(alpha = 0,
                                                   lambda = seq(0.001,0.1,by = 0.001)))

test_class_cv_model 


# Print maximum ROC statistic
max((test_class_cv_model$results)$ROC)

plot(test_class_cv_model)


# best lambda parameter is 0.1 (probably due to corrections)
test_class_cv_model$bestTune

# best coefficients for optimal model
coef(test_class_cv_model$finalModel, test_class_cv_model$bestTune$lambda)
#more readable
round(coef(test_class_cv_model$finalModel, test_class_cv_model$bestTune$lambda), 2)
#############################
## LASSO REGRESSION
#############################

#identify optimal lambda

#example:

# Specify a range for Lambda
lambdas = 10^seq(3, -2, by = -.1)

# LASSO regression involves tuning a hyperparameter lambda, and it runs the model many times for different values of lambda
fit = glmnet(x, as.numeric(y), alpha = 1, lambda = lambdas) # alpha=1 in lasso

# cv.glmnet() uses cross-validation to work out 
cv_fit = cv.glmnet(x, as.numeric(y), alpha = 1, lambda = lambdas)
plot(cv_fit) # lowest point in the curve indicates the optimal lambda
lmin= cv_fit$lambda.min
lmin # log value of lambda that best minimised the error


# Using caret to perform CV
require(caret)
set.seed(1234)
#k is the number of folds for cv, e.g. k=10 for 10-fold cv
#repeated cv is n_resampling (method="repeatedcv")
#length = (k-folds*nresampling) + 1
cvseeds = vector(mode = "list", length = 11) #length 11 because it is k+1
for(i in 1:11) cvseeds[[i]] <- sample.int(1000, 51) #number of tuning parameter combinations

## For the last model:
cvseeds[[11]] <- sample.int(1000, 1) 
#https://stackoverflow.com/questions/32099991/seed-object-for-reproducible-results-in-parallel-operation-in-caret

train_control = trainControl(method="cv", #method
                             number=10, #number of folds
                             savePredictions=T, 
                             classProbs=T,                 # should class probabilities be returned
                             summaryFunction=twoClassSummary,  # results summary function
                             sampling="up", seeds=cvseeds)

trainX = model.matrix(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + 
                        allies + democracy + democratization + percentdems + 
                        openness + liberalization-1, data = train) 

trainY  = as.factor(train$pursue)
levels(trainY) = c("nopursue", "pursue")

set.seed(1234)
test_class_cv_model = train(trainX, trainY, method = "glmnet",
                            family="binomial",metric="ROC",
                            trControl = train_control, 
                            tuneGrid = expand.grid(alpha = 1,
                                                   lambda = lmin))

test_class_cv_model 

# best parameter
test_class_cv_model$bestTune

# best coefficient
coef(test_class_cv_model$finalModel, test_class_cv_model$bestTune$lambda)
#note lasso rejects all variables (Strict lambda)

### can look at wider range of values by varying alpha and lambda
set.seed(0214)
test_class_cv_model = train(trainX, trainY, method = "glmnet",
                            family="binomial",metric="ROC",
                            trControl = train_control, 
                            #new paramater is tune grid which is same number of values
                            tuneGrid = expand.grid(alpha = 0:1,
                                                   lambda = seq(0.001,0.1,by = 0.001)))

test_class_cv_model 

#optimal
test_class_cv_model$bestTune$alpha #picks lasso
test_class_cv_model$bestTune$lambda

# Print maximum ROC statistic
max((test_class_cv_model$results)$ROC)

#plot lambda vs ROC
#compare different alpha (essentially ridge (0) vs lasso (1)) and see how model performance varies
#see that lasso tends to do well as lambda increases 
#(should intuitively makes sense because lasso only shrinks some, not all variables)
plot(test_class_cv_model)

#look at optimal model
#most important variables are industry, rivalry, dispute, and precentdems
coef(test_class_cv_model$finalModel, test_class_cv_model$bestTune$lambda)
round(coef(test_class_cv_model$finalModel, test_class_cv_model$bestTune$lambda), 2)

#note: if you have missing data, can do knn impute using caret (http://rstudio-pubs-static.s3.amazonaws.com/251240_12a8ecea8e144fada41120ddcf52b116.html#glmnet-with-custom-traincontrol-and-custom-tuning-grid)

#############################
## PRINCIPAL COMPONENT REGRESSION
#############################
library(pls)

train = singhway[train_ind, ]
test = singhway[-train_ind, ]

set.seed(0214)
pcr.fit = pcr(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + 
                allies + democracy + democratization + percentdems + 
                openness + liberalization, data = train, 
              scale=TRUE, #scale data (standardize each predictor so scale doesn't affect)
              validation="CV") #this leads to 10-fold cv

#cv score is provided for each component

#pcr reports RMSE so to get usual MSE have to square it
summary(pcr.fit)
#looking at results we see that just 4% of variation in Y explained by 9 comps (imbalance)

#identify simplest model
pcr.MSEP = MSEP(pcr.fit)


#find lowest component
paste( "Minimum MSE of ",  
       pcr.MSEP$val[1,1, ][ which.min(pcr.MSEP$val[1,1, ] )], 
       " was produced with ", 
       sub(" comps","", names(which.min(pcr.MSEP$val[1,1, ] ))), 
       " components")

#see that the smallest validation error occurs when M = 9
#if M=11 then this would be full least squares and no dimensionality reduction

#plot cv scores

validationplot(pcr.fit, val.type="MSEP")

#can see what the scores of each principal component are
coefplot(pcr.fit) #note: unconventoinal way to present coefs; recommend sjPlot 

x = model.matrix(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + 
                   allies + democracy + democratization + percentdems + 
                   openness + liberalization-1, data = test) 

#see how well optimal M=8 does on performance
pcr.pred = predict(pcr.fit, x, ncomp=8)
mean((test$pursue -pcr.pred)^2)

#note the final model is more difficult to interpret because 
#it does not perform any kind of variable selection



#############################
## PARTIAL LEAST SQUARES REGRESSION
#############################

set.seed(0214)
pls.fit = plsr(pursue ~ gdpcap + g2 + industry1 + rivalry + disputes + 
                 allies + democracy + democratization + percentdems + 
                 openness + liberalization, data = train, 
               scale=TRUE,
               validation="CV")

summary(pls.fit)
validationplot(pls.fit, val.type="MSEP")
plot(pls.MSEP$val)
pls.MSEP = MSEP(pls.fit)

#find lowest component
paste( "Minimum MSE of ",  
       pls.MSEP$val[1,1, ][ which.min(pls.MSEP$val[1,1, ] )], 
       " was produced with ", 
       sub(" comps","", names(which.min(pls.MSEP$val[1,1, ] ))), 
       " components")
#lowest cross-validation error now occurs must earlier (m=4)

#looking at results we see that 61% of variation in data explained by 4 comps
pls.pred = predict(pls.fit, x, ncomp=5)
mean((test$pursue-pls.pred)^2)

pls.mse = mean((test$pursue -pls.pred)^2)
pcr.mse = mean((test$pursue -pcr.pred)^2)
pls.mse
pcr.mse
#get nearly identical results to pcr 
