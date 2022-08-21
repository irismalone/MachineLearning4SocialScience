####################
# PSC 8185: ML4SS
# Session 4: Model Assessment and Selection
# February 7, 2022
# Author: Iris Malone
####################
rm(list=ls(all=TRUE))

###################
## CLASS IMBALANCE
###################
#https://www.dropbox.com/s/87njw5bwsjzum0q/onealrussett.csv?dl=1
mids = read.csv("/Users/irismalone/Dropbox/Courses/Machine Learning/R Code/4-Resampling/onealrussett/onealrussett.csv")
mids = read.csv("C:\\Users\\irismalone\\Desktop\\onealrussett.csv")

head(mids)
mids$lncapratio = log(mids$capratio)
#what factors predict a militarized interstate dispute?
#replication of Oneal and Russett (1997)

#problem: disputes are really rare!
table(mids$dispute)
#outcome only occurs in about 5\% of data
947/20043 

#WHY THIS MATTERS - CONSIDER CLASSIFICATOIN WITHOUT BALANCING

# Partition Data into Training and Test Set
smp_size = floor(0.8* nrow(mids))

## set the seed to make your partition reproducible
set.seed(1234)
train_ind = sample(seq_len(nrow(mids)), size = smp_size)

train = mids[train_ind, ]
test = mids[-train_ind, ]

#ESTIMATE CLASSIFICATION MODEL

mids.model = glm(dispute ~ dem + growth  + allies + contig + lncapratio + trade,data=train, family="binomial")

#Make Predictions
logit.train.probs = predict(mids.model, type="response")
hist(logit.train.probs)

logit.test.probs = predict(mids.model, type="response", newdata=test)

#see predicted probability of conflict is skewed towards zero
hist(logit.test.probs)

#create threshold to classify rate - like other binary classification problems assume 0.5
logit.test.binary = ifelse(logit.test.probs > 0.5, 1, 0)

require(caret)
confusionMatrix(as.factor(logit.test.binary), as.factor(test$dispute),  positive="1")

#NIR is 95\%!
#note model accuracy is not a good metric here because of imbalance
#Sensitivity rate is 0%
#Specificity rate is 100%
#kappa score is good measure when dealing with imbalanced data
#kappa score = 0 means poor model

#recall is better for assessing model fit of imbalanced dataset
confusionMatrix(as.factor(logit.test.binary), as.factor(test$dispute), mode="prec_recall", positive="1")
#low recall score = very picky model

#even changing the threshold doesn't help much
logit.test.binary = ifelse(logit.test.probs > 0.15, 1, 0)

require(caret)
confusionMatrix(as.factor(logit.test.binary), as.factor(test$dispute),  positive="1")
#Sensitivity rate is 18%
#Specificity rate is 95%
#kappa score 0.12 (below 0.3-0.4 considered very low/poor model)

confusionMatrix(as.factor(logit.test.binary), as.factor(test$dispute), mode="prec_recall", positive="1")
#recall is 18\% (same as sensitivity)

#NEED TO BALANCE TRAINING SET

table(train$dispute)

library(ROSE)
set.seed(1234)

data.over = ovun.sample(dispute ~ dem + growth  + allies + contig  + lncapratio + trade,data=train, method = "over", seed = 1234)$data
table(data.over$dispute)

data.under = ovun.sample(dispute ~ dem + growth  + allies + contig  + lncapratio + trade,data=train, method = "under", seed = 1234)$data
table(data.under$dispute)

data.both = ovun.sample(dispute ~ dem + growth  + allies + contig  + lncapratio + trade,data=train, method = "both", seed = 1234)$data
table(data.both$dispute)


require(DMwR)
#SMOTE can only balance factor outcome
train$dispute = as.factor(train$dispute)

data.smote=SMOTE(dispute ~ dem + growth  + allies + contig  + lncapratio + trade, train, perc.over = 200, perc.under = 100)
table(data.smote$dispute)

#determine best method
mids.over = glm(dispute ~ dem + growth  + allies + contig  + lncapratio + trade,data=data.over, family="binomial")
mids.under = glm(dispute ~ dem + growth  + allies + contig  + lncapratio + trade,data=data.under, family="binomial")
mids.both = glm(dispute ~ dem + growth  + allies + contig  + lncapratio + trade,data=data.both, family="binomial")
mids.smote = glm(dispute ~ dem + growth  + allies + contig  + lncapratio + trade,data=data.smote, family="binomial")

#make predictions on unseen data
pred.logit.over = predict(mids.over, newdata = test, type="response")
logit.over.binary = ifelse(pred.logit.over > 0.5, 1, 0)

pred.logit.under = predict(mids.under, newdata = test, type="response")
logit.under.binary = ifelse(pred.logit.under > 0.5, 1, 0)

pred.logit.both = predict(mids.both, newdata = test, type="response")
logit.both.binary = ifelse(pred.logit.both > 0.5, 1, 0)

pred.logit.smote = predict(mids.smote, newdata = test, type="response")
logit.smote.binary = ifelse(pred.logit.smote > 0.5, 1, 0)

#check AUC for accuracy
require(pROC)

orig = confusionMatrix(as.factor(logit.test.binary), as.factor(test$dispute), mode="prec_recall", positive="1")
orig$byClass['Recall']
#recall: 0.18

#sometimes people will recommend auc, but this isn't good metric because there's no penalty for FP
roc.curve(test$dispute, logit.test.probs)
auc(test$dispute, logit.test.probs)
#AUC: 75.9%
#to see why compare the imbalanced auc with the oversampled auc

roc.curve(test$dispute, pred.logit.over)
auc(test$dispute, pred.logit.over)
#AUC: 76.3%

#they're nearly identical so even though oversampling is slightly better, it's not that informative

#Look at recall rate for different samples to determine best class imbalance correction

over = confusionMatrix(as.factor(logit.over.binary), as.factor(test$dispute), mode="prec_recall", positive="1")
over$byClass['Recall']
#over: 0.718

both = confusionMatrix(as.factor(logit.both.binary), as.factor(test$dispute), mode="prec_recall", positive="1")
both$byClass['Recall']
#both: 0.718


under = confusionMatrix(as.factor(logit.under.binary), as.factor(test$dispute), mode="prec_recall", positive="1")
under$byClass['Recall']
#under: 0.723


smote = confusionMatrix(as.factor(logit.smote.binary), as.factor(test$dispute), mode="prec_recall", positive="1")
smote$byClass['Recall']
#smote: 0.830

#suggest SMOTE produces best model

#compare histogram of imbalanced and balanced corrections
#see better range of test values in balanced dataset
par(mfrow=c(1, 2))
hist(logit.test.probs)
hist(pred.logit.smote)

#see better range of test values produces more informative set of performance metrics
logit.test.binary = ifelse(pred.logit.smote > 0.5, 1, 0)

confusionMatrix(as.factor(logit.test.binary), as.factor(test$dispute),  positive="1")
#note changes in model
#we not have sensitivity rate of 83%!
#we now have specificity rate of 59%!
#kappa score has increased slightly

#good recall
confusionMatrix(as.factor(logit.test.binary), as.factor(test$dispute),  positive="1", mode="prec_recall")


# Build custom AUC function to extract AUC
# from the caret model object
library(dplyr) # for data manipulation
library(caret) # for model-building
library(DMwR) # for smote implementation
library(purrr) # for functional programming (map)
library(pROC) # for AUC calculation


#################
# Automate Class Imbalance
#################

# We can automate class imbalance corrections using the caret package (much faster)


#to do, first set our cross-validation seeds for reproducibility
set.seed(2022)
#k is the number of folds for cv, e.g. k=10 for 10-fold cv
#repeated cv is n_resampling (method="repeatedcv")
#length = (k-folds*nresampling) + 1
cvseeds = vector(mode = "list", length = 11) #length 11 because it is k+1
for(i in 1:11) cvseeds[[i]] <- sample.int(1000, 51) #number of tuning parameter combinations

## For the last model:
cvseeds[[11]] <- sample.int(1000, 1) 
#https://stackoverflow.com/questions/32099991/seed-object-for-reproducible-results-in-parallel-operation-in-caret

#next create a control function for training

ctrl = trainControl(method = "cv",
                    number = 10,
                    summaryFunction = twoClassSummary,
                    classProbs = TRUE, 
                    allowParallel=T, seeds = cvseeds)


set.seed(1234)
#have to rename levels
train$dispute = as.factor(train$dispute)
test$dispute = as.factor(test$dispute)
levels(train$dispute) = c("peace", "conflict")
levels(test$dispute) = c("peace", "conflict")

set.seed(1234)
#the dataset will take forever to run in class if i use full df, so sample mini
#sampling will create slight difference in results
trainmini=train[sample(nrow(train), 1000), ]


train_control = trainControl(method="cv", #method
                             number=10, #number of folds
                             savePredictions=T, 
                             classProbs=T,  # should class probabilities be returned
                             summaryFunction=twoClassSummary, 
                             seeds=cvseeds)

#original base model for comparison
model = train(dispute ~ dem + growth, data = trainmini, 
              method = "glm", 
              family="binomial", 
              trControl=train_control)
names(model)


table(model$pred[,1], model$pred[,2])

orig_fit = confusionMatrix(model$pred[,1], model$pred[,2], positive="conflict")

confusionMatrix(model$pred[,1], model$pred[,2], mode="prec_recall", positive="conflict")
#as expected poor fit due to class imbalance
orig_fit

orig_fit$byClass



#############
## DOWNSAMPLE
##############

train_control = trainControl(method="cv", #method
                             number=10, #number of folds
                             savePredictions=T, 
                             classProbs=T,                 # should class probabilities be returned
                             summaryFunction=twoClassSummary,  # results summary function
                             sampling="down", #down-sample
                             seeds=cvseeds)

down_model = train(dispute ~ dem + growth, data = trainmini, 
                   method = "glm", 
                   family="binomial", 
                   trControl=train_control)

down_fit = confusionMatrix(down_model$pred[,1], down_model$pred[,2], positive="conflict")
confusionMatrix(down_model$pred[,1], down_model$pred[,2], mode="prec_recall", positive="conflict")

down_fit
down_fit$byClass
#better sensitivity/recall

############
## UPSAMPLE
#############

train_control = trainControl(method="cv", #method
                             number=5, #number of folds
                             savePredictions=T, 
                             classProbs=T,                 # should class probabilities be returned
                             summaryFunction=twoClassSummary,  # results summary function
                             sampling="up",  #up-sample
                             seeds=cvseeds)
up_model = train(dispute ~ dem + growth, data = trainmini, 
                 method = "glm", 
                 family="binomial", 
                 trControl=train_control)
up_fit = confusionMatrix(up_model$pred[,1], up_model$pred[,2], positive="conflict")
confusionMatrix(up_model$pred[,1], up_model$pred[,2], mode="prec_recall", positive="conflict")
up_fit
up_fit$byClass
#even better sensitivity than downsample

##########
## SMOTE
##########

trainmini=train[sample(nrow(train), 1000), ]
train_control = trainControl(method="cv", #method
                             number=10, #number of folds
                             savePredictions=T, 
                             classProbs=T,                 # should class probabilities be returned
                             summaryFunction=twoClassSummary,  # results summary function
                             sampling="smote", #smote
                             seeds=cvseeds)
smote_model = train(dispute ~ dem + growth, data = trainmini, 
                    method = "glm", 
                    family="binomial", 
                    trControl=train_control)

smote_fit = confusionMatrix(smote_model$pred[,1], smote_model$pred[,2], positive="conflict")
confusionMatrix(smote_model$pred[,1], smote_model$pred[,2], mode="prec_recall", positive="conflict")
smote_fit
smote_fit$byClass
#worse sensitivity rate

# Examine results for test set

model_list = list(original = orig_fit$byClass['Recall'],
                  down = down_fit$byClass['Recall'],
                  up = up_fit$byClass['Recall'],
                  SMOTE = smote_fit$byClass['Recall'])
model_list
#up-sampling produces best results (note - these results may differ from before because we automated on just a sample of the data)

#can use similar procedure to tune parameters, e.g. find optimal number of K nearest neighbors

#main takeaway: if you want to boost model specificity, 
#need to correct for imbalance
#model accuracy is a non-informative metric when you perform correction
#changing the threshold will change the sensitivity

###################
## CROSS-VALIDATION
###################
require(boot)

set.seed(1234)

#have to rename levels and turn into factor for classification
train$dispute = as.factor(train$dispute)
test$dispute = as.factor(test$dispute)
levels(train$dispute) = c("peace", "conflict")
levels(test$dispute) = c("peace", "conflict")

####################
## LOOCV
####################

trainmini=train[sample(nrow(train), 1000), ]

#to do, first set our cross-validation seeds for reproducibility
set.seed(2022)
#k is the number of folds for cv, e.g. k=10 for 10-fold cv
#repeated cv is n_resampling (method="repeatedcv")
#length = (k-folds*nresampling) + 1
cvseeds = vector(mode = "list", length = 1001) 
#length 1001 because it is k=n 
#and then k+1 for loocv
for(i in 1:1001) cvseeds[[i]] <- sample.int(1000, 51) #number of tuning parameter combinations

## For the last model:
cvseeds[[11]] <- sample.int(1000, 1) 

train_control = trainControl(method="LOOCV", #method
                             number=1, #number of folds
                             savePredictions=T, 
                             classProbs=T,                 # should class probabilities be returned
                             summaryFunction=twoClassSummary,  # results summary function
                             sampling="smote", seeds=cvseeds) 

loocv_model = train(dispute ~ dem + growth, data = trainmini, 
              method = "glm", 
              family="binomial", 
              trControl=train_control)

names(loocv_model)
loocv_model$results['ROC']
loocv_model$results['Sens']
loocv_model$results['Spec']

####################
## 10-fold CV
####################
set.seed(2022)
#k is the number of folds for cv, e.g. k=10 for 10-fold cv
#repeated cv is n_resampling (method="repeatedcv")
#length = (k-folds*nresampling) + 1
cvseeds = vector(mode = "list", length = 11) #k+1
for(i in 1:11) cvseeds[[i]] <- sample.int(1000, 51) #number of tuning parameter combinations

## For the last model:
cvseeds[[11]] <- sample.int(1000, 1) 

train_control = trainControl(method="cv", #method
                             number=10, #number of folds
                             savePredictions=T, 
                             classProbs=T,                 # should class probabilities be returned
                             summaryFunction=twoClassSummary,  # results summary function
                             sampling="smote", seeds=cvseeds)
cv10_foldmodel = train(dispute ~ dem + growth, data = trainmini, 
                       method = "glm", family="binomial",
                       trControl=train_control)
summary(cv10_foldmodel)
names(cv10_foldmodel)

####################
## 10-fold repeated CV
####################
#10-fold CV mean dividing your training dataset randomly into 10 parts and then using each of 10 parts as testing dataset for the model trained on other 9. 
#We take the average of the 10 error terms thus obtained.

#In 5 repeats of 10 fold CV, we’ll estimate the average of 5 error terms obtained by performing 10 fold CV five times. 
#in each repetition, the folds are split in a different way. 
#increasing number of samples improves robustness of CV results
#Important thing to note is that 5 repeats of 10 fold CV is not same as 50 fold CV.
cvseeds = vector(mode = "list", length = 51) #length 11 because it is k+1
#note length now changes to (5*10)+1
for(i in 1:51) cvseeds[[i]] <- sample.int(1000, 51) #number of tuning parameter combinations

## For the last model:
cvseeds[[51]] <- sample.int(1000, 1) 
#https://stackoverflow.com/questions/32099991/seed-object-for-reproducible-results-in-parallel-operation-in-caret


train_control = trainControl(method="repeatedcv", #method
                             number=10, #number of folds
                             repeats=5, #number of resamples
                             savePredictions=T, 
                             classProbs=T,                 # should class probabilities be returned
                             summaryFunction=twoClassSummary,  # results summary function
                             sampling="smote", seeds=cvseeds)
cv10_fold5repeatedcvmodel = train(dispute ~ dem + growth, data = trainmini, 
                                  method = "glm", family="binomial",
                                  trControl=train_control)
summary(cv10_fold5repeatedcvmodel)
names(cv10_fold5repeatedcvmodel)

##################
## compare different models
loocv_model$results
cv10_foldmodel$results
cv10_fold5repeatedcvmodel$results


###################
## determine optimal model/variable selection using CV
###################
#common problem: need to determine which variables to include in the model
#use a for loop to compare increasingly complex models
ivars = c("allies","contig", "capratio", "trade")
specs = paste0("dispute ~ dem + growth + ", ivars)
specs = c("dispute ~ dem + growth", specs)

#for loop
compare.var = c()

library(caret)
library(plotROC)


compare.accuracy = c()
model.names=c()

set.seed(2022)
#k is the number of folds for cv, e.g. k=10 for 10-fold cv
#repeated cv is n_resampling (method="repeatedcv")
#length = (k-folds*nresampling) + 1
cvseeds = vector(mode = "list", length = 11) #k+1
for(i in 1:11) cvseeds[[i]] <- sample.int(1000, 51) #number of tuning parameter combinations

## For the last model:
cvseeds[[11]] <- sample.int(1000, 1) 

for (i in specs){
  set.seed(1234)
  train_control = trainControl(method="cv", number=10, savePredictions=T, classProbs=T, sampling="up", seeds=cvseeds)
  model = train(formula(i), data = train, method = "glm", family="binomial", trControl=train_control)
  accuracy.metric = model$results$Accuracy
  print(i)
  print(accuracy.metric)
  model.names = rbind(i, model.names)
  compare.accuracy = rbind(accuracy.metric, compare.accuracy)
  #print(compare.accuracy) 
}
hist(compare.accuracy)
cbind(model.names, compare.accuracy)

#classification accuracy highest for dem + growth + contig model
#suggests optimal var selection includes those models
#suggests contig very important predictor of dispute (massive model increase)


###################
## BOOTSTRAP
###################
library(ISLR)
data(Portfolio)
?Portfolio

#We can estimate the value of alpha that minimizes the variance of our investment 
#using computed estimates of the population variances and covariances from our sample of 
#past measurements of X (Apple) and Y (Meta):
#create a function that computes the statistic of interest:

alpha = function(x, y){
  vx = var(x)
  vy = var(y)
  cxy = cov(x, y)
  (vy - cxy) / (vx - vy - 2 * cxy)
}

alpha(Portfolio$X, Portfolio$Y)

#What’s the standard error of alpha?
#We randomly select n observations from the data set in order to produce a bootstrap data set, Z∗1. Note that if an observation is contained in Z∗1, then both its X and Y values are included. We can use Z∗1 to produce a new bootstrap estimate for α, which we call α̂ ∗1.

alpha.fn = function(data, index){
  with(data[index,], alpha(X, Y))
}
##portfolio data frame, rows of the data frame (Values of 1 to N) 
##with the data from the index, compute alpha for X and Y

alpha.fn(Portfolio, 1:100)

##original function using data 1 to N (100), if same as before, working 

set.seed(1)
##since using random sampling, set seed for reproducible results 

##The next command uses the sample() function to randomly select 100 observations from the range 1 to 100, with replacement. This is equivalent to constructing a new bootstrap data set and recomputing alpha based on the new data set
alpha.fn(Portfolio, sample(1:100, 100, replace=TRUE))

##single bootstrap sample of size one, now let's do it 1000x

#we use the boot function, which is part of the boot library, 
#to perform the bootstrap by repeatedly sampling observations from the data set with replacement:

library(boot)
boot.out=boot(Portfolio, alpha.fn, R=1000)
boot.out
#note we get same estimate from homemade bootstrap function as using boot library!

#see histogram and quantile
plot(boot.out)
#see distribution of estimates
names(boot.out) #t is coef
hist(boot.out$t)


#Example: Estimating the Accuracy of a Logit Regression Model

#write a function to extract the coefficients from a model
boot.fn = function(data, index) {
  fit = glm(dispute ~ dem + growth  + allies + contig  + lncapratio + trade, 
            family="binomial", 
            data=data, 
            subset=index) #this subset tells what sample observations to use in creating new dataset
  return(coef(fit))
}


#sample the data with replacement once and estimate the coefs
#note you get slightly different estimates from different samples (even though sample is 16000 obs)
set.seed(1234)
boot.fn(train, sample(dim(train)[1], dim(train)[1], replace=T))

set.seed(2022)
boot.fn(train, sample(dim(train)[1], dim(train)[1], replace=T))

#we can run the model 200 times
#boot(data, statistic, R)

bootresults = boot(train, boot.fn, 200)
names(bootresults)
bootresults$t0
summary(bootresults$t) #will give estimates by variable (not named here)
hist(bootresults$t[,1], main="Bootstrap Estimates of Intercept")
hist(bootresults$t[,2], main="Bootstrap Estimates of Democracy")

summary(mids.model)

mean(bootresults$t[,2]) #approx beta of democracy
sd(bootresults$t[,2]) #approx. sd of democracy

