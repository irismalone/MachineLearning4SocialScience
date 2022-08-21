####################
# PSC 8185: ML4SS
# Session 7: Random Forest, Boosting, BART
# March 7, 2022
# Author: Iris Malone
####################

rm(list=ls(all=TRUE))

#Load Data - keep using Streeter example
rf_data = read.csv("/Users/irismalone/Dropbox/Courses/Machine Learning/R Code/6-CART/jop_analysis_data.csv")
require(caret)
#R package for CART used in ISLR
library(tree)

#prettier visuals for CART
library(rpart)

#package for prettier variable importance plots
require(vip)

#package for partial dependence plots
require(pdp)

library(rpart.plot)

########################
## set-up data
#######################
set.seed(1234)
require(dplyr)
rf_data2 = rf_data %>% 
  mutate(black = ifelse(race == "Black", 1,0)) %>% 
  select(- c(race, crime_Motor.vehicle.theft, warrant_crime_Trespassing, police_tactic_Taser))


smp_size = floor(0.75* nrow(rf_data2))

train_ind = sample(seq_len(nrow(rf_data2)), size = smp_size)

train = rf_data2[train_ind, ]
test = rf_data2[-train_ind, ]

#####################
# SPECIAL TOPIC: HOW TO DEAL WITH MISSING DATA
####################

###################
# VISUALIZE PATTERNS MISSING DATA
###################
require(VIM)
a = aggr(train, plot=T, number=T)
a #no missing data in original df. wow. quite impressive (and unusual!)

#artificially create missing data
library(missMethods) #you'd never actually need to load this package
train_sample = select(train, c(1:5, 223))
trainmini=train_sample[sample(nrow(train_sample), 100), ] #do subset sample to visualize
train_missing = as.data.frame(lapply(trainmini, function(cc) cc[ sample(c(TRUE, NA), prob = c(0.85, 0.15), size = length(cc)/2, replace = TRUE) ]))
a = aggr(train_missing, plot=T, number=T)
a #now see missing data patterns

#####################
# IMPUTATION SOLUTIONS
#####################

####################
# MEAN IMPUTATION
#####################
# Quick code to replace missing values with the mean
#create new imputed dataframe
train_imp_mean = data.frame(
  sapply(
    train_missing,
    function(x) ifelse(is.na(x),
                       mean(x, na.rm = TRUE),
                       x)))

#check imputed data distribution
a = aggr(train_imp_mean, plot=T, number=T)
a #no missing data in new df


####################
# MEDIAN IMPUTATION
#####################
train_imp_median = data.frame(
  sapply(
    train_missing,
    function(x) ifelse(is.na(x),
                       median(x, na.rm = TRUE),
                       x)))

a = aggr(train_imp_median, plot=T, number=T)
a #no missing data in imputed df


####################
# KNN IMPUTATION
####################
?kNN #kNN imputation is in vim package

#default number of neighbor observations is k=5
#could probably do some CV to find optimal number of numbers
train_imp_knn = kNN(train_missing,
                    k=5)

#kNN in VIM package is nice because it scales the data to do the imputation
#and then scales it back so results are more comparable to other methods

a = aggr(train_imp_knn, plot=T, number=T)
a #no missing data in imputed df

####################
# COMPARE FITS
####################
#see artificial missing data in row 2/3
#see that meanimp overestimates, median imp does better, KNN imp sometimes off

cbind(original=trainmini$age, 
      missing=train_missing$age, 
      meanimp=round(train_imp_mean$age,2), 
      medianimp=train_imp_median$age, 
      knnimp = train_imp_knn$age)

#compare summary distribution
summary(trainmini$age) #true distribution
summary(train_imp_mean$age)
summary(train_imp_median$age)
summary(train_imp_knn$age)

#####################
# IMPUTATION IN CARET
#####################
?preProcess
#method:	
# a character vector specifying the type of processing. 
# Possible values are ... "center", "scale", "range", 
#"knnImpute", "bagImpute", "medianImpute", 


####################
# MEDIAN
####################

set.seed(1234)

#create a model for missing values
preProcess_missingdata_model = preProcess(train_missing, 
                                          method='medianImpute')
preProcess_missingdata_model

#make predicted values and apply to new df
train_imp_median = predict(preProcess_missingdata_model, 
                           newdata = train_missing)
anyNA(train_imp_median)

head(trainmini)
head(train_missing)
head(train_imp_median) #see how it fills in the gaps


####################
# KNN
####################

set.seed(1234)
library(RANN) # required for knnImpute
#create new df

#recall KNN works better when you center and scale the data

#knnImpute will automatically do this
preProcess_missingdata_model = preProcess(train_missing, method='knnImpute', k=5)
preProcess_missingdata_model 

# create a model for missing values
# make predicted values
train_imp_knn = predict(preProcess_missingdata_model, newdata = train_missing)
anyNA(train_imp_knn)

head(trainmini)
head(train_missing)
head(train_imp_knn) #note imputed dataset is still scaled and centered
#you can run results the same way, the results are just less interpretable

#in general: can do mean/median imputation pretty well

####################
# RANDOM FOREST
####################
## LOAD ORIGINAL DATA
rf_data2 = rf_data %>% mutate(black = ifelse(race == "Black", 1,0)) %>% 
  select(- c(race, crime_Motor.vehicle.theft, warrant_crime_Trespassing, police_tactic_Taser))

set.seed(1234)
smp_size = floor(0.75* nrow(rf_data2))

train_ind = sample(seq_len(nrow(rf_data2)), size = smp_size)

train = rf_data2[train_ind, ]
test = rf_data2[-train_ind, ]

####################
### RECALL: BAGGING
####################
require(randomForest)
set.seed(1234)
bag.race = randomForest(as.factor(black) ~ ., #formula
                       data = train, #data
                       mtry = dim(train)[2]-1, #number of parameters to put in each bootstrap tree,
                       importance=TRUE) #tell model to calculate variable importance scores


#if you do class, it'll automatically apply threshold 0.5 to make prediction
bag.pred = predict(bag.race, newdata=test, type="class")
head(bag.pred)

confusionMatrix(as.factor(bag.pred), as.factor(test$black), positive="1")


#if you do prob -- or if outcome is numeric, it'll return probability
bag.pred = predict(bag.race, newdata=test, type="prob")
head(bag.pred)
confusionMatrix(as.factor(ifelse(bag.pred[,2] >0.5, 1, 0)), as.factor(test$black), positive="1")


####################
### RANDOM FOREST
####################

set.seed(1234)
rf.race = randomForest(as.factor(black) ~ ., #formula
                        data = train, #data
                        mtry = sqrt(dim(train)[2]-1), #number of parameters to put in each bootstrap tree,
                        importance=TRUE) #tell model to calculate variable importance scores

rf.pred = predict(rf.race, newdata=test, type="class")

confusionMatrix(as.factor(rf.pred), as.factor(test$black), positive="1")

names(rf.race) #can pull out importance scores
importance(rf.race)

#random forest also has built-in varimp function

varImp(rf.race)
?varImpPlot

varImpPlot(rf.race, type=1) #mean decrease in accuracy
varImpPlot(rf.race, type=2) #mean decrease in node impurity

####################
# TUNING MTRY
####################

#random forest has built-in tuning measure

x = model.matrix(black~.-1, data = train) 

y = train$black
y  = as.factor(train$black)

#can create a grid function and iterate through
tuned.mtry = tuneRF(x, y, 
                    mtryStart = 2, 
                    ntreeTry = 500, #number of trees (bootstrap samples)
                    stepFactor = 2, #this will square the number of variables
                    improve = 0)


tuned.mtry[,"mtry"][which.min(tuned.mtry[,"OOBError"])]
mtry_opt = tuned.mtry[,"mtry"][which.min(tuned.mtry[,"OOBError"])]
mtry_opt #finds optimal number is 32
#general rule would be sqrt(222) = 15 so this is quite higher

#####################
## OPTIMAL FOREST MODEL
#####################

set.seed(1234)
rf.race = randomForest(as.factor(black) ~ ., #formula
                       data = train, #data
                       mtry = mtry_opt, #number of parameters to put in each bootstrap tree,
                       importance=TRUE) #tell model to calculate variable importance scores

rf.pred = predict(rf.race, newdata=test, type="class")

confusionMatrix(as.factor(rf.pred), as.factor(test$black), positive="1")

####################
### CARET - RANDOM FOREST
####################


set.seed(1234)
#length is = (n_repeats*nresampling)+1
#for example, in 10-fold cv, k=10 + 1
numtimes = 11
seeds = vector(mode = "list", length = numtimes)

#create series for seeds for each iteration of fold
for(i in 1:10) seeds[[i]] = sample.int(n=1000, numtimes)

#for the last model
seeds[[11]] = sample.int(1000, 1)


#Recall: Bagging
ctrl = trainControl(method="cv", 
                    number=10,
                    seeds=seeds, 
                    index=createFolds(train$black),
                    classProbs = TRUE,
                    savePredictions=TRUE)

#recall caret automatically tries to optimize tuning parameters for us

#if we want caret to not tune, we need to specify expand grid as just one value
mtry = ncol(x)
train$black = as.factor(train$black)
levels(train$black) = c("White", "Black")
test$black = as.factor(test$black)
levels(test$black) = c("White", "Black")

#Recall: Bagging in CARET
#key hyperparameter: m=p
tunegrid = expand.grid(.mtry=mtry)

set.seed(1234)
train.bag.caret = train(black ~., data=train,
                        method = "rf",
                        ntree=500,
                        metric='Accuracy', 
                        tuneGrid=tunegrid,
                        trControl = ctrl)

#tuning parameter held constant at mtry
train.bag.caret

#get training accuracy

names(train.bag.caret)


#Random Forest just shifts mtry
mtry = sqrt(ncol(x))
tunegrid = expand.grid(.mtry=mtry)
set.seed(1234)
train.rf.caret = train(black ~., 
                       data=train,
                        method = "rf", 
                        ntree=500,
                        metric='Accuracy', 
                        tuneGrid=tunegrid,
                        trControl = ctrl)

train.rf.caret


rf.pred = predict(train.rf.caret, newdata=test, type="prob")

confusionMatrix(as.factor(ifelse(rf.pred[,2]>0.5, "Black", "White")), as.factor(test$black), positive="Black")

#######################
## CARET - OPTIMAL MTRY
########################
control = trainControl(method='cv', 
                       number=10, 
                       search='grid',
                       savePredictions=TRUE,
                       classProbs = TRUE)

#this takes awhile to run!

#create tunegrid with values from 2:21 for mtry to tuning model. 
#Our train function will change number of entry variable at each split according to tunegrid. 
#have to do a very small version because there are so many versions
tunegrid = expand.grid(.mtry = seq(5,21, 4)) 
trainmini=train[sample(nrow(train), 100), ]

rf_gridsearch = train(black ~ ., 
                      data = trainmini,
                      method = 'rf',
                      metric = 'Accuracy',
                      tuneGrid = tunegrid)

rf_gridsearch

#see accuracy improves as number of predictors improve
plot(rf_gridsearch)

####################
##SEPARATION PLOT
####################

## setting plot colors (model Greenhill et. al 2011)
Y = as.numeric(as.factor(test$black))-1
Y.hat = rf.pred[,2]
Sep.Data = data.frame(Y.hat, Y)
Sep.Data = Sep.Data[order(Y.hat, +Y), ]

## setting plot colors (to replicate Greenhill et. al 2011 colorscheme)


col = c(rgb(red = 254, green = 232, blue = 200, max = 255), 
        rgb(red = 227, green = 74, blue = 51, max = 255))  

## plotting


library(ggplot2)
library(ggthemes)

theme_set(theme_solarized())


SepPlot = ggplot(data=Sep.Data) +
  geom_rect(aes(xmin = 0, xmax = seq(length.out = length(Y)), ymin = 0, ymax = 1),
            fill = "#FEE8C8") +
  geom_linerange(aes(color = factor(Y), ymin = 0, ymax = 1, x = seq(length.out = length(Y))),
                 alpha = 0.5) +
  geom_line(aes(y = Y.hat, x = seq(length.out = length(Y))), lwd = 0.8) +
  scale_color_manual(values = col) +
  scale_y_continuous(expression(hat("y")), breaks = c(0, 0.25, 0.5, 0.75, 1.0)) + 
  scale_x_continuous("", breaks = NULL) +
  theme(legend.position = "none", panel.background = element_blank(), panel.grid = element_blank(),
        axis.title.y = element_text(face = "bold", angle = 90)) 

SepPlot

#automated package
require(separationplot)
?separationplot
separationplot(Y.hat, Y)

####################
#BOOSTING
####################
rf_data2 = rf_data %>% mutate(black = ifelse(race == "Black", 1,0)) %>% 
  select(- c(race, crime_Motor.vehicle.theft, warrant_crime_Trespassing, police_tactic_Taser))

set.seed(1234)
smp_size = floor(0.75* nrow(rf_data2))

train_ind = sample(seq_len(nrow(rf_data2)), size = smp_size)

train = rf_data2[train_ind, ]
test = rf_data2[-train_ind, ]

require(gbm)
set.seed(1234)
#train$black = ifelse(train$black=="Black",1,0)
#test$black = ifelse(test$black=="Black",1,0)

#this takes a little while
boost.race = gbm(as.numeric(black)~., 
                 data=train,
                 distribution="bernoulli", #classification problem, use "gaussian" for regression 
                 n.trees=50, #this is way too small, generally want 5000, but smaller to run
                 interaction.depth=4,
                 cv.folds=10)

#Check the best iteration number.
summary(boost.race)

#number of optimal times it iterates through trees to learn (out of 50)
best.iter = gbm.perf(boost.race, method="cv")

boost.pred = predict.gbm(boost.race, newdata=test, n.trees=50, type="response")
confusionMatrix(as.factor(ifelse(boost.pred>0.5, "1", "0")), as.factor(test$black), positive="1")

#get pdp
plot.gbm(boost.race, 1, best.iter)

#use caret function

set.seed(1234)
train.gbm.caret = train(as.numeric(black) ~ ., 
                  data = train, 
                 method = "gbm", 
                 trControl = ctrl,
                 verbose=FALSE)
train.gbm.caret
names(train.gbm.caret)


boost.pred = predict(train.gbm.caret, newdata=test, n.trees=5000)
confusionMatrix(as.factor(ifelse(boost.pred>0.5, "1", "0")), as.factor(test$black), positive="1")

####################
#Bayesian Additive Regression Tree
#this typically takes awhile to run so might skip
####################
require(rJava)
require(BayesTree)
require(bartMachine)


set.seed(1234)
trainmini=train[sample(nrow(train), 100), ]

x = model.matrix(black~.-1, data = trainmini) 
x = data.frame(x)
y = ifelse(trainmini$black==1, "Black", "White")
y  = as.factor(trainmini$black)

bart_machine = bartMachineCV(x, y, use_missing_data=FALSE, seed=1234)
# m is the number of trees that it tries
# k is the number of parameters

#bart_machine builds the model - this takes a few min

#we then feed it to do variable selection
set.seed(1234)
bart_model = var_selection_by_permute(bart_machine, 
                                      plot=FALSE, 
                                      bottom_margin=10, 
                                      alpha=0.05)
names(bart_model)
print(bart_model$important_vars_local_names) #shows variable under local distribution
print(bart_model$important_vars_global_max_names) #no variables fall past max threshold

print(bart_model$var_true_props_avg) #average variable inclusion proportions


