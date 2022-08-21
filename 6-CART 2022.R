####################
# PSC 8185: ML4SS
# Session 6: CART and Bagging
# Feb. 28, 2022
# Author: Iris Malone
####################

rm(list=ls(all=TRUE))

library(tree) #package for trees
library(rpart) #alt package for trees
library(caret)
library(randomForest) #package for bagging

######################
## REPLICATION OF STREETER 
#####################
#Load Data
rf_data = read.csv("/Users/irismalone/Dropbox/Courses/Machine Learning/R Code/6-CART/jop_analysis_data.csv")

#``I find that decedent characteristics, criminal activity, threat levels, police actions, 
#and the setting of the lethal interaction are not predictive of race, 
#indicating that the police—given contact—are killing blacks and whites 
#under largely similar circumstances.'' 

#Main Evidence:

#To be predictive of racially-motivated killing, the accuracy rate of the models 
#must perform at least better than a naïve “no information” model 
#that always predicts race to be “white.” 
#Such a model would be accurate 66% of the time since that is the actual 
#percentage of white decedents in the sample.

#The main finding of this paper is that none of the models 
#markedly outperform the no-information model, 
#represented as a vertical line in the figure.

##PREVIEW:

## Uses Lasso, NN, SVM, RF

### ORIGINAL LASSO CODE:
require(glmnet)

# LASSO

table(rf_data$race)
prop.table(table(rf_data$race)) #NIR is 66%

#Make outcome variable binary 
require(dplyr)
lasso_data = rf_data %>% mutate(black = ifelse(race == "Black", 1,0)) %>% select(- c(race))

#### Split data in label and matrix
y = lasso_data$black
x = lasso_data %>% select(-black)
x = as.matrix(x)

### To select the right lambda we are going to use k-fold cross-validation 
lasso_cv = cv.glmnet(x = x, y = y, 
                     family = "binomial", 
                     type.measure = "class", 
                     nfolds = 10) #default is 10-fold cv
lasso_cv

#LASSO Accuracy

#estimate cross-validation error
min(lasso_cv$cvm) 
#estimate overall model accuracy: 1 - error
lasso_accuracy = (1 - min(lasso_cv$cvm))
lasso_accuracy #finds accuracy of 0.71 (see paper)

#reasoning: 0.71 not much bigger than 0.66 so police don't target black people

###########
## What's wrong here?
###########

#No training/test set
#Estimates accuracy based on CV error
#No comparison of accuracy NIR vs lasso rate
#Mild class imbalance (33% of data)

###########
## What if we fix this?
###########

#first let's use optimal lambda to re-estimate model
plot(lasso_cv) # lowest point in the curve indicates the optimal lambda
lmin= lasso_cv$lambda.min
lmin # log value of lambda that best minimised the error

fit = glmnet(x, as.numeric(y), alpha = 1, lambda = lmin) # alpha=1 in lasso

#same results
lasso_cv = cv.glmnet(x = x, y = y, family = "binomial", type.measure = "class", nfolds = 10)
lasso_cv

#get same info from CV
#relevant predictors
lasso_cv$glmnet.fit$beta


#create predictions for full data
lasso.pred = predict(lasso_cv$glmnet.fit, x, s = lasso_cv$lambda.min)
hist(lasso.pred)

#first look at accuracy vs NIR on full set
require(caret)

confusionMatrix(as.factor(ifelse(lasso.pred > 0.5, "Black", "White")), as.factor(ifelse(lasso_data$black==1, "Black", "White")))
#corroborate main results
#accuracy: 0.77
#NIR: 0.66

#BUT....

#p-value good
#kappa: bigger
#f1 score: 0.32

#so this would seemingly overturn her results.

#BUT... 
#there's a risk we see this because she's overfitting to full dataset

#################################
## REPLICATION OF LASSO CORRECTLY
#################################
#Split data into test and training
set.seed(2022)
rf_data2 = rf_data %>% 
  mutate(black = ifelse(race == "Black", 1,0)) %>% 
  select(- c(race))

smp_size = floor(0.8* nrow(rf_data2))

train_ind = sample(seq_len(nrow(rf_data2)), size = smp_size)

train = rf_data2[train_ind, ]
test = rf_data2[-train_ind, ]


#### Split data in label and matrix
y = train$black
x = train %>% select(-black)
x = as.matrix(x)

testx = test %>% select(-black)
testx = as.matrix(testx)

#this takes a few seconds to run
lasso_cv = cv.glmnet(x = x, y = y, family = "binomial",  nfolds = 10)
lasso_cv

#create predictions for test data
lasso.pred = predict(lasso_cv$glmnet.fit, testx, s=lasso_cv$lambda.min)
hist(lasso.pred)

#first look at accuracy vs NIR on full set
require(caret)
confusionMatrix(as.factor(ifelse(lasso.pred > 0.5, "Black", "White")), 
                as.factor(ifelse(test$black ==1, "Black", "White")), 
                positive="Black")

#now see changes in results:
#accuracy: 0.66
#sensitivity: 0.04

#so the results are not that far off from what she reports in the paper, 
#but she just does it wrong...

#two wrongs make a right?

#############################
## CLASSIFICATION TREE
#############################
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

#############################
## PARTITION DATA
#############################
#Building a classification tree:
  
#- Split data into train/test and build tree  

#some variables have only one observation so need to remove
rf_data2 = rf_data %>% mutate(black = ifelse(race == "Black", 1,0)) %>% 
  select(- c(race, crime_Motor.vehicle.theft, warrant_crime_Trespassing, police_tactic_Taser))

set.seed(1234)
smp_size = floor(0.75* nrow(rf_data2))

train_ind = sample(seq_len(nrow(rf_data2)), size = smp_size)

train = rf_data2[train_ind, ]
test = rf_data2[-train_ind, ]

# Build full tree

#############################
## BASIC CLASSIFICATION TREE
#############################

#for classification tree outcome variable is factor
#set-up for regression tree is exactly the same
?tree

#tree(formula, data, weights, subset,
#     na.action = na.pass, control = tree.control(nobs, ...),
#     method = "recursive.partition",
#     split = c("deviance", "gini"), #default is gini
#     model = FALSE, x = FALSE, y = TRUE, wts = TRUE, ...)

#note options to deal with missing data
#method is recursive partition
#splitting criterion is either deviance or gini index
#default is gini

tree.race = tree(as.factor(black) ~ ., data = train)

summary(tree.race)
#summary tells you variables actually used in construction
#summary tells you number of nodes and Gini rate
#the final model only uses 18 out of 226 variables in final construction

#number of terminal nodes: 20
#misclassification rate: 0.19 (gini impurity)


#view the decision tree
plot(tree.race)
text(tree.race, pretty = 1) #pretty will make sure the text is more "readable"
#note this is a super complicated model!


#create predictions: since categorical type is class
tree.race.pred = predict(tree.race, newdata = test)
hist(tree.race.pred)
table(I(tree.race.pred[,2]>=0.5), test$black)

confusionMatrix(as.factor(ifelse(tree.race.pred[,2]>0.5, "1", "0")), as.factor(test$black), positive="1")

#suggestive evidence that factors don't predict racial killings
#however we also know there's a high potential for overfitting here 
#given tree construction and recursive binary splitting approach


##############################
### CROSS-VALIDATION/PRUNING
##############################

# the full tree looks pretty complicated. 
# low accuracy rate could be due to overfitting to training data

#- Prune the tree using a cross-validation approach  

set.seed(1234)
#cv.tree is similar to cv.glmnet; default 10-fold CV
#cv.tree(fullmodel, FUN = prune.misclass)

#Using `FUN = prune.misclass` will prune 
#using the specified classification error rate (gini) (vs deviance).

cv.tree.race = cv.tree(tree.race, FUN = prune.misclass)

cv.tree.race

#results tell us size of tree (varying number of terminal nodes)

#We're looking at *size*, *dev*, and *k*. 

#size is the number of terminal nodes corresponding to each *k* ($\alpha$) value. 

#k is the cost complexity parameter we're using to prune the tree

#Interp: $k = -Inf$ is allowing a full, unpruned tree. 
#$k = 1$ (the highest value in our results) corresponds to a single node tree.

#We make our selection based on *dev* - because we've changed the pruning function,
#this is actually the number of misclassified values. 

min(cv.tree.race$dev)
#for dev: better models = lower dev. 
#The minimum *dev* value is 157. 
#That corresponds to a tree with 12 terminal nodes (shown in *size*) and k of 3.5.
cv.tree.race$size[which.min(cv.tree.race$dev)]

#########
# rule of thumb: plot tree size relative to dev to determine optimal tree
#plot CV results
#we can visualize the best tree size by looking at terminal nodes relative to deviance
plot(cv.tree.race$size, cv.tree.race$dev, type = "b", xlab = "Number of Terminal Nodes", ylab = "Dev")
points(cv.tree.race$size[which.min(cv.tree.race$dev)], min(cv.tree.race$dev),
       pch = 20, col = "red")
#find optimal tree has 12 nodes

#####################
### Build pruned tree  
#####################

# Use CV information to build pruned tree

prune.tree.race = prune.misclass( #full model
                                 tree.race, 
                                 #identify optimal terminal nodes
                                 best = cv.tree.race$size[which.min(cv.tree.race$dev)])

summary(prune.tree.race)
#so optimal model prunes from 18 to 11 variables
#new misclassification rate is 0.21

plot(prune.tree.race)
text(prune.tree.race, pretty = 1)
#so optimal model prunes from 18 to 11 variables 
# now only has 12 nodes (versus 20 in original)
# results in slightly higher misclassification rate


#compare prune model results
#create predictions: since categorical type is class
prune.race.pred = predict(prune.tree.race, newdata = test)

confusionMatrix(as.factor(ifelse(prune.race.pred[,2] >0.5, "1", "0")), as.factor(test$black), positive="1")

#find model seems to predict worse than NIR
#accuracy ~ NIR

#################
## RPART PACKAGE
################

### rpart method is aesthetically nicer, more popular, better maintained, and runs faster

### rpart: Recursive Partitioning And Regression Trees

#According to the help file for rpart:
  
#  > This differs from the tree function in S mainly in its handling of surrogate variables.

#Surrogate variables are used when a value is missing and 
#so a split cannot be completed with the main variable. 
#A surrogate is a different variable is chosen to approximate the first-choice variable 
#in a split.

?rpart
train$black = ifelse(train$black ==1, "Black", "White")
test$black = ifelse(test$black ==1, "Black", "White")

tree.race = rpart(black ~ ., data=train, 
                  method = "class", #this tells us we are doing classification
                  control = list(cp = 0, #uses cp to make splits
                                 #na.action=na.pass, #tells the model to pass by missing values if they exist
                                 minsplit = 10, #minimum number of observations that must exist in node
                                 minbucket = 30)) #min number of observations per region; 
                                #this is generally minsplit/3, but this data requires higher value in order to not stall!

names(tree.race) #more output options to explore

summary(tree.race)

#slightly different output
#tells us all decision rules and observations under each node


print(tree.race) #text version of tree

#instead of k, cost complexity parameter is cp here (not to be confused with AIC)

printcp(tree.race) #prints only the cptable, or for no extra info whatsoever
#shows you number variables (5) used in construction 
#shows you how changing cost complexity parameter affects number terminal nodes (nsplit)
#and gini index (0.34)

#rpart plot is considered prettier
rpart.plot(tree.race) #decision tree
#see p_mk and misclassification rate under each node

# IMO, best looking trees are found using partykit
library(partykit)
rparty.tree.race = as.party(tree.race)
rparty.tree.race

#see number of nodes
#see pmk at bottom of each nodes
plot(rparty.tree.race)

#can look at predictions
tree.race.pred = predict(tree.race, newdata = test)

require(caret)
#get different results since I changed minbucket
confusionMatrix(as.factor(ifelse(tree.race.pred[,1]>0.5, "Black", "White")), 
                as.factor(test$black), 
                positive="Black")

###############################
## CROSS-VALIDATION USING RPART
###############################
# suppose we want to find optimal cost complexity parameter using cross-validation
set.seed(1234)
tree.race = rpart(black ~ ., 
                  data=train, 
                  method = "class", #this tells us we are doing classification
                  control = list(cp = 0, #uses cp to make splits
                                 #na.action=na.pass, #tells the model to pass by missing values if they exist
                                 minsplit = 10, #minimum number of observations that must exist in node
                                 minbucket = 30)) 

names(tree.race)
tree.race

tree.race$cptable
#find optimal cost complexity parameter based on lowest error (xerror)
cp_choose = tree.race$cptable[,1][which.min(tree.race$cptable[,4])]
cp_choose

#prune using optimal cost-complexity
tree.prune.race = prune.rpart(tree.race, cp_choose)

rparty.tree.race = as.party(tree.prune.race)
rparty.tree.race


#plot distribution
plot(rparty.tree.race)


#bit more manageable - this tells us model can have 4 covariates

tree.pred = rpart:::predict.rpart(tree.prune.race, newdata = test)
head(tree.pred)

confusionMatrix(as.factor(ifelse(tree.pred[,1] > 0.5, 
                                 "Black", "White")), as.factor(test$black), positive="Black")


###############
## CARET - RPART
###############

#create range of cost complexity parameters to iterate through
trgrid = expand.grid(cp = seq(0,0.05,0.001))

set.seed(1234)
#length is = (n_repeats*nresampling)+1
#for example, in 5-fold cv, k=5 + 1
numtimes = 6
seeds = vector(mode = "list", length = numtimes)

#create series for seeds for each iteration of fold
for(i in 1:10) seeds[[i]] = sample.int(n=1000, numtimes)

#for the last model
seeds[[11]] = sample.int(1000, 1)

#caret will automatically prune the tree
#i'm using 5-fold cv here just so you can see how I mix it up
ctrl= trainControl(method="cv",
                   number=5,
                   savePredictions=TRUE,
                   classProbs = TRUE, seeds=seeds)
                   
#train$black = as.factor(train$black)
#test$black  = as.factor(test$black)
levels(train$black) = c("Black", "White")
levels(test$black) = c("Black", "White")
train.rpart.caret = train(black ~., data=train,
                            method = "rpart", #this tells it to build tree
                            trControl = ctrl,#note this control function is separate from rpart control function
                            tuneGrid = trgrid,
                            control=rpart::rpart.control(cp=0,
                                              minsplit = 10,
                                               minbucket = 10))

train.rpart.caret

#see accuracy for different cost complexity parameters
#caret/rpart finds optimal cost complexity parameter and prunes tree
#pruned tree is found in finalModel
train.rpart.caret$finalModel

#best complexity parameter
train.rpart.caret$bestTune %>% unlist()

plot(train.rpart.caret$finalModel)
text(train.rpart.caret$finalModel, pretty=1)


tree.pred = predict(train.rpart.caret, newdata = test, type="prob")

#this actually has highest sensitivity yet but lowest accuracy
confusionMatrix(as.factor(ifelse(tree.pred[,1]>0.5, "Black", "White")), as.factor(test$black), positive="Black")


##############################
### VARIABLE IMPORTANCE PLOT
##############################
vi_scores = vi(train.rpart.caret)
vi_scores
vip(vi_scores) #this is nicely a ggplot object so you can manipulate
#most important indicator is if there was a witness followed by mental illnes

##############################
### PARTIAL DEPENDENCE PLOT
##############################

library(ggplot2)
library(pdp)

xgrid = data.frame(age = seq(from = 0, to = 50, length = 51))

a = partial(tree.race,  pred.var = "age", pred.grid = xgrid,  which.class ="Black",train=train) %>% 
  
  autoplot(smooth = TRUE, xlim=c(0, 50), ylab="Pr(Black)") +  theme_light() 

#see marginal effect of age
a

#see likelihood of death falls off sharply
table(I(train$age>45), train$black)


###############
## BAGGING
###############
require(randomForest) #good for bagging and rf

table(train$black)
table(test$black)

set.seed(1234)
bag.race = randomForest(as.factor(black) ~ ., #formula. for classification, RF requires factor response
                        data = train, #data
                        mtry = dim(train)[2]-1, #number of parameters to put in each bootstrap tree,
                        importance=TRUE) #tell model to calculate variable importance scores
#default is 500 trees
#no of variables over bootstrapped samples

bag.race

names(bag.race)
bag.race$err.rate
#see OOB estimate of error rate

bag.race$confusion
#see confusion matrix on training dataset and misclassification rate within class

#see poor sensitivity, but good specificity
#number of tree
table(bag.race$ntree)

#training predictions
head(bag.race$predicted)

#training predictions with probability scores
head(bag.race$votes)

bag.pred = predict(bag.race, newdata=test, type="prob") #to get probabilities given we fed in factor variable
head(bag.pred)
levels(test$black)

confusionMatrix(as.factor(ifelse(bag.pred[,1]>0.5, "Black", "White")), as.factor(test$black), positive="Black")
#see strong sensitivity (0.3) and balanced accuracy just slightly above NIR

##### LOOK AT OOB ERROR
err = bag.race$err.rate
head(err)

#histogram of oob error
hist(err[,"OOB"])

hist(bag.race$votes[,"Black"]) #see probability distribution across hist

oob_err = err[nrow(err), "OOB"]
oob_err

plot(bag.race)
#oob error in black
#see that on averge we get oob error rate or 27%

#"white" racial error in green
#see that error
#"black" racial error in red


#see that as the number of trees grow, we get a better error rate for the majority class
#see that as the number of trees grows, the error rate remains relatively high for minority class

##############################
### VARIABLE IMPORTANCE PLOT
##############################

vi_scores = vi(bag.race)
vip(vi_scores) #this is nicely a ggplot object so you can manipulate
#most important indicator is if there was a witness followed by mental illnes

##############################
### BAGGING IN CARET
##############################

ctrl= trainControl(method="cv",
                   number=10, 
                   savePredictions=TRUE,
                   classProbs = TRUE,
                   seeds=seeds)


#there are no tuning parameter for bagging
set.seed(1234)
train.bag.caret = train(black ~., data=train,
                          method = "treebag", #this tells 
                          tuneLength = 30,
                          trControl = ctrl)

train.bag.caret


bag.pred.caret = predict(train.bag.caret, newdata=test)

confusionMatrix(as.factor(bag.pred.caret), as.factor(test$black), positive="Black")


#VARIABLE IMPORTANCE PLOT
vi_scores = vi(train.bag.caret)
vip(vi_scores) 

vip(vi_scores, num_features=20) 
