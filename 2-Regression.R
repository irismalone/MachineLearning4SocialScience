####################
# PSC 8185: ML4SS
# Week 2: Regression and Classification
# Jan. 24, 2021
# Author: Iris Malone
####################
rm(list=ls(all=TRUE))

################
## REGRESSION 
################
require(Ecdat) #load R package with econometrics data

data(Caschool) # California standardized test scores and school information

?Caschool #tell us info about the data

head(Caschool) #show first 5 rows

names(Caschool) #names of variables

#DV: Test Score 

summary(Caschool$testscr)

#Regression assumes DV has normal distribution - check assumption

hist(Caschool$testscr)

#always look at data beforehand - try to identify key relationships
#It looks like there might be interactions between calwpct and avginc (exponential) 
#or mealpct and avginc (exponential); 
#highest correlated variables are (enrltot, teachers), (enrltot, computers). 
#Some strong correlates between testscr, mealpct, testscr/avginc, avginc/mealpct, and elpct/testscr

suppressMessages(require(car))
scatterplotMatrix(Caschool[5:15])
round(cor(Caschool[5:15]),3)

## Prediction Goal: Determine what factors predict highest test scores

# Partition Data into Training and Test Set
#partition 80% into training and test
smp_size = floor(0.8* nrow(Caschool))

## set the seed to make your partition reproducible
set.seed(1234)
train_ind = sample(seq_len(nrow(Caschool)), size = smp_size)

train = Caschool[train_ind, ]
test = Caschool[-train_ind, ]
dim(train)
dim(test)

## Bivariate Regression

## Motivation: See how reduced free lunch qualification affects test scores
with(train, plot(mealpct, testscr)) 

#estimate linear regression
bivar_model = lm(testscr ~ mealpct, data=train)

summary(bivar_model)
#2 questions: substantive significance? statistical significance?

#substantive significance
#Interp: 1% increase in school's free lunch qualification rate is associated with 
#a 0.61 point decrease in test score
#Relative risk: the mean test score is 681 so this changes relative risk by 0.61/681 (0.09%) tiny!

#statistical significance
#p-val tells us likelihood of observing relationship between X and Y at random. 
#If p-val is signif (p<0.1), then it means we reject the null hypothesis that X & Y have no relationship

#main takeaway: result is stat signif, but maybe not substantively huge (<1% given distribution of DV)

#is it predictive?
#run anova test
base_model = lm(testscr ~ 1, data=train)

bivar_model = lm(testscr ~ mealpct, data=train)

anova(base_model, bivar_model) #so adding this info does improve even if not by a lot


#MODEL DIAGNOSTICS

####
#Check regression assumptions
##

#Plot 1: The first plot depicts residuals versus fitted values. 
#Checks assumption of linearity and homoscedasticity. 
#if violate linearity, red line = curved --> PARTIALLY VIOLATED 

#If violate homoskedasticity, distinct pattern in residuals red line --> LOOKS OK

plot(bivar_model,which=1)

#Plot 2: The second plot checks residuals relative to "normal" observations
plot(bivar_model,which=2)
#Checks assumption of normality
#If not normal, would see s-shaped or non-linear curve ---> LOOKS OK

#Plot 3: Plots RSE vs fitted values
plot(bivar_model,which=3)
#Checks assumption of homoskedasticity
#If violate homoskedasticity, distinct pattern in residuals --> LOOKS MOSTLY OK 

#Plot 4: Cook's Distance
plot(bivar_model, which=4)
#Checks for influential observations/outliers

#Plot 5: Leverage of Key Points
plot(bivar_model, which=5)
#Check for relative influence of outliers; 
#Larger residual + leverage --> more influential
#DO NOT DELETE OUTLIERS

#See all these plots
plot(bivar_model)

## For an example of where model diagnostics are violated, 
## run m = lm(testscr ~ teachers, data=train). See large homoskedasticity violation

#Model Assessment

#make predictions for test data
yhat = predict(bivar_model, newdata=test)
test$yhat = yhat

#Compare actual with predictions
with(test, plot(testscr, yhat))

#Calculate (training) RSE
trainrse = summary(bivar_model)$sigma
trainrse
#calculate test RSE:
testrse = sqrt(sum((test$testscr - yhat)^2)/
                   (nrow(test)-2))
testrse

#Training MSE: 
trainmse = mean(bivar_model$residuals^2, na.rm=T)
trainmse

#Test MSE:

#step 2: calculate test MSE
testmse = mean((test$testscr - yhat)^2)
testmse

##########################
## Multivariate Regression
##########################

multivar_model =lm(testscr ~ mealpct + avginc + elpct,data=train)
summary(multivar_model)

#Results tell us all three variables are stat. signif important:
#mealpct: percent qualifying for reduced-price lunch: 
#         1% increase in students qualifying for reduced price lunch associated with 0.40 drop in test score
#avginc: expenditure per student
#         $1-unit increase in avg inc (logged?) associated with 0.70 increase in test score
#elpct: percent of English learners
#         1% increase in ESL students associated with 0.20 drop in test score


#Model Diagnostics
plot(multivar_model) 

#Model Assessment
#make predictions for test data
yhat_3varmodel = predict(multivar_model, newdata=test)
test$yhat_3varmodel = yhat_3varmodel


#Compare actual with predictions
with(test, plot(testscr, yhat_3varmodel))

#Calculate training RSE
trainrse_multivarmodel = summary(multivar_model)$sigma

#Calculate test RSE:
testrse_multivarmodel = sqrt(sum((test$testscr - yhat_3varmodel)^2)/
         (nrow(test)-2))
testrse_multivarmodel
#Calculate Test MSE:

testmse_multivarmodel = mean((test$testscr - yhat_3varmodel)^2)
testmse_multivarmodel

#Model 3: Kitchen Sink Regression (Add all possible covariates)
kitchensink_model =lm(testscr~ mealpct + avginc + elpct + enrltot + teachers + calwpct + 
        computer + expnstu + str + compstu,data=train)
summary(kitchensink_model)


#make predictions for test data
yhat_kitchensink_model = predict(kitchensink_model , newdata=test)
test$yhat_kitchensink_model = yhat_kitchensink_model

#Compare actual with predictions
with(test, plot(testscr, yhat_kitchensink_model))

#Calculate Training RSE
trainrse_kitchensinkmodel = summary(kitchensink_model)$sigma

#Calculate Test RSE

testrse_kitchensinkmodel = sqrt(sum((test$testscr - yhat_kitchensink_model)^2)/
       (nrow(test)-2))

#Calculate Test MSE:

testmse_kitchensinkmodel = mean((test$testscr - yhat_kitchensink_model)^2)
testmse_kitchensinkmodel

###########
## Assessment Metrics
###########
#Suppose we want to compare 3 models. Which model better predicts test scores?

#R^2?
#No! More complex models always have higher R^2. 
summary(bivar_model)$r.squared
summary(multivar_model)$r.squared 
summary(kitchensink_model)$r.squared 

#Test RSE
paste("Bivariate RSE:" , round(testrse, 2))
paste("Multivariate RSE:" , round(testrse_multivarmodel, 2))
paste("Kitchen Sink RSE:" , round(testrse_kitchensinkmodel, 2))

#Test MSE
paste("Bivariate MSE:" , round(testmse, 2))
paste("Multivariate MSE:" , round(testmse_multivarmodel, 2))
paste("Kitchen Sink MSE:" , round(testmse_kitchensinkmodel, 2))

#A model with a few well-chosen variables performs better than a "kitchen sink" model

#F-Test
#Helps us compare whether new variables between Bivariate Model and 3-Variable Model improve model fit
anova(bivar_model, multivar_model) 
#would reject null hyp that extra variables noninformative

#Compare whether new variables between 3-Variable Model and Kitchen Sink Model improve model fit
anova(multivar_model, kitchensink_model) 
#would fail to reject null hyp that extra variables non-informative

#if given the choice betwen models with comparable metrics, generally prefer the more parsimonious model


################
## CLASSIFICATION 
################

#Qualitative Data
library(foreign)
#explore public opinion data from March 2020
pewdata = read.spss("/users/irismalone/Dropbox/Courses/Machine Learning/Data/pew_Mar20/ATP W63.5.sav", to.data.frame=TRUE)

head(pewdata[1:10,])
#Question: How satisfied are you with the way democracy is working in our country?
#Categorical Responses:
summary(pewdata$Q3_W63.5)

## Aim: Determine what factors predict satisfaction with democracy

## CREATE BINARY DV 
#use simple dv so you know what it means
pewdata$supportdem = ifelse(pewdata$Q3_W63.5 == "Very satisfied" |   
                              pewdata$Q3_W63.5 == "Somewhat satisfied", "Support", "No Support")
pewdata$supportdem[pewdata$Q3_W63.5 == "Refused"] = NA #omit refusals
pewdata = subset(pewdata, is.na(pewdata$supportdem)==FALSE)
table(pewdata$supportdem)
#what proportion satisfied with democracy?
prop.table(table(pewdata$supportdem))
#note: this is relatively balanced data because equitable distribution between responses

#need to turn into categorical/factor for classification analysis
pewdata$supportdem = as.factor(pewdata$supportdem)

#Look at sample of demographic characteristics recorded in survey
head(pewdata)

# Partition Data into Training and Test Set
smp_size = floor(0.8* nrow(pewdata))

## set the seed to make your partition reproductible
set.seed(1234)
train_ind = sample(seq_len(nrow(pewdata)), size = smp_size)

train = pewdata[train_ind, ]
test = pewdata[-train_ind, ]

dim(train)
dim(test)

################
## CREATE MODEL
###############
#focus on just a few IVs: citizenship, education, race, party id
groupvar = c("F_CITIZEN", "F_EDUCCAT",                
             "F_RACECMB", "F_PARTY_FINAL")

#when building out models, it can be easier to just add/update with groupvars
model = as.formula(paste("supportdem", paste(groupvar, collapse=" + "), sep=" ~ "))

model

################
## LOGIT
################
?glm

logit.model = glm(model, data=train, family="binomial")

summary(logit.model)
#interp for binary indicators is RELATIVE to base category

#example 1:
#being non-citizen associated with 0.99 increase in log odds support democracy compared to citizens
#exponentiate to ease interp:
#being non-citizen increases odds of supporting democracy by factor of 2.69

#example 2:
#being democrat associated with 1.94 decrease in log odds support democracy compared to republicans
#being democrat decreases odds of supporting democracy by factor of 0.15
#the odds of supporting current state of democracy for democrats are about 85% lower than the odds for republicans.

coef(logit.model)
summary(logit.model)$coef

#Make Predictions
logit.train.probs = predict(logit.model, type="response")
logit.test.probs = predict(logit.model, type="response", newdata=test)
hist(logit.test.probs)

#create threshold to classify individual respondents
logit.test.binary = ifelse(logit.test.probs > 0.5, "Support", "No Support")

#model predicts about 28% respondents support democracy
table(logit.test.binary)

####################
## MODEL ASSESSMENT
#####################

#Calculate model accuracy
#look at comparison of predicted and actual

table(prediction=logit.test.binary, actual=test$supportdem)
#this is your first confusion matrix!

#No Information Rate - if everyone didn't support, what would be baseline classification rate?
table(test$supportdem)
#nir = tn/(tp + tn)
nir = 990/(782+990)
nir
#nir should make sense - this matches 
prop.table(table(test$supportdem))

#Accuracy $= \frac{TN + TP}{TN + FN + FP + TP}$
logitaccuracy = (828+346)/(828+436+162+346)
logitaccuracy

#logit accuracy looks larger than nir, which is good

#Sensitivity/Recall $= \frac{TP}{TP + FN}$
logitsensitivity = 346/(346+436)
logitsensitivity

#Specificity $= \frac{TN}{TN + FP}$
logitspecificity = 828/(828+162)
logitspecificity

#There's an R package to automate this in caret package
require(caret) 
#Create factor variable to compare predicted and actual
?confusionMatrix
#confusionMatrix(actual, predicted, positive="PREDICTED CLASS")
#set positive class to "support"

#So much information!
#Get accuracy and NIR
#Get sensitivity and specificity

#BE CAREFUL WITH SET-UP
#confusionMatrix(actual, predicted, positive="PREDICTED CLASS")
confusionMatrix(as.factor(logit.test.binary), as.factor(test$supportdem), positive="Support")
nir
logitaccuracy #matches

#Get precision, recall, and F1 score
confusionMatrix(as.factor(logit.test.binary), as.factor(test$supportdem), 
                      mode = "prec_recall", positive="Support")

names(confusionMatrix(as.factor(logit.test.binary), as.factor(test$supportdem), positive="Support"))
#Quickly access all this information by calling "byClass"
logitconfusionmatrix = confusionMatrix(as.factor(logit.test.binary), as.factor(test$supportdem), positive="Support")
logitconfusionmatrix$byClass

################
## KNN 
################

# R package is class
library(class)

set.seed(1234) 
#set seed - this is necessary because if several obs. are tied as nearest neighbors, 
#then R will randomly break the tie

#DATA SET-UP

#need to add small jitter to categorical variables 
#(don't need to add if variables already numeric/continuous, e.g. problem set)
#need to make all categorical explanatory variables numeric
train.vars = with(train, cbind(jitter(as.numeric(F_CITIZEN),0, 0.01), jitter(as.numeric(F_EDUCCAT),0, 0.01), jitter(as.numeric(F_RACECMB),0, 0.01), jitter(as.numeric(F_PARTY_FINAL),0, 0.01)))  
test.vars = with(test, cbind(jitter(as.numeric(F_CITIZEN),0, 0.01), jitter(as.numeric(F_EDUCCAT),0, 0.01), jitter(as.numeric(F_RACECMB),0, 0.01), jitter(as.numeric(F_PARTY_FINAL),0, 0.01)))  

#need to scale the data to avoid producing biased estimates towards variables with higher magnitude
train.vars = scale(train.vars)
test.vars = scale(test.vars)

head(train.vars)

#need to make all categorical outcomes numeric
train.outcome = as.numeric(train$supportdem)-1

#Estimate KNN Model

#knn pred = knn(train, test, train$outcome, k=number outcomes)
knn.pred_k1 = knn(train.vars, test.vars, train.outcome, k=1)
table(knn.pred_k1, test$supportdem)
test.y = 1-(as.numeric(test$supportdem)-1) #turn support into binary and subtract from 1 

#ISLR recommends calculating "Bayes classifier" to estimate error rate
mean(test.y != knn.pred_k1) #Bayes classifier

#can show this is actually just the same as estimating accuracy
table(knn.pred_k1, test$supportdem)
accuracy = (612+452)/(612+452+378+330)
accuracy

knn.pred_k3 = knn(train.vars, test.vars, train.outcome, k=3)
table(knn.pred_k3, test$supportdem)
mean(test.y != knn.pred_k3) #Bayes classifier

knn.pred_k5 = knn(train.vars, test.vars, train.outcome, k=5)
table(knn.pred_k5, test$supportdem)
mean(test.y != knn.pred_k5) #Bayes classifier

#store these results in case we want to compare across multiple models
#Accuracy $= \frac{TN + TP}{TN + FN + FP + TP}$
knn_5_accuracy = (636+440)/(636+440+342+354)
knn_5_accuracy

#Sensitivity/Recall $= \frac{TP}{TP + FN}$
knn_5_sensitivity = 440/(440+342)
knn_5_sensitivity

#Specificity $= \frac{TN}{TN + FP}$
knn_5_specificity = 636/(636+354)
knn_5_specificity

################
## LDA 
################
library(MASS)
?lda
lda.model = lda(model, data=train)

#view model results
#note parameterized means we recover coef results (rigid boundaries)
lda.model

#predict
test.lda.pred = predict(lda.model, newdata=test)
names(test.lda.pred)

#visualize the predictions
#plot posterior distribution of predictions
ldahist(test.lda.pred$x[,1], g=test$supportdem)

test.lda.class = test.lda.pred$class
summary(test.lda.class)

#applying a 50% threshold the posterior probability allows us to recreate the predictions
sum(test.lda.pred$posterior[,1] >= 0.5)

sum(test.lda.pred$posterior[,1] < 0.5)

#view posterior distribution
test.lda.pred$posterior[1:10,1] #predicted probability
test.lda.class[1:10] #how these probabilities translate into classes (e.g. posterior >= 0.5)

cbind(test.lda.pred$posterior[1:10,1], test.lda.class[1:10])

#Model Assessment

library(caret)
confusionMatrix(as.factor(test.lda.pred$class), as.factor(test$supportdem), positive="Support")

#Accuracy $= \frac{TN + TP}{TN + FN + FP + TP}$
ldaaccuracy = (828+343)/(828+343+162+439)
ldaaccuracy

#Sensitivity/Recall $= \frac{TP}{TP + FN}$
ldasensitivity = 343/(343+439)
ldasensitivity

#Specificity $= \frac{TN}{TN + FP}$
ldaspecificity = 828/(828+162)
ldaspecificity


library(caret)
ldaconfusionmatrix = confusionMatrix(as.factor(test.lda.pred$class), as.factor(test$supportdem), positive="Support")
ldaconfusionmatrix$byClass

##########
### look at different ids of cases 
# can be useful when trying to backward induct where the model went wrong and iteratively improve
set.seed(1234)
tp = sample(unique(test$QKEY[test$supportdem =="Support" & test.lda.pred$class=="Support"]), 10) #"true" positives
fp = sample(unique(test$QKEY[test$supportdem =="No Support" & test.lda.pred$class=="Support"]), 10) #"false" positives
fn = sample(unique(test$QKEY[test$supportdem =="Support" & test.lda.pred$class=="No Support"]), 10) #"false" negatives
tn = sample(unique(test$QKEY[test$supportdem =="No Support" & test.lda.pred$class=="No Support"]), 10) #"true" negatives

tp #gives you IDs to go look up

####################
## MODEL COMPARISON 
####################

#Accuracy
paste("Logit Model Accuracy:" , round(logitaccuracy, 3))
paste("KNN=5 Model Accuracy:" , round(knn_5_accuracy, 3))
paste("LDA Model Sensitivity:" , round(ldaaccuracy, 3))

#Sensitivity (True Positive Rate)
paste("Logit Model Sensitivity:" , round(logitsensitivity, 3))
paste("KNN=5 Model Sensitivity:" , round(knn_5_sensitivity, 3))
paste("LDA Model Sensitivity:" , round(ldasensitivity, 3))

#Specificity (True Negative Rate)
paste("Logit Model Specificity:" , round(logitspecificity, 3))
paste("KNN=5 Model Specificity:" , round(knn_5_specificity, 3))
paste("LDA Model Specificity:" , round(ldaspecificity, 3))

#Looks like logit is best overall model; KNN best predicts true positives

###########################
## CAN REDO FOR MULTI-CLASS
###########################

## CREATE BINARY DV 
pewdata$supportdem = ifelse(pewdata$Q3_W63.5 == "Very satisfied", "High Support", NA)
pewdata$supportdem[pewdata$Q3_W63.5 == "Somewhat satisfied"] = "Some Support"
pewdata$supportdem[pewdata$Q3_W63.5 == "Not too satisfied"] = "Low Support"
pewdata$supportdem[pewdata$Q3_W63.5 == "Not at all satisfied"] = "No Support"

pewdata$supportdem[pewdata$Q3_W63.5 == "Refused"] = NA #omit reusals
pewdata = subset(pewdata, is.na(pewdata$supportdem)==FALSE)
table(pewdata$supportdem)
pewdata$supportdem = as.factor(pewdata$supportdem)

# Partition Data into Training and Test Set
smp_size = floor(0.8* nrow(pewdata))

## set the seed to make your partition reproductible
set.seed(1234)
train_ind = sample(seq_len(nrow(pewdata)), size = smp_size)

train = pewdata[train_ind, ]
test = pewdata[-train_ind, ]

################
## CREATE MODEL
################

groupvar = c("F_CITIZEN", "F_EDUCCAT",                
             "F_RACECMB", "F_PARTY_FINAL")

model = as.formula(paste("supportdem", paste(groupvar, collapse=" + "), sep=" ~ "))

################
## KNN 
################

#KNN can't handle categorical features.

library(class)
set.seed(1234) 
#set seed because if several obs. are tied as nearest neighbors, 
#then R will randomly break the tie

#need to create vector/cbind (df doesn't work)
#need to make all categorical explanatory variables numeric
train.vars = with(train, cbind(jitter(as.numeric(F_CITIZEN),0, 0.01), jitter(as.numeric(F_EDUCCAT),0, 0.01), jitter(as.numeric(F_RACECMB),0, 0.01), jitter(as.numeric(F_PARTY_FINAL),0, 0.01)))  
test.vars = with(test, cbind(jitter(as.numeric(F_CITIZEN),0, 0.01), jitter(as.numeric(F_EDUCCAT),0, 0.01), jitter(as.numeric(F_RACECMB),0, 0.01), jitter(as.numeric(F_PARTY_FINAL),0, 0.01)))  
train.vars = scale(train.vars)
test.vars = scale(test.vars)

head(train.vars)

#need to make all categorical outcomes numeric
train.outcome = as.numeric(train$supportdem)
table(train.outcome) #now have categorical variable

#knn pred = knn(train, test, train$outcome, k=number outcomes)

knn.pred_k1 = knn(train.vars, test.vars, train.outcome, k=1)

table(knn.pred_k1, test$supportdem)
#store confusion matrix
cm = as.matrix(table(Actual = test$supportdem, Predicted = knn.pred_k1)) # create the confusion matrix

cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes

knnaccuracy = sum(diag) / n 
knnaccuracy

knnprecision = diag / colsums 
knnrecall = diag / rowsums 
knnf1 = 2 * knnprecision * knnrecall / (knnprecision + knnrecall) 

knnprecision
knnrecall
knnf1

################
## LDA 
################
library(MASS)
lda.model = lda(model, data=train)

#view model results
lda.model

#visualize the predictions
test.lda.pred = predict(lda.model, newdata=test)
names(test.lda.pred)

test.lda.class = test.lda.pred$class
summary(test.lda.class)


#view posterior distribution
test.lda.pred$posterior[1:10,1] #predicted probability
test.lda.class[1:10] #how these probabilities translate into classes

cbind(test.lda.pred$posterior[1:10,1], test.lda.class[1:10])

#plot posterior distribution of predictions
ldahist(test.lda.pred$x[,1], g=test$supportdem)

#Model Assessment

library(caret)
confusionMatrix(as.factor(test.lda.pred$class), as.factor(test$supportdem), positive="High Support")


