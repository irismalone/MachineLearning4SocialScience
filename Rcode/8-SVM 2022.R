####################
# PSC 8185: ML4SS
# Session 8: Support Vector Machines
# March 21, 2022
# Author: Iris Malone
####################

rm(list=ls(all=TRUE))
library (e1071) #svm package
library(ggplot2)
#########################
## MAXIMAL MARGIN CLASSIFIER
#########################

#if linear separable data, then we can use maximal margin

set.seed (1)
x <- matrix(rnorm(20*2), ncol = 2)

#create artificial linear relationship demarcated by -1 or +1
y = ifelse(x[,1] + 2*x[,2]>0, 1, -1)

#svm requires factor variable for classification

df = data.frame(x = x, y
                = as.factor (y))
head(df)
# are the classes linearly separable?
# Plot data
ggplot(data = df, aes(x = x.2, y = x.1, color = y, shape = y)) + 
  geom_point(size = 2) 
#note infinite number hyperplanes

# taking sample of 16 datapoints in training set
train_ind = sample(20, 16)

train = df[train_ind, ]
test = df[-train_ind, ]
dim(train)
dim(test)
# fitting the model
?svm #options

set.seed(1234)
svmfitlinear = svm(
  y~.,
  data = train,
  kernel ="linear", #tells it to use linear kernel (which is basic classifier)
  cost = 10, #tuning parameter
  scale = FALSE) #tells model not to scale and standardize

names(svmfitlinear)
#see different outputs -mostly just ways to tune model (kernel, cost, degree, epsilon)

#plotting
plot(svmfitlinear, train)
#points with “X” are the support vectors (points that affect hyperplane)
#The points marked with an “o” are the other points (no affect hyperplane) 
summary(svmfitlinear)
#tells us 5 support vectors, 3 in one class (yellow) and 2 in the other (red)

#Modification: Use smaller value of the cost parameter

svmfitlinear = svm(
  y~.,
  data = train,
  kernel ="linear", #tells it to use linear kernel (which is basic classifier)
  cost = 0.1, #tuning parameter
  scale = FALSE) #tells model not to scale and standardize


#plotting
plot(svmfitlinear, train)

summary(svmfitlinear) 
#obtain larger number of support vectors (because margin is wider)


########################
## TUNING HYPERPARAMETER BUDGET
########################

#e1071 library has built-in function
set.seed (1234)
tune.out = tune(svm,
                y~.,
                data = train,
                kernel ="linear",
                #feed grid of values to tune cost
                ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 100)))
summary(tune.out)
#see cross-validation results
#finds best cost function is 5 with minimal error 0 (because perfectly separable)

#like lasso/ridge, identify best model by CV and use that for prediction
bestmod = tune.out$best.model
summary(bestmod)

predict = predict(bestmod, test)

#evaluate predictive performance
require(caret)
confusionMatrix(predict, test$y, positive="1")

########################
## SUPPORT VECTOR CLASSIFIER
########################

#######################
## use support vector to model non-linear relationship
#######################

#create data with a non-linear boundary
set.seed (1)
x = matrix (rnorm (1000 * 2) , ncol = 2)
x[1:500 , ] = x[1:500 , ] + 2
x[501:750 , ] = x[501:750 , ] - 2
y = c(rep (1 , 750) , rep (2 , 250))
df = data.frame(x = x, 
                y = as.factor (y))


# plotting shows that the class boundary is non-linear
plot(x, col = (3 - y))

# taking sample of 100 datapoints in training set
train_ind = sample(1000,900)

train = df[train_ind, ]
test = df[-train_ind, ]

#############
# RADIAL KERNEL
##############

svmfitradial = svm(
  y~.,
  data = train,
  kernel ="radial", #tells it to use radial kernel (which is basic classifier)
  gamma=1, #gamma constant
  cost = 1, #tuning parameter
  scale = FALSE) #tells model not to scale and standardize

#plotting shows that SVM has non-linear boundary
plot(svmfitradial, train)
#see radial shape
#x shows us # support vectors 

summary(svmfitradial) 
#find 264 total support vectors

#increasing the cost parameter can reduce the number of training errors
svmfitradial = svm(
  y~.,
  data = train,
  kernel ="radial", #tells it to use radial kernel (which is basic classifier)
  gamma=1, #gamma constant
  cost = 1e5, #tuning parameter
  scale = FALSE) #tells model not to scale and standardize

#plotting shows that SVM has non-linear boundary
plot(svmfitradial , train)

summary(svmfitradial)

################
## CROSS-VALIDATION
set.seed(1234)
#can tune both constant gamma parameter and cost function
tune.out = tune(svm ,
                y~.,
                data = train,
                kernel ="radial",
                ranges = list(cost = c(0.001 , 0.01, 0.1, 1, 5, 10, 100),
                              gamma = c(0.5, 1, 2, 3, 4))) #two different tuning parameters
summary(tune.out)
#see cross-validation results
#best model has cost 0.1 and gamma 1
tune.out$best.parameters

#like lasso/ridge, identify best model by CV and use that for prediction
bestmod = tune.out$best.model

#best mod has 156, 153
summary(bestmod)

predict = predict(bestmod, test)

#evaluate predictive performance
require(caret)
confusionMatrix(predict, test$y)
#see misclassifications



########################
## ROC CURVES
########################
library(ROCR)

#write function to get roc plots 
rocplot = function (pred , truth , ...) {
  predob = prediction (pred , truth)
  perf = performance (predob ,"tpr","fpr")
  plot(perf , ...)
}

#do optimal model
svmfit.opt = svm(
  y~.,
  data = train,
  kernel ="radial",
  gamma = 1,
  cost = 0.1,
  decision.values = T
)
#decision values TRUE means we want it to return a vector of predictions (on training)

fitted =attributes(predict(svmfit.opt , #model
                             train, #newdata 
                             decision.values =TRUE))$decision.values 

rocplot(1-fitted, train["y"], main="Training Data")


#increase gamma to see even more flexible model
svmfit.flex = svm(
  y~.,
  data = train,
  kernel ="radial",
  gamma = 50,
  cost = 0.1,
  decision.values = T
)

fitted =attributes (predict (svmfit.flex , #model
                             train, #newdata 
                             decision.values =TRUE))$decision.values

rocplot(1-fitted, train["y"], add=T, col="red")

#see slightly better AUC when we increase model flexibility 
#(allow for more misclassifications)

#plot ROC for test data

fitted.test =attributes (predict (svmfit.opt , #model
                             test, #newdata 
                             decision.values =TRUE))$decision.values

fitted.test.flex =attributes (predict (svmfit.flex , #model
                                  test, #newdata 
                                  decision.values =TRUE))$decision.values

rocplot(1-fitted.test, test["y"], main="Test Data")
rocplot(1-fitted.test.flex, test["y"], add=T, col="red")
#see increased flexibility doesn't result in as good ROC for test data

#alternative way to get ROC/AUC

#for test data
require(pROC)
test$pred = 1-fitted.test.flex
rf.roc.oos = roc(as.numeric(test$y), test$pred)
plot(test$y, test$pred)
auc(rf.roc.oos) #AUC 
ggroc(list(rf.roc.oos),aes=c("linetype"), legacy.axes=T)  + theme_light() + 
  geom_segment(aes(x = 0, xend = 1, y = 0, yend = 1), color="darkgrey", linetype="dashed") + 
  theme(legend.box.background = element_rect(color="black", size=0.5),
        legend.position="bottom", legend.text=element_text(size=12))

########################
## MULTI-CLASS
########################
set.seed(1234)
# creating a matrix from normal distribution of dimentions 50*2
x = rbind(x, matrix(rnorm(500*2), ncol=2))
# creating a variable y 
y = c(y, rep(0,500))
# applying some transformation
x[y==0,2]=x[y==0,2]+2
#creating data frame 
df = data.frame(x=x, y=as.factor(y))


# taking sample of 100 datapoints in training set
train_ind = sample(1000,900)

train = df[train_ind, ]
test = df[-train_ind, ]

par(mfrow=c(1,1))
# plotting matrix
plot(x,col=(y+1))

# fitting the model
svmfitmulticlass=svm(y~., 
           data=train, 
           kernel="radial", 
           cost=10, 
           gamma=1)

# plotting the model
plot(svmfitmulticlass, train)


predict = predict(svmfitmulticlass, test)


#evaluate predictive performance
require(caret)
confusionMatrix(predict, test$y)
#radial is actually doing a pretty good job classifying multiclass
#very high sensitivity and specificity scores

######################
## REAL-WORLD EXAMPLE
######################
# Load the data
data("PimaIndiansDiabetes2", package = "mlbench")
#health data of Pima Native Americans; 
#outcome is whether they had diabetes; 
#predictors = different potential risk factors

pima.data = na.omit(PimaIndiansDiabetes2)

#scale the data to standardize
pima.scale = as.data.frame(scale(pima.data[,-9]))

#visualize the data
require(reshape2)
pima.scale$diabetes = pima.data$diabetes

pima.scale.melt = melt(pima.scale, id.var="diabetes")
#boxplot of different attributes
ggplot(data=pima.scale.melt, aes(x=diabetes, y=value)) +geom_boxplot()+facet_wrap(~variable, ncol=2)

cor(pima.scale[-9])

set.seed(1234)
smp_size = floor(0.8* nrow(pima.scale))

train_ind = sample(seq_len(nrow(pima.scale)), size = smp_size)

train = pima.scale[train_ind, ]
test = pima.scale[-train_ind, ]

head(train)
#see approx 7 different covariates which could predict diabetes 

#with RW data we don't know right kernel so 
#unfortunately have to just compare lots diff specifications
#SVM modeling
set.seed(1234)
linear.tune = tune.svm(diabetes~., 
                       data=train, 
                       kernel="linear", 
                       cost=c(0.001, 0.01, 0.1, 1,5,10))
linear.tune$best.parameters
summary(linear.tune)
best.linear = linear.tune$best.model
preds = predict(best.linear, newdata=test)
require(caret)
linear.cm = confusionMatrix(as.factor(preds), as.factor(test$diabetes), positive="pos")

#poly
set.seed(1234)
poly.tune = tune.svm(diabetes~., 
                       data=train, 
                       kernel="poly", 
                       cost=c(0.001, 0.01, 0.1, 1,5,10))
summary(poly.tune)
best.poly = poly.tune$best.model
preds = predict(best.poly, newdata=test)
require(caret)
poly.cm = confusionMatrix(as.factor(preds), as.factor(test$diabetes), positive="pos")

#radial

set.seed(1234)
radial.tune = tune.svm(diabetes~., 
                     data=train, 
                     kernel="radial", 
                     cost=c(0.001, 0.01, 0.1, 1,5,10))
summary(radial.tune)
best.radial = radial.tune$best.model
preds = predict(best.radial, newdata=test)
require(caret)
radial.cm = confusionMatrix(as.factor(preds), as.factor(test$diabetes), positive="pos")

linear.cm$byClass["Balanced Accuracy"]
poly.cm$byClass["Balanced Accuracy"]
radial.cm$byClass["Balanced Accuracy"]

#linear cm looks like best
########################
## SUPPORT VECTOR MACHINE W/CARET
########################

#####################
# EXAMPLE IN CARET:
#####################

rm(list=ls(all=TRUE))

data("PimaIndiansDiabetes2", package = "mlbench")
pima.data = na.omit(PimaIndiansDiabetes2)
#predict whether an individual tests positive for clinical diabets
# taking sample of 100 datapoints in training set
set.seed(1234)
cvseeds = vector(mode = "list", length = 11) #length 11 because it is k+1
for(i in 1:11) cvseeds[[i]] <- sample.int(1000, 51) #sample needs to be suff large 

## For the last model:
cvseeds[[11]] <- sample.int(1000, 1)

set.seed(1234)
smp_size = floor(0.8* nrow(pima.data))

train_ind = sample(seq_len(nrow(pima.data)), size = smp_size)

train = pima.data[train_ind, ]
test = pima.data[-train_ind, ]

# Inspect the data
head(pima.data)

require(caret)
# Set up  k-fold Cross Validation
train_control = trainControl(method="cv", 
                             number=10, seeds=cvseeds)

# SVM linear classifier
# Fit the model 
set.seed(1234)
svm.caret = train(diabetes ~., 
              data = train, 
              method = "svmLinear", 
              trControl = train_control,  
              preProcess = c("center","scale"))
#View the model
svm.caret


#optimize hyperparameters
# Fit the model 
svm.caret.tune = train(diabetes ~., 
              data = train, 
              method = "svmLinear", 
              trControl = train_control,  
              preProcess = c("center","scale"), 
              tuneGrid = expand.grid(C = seq(0, 2, length = 20))) #cost parameter
#View the model
svm.caret.tune 

# Plot model accuracy vs different values of Cost
plot(svm.caret.tune)

# Print the best tuning parameter C that
# maximizes model accuracy
svm.caret.tune$bestTune
library(kernlab)
pred.linear = predict(object=svm.caret$finalModel, newdata=test[,-9])
linear.cm = confusionMatrix(as.factor(pred.linear), test$diabetes, positive="pos")

#computing SVM using polynomial function:

# Fit the model 
svm.caret.poly = train(diabetes ~., 
                       data = train, 
                       method = "svmPoly", 
                       trControl = train_control, 
                       preProcess = c("center","scale"))
# Print the best tuning parameter sigma and C that maximizes model accuracy
svm.caret.poly$bestTune


pred.poly = predict(svm.caret.poly$finalModel, newdata=test[,-9])
poly.cm = confusionMatrix(as.factor(pred.poly), test$diabetes, positive="pos")



#Computing SVM using radial basis kernel:

# Fit the model 
svm.caret.radial = train(diabetes ~., 
              data = train, 
              method = "svmRadial", 
              trControl = train_control, 
              preProcess = c("center","scale"))
# Print the best tuning parameter sigma and C that maximizes model accuracy
svm.caret.radial$bestTune


pred.radial = predict(svm.caret.radial$finalModel, newdata=test[,-9])
radial.cm = confusionMatrix(as.factor(pred.radial), test$diabetes, positive="pos")



#plot training results
results = resamples(list(svm.linear=svm.caret.tune, svm.poly=svm.caret.poly, svm.radial=svm.caret.radial))
summary(results)
dotplot(results)

#compare confusion matrix
linear.cm$byClass["Balanced Accuracy"]
poly.cm$byClass["Balanced Accuracy"]
radial.cm$byClass["Balanced Accuracy"]

