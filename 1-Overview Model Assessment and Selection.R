####################
# PSC 8185: ML4SS
# Week 1: Overview to ML
# Jan. 10, 2022
# Author: Iris Malone
####################
rm(list=ls(all=TRUE))

# Step 1: Recipe for Building a ML Model

# Create Data
set.seed(1234)
n = 100

#Our true data
x = runif(n, 0, 4)
y = 4 -2*x + x**2 + rnorm(n, 0, 2) # true f
df = data.frame(x, y)

# Partition Data into Training and Test Set
# 80% of data will be in training set and 20% in test set
smp_size = floor(0.8* nrow(df))

## set the seed to make your partition reproductible
set.seed(1234)
train_ind = sample(seq_len(nrow(df)), size = smp_size)

train = df[train_ind, ]
test = df[-train_ind, ]

# Plot Training and Test Data
#training data
with(train, plot(y ~ x, pch=21, col="black", bg="red", cex=1.4,
                 xlim=c(0,4), ylim=c(0,16), xlab="X", ylab="Y", 
                 main="Training and Testing Data"))

with(test, points(y ~ x, pch=21, col="black", bg="dodgerblue", cex=1.4))

# Step 2: Make a `Guess` about the Best Functional Form ($f$) 
# to Describe Relationship between X & Y

############
# Example 1: A Non-Systematic Approach
############

#Estimate our functional form
x_curve <- seq(-1, 5, by=0.001)
y_curve_1 <- 6 - 2.5 * x_curve + x_curve**2

# Plot our Estimate on Top of the Data to See How "Good" It Is

plot(y_curve_1 ~ x_curve, pch=".", xlim=c(0,4), ylim=c(0,15), 
     xlab="X", ylab="Y", main="Model 1")
lines(y_curve_1 ~ x_curve, lwd=2)
with(train, points(y ~ x, pch=21, col="black", bg="red", cex=1.4))

# Step 3: Model Assessment - 
# We will see how good a fit the data is to our $f$ is by estimating the **mean squared error**

# Procedure to Estimate MSE:

#Make training predictions

train$pred_1 = 6 - 2.5 * train$x + train$x^2

# Subtract the new Y value from the original to get the training error:
train$error_1 = train$y - train$pred_1

# Square the errors;
train$error_1_sq = train$error_1^2

# Add up the errors;
sum_error_1_sq = sum(train$error_1_sq)

# Find the mean:
train_mse_1_long = sum_error_1_sq/dim(train)[1]
train_mse_1_long

#one step calculation
train_mse_1_short = mean(train$error_1^2)
train_mse_1_short

############
# Example 2: A More Systematic Approach
############

# Model selection involves finding the minimum MSE for a given model's complexity.

# We will compare four models and determine which model best fits our true data.

# In each case we estimate our model based on the **training** data only. 

# Model 1 is the best-fitting linear model (least complex). 
mod1 = lm(y ~ x, data=train)

# Model 2 is the quadratic model.
mod2 = lm(y ~ poly(x, 2), data=train)

# Model 3 is the cubic model.
mod3 = lm(y ~ poly(x, 3), data=train)

# Model 4 is the degree 9 polynomial model (most complex). 
mod4 = lm(y ~ poly(x, 9), data=train)

# We use the model we created to make a series of predictions

# We first look at predicted values for the training data. 
# (How well does the model fit on the data it knows?)

fitted1 = predict(mod1, data.frame(x=train$x))
fitted2 = predict(mod2, data.frame(x=train$x))
fitted3 = predict(mod3, data.frame(x=train$x))
fitted4 = predict(mod4, data.frame(x=train$x))

# Visual Inspection of how training data (green) and model predictions (black) compare for given x
par(mfrow=c(1,4))

# Model 1 is too simple. 
# This linear model fails to capture the apparently nonlinear relationship in the training data. 
# This is a case of  **underfitting**, where the model being considered in too simple to capture the true nature of the relationship. 

with(train, plot(y ~ x, pch=".", xlim=c(0,4), ylim=c(0,15), 
                 main="Model 1 (Linear)"))
lines(fitted1 ~ train$x, col="black", lwd=2.5)
with(train, points(y ~ x, pch=21, col="black", bg="limegreen", cex=1.4))

with(train, plot(y ~ x, pch=".", xlim=c(0,4), ylim=c(0,15), 
                 main="Model 2 (Quadratic)"))
points(fitted2 ~ train$x, col="black", lwd=2.5, bg="red")
with(train, points(y ~ x, pch=21, col="black", bg="limegreen", cex=1.4))

with(train, plot(y ~ x, pch=".", xlim=c(0,4), ylim=c(0,15), 
                 main="Model 3 (Cubic)"))
points(fitted3 ~ train$x, col="black", lwd=2.5)
with(train, points(y ~ x, pch=21, col="black", bg="limegreen", cex=1.4))

# Model 4, on the other hand, seems unnecessarily complicated. 
# The curve passes near more training points, but indicates that there are complexities in the relationship that probably don't actually exist. 
# This is a classic case of **overfitting**, where the model being considered is too flexible, and trains itself to the noise in the data. 
# An overfit model will perform unreasonably well on the training data, but will fail to generalize well to new, unseen observations. 

with(train,plot(y ~ x, pch=".", xlim=c(0,4), ylim=c(0,15), 
                main="Model 4 (Degree 9)"))
points(fitted4 ~ train$x, col="black", lwd=3)
with(train,points(y ~ x, pch=21, col="black", bg="limegreen", cex=0.9))

# We can see how good a model is based on how much estimates are "biased"
# We can measure bias as $y - \hat{y}$ and display the residuals in red

# Look at residuals
# Model 4 might have lower residuals, but runs risk of overfitting
with(train, plot(y ~ x, pch=".", xlim=c(0,4), ylim=c(0,15), 
                 main="Model 1"))
lines(fitted1 ~ train$x, col="black", lwd=2.5)
segments(train$x, train$y, train$x,  mod1$fitted.values, col="firebrick3", lwd=2)
with(train, points(y ~ x, pch=21, col="black", bg="limegreen", cex=1.4))

with(train, plot(y ~ x, pch=".", xlim=c(0,4), ylim=c(0,15), 
                 main="Model 2"))
points(fitted2 ~ train$x, col="black", lwd=2.5, bg="red")
segments(train$x, train$y, train$x,  mod2$fitted.values, col="firebrick3", lwd=2)
with(train, points(y ~ x, pch=21, col="black", bg="limegreen", cex=1.4))

with(train, plot(y ~ x, pch=".", xlim=c(0,4), ylim=c(0,15), 
                 main="Model 3"))
points(fitted3 ~ train$x, col="black", lwd=2.5)
segments(train$x, train$y, train$x, mod3$fitted.values, col="firebrick3", lwd=2)
with(train, points(y ~ x, pch=21, col="black", bg="limegreen", cex=1.4))

with(train,plot(y ~ x, pch=".", xlim=c(0,4), ylim=c(0,15), 
                main="Model 4 (Degree 9)"))
points(fitted4 ~ train$x, col="black", lwd=3)
segments(train$x, train$y, train$x, mod4$fitted.values, col="firebrick3", lwd=2)
with(train,points(y ~ x, pch=21, col="black", bg="limegreen", cex=0.9))

par(mfrow=c(1,1))

# Estimate MSE across models

mse1 = mean(mod1$residuals^2)
mse2 = mean(mod2$residuals^2)
mse3 = mean(mod3$residuals^2)
mse4 = mean(mod4$residuals^2)

train_mse = c(mse1, mse2, mse3, mse4)
names(train_mse) = c("Model 1", "Model 2", "Model 3", "Model 4") 

#Results show Model 4 has lowest training MSE, but we think it might be overfitting.

train_mse

# We can better assess which model is best by examining **test MSE**.

# We again use the model created by looking at the training data
mod1 = lm(y ~ x, data=train)
mod2 = lm(y ~ poly(x, 2), data=train)
mod3 = lm(y ~ poly(x, 3), data=train)
mod4 = lm(y ~ poly(x, 9), data=train)

# We now make a series of predictions based on the test data
pred_test1 = predict(mod1, data.frame(x=test$x))
pred_test2 = predict(mod2, data.frame(x=test$x))
pred_test3 = predict(mod3, data.frame(x=test$x))
pred_test4 = predict(mod4, data.frame(x=test$x))

#Estimate test MSE by looking at how close \hat{f}(x)_test is to y_test
test_mse1 = mean((test$y - pred_test1)^2 )
test_mse2 = mean((test$y - pred_test2)^2 )
test_mse3 = mean((test$y - pred_test3)^2 )
test_mse4 = mean((test$y - pred_test4)^2 )

test_mse = c(test_mse1, test_mse2, test_mse3, test_mse4)
names(test_mse) = c("Model 1", "Model 2", "Model 3", "Model 4") 

# We can now compare training and test MSE:
paste("Train MSE:")
train_mse
paste("Test MSE:")
test_mse

# Results show difference between training and test MSE.
# We see high test MSE for Model 1 consistent with underfitting.
# We see very high test MSE for Model 4 consistent with extreme overfitting. 

# Model 3 has the lowest validation MSE. This is our **optimal** model.

##################
# Main Takeaways:
##################

# Assess model performance based on MSE
# Select `best' model based on test MSE
# As model flexibility increases, training MSE decreases. 
# As model flexibility increased, test MSE first decreases, and then increases. 
