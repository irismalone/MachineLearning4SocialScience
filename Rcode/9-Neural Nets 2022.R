####################
# PSC 8185: ML4SS
# Session 9: Neural Nets
# March 28, 2022
# Author: Iris Malone
####################
rm(list=ls(all=TRUE))

# Neural Nets
# 3 Different Applications


########################################
#Basic Feedforward Neural Net
########################################
require(nnet)

require(Ecdat)
data(Fishing)
head(Fishing)

?Fishing

#4 different outputs
table(Fishing$mode)


set.seed(1234)
smp_size = floor(0.6* nrow(Fishing))

train_ind = sample(seq_len(nrow(Fishing)), size = smp_size)

train = Fishing[train_ind, ]
test = Fishing[-train_ind, ]

dim(train) #there are 11 predictors

?nnet
#key inputs are neurons, learning rate, epochs

#For more see: https://medium.com/@yolandawiyono98/ann-classification-with-nnet-package-in-r-3c4dc14d1f14
set.seed(1234)
model = nnet(mode~., #each predictor is a different input
             data=train,
             size=12, #number of neurons in hidden layer, typically predictors + 1
             decay=5e-4,  #learning rate for SGD
             maxit=500) #number of epochs

model

#a 11-12-4 network
#11 input layers
#12 hidden layers
#4 outputs 

#If we had single-class problem we'd see 1 output for each class, e.g.

#model = nnet(I(mode=="pier")~., #each predictor is a different input
#             data=train,
#             size=12, #number of neurons in hidden layer, typically predictors + 1
#             decay=5e-4,  #learning rate for SGD
#             maxit=200) #number of epochs

#> model
#a 11-12-1 network with 157 weights
#inputs: price catch pbeach ppier pboat pcharter cbeach cpier cboat ccharter income 
#output(s): I(mode == "pier") 

summary(model) #see weighted inputs for each i (input), h (hidden), o (output), or b (bias)

names(model)

model
#Create predictions

pred = predict(model, test, type="class")
table(pred) #not a great model - misses an entire class

table(test$mode, pred)
dim(test)

acc = (66+152+186)/473
acc
accuracy = mean(pred == test$mode)
accuracy

#import the function from Github
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

#visualize the neural net
plot.nnet(model)


# Same Function in Caret
#Fit model
require(caret)
#takes awhile to run so start 
model = train(mode~., 
              data=train, 
              method='nnet', 
              linout=TRUE, 
              trace = FALSE,
               #Grid of tuning parameters to try:
               tuneGrid=expand.grid(.size=c(1,5,10, 15, 20), #number neurons
                                    #affect bias-var tradeoff
                                    .decay=c(0,0.001,0.1))) 

model
#15 neurons, decay .1

pred = predict(model$finalModel, test)
head(pred) 
#takes outcome with largest probability and assigns it to that class
pred = predict(model$finalModel, test, type="class")
head(pred)

table(test$mode, pred)

 # slightly diff accuracy due to diff hyperparameter
accuracy = mean(pred == test$mode)
accuracy

#########################
#Recurrent Neural Net
#########################

# Application: Predict foreign exchange rates

ssh = suppressPackageStartupMessages

#time series data set-up packages
ssh(library(timeSeries))
ssh(library(tseries))
ssh(library(aTSA))

#forecast tools
ssh(library(forecast))
ssh(library(rugarch))
ssh(library(ModelMetrics))

#keras - deep learning package
ssh(library(keras))

#data from time series package
data(USDCHF)
length(USDCHF)

#transform data into ts
data = ts(USDCHF, frequency = 365) #freq is min amount of time
plot(data) #see relative exchange rate over time

#traditional forecasting approach: ARIMA

#with forecasting, we split data by date
dim(data)
train = data[1:(length(data)-10-1)] 
test = data[(length(data)-10):length(data) ] #forecast out 11 days

length(train)
length(test)

arima = auto.arima(train)
summary(arima)

#plot time series data
plot.ts(train, main = "Exchange Rate")

#see: https://rpubs.com/kapage/523169
#Hidden layers creation
require(forecast)
alpha = 1.5^(-10)
hid.layer = length(train)/(alpha*(length(train) + 11))


lambda.train = BoxCox.lambda(train) #transform data to be stationary

library(forecast)
#Fitting nnetar
#nnetar function in the forecast package fits a single hidden layer 
#neural network model to a timeseries object
dnntrainpred = nnetar(train, lambda = lambda.train)

# Forecasting 
#this takes awhile
dnnforecast = forecast(dnntrainpred, #ts object
                       h = 11, #number of periods for forecasting
                       PI = TRUE) #whether to plot prediction intervals
testforecast = data.frame(dnnforecast)
head(testforecast)
testmse = mean(test- testforecast[,1])^2
testmse
accuracy(dnnforecast)


###################
## RNN LSTM
##################

#Alternative (best?) way is to build RNN LSTM
#
maxlen = 7 #batch size, length of each sequence the model will study, e.g. 7 days

#create empty matrix to store values
#we'll predict out much further (100 days) and much faster
train = data[1:(length(data)-99-1)] 
test = data[(length(data)-99):length(data) ] #forecast out 100 days

length(train)
length(test)

exch_matrix = matrix(0, 
                     nrow = length(train)-maxlen-1, 
                     ncol = maxlen+1) 

for(i in 1:(length(train)-maxlen-1)){
  exch_matrix[i,] = train[i:(i+maxlen)]
  }
head(exch_matrix) 

#populate the frame, note it's 
#t, t+1, t+2, t+3, ..., t+7
#t+1, t+2, t+3, ..., t+8
#...
#t+5, t+6, t+7, ..., t+12

#partition train inputs and vector
x_train = exch_matrix[, -ncol(exch_matrix)] 
#this will be matrix with:
#t, t+1, t+2, t+3, ..., t+6
#t+1, t+2, t+3, t+4, ..., t+7

y_train = exch_matrix[, ncol(exch_matrix)]
#this has output
#[t+7, t+8, t+9, ..., t+12]

#rnn expects specific structure:
#shape (examples, maxlen, number of features)
dim(x_train) #this is currently just 2d so have to reshape

x_train = array_reshape(x_train, 
                        dim = c((length(train)-maxlen-1), 
                                maxlen, 1))
dim(x_train)
#this tells us we have 62388 training examples
#we are predicting forward 1 step
#looking at 7 input (7 previous values)

#next we build the model:

#create blank keras
model = keras_model_sequential()

model %>% 
  #this is the input layer information:
  layer_dense(input_shape = dim(x_train)[-1], #number of inputs
              activation = "relu", #activation function to apply
              units=maxlen) %>% 
  #this is the hidden layer information:
  #this is the number of neurons in hidden layers
  layer_simple_rnn(units=10) %>% 
  layer_dropout(rate = 0.1) %>% #dropout layer to prevent overfitting
   #just have one hidden layer
  layer_dense(units = 1)
summary(model)

#train the model
model %>% compile(
  loss = "mse", #minimize loss function via mse
  optimizer= "adam", #loss function
  metric = "mae" ) 

history = model %>% 
    fit(x_train, y_train, 
        epochs = 5, #number of passes - in practice is way too low! (50, 100, 500 more common)
        batch_size = 32, #number of obs per sequence
        validation_split=0.1) #validation

plot(history)  
save_model_hdf5(model, "rnn_model.h5")
rnn_model = load_model_hdf5("rnn_model.h5")

#get predictions
maxlen = 7
exch_matrix2 = matrix(0, nrow = length(data)-maxlen-1, ncol = maxlen+1) 
for(i in 1:(length(data)-maxlen-1)){
  exch_matrix2[i,] = data[i:(i+maxlen)]
}
x_train2 = exch_matrix2[, -ncol(exch_matrix2)]
y_train2 = exch_matrix2[, ncol(exch_matrix2)]
x_train2 = array_reshape(x_train2, dim = c((length(data)-maxlen-1), maxlen, 1))

pred = rnn_model %>% predict(x_train2)

#use tibble package to create new data frame comparing observed and expected
df_eval_rnn = tibble::tibble(y_rnn=y_train2[(length(y_train2)-99):length(y_train2)],
                              yhat_rnn=as.vector(pred)[(length(y_train2)-99):length(y_train2)])

head(df_eval_rnn)
rmse = c(
          rmse(df_eval_rnn$y_rnn, df_eval_rnn$yhat_rnn) )
mae = c( 
         mae(df_eval_rnn$y_rnn, df_eval_rnn$yhat_rnn) )
df = tibble::tibble(model=c("RNN"), rmse, mae)
df

###########################
#Convolutional Neural Net
###########################

#Application: Digit Classification
#walking through r code here: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/4462572393058129/1806228006848429/latest.html
#mnist dataset included in keras
mnist = dataset_mnist()
head(mnist)
#already partitioned
names(mnist)

#create training and test 
x_train = mnist$train$x
y_train = mnist$train$y
x_test = mnist$test$x
y_test = mnist$test$y

#plot an image using image() function
index_image = 28 ## change this index to see different image.
input_matrix = x_train[index_image,1:28,1:28]
output_matrix = apply(input_matrix, 2, rev)
output_matrix = t(output_matrix)
image(1:28, 1:28, output_matrix, col=gray.colors(256), xlab=paste('Image for digit of: ', y_train[index_image]), ylab="")

input_matrix #this is matrix for image 28 broken into 1 and 0



# Input image dimensions
img_rows = 28
img_cols = 28

#data preprocessing
#for CNN the input of a MxN image is a MxNxK 3D arrays with K specific channels. 

#channels are based on the complexity of the image
#a greyscale MxN image has only one channel, and the input is MxNx1 tensor. 
#An MxN 8-bit per channel RGB image has three channels with 3 MxN array with values between 0 and 255, 
# so the input is MxNx3 tensor. 

#For RGB color image, the number of channels is 3 
#and we need to replace "1" with "3" for the code cell below if the input image is RGB format.

#add channel into the dimension in last column
x_train = array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test = array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape = c(img_rows, img_cols, 1)

#scale the data for optimization
summary(x_train) #range of values is 0-255 so scale to be between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# Convert class vectors to binary class matrices
num_classes = 10 #number of distinct numbers
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#CNN model contains a series 2D convolutional layers which contains a few parameters: 
  #(1) the kernel_size which is typically 3x3 or 5x5; 
  #(2) the number of filters, which corresponding to the number of channels (i.e. the 3rd dimension) 
  #in the output tensor; 
  #(3) activation funtion.

# define model structure 
cnn_model = keras_model_sequential() 
cnn_model %>%
  #apply kernel filter to input data
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', 
                input_shape = input_shape) %>% 
  #pooling will be used to create new feature map
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  #typically add multiple layers to improve performance
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

summary(cnn_model)

# Compile model
cnn_model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Define a few parameters to be used in the CNN model
batch_size = 128
epochs = 10 #this is again way too small but will expedite process

#this will take a few minutes due to size of training and batch_size
#train the model
history = cnn_model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)

plot(history)
#major reduction in training error as model passes through data more

#get test model accuracy
cnn_model %>% evaluate(x_test, y_test)

# model prediction
cnn_pred = cnn_model %>% 
  predict_classes(x_test) #model will predict which number is in an image from 1 to 10

head(cnn_pred, n=50)

## number of mis-classified images
sum(cnn_pred != mnist$test$y)

missed_image = mnist$test$x[cnn_pred != mnist$test$y,,]
missed_digit = mnist$test$y[cnn_pred != mnist$test$y]
missed_pred = cnn_pred[cnn_pred != mnist$test$y]

predictions = predict_classes(cnn_model, x_test)
probabilities = predict_proba(cnn_model, x_test)
head(probabilities) #probability of being assigned to a given class per observation

index_image = 6 ## change this index to see different image.
input_matrix = missed_image[index_image,1:28,1:28]
output_matrix = apply(input_matrix, 2, rev)
output_matrix = t(output_matrix)
image(1:28, 1:28, output_matrix, col=gray.colors(256), xlab=paste('Image for digit ', missed_digit[index_image], ', wrongly predicted as ', missed_pred[index_image]), ylab="")


