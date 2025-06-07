####################
# PSC 8185: ML4SS
# Session 10: Unsupervised Learning
# April 4, 2022
# Author: Iris Malone
####################
rm(list=ls(all=TRUE))

################################
## PRINCIPAL COMPONENT ANALYSIS
################################

### LOOK AT USARRESTS DATA
states = row.names(USArrests)
states
names(USArrests)

#check the mean and variance of the USArrests dataset.
apply(USArrests, 2, mean)

apply(USArrests, 2, var)
#recall - we center and scale so that we get variation around var, not mean


#Use prcomp function in stats package
?prcomp
pr.out = prcomp(USArrests, scale = TRUE)
names(pr.out)

#print out gives us the eigenvalues 
pr.out

#Rotation component corresponds to the matrix 
#whose columns contain the eigenvectors
loadings = pr.out$rotation
loadings

#rows of matrix are called the eigenvectors
#these specify the orientation of the principal components relative to the original variables. 
#The elements of each eigenvector are the loadings
#loadings describe how much each variable contributes to a particular principal component. 
#variables with large (absolute value) loadings contribute more to that PC
#Sign of  a loading indicates whether a variable and a principal component are positively or negatively correlated.

#The center and scale components contain the mean and standard deviations prior to scaling.
pr.out$center
pr.out$scale

#The plot is showing:
#the score of each case (e.g., state) on the first two principal components
#the loading of each variable (e.g. crime) on the first two principal components.
#The left and bottom axes are showing [normalized] principal component scores; 
#the top and right axes are showing the loadings.
biplot(pr.out)


#alternate way to visualize (much prettier)
require(ggfortify)
#plot vectors, but not observations
autoplot(prcomp(USArrests, center=T, scale=T),  size=0, loadings=FALSE, 
         loadings.colour='blue', loadings.label = T, loadings.label.size = 6, 
         loadings.label.colour='blue',loadings.label.repel=T, 
         frame=T, label=F, label.size=4) + 
  geom_jitter(size=0) + theme_bw()  + theme(legend.position="none") 

#plot vectors and observations
autoplot(prcomp(USArrests, center=T, scale=T),  size=0, loadings=FALSE, 
         loadings.colour='red', loadings.label = T, loadings.label.size = 6, 
         loadings.label.colour='red',loadings.label.repel=T, 
         frame=T, label=T, label.size=4) + 
  geom_jitter(size=0) + theme_bw()  + theme(legend.position="none") 

#can compute the variance associated with each principal component 
#from the standard deviation returned by prcomp().

pr.out$sdev

#get variance
pr.var = pr.out$sdev^2
pr.var

#we can compute the proportional variance explained (PVE)
pve = pr.var/sum(pr.var)
pve #this shows us how much variance explained by each component
#pc 1 explains 62% of variance, pc2 explains 25%

#we can plot PVE

plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained ", ylim = c(0, 1), type = "b")

#Can plot cumulative PVE
plot(cumsum(pve), xlab = "Principal Component ", ylab = " Cumulative Proportion of Variance Explained ", ylim = c(0, 1), type = "b")
#interp: tells us that four dimensions can explain lots of variation
#approx. 87% of all variation in the data can be collapsed to 2 dimensions
#implies lots of variables are explaining the same thing - can build good parsimonious model
########################
# K-MEANS CLUSTERING
#######################
set.seed(2)
x = matrix(rnorm(50 * 2), ncol = 2)
x[1:25, 1] = x[1:25, 1] + 3
x[1:25, 2] = x[1:25, 2] - 4
head(x) #2d data so can plot
plot(x)

?kmeans

#init. cluster data into 2
#control initial cluster assignments with nstart
km.out = kmeans(x, 2, nstart = 20) #generate 20 initial random centroids and choose the best one for the algorithm

#summary of clustering
names(km.out)
table(km.out$cluster)
#see 25 obs. assigned to each cluster

#plot the clusters
plot(x, col = (km.out$cluster + 1), main = "K-Means Clustering Results with K=2", xlab = "", ylab = "", pch = 20, cex = 2)

#can run with different clusters
set.seed(4)
km.out = kmeans(x, 4, nstart = 20)
km.out
table(km.out$cluster)

plot(x, col = (km.out$cluster + 1), main = "K-Means Clustering Results with K=4", xlab = "", ylab = "", pch = 20, cex = 2)


########################
# HIERARCHICAL CLUSTERING
#######################
#use hierarchical clustering on artificial dataset
?hclust
#d are the pairs of objects we want to compare
#method is how we want to build the dendogram and find clusters
hc.complete = hclust(dist(x), method = "complete")
hc.complete


#pick linkage function using hclust 
hc.average = hclust(dist(x), method = "average")
hc.single = hclust(dist(x), method = "single")

#compare different linkage functions by looking at dendogram
par(mfrow = c(1, 3))
plot(hc.complete, main = "Complete Linkage", xlab = "", sub = "", cex = 0.9)
plot(hc.average, main = "Average Linkage", xlab = "", sub = "", cex = 0.9)
plot(hc.single, main = "Single Linkage", xlab = "", sub = "", cex = 0.9)

#we specify clusters by where we want to cut the tree
#recall small heights lead to more clusters; larger heights lead to fewer clusters

#different linkages will result in different classifications across observations
#this tells us we want 2 clusters
#see different cuts across linkages
?cutree #specify number of clusters
cutree(hc.complete, 2)

cutree(hc.average, 2)

cutree(hc.single, 2)
#see single assigns fewer observations to cluster 2

par(mfrow = c(1, 3))
plot(hc.complete, main = "Complete Linkage", xlab = "", sub = "", cex = 0.9)
abline(h=5.0, col="red")
plot(hc.average, main = "Average Linkage", xlab = "", sub = "", cex = 0.9)
abline(h=3.2, col="red")
plot(hc.single, main = "Single Linkage", xlab = "", sub = "", cex = 0.9)
abline(h=1.39, col="red")

cutree(hc.single, 4)

par(mfrow = c(1, 2))
plot(hc.single, main = "Single Linkage with 2 clusters", xlab = "", sub = "", cex = 0.9)
abline(h=1.39, col="red")

plot(hc.single, main = "Single Linkage with 4 clusters", xlab = "", sub = "", cex = 0.9)
abline(h=1.18, col="red")

#see different heights result in different clusters

#scale the data and plot dendogram
xsc = scale(x)
plot(hclust(dist(xsc), method = "complete"), main = "Hierarchical Clustering with Scaled Features ")

x = matrix(rnorm(30 * 3), ncol = 3)
dd = as.dist(1 - cor(t(x)))
plot(hclust(dd, method = "complete"), main = "Complete Linkage with Correlation -Based Distance", xlab = "", sub = "")

################################
## REAL-WORLD EXAMPLE: REPRESSION
################################
#CIRI dataset tries to measure repression/human rights quality in a given state
#would take info about lots of different characteristics and then create composite score PHYSINT
#0=not free
#8 = total free

#RQ: What makes a regime 'repressive'?

#can apply PCA and clustering here to see (improve on?) their classification 
#see more here: https://en.wikipedia.org/wiki/CIRI_Human_Rights_Data_Project
#see coding variablees here: https://drive.google.com/file/d/0BxDpF6GQ-6fbY25CYVRIOTJ2MHM/edit
#what makes a state have better human rights protection?
dfciri = read.csv("cirirepression.csv")
head(dfciri)
dim(dfciri)
table(dfciri$year)
dfciri = subset(dfciri, dfciri$year==2011)
#append row names so we can see observation data later on
row.names(dfciri) = paste(dfciri$CTRY) 

require(dplyr)
pr = dfciri %>% 
  select(- c(CTRY, year, CIRI, ccode, POLITY, UNCTRY, UNREG, UNSUBREG,
             OLD_EMPINX, OLD_MOVE, OLD_RELFRE, WOSOC, PHYSINT))
#select variable based on governmen (polity), regime type,
             #freedom movement, freedom speech, physical integrity

head(pr)
dim(pr) #15 different factors

pr =na.omit(pr)
require(corrplot)
M = cor(pr, use="complete")
par(mfrow = c(1, 1))

corrplot(M)

corrplot(M, method=c("number"))

require(VIM)

pr.out = prcomp(pr, scale=T)
#plot vectors and observations
loadings = pr.out$rotation
loadings
#
#level of empowerment, freedom of assembly, freedom of movement negatively correlated with PC2
par(mfrow = c(1, 1))
autoplot(prcomp(pr, center=T, scale=T),  size=0, loadings=FALSE, 
         loadings.colour='red', loadings.label = T, loadings.label.size = 6, 
         loadings.label.colour='red',loadings.label.repel=T, 
         frame=T, label=F, label.size=4) + 
  geom_jitter(size=0) + theme_bw()  + theme(legend.position="none") 


autoplot(prcomp(pr, center=T, scale=T),  size=0, loadings=FALSE, 
         loadings.colour='red', loadings.label = T, loadings.label.size = 6, 
         loadings.label.colour='red',loadings.label.repel=T, 
         frame=T, label=T, label.size=4) + 
  geom_jitter(size=0) + theme_bw()  + theme(legend.position="none") 

#looks like most repressive countries are Saudi Arabia, Burma, Iran, China
#looks like least repressive states are European (e.g Luxembourg, San Marino)
#interpret PC1 - explains most variation between 'free' and 'not free'

#can compute the variance associated with each principal component 
#from the standard deviation returned by prcomp().
pr.out=prcomp(pr, scale=T, center=T)
pr.out$sdev

pr.var = pr.out$sdev^2
pr.var

#we can compute the proportional variance explained (PVE)
pve = pr.var/sum(pr.var)
round(pve,3) #this shows us how much variance explained by each component
#find pc1 explains 51% of variation and then steep drop-off
#interp: low levels freedom speech/movement tend to correlate with 
#relatively high levels of torture, killing, imprisonment

#we can plot PVE

plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained ", ylim = c(0, 1), type = "b")

#Can plot cumulative PVE
plot(cumsum(pve), xlab = "Principal Component ", ylab = " Cumulative Proportion of Variance Explained ", ylim = c(0, 1), type = "b")

#####
#K-Means Clustering
#####

#physint score takes sum of measures and classifies
km.out = kmeans(pr, 9, nstart=20)
#km.out
table(km.out$cluster)
table(dfciri$PHYSINT)

set.seed(1234)
#check - how well do these groupings make sense? do some seem more 'free' than others?
#if clusters seem off, need to go back and iterate to get `reasonable` groupings
sample(km.out$cluster[km.out$cluster==1], 5) 
sample(km.out$cluster[km.out$cluster==4], 5) 
sample(km.out$cluster[km.out$cluster==8], 5) 
sample(km.out$cluster[km.out$cluster==3], 5) 

#####
# Hierarchical Clustering
#####
hc.complete = hclust(dist(pr), method = "complete")
hc.complete


#pick linkage function using hclust 
hc.average = hclust(dist(pr), method = "average")
hc.single = hclust(dist(pr), method = "single")

#compare different linkage functions by looking at dendogram
par(mfrow = c(1, 1))
plot(hc.complete, main = "Complete Linkage for CIRI", xlab = "", sub = "", cex = 0.9)
plot(hc.average, main = "Average Linkage for CIRI", xlab = "", sub = "", cex = 0.9)
plot(hc.single, main = "Single Linkage for CIRI", xlab = "", sub = "", cex = 0.9)
cutree(hc.complete, 8)
table(cutree(hc.complete, 8))

cutree(hc.single, 8)
table(cutree(hc.single, 8))
