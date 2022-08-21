#################
## MISSING DATA
#################

require("mice")

# start with a complete data set
require(ISLR)
data(iris)
head(iris)
require(VIM)
pdf("base.pdf", 12, 8)
aggr(College, prop = T, numbers = T)
dev.off()

library(missMethods)
library(ggplot2)

set.seed(123)

make_simple_MDplot <- function(ds_comp, ds_mis) {
  ds_comp$missX <- is.na(ds_mis$X)
  ggplot(ds_comp, aes(x = X, y = Y, col = missX)) +
    geom_point()
}

# generate complete data frame
ds_comp = data.frame(X = rnorm(100), Y = rnorm(100))

ds_mcar = delete_MCAR(ds_comp, 0.3, "X")
pdf("mcar.pdf", 12, 8)
make_simple_MDplot(ds_comp, ds_mcar)
dev.off()
pdf("mcar2.pdf", 12, 8)
aggr(ds_mcar, prop = T, numbers = T)
dev.off()
ds_mar <- delete_MAR_censoring(ds_comp, 0.3, "X", cols_ctrl = "Y")
pdf("mar.pdf", 12, 8)

make_simple_MDplot(ds_comp, ds_mar)
dev.off()

pdf("mar2.pdf", 12, 8)
aggr(ds_mar, prop = T, numbers = T)
dev.off()

ds_mnar <- delete_MNAR_censoring(ds_comp, 0.3, "X")

pdf("mnar.pdf", 12, 8)
make_simple_MDplot(ds_comp, ds_mnar)
dev.off()

aggr(ds_mnar, prop = T, numbers = T)

set.seed(2016)
testdata <- as.data.frame(MASS::mvrnorm(n = 10000, 
                                        mu = c(10, 5, 0), 
                                        Sigma = matrix(data = c(1.0, 0.2, 0.2, 
                                                                0.2, 1.0, 0.2, 
                                                                0.2, 0.2, 1.0), 
                                                       nrow = 3, 
                                                       byrow = T)))
summary(testdata)