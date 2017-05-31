##############################
#HW 3 Christian Gao- Modeling#
##############################
library(h2o)
library(gbm)
library(randomForest)
library(xgboost)

###############
##### GBM #####
###############

###GBM- H2o###
h2o.init(nthreads = -1)
df <- h2o.importFile("data/sentiment_df.csv")

splits <- h2o.splitFrame(
  df,           ##  splitting the H2O frame we read above
  ratios = c(.1,.05),   ##  create splits 
  seed=1234)    

train <- h2o.assign(splits[[1]], "train.hex")   
valid <- h2o.assign(splits[[2]], "valid.hex")

#GBM Base Model#

gbm_base<-h2o.gbm(y = "Sentiment", training_frame = train)
gbm_base
h2o.auc(h2o.performance(gbm_base, newdata = valid)) 
#

#Increase Tress
gbm_1 <- h2o.gbm(y = "Sentiment", training_frame = train, distribution = "bernoulli", 
               ntrees = 300, max_depth = 5, learn_rate = 0.1, 
               nbins = 20, seed = 123)
gbm_1
h2o.auc(h2o.performance(gbm_1, newdata = valid))
#0.7704041

#Increase Learning Rate
gbm_2 <- h2o.gbm(y = "Sentiment", training_frame = train, distribution = "bernoulli", 
                 ntrees = 50, max_depth = 5, learn_rate = 0.2, 
                 nbins = 20, seed = 123)
gbm_2
h2o.auc(h2o.performance(gbm_2, newdata = valid))
#0.7704041

#Increase Depth
gbm_3 <- h2o.gbm(y = "Sentiment", training_frame = train, distribution = "bernoulli", 
                 ntrees = 50, max_depth = 20, learn_rate = 0.1, 
                 nbins = 20, seed = 123)
gbm_3
h2o.auc(h2o.performance(gbm_3, newdata = valid))
#0.752772

#Increase Data with stopping
splits_2 <- h2o.splitFrame(
  df,           ##  splitting the H2O frame we read above
  ratios = .8,   ##  create splits 
  seed=1234)    

train_2 <- h2o.assign(splits_2[[1]], "train.hex")   
valid_2 <- h2o.assign(splits_2[[2]], "valid.hex")

system.time({gbm_4 <- h2o.gbm(y = "Sentiment", training_frame = train_2, distribution = "bernoulli", 
                 ntrees = 400, max_depth = 10, learn_rate = 0.02, 
                 nbins = 30, seed = 123,
                 stopping_rounds=5, stopping_tolerance=0.005,stopping_metric="AUC")})
gbm_4
h2o.auc(h2o.performance(gbm_4, newdata = valid_2))
plot(h2o.performance(gbm_4, newdata = valid_2),col = "blue",main = "True Positives vs False Positives GBM")
#0.7833309

###############
#Using XGboost#
###############

library(readr)
library(xgboost)
library(ROCR)

sentiment_df <- read_csv("data/sentiment_df.csv")

set.seed(123)
N <- nrow(sentiment_df)
idx <- sample(1:N, 0.6*N)
d_train <- sentiment_df[idx,]
d_test <- sentiment_df[-idx,]

### Base Case ###

X <- Matrix::sparse.model.matrix(dep_delayed_15min ~ . - 1, data = sentiment_df)
X_train <- X[idx,]
X_test <- X[-idx,]

dxgb_train <- xgb.DMatrix(data = X_train, label = ifelse(d_train$Sentiment=='Y',1,0))


system.time({
  n_proc <- parallel::detectCores()
  md <- xgb.train(data = dxgb_train, nthread = n_proc, objective = "binary:logistic", 
                  nround = 300, max_depth = 20, eta = 0.1)
})


predictions <- predict(md, newdata = X_test)
sentiment_pred <- prediction(predictions, d_test$dep_delayed_15min)
performance(sentiment_pred, "auc")@y.values[[1]]

