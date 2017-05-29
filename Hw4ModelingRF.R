##############################
#HW 3 Christian Gao- Modeling#
##############################
library(h2o)
h2o.init(nthreads = -1)

df <- h2o.importFile("data/sentiment_df.csv")
splits <- h2o.splitFrame(df,ratios = c(.8,.1),seed=1234)
train <- h2o.assign(splits[[1]], "train.hex")   
valid <- h2o.assign(splits[[2]], "valid.hex")  
test <- h2o.assign(splits[[3]], "test.hex") 

################
# Random Forest#
################

rf_base<-h2o.randomForest(training_frame = train, validation_frame = valid, y="Sentiment",seed=12345,ntrees=200)

h2o.auc(h2o.performance(rf_base, newdata = valid)) 
#

rf_pred_base<-h2o.predict(object = rf_base, newdata = test)
rf_pred_base

h2o.hit_ratio_table(rf_base,valid = T)[1,2]             ## validation set accuracy
mean(rf_pred_base$predict==test$Cover_Type)

###Increase Tree Depth###

###Increase Trees###
rf1<-h2o.randomForest(training_frame = train, validation_frame = valid, y="Sentiment",seed=12345,ntrees=1000)

h2o.auc(h2o.performance(rf1, newdata = valid)) 
#

rf_pred_1<-h2o.predict(object = rf1, newdata = test)
rf_pred_1

h2o.hit_ratio_table(rf1,valid = T)[1,2]             ## validation set accuracy
mean(rf_pred_1$predict==test$Cover_Type)

###Increase Tree Split Candidates###


