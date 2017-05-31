##############################
#HW 3 Christian Gao- Modeling#
##############################
library(h2o)
library(randomForest)
h2o.init(nthreads = -1)

df <- h2o.importFile("data/sentiment_df.csv")
splits <- h2o.splitFrame(df,ratios = c(.8,.1),seed=1234)
train <- h2o.assign(splits[[1]], "train.hex")   
valid <- h2o.assign(splits[[2]], "valid.hex")  
test <- h2o.assign(splits[[3]], "test.hex") 

################
# Random Forest#
################
system.time({
rf_base<-h2o.randomForest(training_frame = train, validation_frame = valid, y="Sentiment",seed=12345,ntrees=200)
})

h2o.auc(h2o.performance(rf_base, newdata = valid)) 
#

rf_pred_base<-h2o.predict(object = rf_base, newdata = test)
rf_pred_base

h2o.hit_ratio_table(rf_base,valid = T)[1,2]             ## validation set accuracy
mean(rf_pred_base$predict==test$Cover_Type)

###Increase Tree Depth###
system.time({
  rf3<-h2o.randomForest(training_frame = train, validation_frame = valid, y="Sentiment",seed=12345,ntrees=50,max_depth = 50
                        ,stopping_rounds = 10,stopping_metric="AUC",stopping_tolerance=.)
})
h2o.auc(h2o.performance(rf3, newdata = valid)) 
#

rf_pred_3<-h2o.predict(object = rf3, newdata = test)
rf_pred_3

h2o.hit_ratio_table(rf3,valid = T)[1,2]             ## validation set accuracy
mean(rf_pred_3$predict==test$Cover_Type)

###Increase Trees with stopping###
system.time({
rf3<-h2o.randomForest(training_frame = train, validation_frame = valid, y="Sentiment",seed=12345,ntrees=400
                      ,stopping_rounds = 10,stopping_metric="AUC",stopping_tolerance=.01)
})
h2o.auc(h2o.performance(rf3, newdata = valid)) 
#

rf_pred_3<-h2o.predict(object = rf3, newdata = test)
rf_pred_3

h2o.hit_ratio_table(rf3,valid = T)[1,2]             ## validation set accuracy
mean(rf_pred_3$predict==test$Cover_Type)

###Increase Tree Split Candidates###

system.time({
  rf4<-h2o.randomForest(training_frame = train, validation_frame = valid, y="Sentiment",seed=12345,ntrees=200,mtries=10)
})

h2o.auc(h2o.performance(rf4, newdata = test)) 
#0.7763487

rf_pred_4<-h2o.predict(object = rf4, newdata = test)
rf_pred_4

mean(rf_pred_1$predict==test$Sentiment)

# user  system elapsed 
#8.982   1.861 884.537 

###Decrease Tree Split Candidates###

system.time({
  rf5<-h2o.randomForest(training_frame = train, validation_frame = valid, y="≈",seed=12345,ntrees=200,mtries=4)
})

h2o.auc(h2o.performance(rf5, newdata = test)) 
plot(h2o.performance(rf5, newdata = test),col = "purple",main ="True Positive vs False Positive RF")
#0.7746153

#user  system elapsed 
#4.845   0.939 436.102 

rf_pred_5<-h2o.predict(object = rf5, newdata = test)
rf_pred_5

mean(rf_pred_5$predict==test$Sentiment)
#0.6791926

###Using Random Forest Package###
sentiment_df <- read.csv("data/sentiment_df.csv")
set.seed(123)
N <- nrow(sentiment_df)
idx <- sample(1:N, 0.8*N)
d_train <- sentiment_df[idx,]
d_test <- sentiment_df[-idx,]
model <- randomForest(Sentiment ~ neg_word_count + pos_word_count + deg_word_count +
                      emoji_count +punct_count, data = sentiment_df,
                      na.action =na.roughfix, ntree=1)

model


