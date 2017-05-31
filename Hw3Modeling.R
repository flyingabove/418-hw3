##############################
#HW 3 Christian Gao- Modeling#
##############################
library(h2o)
library(gbm)
library(randomForest)
library(xgboost)


training_index<-sample(1:1578627,1000000)
training_set<-sentiment_total[training_index,]
test_set<-sentiment_total[!training_index,]

sentiment_df<-as.data.frame(sentiment_total)
sentiment_df<-sentiment_df[c("Sentiment","neg_word_1", "neg_word_2", 
                           "neg_word_count", "pos_word_1","pos_word_2", "pos_word_count", "deg_word_1", "deg_word_2",
                           "deg_word_count", "emoji_1", "emoji_2", "emoji_count", "all_punct", "punct1", "punct2","punct_count")]
sentiment_df$Sentiment[sentiment_df$Sentiment==1]="P"
sentiment_df$Sentiment[sentiment_df$Sentiment==0]="N"


###############
##### GBM #####
###############

###GBM- H2o###
h2o.init(nthreads = -1)

write.csv(sentiment_df,"data/sentiment_df.csv",row.names = FALSE)
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

#Increase Tress
gbm_1 <- h2o.gbm(y = "Sentiment", training_frame = train, distribution = "bernoulli", 
               ntrees = 300, max_depth = 5, learn_rate = 0.1, 
               nbins = 20, seed = 123)
gbm_1
h2o.auc(h2o.performance(gbm_1, newdata = valid)) 

#Increase Learning Rate
gbm_2 <- h2o.gbm(y = "Sentiment", training_frame = train, distribution = "bernoulli", 
                 ntrees = 50, max_depth = 5, learn_rate = 0.2, 
                 nbins = 20, seed = 123)
gbm_2
h2o.auc(h2o.performance(gbm_2, newdata = valid)) 

#Increase Depth
gbm_3 <- h2o.gbm(y = "Sentiment", training_frame = train, distribution = "bernoulli", 
                 ntrees = 50, max_depth = 20, learn_rate = 0.1, 
                 nbins = 20, seed = 123)
gbm_3
h2o.auc(h2o.performance(gbm_3, newdata = valid)) 

#Increase Data with stopping
splits_2 <- h2o.splitFrame(
  df,           ##  splitting the H2O frame we read above
  ratios = .8,   ##  create splits 
  seed=1234)    

train_2 <- h2o.assign(splits_2[[1]], "train.hex")   
valid_2 <- h2o.assign(splits_2[[2]], "valid.hex")

gbm_4 <- h2o.gbm(y = "Sentiment", training_frame = train_2, distribution = "bernoulli", 
                 ntrees = 1000, max_depth = 15, learn_rate = 0.05, 
                 nbins = 30, seed = 123,
                 stopping_rounds=5, stopping_tolerance=0.005,stopping_metric="AUC")
gbm_4
h2o.auc(h2o.performance(gbm_4, newdata = valid_2))
#0.7790308

rf1 <- h2o.randomForest(        
  training_frame = train,        
  validation_frame = valid,      
  x=1:12,                        
  y=13,                          
  model_id = "rf_covType_v1",    
  ntrees = 200,                  
  stopping_rounds = 2,           
  score_each_iteration = T,      
  seed = 1000000)




