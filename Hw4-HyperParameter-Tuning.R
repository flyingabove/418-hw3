##############################
#HW 3 Christian Gao- Modeling#
##############################
library(h2o)

###############
##### GBM #####
###############

###GBM- H2o###

h2o.init(nthreads = -1,min_mem_size = "12g")
df <- h2o.importFile(path="/home/christian/Documents/christian-418-hw4/data/sentiment_df-gbm.csv")

splits <- h2o.splitFrame(
  df,           ##  splitting the H2O frame we read above
  ratios = c(.1,.05),   ##  create splits 
  seed=1234)    

train <- h2o.assign(splits[[1]], "train.hex")   
valid <- h2o.assign(splits[[2]], "valid.hex")

###GBM Base Model###

gbm_base<-h2o.gbm(y = "Sentiment", training_frame = train)
gbm_base
h2o.auc(h2o.performance(gbm_base, newdata = valid))
plot(h2o.performance(gbm_base, newdata = valid),col = "blue",main = "True Positives vs False Positives GBM")

###GBM with hyper parameter tuning###

ntrees_opts = c(1000,1500,3000)       # early stopping will stop earlier
max_depth_opts = seq(1,20)
#min_rows_opts = c(1,5,10,20,50,100)
learn_rate_opts = seq(0.001,0.02,0.001)
sample_rate_opts = seq(0.3,1,0.05)
col_sample_rate_opts = seq(0.3,1,0.05)
col_sample_rate_per_tree_opts = seq(0.3,1,0.05)

hyper_params = list( ntrees = ntrees_opts, 
                     max_depth = max_depth_opts, 
                     #min_rows = min_rows_opts, 
                     learn_rate = learn_rate_opts,
                     sample_rate = sample_rate_opts,
                     col_sample_rate = col_sample_rate_opts,
                     col_sample_rate_per_tree = col_sample_rate_per_tree_opts
)


# Search a random subset of these hyper-parmameters. Max runtime 
# and max models are enforced, and the search will stop after we 
# don't improve much over the best 5 random models.
search_criteria = list(strategy = "RandomDiscrete", 
                       max_runtime_secs = 600, 
                       max_models = 100, 
                       stopping_metric = "AUTO", 
                       stopping_tolerance = 0.00001, 
                       stopping_rounds = 5, 
                       seed = 123456)

system.time(
gbm_grid <- h2o.grid(algorithm = "gbm", 
                     grid_id = "mygrid",
                     x = c(2:17),
                     y = 1, 
                  
                     training_frame = train,
                     validation_frame = valid,
                     nfolds = 0,
                     
                     distribution="bernoulli",
                     
                     stopping_rounds = 2,
                     stopping_tolerance=0.005,
                     stopping_metric = "AUC",
                     nbins = 30,
                     
                     # how often to score (affects early stopping):
                     score_tree_interval = 100, 
                     
                     ## seed to control the sampling of the 
                     ## Cartesian hyper-parameter space:
                     seed = 123456,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)
)

gbm_sorted_grid <- h2o.getGrid(grid_id = "mygrid", sort_by = "auc",decreasing = TRUE)
print(gbm_sorted_grid)

best_model <- h2o.getModel(gbm_sorted_grid@model_ids[[1]])
summary(best_model)

h2o.auc(h2o.performance(best_model, newdata = valid))
plot(h2o.performance(best_model, newdata = valid),col = "blue",main = "True Positives vs False Positives GBM")

h2o.shutdown()
