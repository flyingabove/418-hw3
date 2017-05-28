##############################
#HW 3 Christian Gao- Modeling#
##############################
library(h2o)
library(gbm)
h2o.init()

iris

### H2o GBM ###

df = as.h2o(iris)

splits <- h2o.splitFrame(
  df,           ##  splitting the H2O frame we read above
  c(0.6,0.2),   ##  create splits of 60% and 20%; 
  seed=1234)    ##  setting a seed will ensure reproducible results (not R's seed)

train <- h2o.assign(splits[[1]], "train.hex")   

valid <- h2o.assign(splits[[2]], "valid.hex")

gbm2 <- h2o.gbm(
  training_frame = train,     ##
  validation_frame = valid,   ##
  y="Sepal.Length",                       ## 
  ntrees = 20,                ## decrease the trees, mostly to allow for run time
  learn_rate = 0.2,           ## increase the learning rate (from 0.1)
  max_depth = 10,             ## increase the depth (from 5)
  stopping_rounds = 2,        ## 
  stopping_tolerance = 0.01,  ##
  score_each_iteration = T,   ##
  model_id = "gbm_covType2",  ##
  seed = 2000000)             ##

summary(gbm2)
h2o.accuracy(gbm2)
h2o.metric(gbm2)

   ## review the new model's accuracy
###############################################################################

## This has moved us in the right direction, but still lower accuracy 
##  than the random forest.
## And it still has not converged, so we can make it more aggressive.
## We can now add the stochastic nature of random forest into the GBM
##  using some of the new H2O settings. This will help generalize 
##  and also provide a quicker runtime, so we can add a few more trees.

gbm3 <- h2o.gbm(
  training_frame = train,     ##
  validation_frame = valid,   ##
  x=1:12,                     ##
  y=13,                       ## 
  ntrees = 30,                ## add a few trees (from 20, though default is 50)
  learn_rate = 0.3,           ## increase the learning rate even further
  max_depth = 10,             ## 
  sample_rate = 0.7,          ## use a random 70% of the rows to fit each tree
  col_sample_rate = 0.7,       ## use 70% of the columns to fit each tree
  stopping_rounds = 2,        ## 
  stopping_tolerance = 0.01,  ##
  score_each_iteration = T,   ##
  model_id = "gbm_covType3",  ##
  seed = 2000000)             ##
###############################################################################

summary(gbm3)
h2o.hit_ratio_table(rf1,valid = T)[1,2]     ## review the random forest accuracy
h2o.hit_ratio_table(gbm1,valid = T)[1,2]    ## review the first model's accuracy
h2o.hit_ratio_table(gbm2,valid = T)[1,2]    ## review the second model's accuracy
h2o.hit_ratio_table(gbm3,valid = T)[1,2]    ## review the newest model's accuracy
###############################################################################

## Now the GBM is close to the initial random forest.
## However, we used a default random forest. 
## Random forest's primary strength is how well it runs with standard
##  parameters. And while there are only a few parameters to tune, we can 
##  experiment with those to see if it will make a difference.
## The main parameters to tune are the tree depth and the mtries, which
##  is the number of predictors to use.
## The default depth of trees is 20. It is common to increase this number,
##  to the point that in some implementations, the depth is unlimited.
##  We will increase ours from 20 to 30.
## Note that the default mtries depends on whether classification or regression
##  is being run. The default for classification is one-third of the columns.
##  The default for regression is the square root of the number of columns.

rf2 <- h2o.randomForest(        ##
  training_frame = train,       ##
  validation_frame = valid,     ##
  x=1:12,                       ##
  y=13,                         ##
  model_id = "rf_covType2",     ## 
  ntrees = 200,                 ##
  max_depth = 30,               ## Increase depth, from 20
  stopping_rounds = 2,          ##
  stopping_tolerance = 1e-2,    ##
  score_each_iteration = T,     ##
  seed=3000000)                 ##
###############################################################################
summary(rf2)
h2o.hit_ratio_table(gbm3,valid = T)[1,2]    ## review the newest GBM accuracy
h2o.hit_ratio_table(rf1,valid = T)[1,2]     ## original random forest accuracy
h2o.hit_ratio_table(rf2,valid = T)[1,2]     ## newest random forest accuracy
###############################################################################

## So we now have our accuracy up beyond 95%. 
## We have witheld an extra test set to ensure that after all the parameter
##  tuning we have done, repeatedly applied to the validation data, that our
##  model produces similar results against the third data set. 

## Create predictions using our latest RF model against the test set.
finalRf_predictions<-h2o.predict(
  object = rf2
  ,newdata = test)

## Glance at what that prediction set looks like
## We see a final prediction in the "predict" column,
##  and then the predicted probabilities per class.
finalRf_predictions

## Compare these predictions to the accuracy we got from our experimentation
h2o.hit_ratio_table(rf2,valid = T)[1,2]             ## validation set accuracy
mean(finalRf_predictions$predict==test$Cover_Type)  ## test set accuracy

## We have very similar error rates on both sets, so it would not seem
##  that we have overfit the validation set through our experimentation.
##
## This concludes the demo, but what might we try next, if we were to continue?
##
## We could further experiment with deeper trees or a higher percentage of
##  columns used (mtries).
## Also we could experiment with the nbins and nbins_cats settings to control
##  the H2O splitting.
## The general guidance is to lower the number to increase generalization
##  (avoid overfitting), increase to better fit the distribution.
## A good example of adjusting this value is for nbins_cats to be increased
##  to match the number of values in a category. Though usually unnecessary,
##  if a problem has a very important categorical predictor, this can 
##  improve performance.
##
## Also, we can tune our GBM more and surely get better performance.
## The GBM will converge a little slower for optimal accuracy, so if we 
##  were to relax our runtime requirements a little bit, we could balance
##  the learn rate and number of trees used.
## In a production setting where fine-grain accuracy is beneficial, it is 
##  common to set the learn rate to a very small number, such as 0.01 or less,
##  and add trees to match. Use of early stopping is very powerful to allow 
##  the setting of a low learning rate and then building as many trees as 
##  needed until the desired convergence is met.
## As with random forest, we can also adjust nbins and nbins_cats.


### All done, shutdown H2O    
h2o.shutdown(prompt=FALSE)