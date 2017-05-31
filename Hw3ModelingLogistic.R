library(h2o)
library(pROC)

h2o.init(nthreads = -1)

#####################
#Logistic Regression#
#####################

df <- h2o.importFile("data/sentiment_df.csv")
splits <- h2o.splitFrame(df,ratios = c(.8,.1),seed=1234)
train <- h2o.assign(splits[[1]], "train.hex")   
valid <- h2o.assign(splits[[2]], "valid.hex")  
test <- h2o.assign(splits[[3]], "test.hex") 

### Base Regression ###
system.time({
  glm_base <- h2o.glm(y = "Sentiment", training_frame = train, validation_frame = valid,
                family = "binomial", alpha = 1, lambda = 0)
})

h2o.auc(h2o.performance(glm_base, test))
plot(h2o.performance(glm_base, test),col = "red",main = "True Positive vs False Positive Logistic Reg")
?h2o.performance

#0.7775226

### Small Lambda ###
system.time({
  glm_1 <- h2o.glm(y = "Sentiment", training_frame = train, validation_frame = valid,
                      family = "binomial", alpha = 1, lambda = 0.001)
})

h2o.auc(h2o.performance(glm_1, test))
#0.745197

### High Lambda ###
system.time({
  glm_2 <- h2o.glm(y = "Sentiment", training_frame = train, validation_frame = valid,
                      family = "binomial", alpha = 1, lambda = .4)
})

h2o.auc(h2o.performance(glm_2, test))
#0.5

### Alternate GLM Package ###
sentiment_df <- read.csv("data/sentiment_df.csv")
set.seed(123)
N <- nrow(sentiment_df)
idx <- sample(1:N, 0.8*N)
d_train <- sentiment_df[idx,]
d_test <- sentiment_df[-idx,]

glm_r <- glm(Sentiment ~ ., data = d_train, family = "binomial")
summary(glm_r)
prob=predict(glm_r,type="response")
d_train$prob = prob
roccurve <- roc(Sentiment ~ prob, data = d_test)
auc(roccurve)

