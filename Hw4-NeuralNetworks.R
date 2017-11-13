library(keras)
library(magrittr)
library(data.table)
library(sfsmisc)

###Cleaning###
sentiment_df<-fread("data/sentiment_raw.csv")
sentiment_df<-sentiment_df[1:10000]
sentiment_df$ascii<-iconv(sentiment_df$JustText, "latin1", "ASCII", sub="")
sentiment_df$no_repeats<-gsub("([[:alpha:]])\\1{2,}", "\\1", sentiment_df$ascii)
sentiment_df$no_repeats<-gsub("(?<=[\\s])\\s*|^\\s+|\\s+$","",perl=TRUE,sentiment_df$no_repeats)
sentiment_df<-sentiment_df[sapply(gregexpr("\\W+", sentiment_df$no_repeats), length) >1,]

sentiment_df_small<-sentiment_df
writeLines(sentiment_df_small$no_repeats,"data/smalltest.txt")
system("grep -o -E '\\w+' smalltest.txt | sort -u -f > wordlist.txt")
factor_list<-factor(readLines("data/wordList.txt"))
text_raw_list<-strsplit(sentiment_df_small$no_repeats,split = " ")

get_training<-function(string_in,levels){
  y2<-factor(string_in,levels = levels); 
  result<-unclass(y2) %>% as.numeric 
  if(NA %in% result){
    print(string_in)
    print(result)
  }
  result
}

nn_predictors<-lapply(text_raw_list,get_training,levels = factor_list)

training_index<-sample(1:length(nn_predictors),length(nn_predictors)*.75)
x_train<- nn_predictors[training_index]
x_test <- nn_predictors[-training_index]

y_train<- sentiment_df_small$Sentiment[training_index]
y_test <- sentiment_df_small$Sentiment[-training_index]

### Test and Train ###

max_features <- length(factor_list)
maxlen <- 80  # cut texts after this number of words 
batch_size <- 128

print('Loading data...\n')
cat(length(x_train), 'train sequences\n')
cat(length(x_test), 'test sequences\n')

print('Pad sequences (samples x time)\n')
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)
cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')

print('Build model...\n')
model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>% 
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

#Model Settings
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

print('Train...\n')
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 1,
  validation_data = list(x_test, y_test)
)

save_model_hdf5(model, filepath = "data/nn_small_model", overwrite = TRUE,
                include_optimizer = TRUE)

model<-load_model_hdf5(filepath="data/nn_small_model")

scores <- model %>% evaluate(
  x_test, y_test,
  batch_size = batch_size
)

#Results

model
cat('Test score:', scores[[1]])
cat('Test accuracy', scores[[2]])

predictions <-predict_classes(model,x_test)
probs <-predict_proba(model,x_test)

#Confusion Matrix
table(y_test, predictions)

#AUC
generate_auc<-function(probs,predictionsm,dens){
  
  getTPR<-function(y_test,predictions){
    
    rates<-table(y_test, predictions)%>%as.data.frame
    TP<-(rates%>%subset(y_test == 0)%>%subset(predictions == 0))$Freq
    if(length(TP)==0)
      TP = 0
    TN<-(rates%>%subset(y_test == 0)%>%subset(predictions == 1))$Freq
    if(length(TN)==0)
      TN = 0
    return(TP/(TP+TN))
  }
  
  getFPR<-function(y_test, predictions){
    
    rates<-table(y_test, predictions)%>%as.data.frame
    FP<-(rates%>%subset(y_test == 1)%>%subset(predictions == 0))$Freq
    if(length(FP)==0)
      FP = 0
    FN<-(rates%>%subset(y_test == 1)%>%subset(predictions == 1))$Freq
    if(length(FN)==0)
      FN = 0
    return(FP/(FP+FN))
  }

  pred_list = lapply(seq(-.5,.5,dens),function(addit){round(probs+addit)})
                                                                  
  TPR_list = sapply(X = pred_list ,FUN=getTPR, y_test =y_test)
  FPR_list = sapply(X = pred_list ,FUN=getFPR, y_test =y_test)
  
  plot(FPR_list,TPR_list, main = "NN ROC curve" , xlab = "False Positive", ylab = "TRUE positive", col = "blue")
  
  auc = integrate.xy(FPR_list,TPR_list)
  
  return(auc)
}
#
auc<-generate_auc(probs,predictions,.001)

cat("FINAL AUC: " +auc)


