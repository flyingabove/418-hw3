####################
#HW 3 Christian Gao#
####################

### Libs ###
library(data.table)
library(qdapRegex)
library(coreNLP)
library(openNLP)
library(NLP)
library(magrittr)
options(java.parameters = "-Xmx8000m")

#Dataset Downloaded At: http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip
# install.packages(
#   "http://datacube.wu.ac.at/src/contrib/openNLPmodels.en_1.5-1.tar.gz",  
#   repos=NULL, 
#   type="source"
# )

#######################
#Training and Test Set#
#######################

sentiment_total<-fread("Sentiment Analysis Dataset.csv",sep=",")
training_index<-sample(1:1578627,1000000)
training_set<-sentiment_total[training_index,]
test_set<-sentiment_total[!training_index,]

sentiment_total$RawText<-sentiment_total$SentimentText

##########
#Cleaning#
##########

#sentiment_total_copy<-sentiment_total
#sentiment_total<-sentiment_total_copy

#Basic Cleaning

sentiment_total$SentimentText<-gsub("([[:alpha:]])\\1{2,}", "\\1\\1\\1", sentiment_total$SentimentText)
sentiment_total$SentimentText<-gsub("&lt;|&quot;|&amp;|&gt;","",sentiment_total$SentimentText)
sentiment_total$SentimentText<-gsub("^- ","",sentiment_total$SentimentText)
sentiment_total$SentimentText<-gsub("(?<=[\\s])\\s*|^\\s+|\\s+$","",perl=TRUE,sentiment_total$SentimentText)
sentiment_total$SentimentText<-gsub(" ?(f|ht)tp(s?)://(.*)[.][a-z]+", "", sentiment_total$SentimentText)

#Filter out Emoticons
sentiment_total$Emoticons<-sapply(ex_emoticon(sentiment_total$SentimentText),paste,collapse = " ")
sentiment_total$Emoticons[sentiment_total$Emoticons=="NA"] = ""
emo_1<-grepl("\\(\\:",sentiment_total$SentimentText)
sentiment_total$Emoticons[emo_1]<-paste(sentiment_total$Emoticons[emo_1],"(:",sep=" ")
emo_2<-grepl("\\)\\:",sentiment_total$SentimentText)
sentiment_total$Emoticons[emo_2]<-paste(sentiment_total$Emoticons[emo_2],"):",sep=" ")
emo_3<-grepl("\\:'-\\(",sentiment_total$SentimentText)
sentiment_total$Emoticons[emo_3]<-paste(sentiment_total$Emoticons[emo_3],"\\:'-\\(",sep=" ")

sentiment_total$SentimentText<-gsub("\\(\\:","",sentiment_total$SentimentText)
sentiment_total$SentimentText<-gsub("\\)\\:","",sentiment_total$SentimentText)
sentiment_total$SentimentText<-gsub("\\:'-\\(","",sentiment_total$SentimentText)
sentiment_total$SentimentText<-sapply(rm_emoticon(sentiment_total$SentimentText),paste,collapse = " ")
sentiment_total$Emoticons<-gsub("([^[:alpha:]])\\1{2,}", "\\1", sentiment_total$Emoticons)

#lower case
sentiment_total$SentimentText<-tolower(sentiment_total$SentimentText)

#Make Just Text
sentiment_total$JustText<-gsub("[^[:alpha:][:space:]]","",sentiment_total$SentimentText)
sentiment_total$JustText<-gsub("(?<=[\\s])\\s*|^\\s+|\\s+$","",perl=TRUE,sentiment_total$JustText)

#Make Just Sentence for POS
sentiment_total$JustSentence<-gsub('[^[:alpha:][:space:]\\.\\?\\!\\"\\:]',"",sentiment_total$SentimentText)
sentiment_total$JustSentence<-gsub("([^[:alpha:]])\\1{2,}", "\\1",sentiment_total$JustSentence)
sentiment_total$JustSentence<-gsub("^[^[:alpha:]]","",perl=TRUE,sentiment_total$JustSentence)
sentiment_total$JustSentence<-gsub("(?<=[\\s])\\s*|^\\s+|\\s+$","",perl=TRUE,sentiment_total$JustSentence)

#Remove NullText
sentiment_total<-sentiment_total[!sentiment_total$SentimentText=="",]

#Find Punctuation
sentiment_total$Punctuation = gsub("[[:alpha:][:space:]]|[0-9]","",sentiment_total$SentimentText)

####################
#Feature Generation#
####################

#VP1 VP2 NP1 NP2
#Verb, Number of Verbs
#Noun, Number of Nouns
#Emoticon
#Negative Words, number of neg, positive, number of positive
#Negation Words, number of neg
#Intesity Words, Number of Intensity words
#Number of Words, First Word Last Word, Number of Unique Words
#Punctuation, number of puct

###Install###
#downloadCoreNLP()
#downloadCoreNLP(type = c("english"))
initCoreNLP()
options(expressions = 5000)

get_verb_phrases<-function(nlp_tree){
  verb_list<-list()
  
  gen_verb_phrase<-function(nlp_tree,verb_list){
    children = nlp_tree$children
    
    for(child in children){
      #print(child)
      if(length(child)==1){
        return(list())
      }
      else if(child$value == "VP"){
        vp_string<-paste(capture.output(print(child)), collapse = " ")
        vp_string<-gsub("[^a-z[:space:]]","",vp_string,perl=TRUE)
        vp_string<-gsub("(?<=[\\s])\\s*|^\\s+|\\s+$","",vp_string,perl=TRUE)
        return(c(vp_string , verb_list))
      }
      else{
        verb_list = c(verb_list,gen_verb_phrase(child,verb_list))
      }
    }
    return(verb_list)
  }
  
  verb_list<-gen_verb_phrase(nlp_tree,verb_list)
  
  return(unlist(verb_list))
}

###Set Annotators###

s = as.String(sentiment_total$JustSentence[1:20])

pos_tag_annotator <- Maxent_POS_Tag_Annotator()
sent_token_annotator <- Maxent_Sent_Token_Annotator()
word_token_annotator <- Maxent_Word_Token_Annotator()
parse_annotator <- Parse_Annotator()

create_data<-function(s){
  s<-as.String(s)
  a2 <- annotate(s, list(sent_token_annotator, word_token_annotator))
  ## Variant with POS tag probabilities as (additional) features.
  p<-parse_annotator(s,a2)
  ptexts <- sapply(p$features, `[[`, "parse")
  nlp_tree=lapply(ptexts,Tree_parse)
  test=as.list(lapply(nlp_tree,get_verb_phrases))
  verb_phrases<-lapply(nlp_tree,get_verb_phrases)%>%unlist%>%unique
  verb_phrase[1:2]
}

# create_data2<-function(s){
#   s<-as.String(s)
#   a2 <- annotate(s, list(sent_token_annotator, word_token_annotator, pos_tag_annotator))
#   ## Variant with POS tag probabilities as (additional) features.
#   p<-parse_annotator(s,a2)
#   ptexts <- sapply(p$features, `[[`, "parse")
#   nlp_trees=lapply(ptexts,Tree_parse)
#   verb_phrases<-lapply(X=nlp_trees,FUN=get_verb_phrases)
#   verb_df<-do.call(rbind,verb_phrases)
#   verb_df
# }

system.time(verb_list<-lapply(sentiment_total$JustSentence[1:20],FUN=create_data))
verb_df<-do.call(rbind,verb_list)
### Negative and Positive Words ###

#Data From: https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/

negative_words<-readLines("negative-words.txt")
positive_words<-readLines("positive-words.txt")
adverbs_of_degree<-readLines("adverbs-of-degree.txt")

get_neg_pos_features<-function(tweet){
  tweet_array<-strsplit(tweet,split = " ")[[1]]
  
  neg_word_array<-tweet_array[tweet_array%in%negative_words]
  pos_word_array<-tweet_array[tweet_array%in%positive_words]
  degree_word_array<-tweet_array[tweet_array%in%adverbs_of_degree]
  
  neg_word_list = c(neg_word_array[1:2],length(neg_word_array),pos_word_array[1:2],length(pos_word_array),degree_word_array[1:2],length(degree_word_array))
  return(neg_word_list)
}

word_pos_neg_list<-lapply(sentiment_total$JustText,FUN=get_neg_pos_features)
word_pos_neg_df<-do.call(rbind,word_pos_neg_list)%>%as.data.frame
names(word_pos_neg_df)<-c("neg_word_1","neg_word_2","neg_word_count","pos_word_1","pos_word_2","pos_word_count","deg_word_1","deg_word_2","deg_word_count")
View(word_pos_neg_df)
sentiment_total<-cbind(sentiment_total,word_pos_neg_df)

### Emojis ###

get_emoticon_features<-function(emojis){
  if(emojis=="")
    return(c(NA,NA,NA))
  else{
    emoji_array<-strsplit(emojis,split = " ")[[1]]
    emoji_list<-c(emoji_array[1:2],length(emoji_array))
  }
  return(emoji_list)
}

emojis_list<-lapply(sentiment_total$Emoticons,FUN=get_emoticon_features)
emojis_df<-do.call(rbind,emojis_list)%>%as.data.frame
names(emojis_df)<-c("emoji_1","emoji_2","emoji_count")
View(emojis_df)
sentiment_total<-cbind(sentiment_total,emojis_df)

###Puctuation###

get_punctuation_features<-function(punct){
  if(punct=="")
    return(c(NA,NA,NA,NA))
  else{
    punct_array<-strsplit(punct,split = "")[[1]]
    punct_array<-unique(punct_array)
    punct_list<-c(paste(punct_array,collapse = ""),punct_array[1:2],length(punct_array))
  }
  return(punct_list)
}

punct_list<-lapply(sentiment_total$Punctuation,FUN=get_punctuation_features)
punct_df<-do.call(rbind,punct_list)%>%as.data.frame
names(punct_df)<-c("all_punct","punct1","punct2","punct_cout")
View(punct_df)
sentiment_total<-cbind(sentiment_total,punct_df)

###Word Count###

get_punctuation_features<-function(punct){
  if(punct=="")
    return(c(NA,NA,NA,NA))
  else{
    punct_array<-strsplit(punct,split = "")[[1]]
    punct_array<-unique(punct_array)
    punct_list<-c(paste(punct_array,collapse = ""),punct_array[1:2],length(punct_array))
  }
  return(punct_list)
}

punct_list<-lapply(sentiment_total$Punctuation,FUN=get_punctuation_features)
punct_df<-do.call(rbind,punct_list)%>%as.data.frame
names(punct_df)<-c("all_punct","punct1","punct2","punct_count")
View(punct_df)
sentiment_total<-cbind(sentiment_total,punct_df)

### Final Data cleaning ###

sentiment_df<-as.data.frame(sentiment_total)
sentiment_df<-sentiment_df[c("Sentiment","neg_word_1", "neg_word_2", 
                             "neg_word_count", "pos_word_1","pos_word_2", "pos_word_count", "deg_word_1", "deg_word_2",
                             "deg_word_count", "emoji_1", "emoji_2", "emoji_count", "all_punct", "punct1", "punct2","punct_count")]
sentiment_df$Sentiment[sentiment_df$Sentiment==1]="P"
sentiment_df$Sentiment[sentiment_df$Sentiment==0]="N"
write.csv(sentiment_df,"data/sentiment_df.csv",row.names = FALSE)
