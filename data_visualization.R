library(gdata)
library(dplyr)
library(qdap)
require(stringi)

count_word <- function(search_term){
 count = stri_count(search_term,regex="\\S+")
 return (count)
}

train_data = read.csv('train.csv')
train = train_data
train$search_term = sapply(train$search_term,count_word)
plot_data = data.frame(aggregate(relevance ~ search_term, train, FUN=mean))

plot(plot_data,type='h',main="No.of words in search term vs the avg relevance",xlab='words in search term',ylab='avg relevance',col='dark red')