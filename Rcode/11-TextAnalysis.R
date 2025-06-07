####################
# PSC 8185: ML4SS
# Session 11: Pre-Processing and Dictionary Methods
# April 18, 2022
# Author: Rochelle Terman
####################
rm(list=ls(all=TRUE))

#################
## PRE-PROCESSING
#################

## 1. Preprocessing a Corpus

library(tm) # Framework for text mining
library(dplyr) # Data preparation and pipes $>$
library(tidytext)
#For this unit, we will be using a section of Machiavelli's Prince as our corpus. 
#Since The Prince is a monograph, we have already "chunked" the text, 
#so that each short paragraph or "chunk" is considered a "document."

#read in CSV file
princecorpus =read.csv("/users/irismalone/Dropbox/Courses/Machine Learning/R Code/11-TextAnalysis/mach.csv") 

#use tm package to turn into corpus
princecorpus$text = princecorpus[,2]

#get unique identifier for each row
princecorpus$chapter = rownames(princecorpus)


### 1.2 Preprocessing functions 

#Many text analysis applications follow a similar 'recipe' for preprecessing, 
#involving (the order of these steps might differ as per application):

##1. Tokenizing the text to unigrams (or bigrams, or trigrams)
#2. Converting all characters to lowercase
#3. Removing punctuation
#4. Removing numbers
#5. Removing Stop Words, including custom stop words
#6. "Stemming" words, or lemmatization. There are several stemming alogrithms. Porter is the most popular.
#7. Creating a Document-Term Matrix


###############
## OPTION 1: MANUALLY CLEAN
##############

require(dplyr)
require(tidytext) #loads many tidy packages for text analysis
?tidytext


text_cleaning_tokens = princecorpus %>% 
  tidytext::unnest_tokens(word, text) #apply bag of words assumption to text column to extract words and remove capitalization

head(text_cleaning_tokens)

#number of words in the Prince
dim(text_cleaning_tokens)

text_cleaning_tokens$word = gsub('[[:punct:]]+', '', text_cleaning_tokens$word) #remove punctuation
text_cleaning_tokens$word = gsub('[[:digit:]]+', '', text_cleaning_tokens$word) #remove numbers
head(text_cleaning_tokens)

dim(text_cleaning_tokens)

#view stop words
require(stopwords) #this is already loaded in the tidytext
?stopwords
head(stop_words)
tail(stop_words)

#remove stop words
text_cleaning_tokens = text_cleaning_tokens %>% 
  anti_join(stop_words) #anti_join will remove all stop words from text

text_cleaning_tokens = text_cleaning_tokens %>% 
   filter(!(word=="")) #also remove blanks

head(text_cleaning_tokens)

#apply Snowball stemmer
library(SnowballC)
?SnowballC

#example of how stemming
vector = c("run", "ran", "running")
wordStem(vector)

#get tokens
tokens = text_cleaning_tokens %>%
  mutate(stem = wordStem(word)) #wordStem is in Snowball package

head(tokens)
table(length(unique(tokens$stem))) #2197 unique tokens

#alternatively: can lemmatize

library(textstem)
#example of how lemma works
vector = c("run", "ran", "running")
lemmatize_words(vector)

tokens_lemma = text_cleaning_tokens %>%
  mutate(lemma = lemmatize_words(word))
head(tokens_lemma)
table(length(unique(tokens_lemma$lemma))) #2333 unique tokens

##############
## OPTION 2: BUILD FUNCTION
##############
# apply textcleaner function. note: we only clean the text without convert it to dtm
# build textcleaner function

#we can automate this into tidy function 
require(tidyr)
require(stringr)
require(textclean)
?textclean

textcleaner_2 = function(x){
  x = as.character(x)
  
  x = x %>%
    str_to_lower() %>%  # convert all the string to low alphabet
    replace_contraction() %>% # replace contraction to their multi-word forms
    replace_internet_slang() %>% # replace internet slang to normal words
    #replace_emoji() %>% # replace emoji to words
    #replace_emoticon() %>% # replace emoticon to words
    replace_hash(replacement = "") %>% # remove hashtag
    replace_word_elongation() %>% # replace informal writing with known semantic replacements
    replace_number(remove = T) %>% # remove number
    replace_date(replacement = "") %>% # remove date
    replace_time(replacement = "") %>% # remove time
    str_remove_all(pattern = "[[:punct:]]") %>% # remove punctuation
    str_remove_all(pattern = "[^\\s]*[0-9][^\\s]*") %>% # remove mixed string n number
    str_squish() %>% # reduces repeated whitespace inside a string.
    str_trim() # removes whitespace from start and end of string
  return(as.data.frame(x))
  
}

clean_prince = textcleaner_2(princecorpus$text)
head(clean_prince) #this applies to all text in chapter

clean_prince = clean_prince %>% mutate(chapter = rownames(clean_prince))
head(clean_prince)


require(SnowballC)
# create dtm
require(tidytext)
require(dplyr)
require(tm)
require(textmineR)

#add custom stop words
customstopwords = c('niccolo', 'macchiavelli')
#remove meta data
?CreateDtm
dtm_prince = CreateDtm(doc_vec = clean_prince$x, #names(clean_prince) where text is stored
                        doc_names = clean_prince$chapter, #unique identifier
                        ngram_window = c(1,2), #ngram - specify unigram c(1), bigrams (2), or both c(1, 2)
                        stopword_vec = c(stopwords("en"), customstopwords), #remove stopwords
                        verbose = F,
                        remove_numbers = TRUE,
                        remove_punctuation = TRUE,
                        lower=TRUE,
                        stem_lemma_function = function(x) SnowballC::wordStem(x, "porter")
)

head(dtm_prince)
dtm_prince = dtm_prince[,colSums(dtm_prince)>2]

head(dtm_prince)


## 2. Exploring the DTM

### 2.1 Dimensions

#look at the structure of our DTM. 
   
# how many documents? how many terms?
dim(dtm_prince) #188 docs, 1317 distinct tokens
  

### 2.2 Frequencies

#We can obtain the term frequencies as a vector by converting the document term matrix into a matrix 
#and using `colSums` to sum the column counts:
  
   
# how many terms?
freq = colSums(as.matrix(dtm_prince))
freq[1:5]
length(freq)
  

#By ordering the frequencies we can list the most frequent terms and the least frequent terms.

 
# order
sorted = sort(freq, decreasing = T)
# most frequent terms
head(sorted)
# least frequent
tail(sorted)
  

### 2.3 Plotting frequencies

#Let's make a plot that shows the frequency of frequencies for the terms. 
#(For example, how many words are used only once? 5 times? 10 times?)

 
# frequency of frenquencies
head(table(freq),15)
tail(table(freq),15)
# plot
plot(table(freq))
  

#What does this tell us about the nature of language?
#it follow' Zipf's law
#word frequency is inversely proportional to its rank 

### 2.4 Exploring common words


###look at most common words


#####
#look at book

freq_words = tokens %>%
  count(stem, sort = TRUE) %>% filter(n > 5) 

#shop top 20 words
freq_words %>% 
  mutate(stem = reorder(stem, n)) %>%
  group_by(stem) %>%
  top_n(20)

#plot most frequent words
require(ggplot2)
require(dplyr)
freq_words %>% 
  mutate(stem = reorder(stem, n)) %>%
  group_by(stem) %>%
  top_n(20) %>% 
  arrange(desc(n)) %>%
  top_n(20)  %>% 
  ungroup() %>% 
  mutate(stem = factor(paste(stem, sep = "__"), 
                       levels = rev(paste(stem,  sep = "__"))))  %>%
  
  top_n(20) %>%
  
  ggplot(aes(stem, n)) +
  geom_col(alpha=0.8, show.legend = FALSE)+
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) +
  coord_flip()  +
  theme_bw() +
  theme(panel.grid.major.y = element_blank()) + 
  theme(legend.position = "bottom",
        axis.ticks.y = element_blank())  + xlab("Word Stem") + ylab("Frequency")



## 3. Exporting the DTM

### 3.1
#We can convert a DTM to a matrix or data.frame in order to write to a csv, add meta data, etc.

#First create an object that converts the `dtm` to a dataframe (we first have to convert to matrix, and then to dataframe)

 
# coerce into dataframe
dtm = as.data.frame(as.matrix(dtm_prince))
names(dtm)  # names of documents
  

### 3.2
#Now suppose we want to add a column called `doc_section` to differentiate different parts
#For the first 100 rows, the value of this column should be "Section 1". 
#For documents 101-188, the section should be "Section 2".

 
# add fake column for section
dtm$doc_section = "NA"
dtm$doc_section[1:100] = "Section 1"
dtm$doc_section[101:188] = "Section 2"
dtm$doc_section = as.factor(dtm$doc_section)
# check to see if they're the same number of documents per author
summary(dtm$doc_section)
  
dim(dtm)
names(dtm)
dtm
table(dtm$ruler) #dtm across segments

### 3.3

#Export the dataframe as a csv.

 
# don't run this unless you want a very big file! 
#write.csv(dtm, "dtm.csv")
  

##########################
## DICTIONARY-BASED METHODS
##########################


## 1. Comparing Songs from Taylor Swift's catalogue

#Motivation: code Taylor Swift songs 

#Load the code below to get started.
library(tidytext)
ts = read.csv("taylor_swift.csv")
  
#use tm package to turn into corpus
ts$text = ts$lyrics

#get unique identifier for each row
ts$id = rownames(ts)

#preprocess the corpus
  
# preprocess and create DTM
require(tidyr)
require(stringr)
require(textclean)
?textclean

textcleaner_2 = function(x){
  x = as.character(x)
  
  x = x %>%
    str_to_lower() %>%  # convert all the string to low alphabet
    replace_contraction() %>% # replace contraction to their multi-word forms
    replace_internet_slang() %>% # replace internet slang to normal words
    #replace_emoji() %>% # replace emoji to words
    #replace_emoticon() %>% # replace emoticon to words
    replace_hash(replacement = "") %>% # remove hashtag
    replace_word_elongation() %>% # replace informal writing with known semantic replacements
    replace_number(remove = T) %>% # remove number
    replace_date(replacement = "") %>% # remove date
    replace_time(replacement = "") %>% # remove time
    str_remove_all(pattern = "[[:punct:]]") %>% # remove punctuation
    str_remove_all(pattern = "[^\\s]*[0-9][^\\s]*") %>% # remove mixed string n number
    str_squish() %>% # reduces repeated whitespace inside a string.
    str_trim() # removes whitespace from start and end of string
  return(as.data.frame(x))
  
}

clean_ts = textcleaner_2(ts$text)
head(clean_ts) #this applies to all text in chapter

clean_ts = clean_ts %>% 
  mutate(song = rownames(clean_ts))


require(SnowballC)
# create dtm
require(tidytext)
require(dplyr)
require(tm)
require(textmineR)

#add custom stop words

dtm_ts = CreateDtm(doc_vec = clean_ts$x, 
                       doc_names = clean_ts$song, #unique identifier
                       #ngram_window = c(1), #if just interested in unigram can x out
                       stopword_vec = c(stopwords("en")), #remove stopwords
                       verbose = F,
                       remove_numbers = TRUE,
                       remove_punctuation = TRUE,
                       lower=TRUE,
                       stem_lemma_function = function(x) SnowballC::wordStem(x, "porter")
)


## 2. Setting up the sentiment dictionary

## 2.1

#Use sentiment dictionaries from the `tidytext` package. Using the `get_sentiments` function, load the "bing" dictionary and store it in an object called `sent`. 

?get_sentiments #look at different options
sent = get_sentiments("bing")
head(sent)
  

## 2.2

#Add a column to `sent` called `score`. 
#This column should hold a "1" for positive words and "-1" for negative words.

  
sent$score = ifelse(sent$sentiment=="positive", 1, -1)
  

## 3. Scoring the songs

## 3.1 

#Score each song. 

#First, create a dataframe that holds all the words in our dtm along with their sentiment score.

# get all the words in our dtm and put it in a dataframe
words = data.frame(word = colnames(dtm_ts))
head(words)
# get their sentiment scores
words = merge(words, sent, all.x = T)
head(words)
# replace NAs with 0s
words$score[is.na(words$score)] = 0
head(words)
  

## 3.2

#We can now use matrix algebra (!!) to multiply our dtm by the scoring vector. 
#This will return to us a score for each document (i.e., song).

  
# calculate documents scores with matrix algebra! 
scores = as.matrix(dtm_ts) %*% words$score
# put it in the original documents data frame
ts$sentiment = scores
head(ts)  

#Which song is happiest? Go listen to the song and see if you agree.

#alternate way:
ts2= with(ts, data.frame(artist, year, album, track_title, id, text))
head(ts2)
text_cleaning_tokens = ts2 %>% 
  tidytext::unnest_tokens(word, text) #apply bag of words assumption to text column to extract words and remove capitalization


#remove stop words
text_cleaning_tokens = text_cleaning_tokens %>% 
  anti_join(stop_words)

text_cleaning_tokens = text_cleaning_tokens %>% filter(!(word==""))

head(text_cleaning_tokens)

#apply Snowball stemmer
library(SnowballC)
?SnowballC

#get tokens
tokens = text_cleaning_tokens %>%
  mutate(stem = wordStem(word)) 

library(tidyr)
library(textdata)

ts_sentiment = tokens %>%
  inner_join(get_sentiments("bing")) %>% #apply sentiment diary
  count(album, index = track_title, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>% 
  mutate(sentiment = positive - negative) #get net sentiment

head(ts_sentiment) #sentiment score (pos-neg) per song

library(ggplot2)

ggplot(ts_sentiment, aes(index, sentiment, fill = album)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~album, ncol = 3, scales = "free_x") + xlab("Song") + ylab("Sentiment Score") + theme_bw() +
  theme(axis.text.x = element_text(angle = 90))
  
#find out what the most and least positive Taylor Swift album is.
tapply(ts$sentiment, ts_sentiment$album, summary) 

tapply(ts$sentiment, ts_sentiment$album, mean) 

