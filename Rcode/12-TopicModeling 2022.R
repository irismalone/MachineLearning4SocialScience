####################
# PSC 8185: ML4SS
# Session 12: Document Clustering and Topic Modeling
# April 25, 2022
# Author: Iris Malone
####################
rm(list=ls(all=TRUE))

############################
## CREATING CUSTOM DICTIONARIES
############################

#example: compare Biden and Trump inaugurals
library(quanteda)
# some operations on the inaugural corpus
summary(data_corpus_inaugural)

trump = data_corpus_inaugural[58]
head(trump)

biden = data_corpus_inaugural[59]
head(biden)


#we can automate this into tidy function 
require(tidyr)
require(stringr)
require(textclean)
require(tidytext)
require(dplyr)
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

biden$chapter = rownames(biden)

clean_biden = textcleaner_2(biden)

clean_biden = clean_biden %>% mutate(chapter = rownames(clean_biden))

text_cleaning_tokens = clean_biden %>% 
  tidytext::unnest_tokens(word, x)
text_cleaning_tokens$word = gsub('[[:digit:]]+', '', text_cleaning_tokens$word)
text_cleaning_tokens$word = gsub('[[:punct:]]+', '', text_cleaning_tokens$word)


text_cleaning_tokens = text_cleaning_tokens %>% 
  anti_join(stop_words)
text_cleaning_tokens = text_cleaning_tokens %>% filter(!(word==""))

library(SnowballC)

biden_tokens = text_cleaning_tokens %>%
  mutate(stem = wordStem(word))

###apply TF-IDF to find most distinctive words

#apply bind_tf_idf to get distinctive words
?bind_tf_idf
#input is df, terms (stems, lemma, or words), document id, n counts
biden_count= biden_tokens %>%
  count(chapter, stem, sort = TRUE) %>% 
  filter(n > 5)  %>% #only keep if mentioned more than 5 times
  bind_tf_idf(stem,chapter, n)
head(biden_count)

#get tokens from Trump inaugural
clean_trump = textcleaner_2(trump)

clean_trump = clean_trump %>% mutate(chapter = rownames(clean_trump))

text_cleaning_tokens = clean_trump %>% 
  tidytext::unnest_tokens(word, x)
text_cleaning_tokens$word = gsub('[[:digit:]]+', '', text_cleaning_tokens$word)
text_cleaning_tokens$word = gsub('[[:punct:]]+', '', text_cleaning_tokens$word)


text_cleaning_tokens = text_cleaning_tokens %>% 
  anti_join(stop_words)
text_cleaning_tokens = text_cleaning_tokens %>% filter(!(word==""))

library(SnowballC)

trump_tokens = text_cleaning_tokens %>%
  mutate(stem = wordStem(word))

###apply TF-IDF to find most distinctive words

trump_count= trump_tokens %>%
  count(chapter, stem, sort = TRUE) %>% filter(n > 5)  %>%
  bind_tf_idf(stem,chapter, n)
head(trump_count)

############################
# compare tf
############################
trump_tokens$pres = "Trump"
biden_tokens$pres = "Biden"
tokens = rbind(trump_tokens, biden_tokens)
head(tokens)


#look at tf-idf
require(ggplot2)
idfstem = tokens %>%
  count(pres, stem, sort = TRUE)  %>%
  bind_tf_idf(stem, pres, n) %>%
  arrange(-tf_idf) %>%
  group_by(pres) %>%
  top_n(20) %>%
  ungroup %>%
  mutate(stem = reorder(stem, tf_idf)) %>%
  ggplot(aes(stem, tf_idf, fill = pres)) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ pres, scales = "free") +
  coord_flip() + theme_bw() +
  scale_colour_brewer(palette = "Dark2") + xlab("Word Stem") + ylab("TF-IDF")

idfstem

require(ggwordcloud)
set.seed(100)
tokens %>%
  count(pres, stem, sort = TRUE)  %>%
  bind_tf_idf(stem, pres, n) %>%
  ggplot(aes(label = stem,  color = pres, size = n)) +
  geom_text_wordcloud_area(rm_outside = TRUE, max_steps = 1,
                           grid_size = 1, eccentricity = .9, area_corr = TRUE) +
  facet_wrap(~pres) + theme_bw()  +
  scale_size_area(max_size = 20)



###########################
## CLUSTERING
###########################
load("/users/irismalone/Dropbox/Courses/Machine Learning/R Code/12-TopicModeling/FlakeMatrix.RData") 

#Analyze series of press releases by fmr Republican Senator Jeff Flake
dim(flake_matrix) #602 press releases
head(flake_matrix)

# Euclidean Distance
#calculate euclidean distance between documents
euclid.dist = dist(flake_matrix, method = "euclidean")
euclid.dist = as.matrix(euclid.dist)
dim(euclid.dist)
head(euclid.dist)

#find press releases which are closest in space
get.closest = function(matrix, index){
  row = matrix[index,]
  closest = order(row)[2]
  return(closest)
}

#example: find which press release is closest to press release 1
get.closest(euclid.dist, 1)
rowSums(flake_matrix)[1]
rowSums(flake_matrix)[313]
#we can systematically identify closest document for each document 

indices = 1:nrow(euclid.dist)
sapply(indices, get.closest, matrix = euclid.dist)


#Visualizing distance between documents
# fit the scale
fit = cmdscale(euclid.dist, eig=TRUE, k=2) # k is the number of dim
# plot solution 
x = fit$points[,1]
y = fit$points[,2]
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2", 
     main="Metric	MDS",	type="n")
text(x, y, labels = 1:nrow(flake_matrix), cex=.7)
# find the average length of the documents
mean(rowSums(flake_matrix))
# now look the length of the outliers
rowSums(flake_matrix)[45]
rowSums(flake_matrix)[150]

# Cosine Similarity
#Unlike the `dist` function which compares distances between rows, 
#the `cosine` function compares distances between columns. 
#This means that we have the **transpose** our matrix before passing it into the `cosine` function.

# transpose matrix
flake_matrix_t = t(flake_matrix)
# calculate cosine metric
require(lsa)
cos.sim = cosine(flake_matrix_t)

# convert to dissimilarity distances
cos.dist = as.matrix(1-cos.sim) 

#determine which document is closest
sapply(indices, get.closest, matrix = cos.dist)

#visualize distance between different clusters
# fit the scale
fit = cmdscale(cos.dist, eig=TRUE, k=2) # k is the number of dim
# plot solution 
x = fit$points[,1]
y = fit$points[,2]
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2", 
     main="Metric	MDS (Multidimensional scaling)",	type="n")
text(x, y, labels = 1:nrow(flake_matrix), cex=.7)

#######################
## K-MEANS CLUSTERING
#######################
#to use k-means recall we have to normalize the data
euclid.lengths = sqrt(rowSums(flake_matrix^2))
flake_norm = flake_matrix/euclid.lengths

k = 3 # assign k = 3
# Recall that kmeans depends on the initial starting values so have to set random seed
set.seed(12345) 
k_cluster= kmeans(flake_norm, centers = k)

#look at the cluster assignments by examining `k_cluster$cluster`

# get cluster assignments of first 10 documents:
head(k_cluster$cluster, 10)

#examine number of documents assigned to each cluster
k_cluster$size

####################
#interpreting the results
####################

#identify the 10 biggest words in each cluster

## First, we're going to create a matrix to store the key words.  
key_words = matrix(NA, nrow = k, ncol=10)
## Now, we iterate over the clusters 
for(z in 1:k){
  ## we want to identify the ten most prevalent words, on average, for the cluster. 
  ## To do that, we can use the k_cluster$centers object to get the cluster centroid.
  ## We then can use the sort function and select the ten most prevalent words.
  ten_most = sort(k_cluster$center[z,], decreasing=T)[1:10]
  
  ## `ten_most` gives us a named vector.
  ## Since we're just interested in the top words, we grab the names of this object and store them.
  key_words[z,]= names(ten_most)
}

key_words
#the results suggest first cluster is about pork, 
#second cluster is about appropriations, and third is meta-data?

#we might be interested in more distinct words
key_words2= matrix(NA, nrow=k, ncol=10)
for(z in 1:k){
  diff= k_cluster$center[z,] - apply(k_cluster$center[-z, ], 2, mean)
  key_words2[z,]= names(sort(diff, decreasing=T)[1:10])
}
key_words2

############################
## TOPIC MODELING
############################


#we can automate this into tidy function 
require(tidyr)
require(stringr)
require(textclean)
?textclean

#take biden inaugural - it's a very small text so can't model much but
#we might expect him to talk about different topics (e.g. foreign policy, domestic policy, economy)
sentences=corpus(biden) %>%
  corpus_reshape(to = "sentences")

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

clean = textcleaner_2(sentences)

clean = clean %>% mutate(chapter = rownames(clean))
head(clean)

require(SnowballC)
# create dtm
require(tidytext)
require(dplyr)
require(tm)
require(textmineR)
require(textstem) #for lemmatization
custom_stopwords=c("will", "you", "I", "american", "america")
#create dtm
head(clean)
dtm_biden = CreateDtm(doc_vec = clean$x, #names(clean_dtm) where text is stored
                       doc_names = clean$chapter, #unique identifier
                       #ngram_window = c(1,2), #ngram - specify unigram c(1), bigrams (2), or both c(1, 2)
                       stopword_vec = c(stopwords("smart"), custom_stopwords), #remove stopwords
                       verbose = F,
                       remove_numbers = TRUE,
                       remove_punctuation = TRUE,
                       lower=TRUE,
                      stem_lemma_function = function(x) lemmatize_words(x))

#use LDA model fit for topic
set.seed(1234)
mod_lda_biden = FitLdaModel(dtm = dtm_biden,
                              k = 4, # number of topic
                              iterations = 500,
                              burnin = 180,
                              alpha = 0.1,beta = 0.05,
                              optimize_alpha = T,
                              calc_likelihood = T,
                              calc_coherence = T,
                              calc_r2 = T,
                              smooth=FALSE)

mod_lda_biden$r2
#get top terms based on pi/phi (pr word | topic)
mod_lda_biden$top_terms = GetTopTerms(phi = mod_lda_biden$phi,M = 5)

data.frame(mod_lda_biden$top_terms)
#coherence measures frequency of terms in given topic - higher coherence better
mod_lda_biden$coherence
#get prevalence
mod_lda_biden$prevalence = colSums(mod_lda_biden$theta)/sum(mod_lda_biden$theta)*100
mod_lda_biden$prevalence
mod_lda_biden$summary = data.frame(topic = rownames(mod_lda_biden$phi),
                                    coherence = round(mod_lda_biden$coherence,3),
                                    prevalence = round(mod_lda_biden$prevalence,3),
                                    top_terms = apply(mod_lda_biden$top_terms,2,function(x){paste(x,collapse = ", ")}))

#summarize information
mod_lda_biden$summary 


#get phi probabilities
GetTerms = function (phi, M) 
{
  result = apply(phi, 1, function(x) {
    print(x)[order(x, decreasing = TRUE)][1:M]
    #names(x)[order(x, decreasing = TRUE)][1:M]
  })
  return(result)
}

#get top 40 terms from each topic
beta=GetTerms(phi = mod_lda_biden$phi,M = 40)
library(reshape2)
longbeta = melt(beta)
words = GetTopTerms(phi = mod_lda_biden$phi,M = 40)
longwords = melt(words)
top_terms = as.data.frame(merge(longbeta, longwords, by=c("Var1", "Var2")))
colnames(top_terms)[1] = "Rank"
colnames(top_terms)[2] = "Topic"
colnames(top_terms)[3] = "Value"
colnames(top_terms)[4] = "Term"

top_terms$Topic
topics=c("t_2", "t_3", "t_4") #top 3 topics in terms of coherence
top_terms2 = subset(top_terms, top_terms$Topic %in% topics)
top_terms2$Topic = factor(top_terms2$Topic)
top_terms2$size = top_terms2$Value*50
levels(top_terms2$Topic)

require(ggplot2)
require(ggwordcloud)
top_terms2 %>%
  ggplot(aes(label = Term,  color = Topic, size = size)) +
  geom_text_wordcloud_area(rm_outside = TRUE, max_steps = 1,
                           grid_size = 1, eccentricity = .9, area_corr = TRUE) +
  facet_wrap(~Topic) + theme_bw()  +
  scale_size_area(max_size = 20)


############################
## APPLICATION
############################


## Motivation: Want custom dictionaries for classification
## Classic Example: The Federalist Papers - who authored the disputed papers?
## Solution: ML https://priceonomics.com/how-statistics-solved-a-175-year-old-mystery-about/
library(tidyverse)
library(tidytext)
library(gutenbergr) #Rdata for all public texts on Gutenberg project
library(glmnet)

#Federalist Papers is number 1404
papers = gutenberg_download(1404)
head(papers, n = 10)

#Divide each paper up into segments by sentences - this gives model more to train

papers_sentences = pull(papers, text) %>% 
  str_c(collapse = " ") %>%
  str_split(pattern = "\\.|\\?|\\!") %>%
  unlist() %>%
  tibble(text = .) %>%
  mutate(sentence = row_number())

head(papers_sentences)

#assign known authors to known documents
hamilton = c(1, 6:9, 11:13, 15:17, 21:36, 59:61, 65:85)
madison = c(10, 14, 18:20, 37:48)
jay = c(2:5, 64)
unknown = c(49:58, 62:63)

#look for papers that start with the line "FEDERALIST NO." since that indicates new document
#label author to all sentences within those documents
papers_words = papers_sentences %>%
  mutate(no = cumsum(str_detect(text, regex("FEDERALIST No",
                                            ignore_case = TRUE)))) %>%
  unnest_tokens(word, text) %>%
  mutate(author = case_when(no %in% hamilton ~ "hamilton",
                            no %in% madison ~ "madison",
                            no %in% jay ~ "jay",
                            no %in% unknown ~ "unknown"),
         id = paste(no, sentence, sep = "-"))

#no surprise: hamilton wrote the most
papers_words %>%
  count(author)

#exclude jay since he got sick so early
papers_words = papers_words %>%
  filter(author != "jay")

#create DTM
papers_dtm = papers_words %>%
  count(id, word, sort = TRUE) %>%
  cast_sparse(id, word, n)

#define a response variable 'author'
meta = data.frame(id = dimnames(papers_dtm)[[1]]) %>%
  left_join(papers_words[!duplicated(papers_words$id), ], by = "id") %>%
  mutate(y = as.numeric(author == "hamilton"),
         train = author != "unknown")

#for each unique id, we've assigned an author to a word
head(meta)

#run lasso regression to identify which words are most distinctive of a given author

predictor = papers_dtm[meta$train, ]
response = meta$y[meta$train]

model = cv.glmnet(predictor, response, family = "binomial", alpha = 0.9)
plot(model$lambda)

meta = meta %>%
  mutate(pred = predict(model, newx = as.matrix(papers_dtm), type = "response",
                        s = model$lambda.min) %>% as.numeric())


#plot training results
meta %>%
  filter(train) %>%
  ggplot(aes(factor(no), pred)) + 
  geom_boxplot(aes(fill = author)) +
  theme_minimal() +
  labs(y = "predicted probability",
       x = "Article number") +
  theme(legend.position = "top") +
  scale_fill_manual(values = c("#304890", "#6A7E50")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#plot the predicted probability of the unknown authors
meta %>%
  ggplot(aes(factor(no), pred)) + 
  geom_boxplot(aes(fill = author)) +
  theme_minimal() +
  labs(y = "predicted probability",
       x = "Article number") +
  theme(legend.position = "top") +
  scale_fill_manual(values = c("#304890", "#6A7E50", "#D6BBD0")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
#seems more likely unknown papers are hamilton, but madison could have edited

