# Quora-question-pair-similarity


Quora is the place where questions get answers. This website helps online users raise queries for knowing the unknown and reply to ones they are good at answering. 
There are 7.6 billion people living on the planet and over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

### About the problem 


 Quora has given an (almost) real-world dataset of question pairs, with the label of is_duplicate along with every question pair. The objective was to minimize the logloss of predictions on duplicacy in the testing dataset. Given a pair of questions q1 and q2, train a model that learns the function: <br>
f(q1, q2) → 0 or 1 <br>
where 1 represents that q1 and q2 have the same intent and 0 otherwise.


###  Problem Statement 
- Identify which questions asked on Quora are duplicates of questions that have already been asked. 
- This could be useful to instantly provide answers to questions that have already been answered. 
- We are tasked with predicting whether a pair of questions are duplicates or not. 



### Real world/Business Objectives and Constraints 

1. The cost of a mis-classification can be very high.
2. You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
3. No strict latency concerns.
4. Interpretability is partially important.

### Data Overview

- Source : https://www.kaggle.com/c/quora-question-pairs

<p> 
- Data will be in a file Train.csv <br>
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate <br>
- Size of Train.csv - 60MB <br>
- Number of rows in Train.csv = 404,290
</p>



###  Performance Metric
 
* log-loss 
* Binary Confusion Matrix

###  Some Data Analysis



###  Load the data

As it's a high-level dataset, entry-level machines with 8GB RAM will take so much of time for the executions. That's why I switched to Kaggle Kernel for this case study. First I load some basic requirement packages for this case study.



###  Preprocessing

    1. Deduplication of entries
    2. Checking for NULL values
    3. Preprocessing of Text:
         - Removing html tags 
         - Removing Punctuations
         - Performing stemming
         - Removing Stopwords
         - Expanding contractions etc.
         
    4. Removing  empty question rows if any exist
    
    
### Feature Extraction
 
 __Basic Feature Extraction__
 - ____freq_qid1____ = Frequency of qid1's
 - ____freq_qid2____ = Frequency of qid2's 
 - ____q1len____ = Length of q1
 - ____q2len____ = Length of q2
 - ____q1_n_words____ = Number of words in Question 1
 - ____q2_n_words____ = Number of words in Question 2
 - ____word_Common____ = (Number of common unique words in Question 1 and Question 2)
 - ____word_Total____ =(Total num of words in Question 1 + Total num of words in Question 2)
 - ____word_share____ = (word_common)/(word_Total)
 - ____freq_q1+freq_q2____ = sum total of frequency of qid1 and qid2 
 - ____freq_q1-freq_q2____ = absolute difference of frequency of qid1 and qid2 
 
 
__Advance Feature Extraction__

Definition:
- __Token__: You get a token by splitting sentence a space
- __Stop_Word__ : stop words as per NLTK.
- __Word__ : A token that is not a stop_word


Features:
- __cwc_min__ :  Ratio of common_word_count to min lenghth of word count of Q1 and Q2 <br>cwc_min = common_word_count / (min(len(q1_words), len(q2_words))
<br>
<br>
- __cwc_max__ :  Ratio of common_word_count to max lenghth of word count of Q1 and Q2 <br>cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
<br>
<br>
- __csc_min__ :  Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2 <br> csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
<br>
<br>
- __csc_max__ :  Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2<br>csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
<br>
<br>
- __ctc_min__ :  Ratio of common_token_count to min lenghth of token count of Q1 and Q2<br>ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
<br>
<br>

- __ctc_max__ :  Ratio of common_token_count to max lenghth of token count of Q1 and Q2<br>ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
<br>
<br>
        
- __last_word_eq__ :  Check if First word of both questions is equal or not<br>last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])
<br>
<br>

- __first_word_eq__ :  Check if First word of both questions is equal or not<br>first_word_eq = int(q1_tokens[0] == q2_tokens[0])
<br>
<br>
        
- __abs_len_diff__ :  Abs. length difference<br>abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
<br>
<br>

- __mean_len__ :  Average Token Length of both Questions<br>mean_len = (len(q1_tokens) + len(q2_tokens))/2
<br>
<br>


- __fuzz_ratio__ :  https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>

- __fuzz_partial_ratio__ :  https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>


- __token_sort_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>


- __token_set_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>





- __longest_substr_ratio__ :   Ratio of length longest common substring to min length of the token count of Q1 and Q2
longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))
Some Features analysis and visualizations
word_share - We can check from below that it is overlapping a bit, but it is giving some classifiable score for dissimilar questions.

### Some Features analysis and visualizations

- word_share - We can check from below that it is overlaping a bit, but it is giving some classifiable score for disimilar questions.




- Word Common - it is almost overlaping.


-  Pair plot of features ['ctc_min', 'cwc_min', 'csc_min', 'token_sort_ratio'] 




### Machine Learning Models

- Building a random model (Finding worst-case log-loss)
> Using a random model to get the upper bound for the log-loss. I want the value of log-loss to be lesser than the log-loss of a random model. Trained a random model to check       Worst case log loss and got log loss as ** 0.887019447220453 **.
- Random train test split( 70:30) 

> Splited the data in to 70 30 ratio

- Vectorization

> Vectorized the data with simple TF-IDF vectors.

- Models

> Trained LR, SVM models and also tuned hyperparameters using Random Search. Hyperparameter tuned XgBoost using RandomSearch to reduce the log-loss. Below are models and their  log loss scores.

- Featurized text data with tfidf weighted word-vectors
- Models
> Trained LR, SVM models with tfidf weighted word-vectors and also tuned hyperparameters using Random Search. Hyperparameter tuned XgBoost using RandomSearch to reduce the log-loss. Below are models and their  log loss scores






### Reference:

1. Kaggle Discussion  https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments
2. Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
3. Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
4. Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30
