#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 13:01:57 2021
@author: ArghaSarker
"""
# importing the necessary library and files
from script.preprocessing.preprocessor import Preprocessor
from script.util import COLUMN_TWEET
from nltk.corpus import stopwords
import ast
import nltk
import re
import string

from nltk.stem.wordnet import WordNetLemmatizer



class Tokenizer(Preprocessor):
    """Tokenize the tweets (sentences into words )"""
    
    # initializing inti to handle the class 
    
    def __init__(self, input_column, output_column):
        """input column takes the input and stores it 
        from data and output_column store our tokenized 
        data and sends the result out"""
        
        # input column "tweet", new output column
        super().__init__([input_column], output_column)
        
   
    # now we get values from the preprocessor class
    
    def _get_values(self, inputs):
        # defining method for removing url
        def remove_url(text_data):
            return re.sub(r"http\S+", "", text_data)
        
        
        # defining punct list
        regular_punct = list(string.punctuation)
        
        # remove  punctuiation
        def remove_punctuation(text,punct_list):
            for punc in punct_list:
                if punc in text:
                    text = text.replace(punc, ' ').lower()
            return text.strip()
            
        # changes the tweets into first sentences and then to words and outputs as token
        # holdes our tokenized words
        tokenized = []
        filtered_sentence = []
        
        lemmatizer = WordNetLemmatizer()
        
        
        stop_words = set(stopwords.words('english'))
        
        for tweeter in inputs[0]: 
           
            #tweet_punct = remove_punctuation(tweeter,regular_punct)
            tweet_url = remove_url(tweeter)
            tweet= remove_punctuation(tweet_url,regular_punct)
            
            
            # setp 1: convert the tweets into sentences
            
            sentences = nltk.sent_tokenize(tweet) # (tweet, language = "English") , 
                                                    #can it solve the problem of selecting the only english tweets?? have to 
            tokenized_tweet = []
            lemma_word_list = []
            #step 2: convert sentences into word tokens
            
            for sentence in sentences: 
                words= nltk.word_tokenize(sentence)
                
                #Step 3: Remove stop words from the word tokens 
                #stop_removed = [word for word in words if not word in stop_words]
                stop_removed = []
                
                # for word in words:
                #     if not word in stop_words:
                
                #         lem_word = lemmatizer.lemmatize(word)
                #         stop_removed.append(lem_word)
                        
                for word in words:
                    
                      # first lemetize and then remove stop  
                    lem_word = lemmatizer.lemmatize(word)
                    if not lem_word in stop_words:
                        stop_removed.append(lem_word)
                
                        
                

                tokenized_tweet += stop_removed
                
                #without_stop_list = ast.literal_eval(stop_removed)
                
                
                    
                #tokenized_tweet += stop_removed
                
            tokenized.append(str(tokenized_tweet))
            
        
        return tokenized