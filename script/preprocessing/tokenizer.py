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
        # changes the tweets into first sentences and then to words and outputs as token
        # holdes our tokenized words
        tokenized = []
        filtered_sentence = []
        
        lemmatizer = WordNetLemmatizer()
        
        
        stop_words = set(stopwords.words('english'))
        
        for tweet in inputs[0]: 
            
            # setp 1: convert the tweets into sentences
            
            sentences = nltk.sent_tokenize(tweet) # (tweet, language = "English") , 
                                                    #can it solve the problem of selecting the only english tweets?? have to 
            tokenized_tweet = []
            lemma_word_list = []
            #step 2: convert sentences into word tokens
            
            for sentence in sentences: 
                words= nltk.word_tokenize(sentence)
                
                #Step 3: Remove stop words from the word tokens 
                stop_removed = [word for word in words if not word in stop_words]
                
                #without_stop_list = ast.literal_eval(stop_removed)
                
                
                    
                tokenized_tweet += stop_removed
                    
                
                
         
                
            
            tokenized.append(str(tokenized_tweet))
            
        
        return tokenized
            
            
                
                
                
            
        
        
        
    
    
    