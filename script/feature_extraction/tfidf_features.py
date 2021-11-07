"""
Created on Friday Oct 29  2021

@author: yesidc
"""


import numpy as np
from script.feature_extraction.feature_extractor import FeatureExtractor


from sklearn.feature_extraction.text import TfidfVectorizer

class tfidf_vectors (FeatureExtractor):

    def __init__(self,input_column):
        super().__init__([input_column], "{0}_tfidf".format(input_column))


    def _get_values (self,inputs):
        """Given the tweet_no_punctuation column, extracts tfidf vector representation for each tweet"""
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                stop_words='english')
        features_result = tfidf.fit_transform(inputs[0]).toarray()
        return features_result


