"""
Created on Thursday Oct 25  2021

@author: yesidc
"""

import pandas as pd
import numpy as np
from script.feature_extraction.feature_extractor import FeatureExtractor
import ast

class contain_websites (FeatureExtractor):

    def __init__(self,input_column):
        super().__init__([input_column], "{0}_websites".format(input_column))


    def _get_values (self,inputs):
        """Given the urls column, returns how many websites the tweet contains"""
        result = inputs[0].values
        result = np.array([len(ast.literal_eval(x)) for x in result]).reshape(-1, 1)
        return result
