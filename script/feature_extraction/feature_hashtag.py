"""
Created on Friday Nov 5  2021

@author: yesidc
"""

import pandas as pd
from script.feature_extraction.feature_extractor import FeatureExtractor

class extract_num_hashtags (FeatureExtractor):

    def __init__(self,input_column):
        super().__init__([input_column], "{0}_num_hashtags".format(input_column))


    def _get_values (self,inputs):
        """Given the hashtags column, extracts the number hashtags per tweet"""

        result= inputs[0].map(lambda x: len(x[1:-1].split(', ')))
        result= result.values
        #result = pd.DatetimeIndex(inputs[0]).month.values
        result = result.reshape(-1, 1)
        return result
