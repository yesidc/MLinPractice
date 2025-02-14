"""
Created on Wed Oct 24  2021

@author: yesidc
"""

import pandas as pd
from script.feature_extraction.feature_extractor import FeatureExtractor

class extract_month (FeatureExtractor):

    def __init__(self,input_column):
        super().__init__([input_column], "{0}_month".format(input_column))


    def _get_values (self,inputs):
        """Given the date column, extracts the month the tweet was posted"""
        result = pd.DatetimeIndex(inputs[0]).month.values
        result = result.reshape(-1, 1)
        return result
