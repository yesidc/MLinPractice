"""
Created on Sunday Nov 7  2021

@author: yesidc
"""


import pandas as pd
from script.feature_extraction.feature_extractor import FeatureExtractor

class extract_hour (FeatureExtractor):

    def __init__(self,input_column):
        super().__init__([input_column], "{0}_hour".format(input_column))


    def _get_values (self,inputs):
        """Given the time column, extracts the hour the tweet was posted"""
        result = pd.DatetimeIndex(inputs[0]).hour.values
        result = result.reshape(-1, 1)
        return result