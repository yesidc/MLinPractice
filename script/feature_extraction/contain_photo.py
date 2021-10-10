import pandas as pd
import numpy as np
from script.feature_extraction.feature_extractor import FeatureExtractor

class contain_pthotos (FeatureExtractor):

    def __init__(self,input_column):
        super().__init__([input_column], "{0}_photo".format(input_column))


    def _get_values (self,inputs):
        """Given the photos column, returns 1 if a photo was included in the post, 0 otherwise"""
        result = inputs[0].values
        result = np.array([1 if len(x)>2 else 0 for x in result]).reshape(-1,1)
        return result
