#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:28:00 2021

@author: ArghaSarker
"""

import unittest
import pandas as pd
from script.preprocessing.tokenizer import Tokenizer

class TokenizerTest(unittest.TestCase):
    
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.tokenizer = Tokenizer(self.INPUT_COLUMN,self.OUTPUT_COLUMN)
        
        
    def test_input_columns(self):
        self.assertListEqual(self.tokenizer._input_columns, [self.INPUT_COLUMN])
        
    def test_output_column(self):
        self.assertEqual(self.tokenizer._output_column, self.OUTPUT_COLUMN)

    def test_preprocesisng(self):
        input_text = "Interested in entering the world of Data Analysis, but worried you have no experience or are just a beginner programmer?   There's an apprenticeship for you, apply here:  https://t.co/lxJv1ahq8g  #YouMakeTheBBC  https://t.co/G2Bulw5XEq"  
        output_text = "['interested', 'entering', 'world', 'data', 'analysis', 'worried', 'experience', 'beginner', 'programmer', 'apprenticeship', 'apply', 'youmakethebbc']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN]= [input_text]
        
        tokenized = self.tokenizer.fit_transform(input_df)
        
    
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text)
        
    if __name__ == '__main__':
        unittest.main()