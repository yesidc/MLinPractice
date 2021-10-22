#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from script.util import COLUMN_TWEET,SUFFIX_TOKENIZED,COLUMN_PUNCTUATION,CLEANED_TWEET 
from script.preprocessing.punctuation_remover import PunctuationRemover
from script.preprocessing.tokenizer import Tokenizer

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-p", "--punctuation", action = "store_true", help = "remove punctuation")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
# tokenizer initianization 

parser.add_argument("-t", "--tokenize", action = "store_true", help = "Tokenize the tweets into words")
#parser.add_argument("--tokenize_input", help = "input column to tokenize", default = COLUMN_TWEET)
parser.add_argument("--tokenize_input", help = "input column to tokenize", default = COLUMN_TWEET )


args = parser.parse_args()

# load data Kept it 30 for shorter machine load
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n", nrows= 50)







# collect all preprocessors
preprocessors = []
if args.punctuation:
    preprocessors.append(PunctuationRemover())
if args.tokenize: 
    #preprocessors.append(Tokenizer(args.tokenize_input, args.tokenize_input + SUFFIX_TOKENIZED))
    preprocessors.append(Tokenizer(args.tokenize_input, CLEANED_TWEET))
# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)

# store the results
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

# create a pipeline if necessary and store it as pickle file
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)