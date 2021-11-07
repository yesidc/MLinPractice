#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.

Created on Wed Sep 29 11:00:24 2021

@author: lbechberger
"""



import argparse, csv, pickle
import pandas as pd
import numpy as np
from script.feature_extraction.character_length import CharacterLength
from script.feature_extraction.feature_collector import FeatureCollector
from script.feature_extraction.month_tweet import extract_month
from script.feature_extraction.contain_photo import contain_pthotos
from script.feature_extraction.contain_website import contain_websites
from script.feature_extraction.tfidf_features import tfidf_vectors
from script.feature_extraction.feature_hashtag import extract_num_hashtags
from script.feature_extraction.feature_hour import extract_hour
from script.util import COLUMN_TWEET, COLUMN_LABEL,COLUMN_DATE, COLUMN_PHOTOS,COLUMN_URLS,COLUMN_PUNCTUATION,COLUMN_HASHTAG,COLUMN_TIME


# setting up CLI
parser = argparse.ArgumentParser(description = "Feature Extraction")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
parser.add_argument("-c", "--char_length", action = "store_true", help = "compute the number of characters in the tweet")
parser.add_argument("-m", "--month_tweet", action= "store_true", help= "retrieve the month the tweet was posted")
parser.add_argument("-p", "--contain_photo", action= "store_true", help= "returns 1 if the post contains a photo; 0 otherwise")
parser.add_argument("-w", "--contain_website", action= "store_true", help= "returns 1 if the post contains a website; 0 otherwise")
parser.add_argument("-t", "--tfidf_vector", action= "store_true", help= "Extracts tfidf of each tweet")
parser.add_argument("-n", "--num_hashtags", action= "store_true", help= "Retrieves the number of hashtags per tweet")
parser.add_argument("-d", "--time_hour", action= "store_true", help= "Retrieves the hour the tweet was posted")
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

#TODO change readme reference to code --> change it to script
#TODO delete this line of code
df = df.iloc[40:70]


if args.import_file is not None:
    # simply import an exisiting FeatureCollector
    with open(args.import_file, "rb") as f_in:
        feature_collector = pickle.load(f_in)

else:    # need to create FeatureCollector manually

    # collect all feature extractors
    features = []
    if args.char_length:
        # character length of original tweet (without any changes)
        features.append(CharacterLength(COLUMN_TWEET))
    if args.month_tweet:
        #month tweet was posted
        features.append(extract_month(COLUMN_DATE))
    if args.contain_photo:
        #whether photo was included in the tweet
        features.append(contain_pthotos(COLUMN_PHOTOS))

    if args.contain_website:
        #whether a website was included in the tweet.
        features.append(contain_websites(COLUMN_URLS))
    if args.tfidf_vector:
        features.append(tfidf_vectors(COLUMN_PUNCTUATION))

    if args.num_hashtags:
        features.append(extract_num_hashtags(COLUMN_HASHTAG))
    if args.time_hour:
        features.append(extract_hour(COLUMN_TIME))


    # create overall FeatureCollector
    feature_collector = FeatureCollector(features)


    
    # fit it on the given data set (assumed to be training data)
    feature_collector.fit(df)


# apply the given FeatureCollector on the current data set
# maps the pandas DataFrame to an numpy array
feature_array = feature_collector.transform(df)

# get label array
label_array = np.array(df[COLUMN_LABEL])
label_array = label_array.reshape(-1, 1)

# store the results
results = {"features": feature_array, "labels": label_array, 
           "feature_names": feature_collector.get_feature_names()}
with open(args.output_file, 'wb') as f_out:
    pickle.dump(results, f_out)

# export the FeatureCollector as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(feature_collector, f_out)