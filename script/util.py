#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""
import matplotlib.pyplot as plt


def scatter_plot_pca(pca,path_to_save,graph_name):
    plt.scatter(pca[:, 0], pca[:, 1])
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.savefig(path_to_save+"/PCA_scatter_plot_"+graph_name+".png")

# column names for the original data frame

COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"
COLUMN_DATE = "date"
COLUMN_PHOTOS = "photos"
COLUMN_URLS = "urls"
# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"
COLUMN_HASHTAG ='hashtags'
COLUMN_TIME="time"


SUFFIX_TOKENIZED= "_tokenized"

CLEANED_TWEET = "cleaned_tweet"