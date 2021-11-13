#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""
import matplotlib.pyplot as plt
import pandas as pd

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
SUFFIX_TOKENIZED= "_tokenized"
CLEANED_TWEET = "cleaned_tweet"



#Function adapted from code availabe here https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python
def plot_pca(pca,path_to_save,graph_name,df):
    pcd_df=pd.DataFrame(data=pca, columns=['principal component 1', 'principal component 2'])

    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title("PCA of tweet dataset", fontsize=20)
    targets = [True, False]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        indicesToKeep = df['labels'] == target
        plt.scatter(pcd_df.loc[indicesToKeep, 'principal component 1']
                    , pcd_df.loc[indicesToKeep, 'principal component 2'], c=color, s=50)

    plt.legend(targets, prop={'size': 15})
    plt.savefig(path_to_save + "/PCA_" + graph_name + ".png")