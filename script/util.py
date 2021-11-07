#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""
import matplotlib.pyplot as plt

#Function adapted from code availabe here https://ostwalprasad.github.io/machine-learning/PCA-using-python.html
def scatter_plot_pca(pca,path_to_save,graph_name):
    plt.scatter(pca[:, 0], pca[:, 1])
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.savefig(path_to_save+"/PCA_scatter_plot_"+graph_name+".png")

#Function adapted from code availabe here https://ostwalprasad.github.io/machine-learning/PCA-using-python.html
def plot_pca_biplot(path_to_save,graph_name,score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    plt.scatter(xs , ys , s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='green', ha='center',
                     va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.savefig(path_to_save + "/PCA_biplot_"+ graph_name +".png")



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