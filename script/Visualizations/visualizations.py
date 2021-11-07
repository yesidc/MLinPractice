import argparse, csv, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
import seaborn as sn
from script.feature_extraction.character_length import CharacterLength
from script.feature_extraction.feature_collector import FeatureCollector
from script.util import COLUMN_TWEET, COLUMN_LABEL

# setting up CLI
parser = argparse.ArgumentParser(description="Visualization")
parser.add_argument("input_file", help="path to the input csv file")
parser.add_argument("output_file", help="path to the output folder where images are saved")
parser.add_argument("-d", "--default_feat_visualizations", action="store_true", help="generates all png features visualization graphs")
parser.add_argument("-v", "--features_variance", action="store_true", help = "generates the csv data variance as png")
parser.add_argument("-des", "--data_description", action="store_true", help = "generates the csv description of the data as png")
parser.add_argument("-mg", "--group_means", action="store_true", help = "generates png image of the features group means")
parser.add_argument("-fmv", "--feature_mean_var", action="store_true", help = "generates png image of the features mean and variance")
parser.add_argument("-fc", "--feature_correlations", action="store_true", help = "generates png image of the features correlation before and after feature selection")
parser.add_argument("-p", "--pairplot_correlations", action="store_true", help = "generates png image of the pairwise relationship of the features distributions in the dataset")
parser.add_argument("-vir", "--tweets_virality", action="store_true", help = "generates png images of the virality of tweets by feature")
parser.add_argument("-t", "--time_virality", action="store_true", help = "generates png images of the virality of tweets by time features")
parser.add_argument("-dv", "--describe_virality", action="store_true", help = "generates png images of the virality description of tweets by features")


args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")


def describe(data):
    """
    :param data: a csv file.
    :return: the csv general description of the data as png.
    """
    description = pd.DataFrame(data.describe())
    dfi.export(description, args.output_file + "/description_data.png")

if args.data_description:
    describe(df)


# clean, process, and group data features
df_clean = df[["likes_count", "replies_count", "retweets_count", "language", "video", "label", "date", "time"]]
# extract the amount of photos, urls, hashtags, and the hours when tweets were made
df_clean["photos"] = df["photos"].map(lambda x: len(x[1:-1].split(', ')))
df_clean["urls"] = df["urls"].map(lambda x: len(x[1:-1].split(', ')))
df_clean["hashtags"] = df["hashtags"].map(lambda x: len(x[1:-1].split(', ')))
df_clean["tweet_hour"] = pd.DataFrame(pd.DatetimeIndex(df['time']).hour.values)


# Variance can help us identify which features values change the most. Those that have
# a low variance means numbers stay pretty much same and will probably not tell us much.
# On the contrary, those with high variance can tell us what us going on with the data.
def variance(data, data2):
    """
    :param data: uncleaned .csv file
    :param data2: cleaned .csv file
    :return: the csv data variance as png for cleaned and uncleaned data.
    """
    var_img1 = pd.DataFrame({"Variance": data.var()})
    var_img2 = pd.DataFrame({"Variance cleaned": data2.var()})

    dfi.export(var_img1, args.output_file + "/features_variance1.png")
    dfi.export(var_img2, args.output_file + "/features_variance_cleaned.png")

# check the variance of initial df and df_clean
if args.features_variance:
    variance(df, df_clean)

# group the cleaned data by groups
groups = df_clean.groupby('label')

def groups_means(data):
    """
    :param data: a DataFrame file.
    :return: an image of the grouped means by label
    """
    fig, ax = plt.subplots()
    # groups means by language
    grouped_mean_lang = data.groupby('label').mean()
    img = np.log10(grouped_mean_lang).plot(kind='bar')
    plt.xticks(rotation=0, ha='right')
    plt.title("Features means grouped by label")
    plt.savefig(args.output_file + "/features_means_grouped_by_label.png")


if args.group_means:
    groups_means(df_clean)

def feature_mean_var_by_label(data):
    """
    :param data: a DataFrame file
    :return: generates png image of the features mean and variance
    """
    # group mean and variance by label which show interesting features to be those to the left
    feature_means_label = data.groupby('label').mean()
    feature_var_label = data.groupby('label').var()

    # means
    np.log10(feature_means_label).plot(kind='box')
    plt.xticks(rotation=18, ha='right')
    plt.title("Feature mean by label")
    plt.savefig(args.output_file + "/features_means_by_label.png")
    # variances
    np.log10(feature_var_label).plot(kind='box')
    plt.xticks(rotation=18, ha='right')
    plt.title("Feature variance by label")
    plt.savefig(args.output_file + "/features_variance_by_label.png")

if args.feature_mean_var:
    feature_mean_var_by_label(df_clean)




# heat map correlation of features after selection (fig. bottom)
def feature_correlation(data, data_cleaned):
    """
    :param data: a Dataframe file.
    :param data_cleaned: a DataFrame file after feature selection.
    :return: generates png image of the features correlation before and after feature selection.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    # Heatmap help us find the correlation between the different data features (fig. top)
    sn.heatmap(data.corr(), ax=ax1)
    ax1.set_title("Feature correlation initial tweets data")
    # Heat map correlation of features after selection (fig. bottom)
    sn.heatmap(data_cleaned.corr(), annot=True, ax=ax2)
    ax2.set_title("Feature correlation after selection")
    fig.tight_layout()
    plt.savefig(args.output_file + "/feature_selection_by_correlation.png")

if args.feature_correlations:
    feature_correlation(df, df_clean)


def pairplot_correlations(data, sample=30000):
    """
    :param data: a DataFrame file.
    :param sample: the amount of data points to sample from (default = 30000).
    :return: generates a pairwise plot relationship of the data set.
    """
    # pairwise relationship of the features distributions in the dataset.
    sn.pairplot(data.sample(sample), hue="label")
    plt.savefig(args.output_file + "/feature_selection_by_correlation_pairplot.png")

if args.pairplot_correlations:
    pairplot_correlations(df_clean)


# explore tweets virality by features relations
def scatter_viral(data, x, y, xlog=False, ylog=False, plot_title="", save=""):
    """
    :param data: a DataFrame file.
    :param x: feature to project on the x-axis.
    :param y: feature to project on the y-axis.
    :param xlog: bool. When xlog=True, projects the logarithm of values on the respective axis.
    :param ylog: bool. When xlog=True, projects the logarithm of values on the respective axis.
    :param plot_title: string sentence to set as plot's title.
    :param save: string non-spaced sentence to save the image followed by .png
    :return: generates a scatter plot of the virality relationship between the x and y features specified
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    for name, group in data:
        ax.plot(group[x], group[y], marker='o', linestyle='', ms=1, label=name, alpha=0.3)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.title.set_text(plot_title)
    # reset the legend text name
    L = ax.legend()
    L.get_texts()[0].set_text("Not-viral")
    L.get_texts()[1].set_text("Viral")

    # save images to the visualization folder as indicated by the parser
    plt.savefig(args.output_file + save)


def tweets_virality_features(groups):
    scatter_viral(groups, "likes_count", "retweets_count", True, True,
                  plot_title="Feature selection for virality: retweets as a function of likes",
                  save="/retweets_likes.png")
    scatter_viral(groups, "likes_count", "replies_count", True, True,
                  plot_title="Feature selection for virality: replies as a function of likes",
                  save="/replies_likes.png")
    scatter_viral(groups, "retweets_count", "replies_count", True, True,
                  plot_title="Feature selection for virality: replies as a function of retweets",
                  save="/replies_retweets.png")
    scatter_viral(groups, "likes_count", "language", True, False,
                  plot_title="Feature selection for virality: language as a function of likes",
                  save="/language_likes.png")
    scatter_viral(groups, "likes_count", "hashtags", True, False,
                  plot_title="Feature selection for virality: hashtags as a function of likes",
                  save="/hashtags_likes.png")
    scatter_viral(groups, "likes_count", "photos", True, False,
                  plot_title="Feature selection for virality: photos as a function of likes", save="/photos_likes.png")
    scatter_viral(groups, "tweet_hour", "likes_count", False, True,
                  plot_title="likes as a function of tweet's hour", save="/likes_tweet_hour.png")

if args.tweets_virality:
    tweets_virality_features(groups)


# explore tweets virality by date and time features.
def count_tweets_per_creation_date(data):
    """
    :param data: a DataFrame file
    :return: generates histograms of the amount of tweets made per date and time and their virality.
    """
    data['tweets_per_year'] = pd.DataFrame(pd.DatetimeIndex(data['date']).year.values)
    data['tweets_per_month'] = pd.DataFrame(pd.DatetimeIndex(data['date']).month.values)
    data['tweets_per_day'] = pd.DataFrame(pd.DatetimeIndex(data['date']).day.values)
    data['tweets_per_hour'] = pd.DataFrame(pd.DatetimeIndex(data['time']).hour.values)

    column_dates = ["tweets_per_year", "tweets_per_month","tweets_per_day","tweets_per_hour"]
    plt.figure(figsize=(16, 10))

    # amount of tweets per date
    for i, name in enumerate(column_dates):
        plt.subplot(2,2, i+1)
        sn.histplot(data, x = name, discrete = True)
    plt.suptitle("Amount of tweets per creation date", fontsize=20)
    plt.savefig(args.output_file + "/tweets_amount_per_creation_date.png")

    # virality of tweets per date
    plt.figure(figsize=(16, 10))
    for i, name in enumerate(column_dates):
        plt.subplot(2, 2, i + 1)
        sn.countplot(name, data=data, hue='label')
    plt.suptitle("Virality of tweets per creation date", fontsize=20)
    plt.savefig(args.output_file + "/tweets_virality_per_creation_date.png")

if args.time_virality:
    count_tweets_per_creation_date(df_clean)


# explore tweets virality by describing the data to confirm the visualizations information
def describe_features_virality(df):
    """
    :param df: a DataFrame file.
    :return: generates the dataFrame image of the features description.
    """
    # select and group the features by label
    likes_to_viral = df[["likes_count", "label"]]
    retweets_to_viral = df[["retweets_count", "label"]]
    replies_to_viral = df[["replies_count", "label"]]

    likes_groups = likes_to_viral.groupby('label')
    retweets_groups = retweets_to_viral.groupby('label')
    replies_groups = replies_to_viral.groupby('label')

    # Learning: Likely not viral if likes < 50
    desc1 = likes_groups.describe()
    dfi.export(desc1, args.output_file + "/likes_description.png")

    # Learning: Likely not viral if retweets < 47
    desc2 = retweets_groups.describe()
    dfi.export(desc2, args.output_file + "/retweets_description.png")

    # Learning: Does not explain virality as well, given percentile distributions
    # are flat between true and false labeled tweets
    desc3 = replies_groups.describe()
    dfi.export(desc3, args.output_file + "/replies_description.png")

if args.describe_virality:
    describe_features_virality(df_clean)

# default (-d) image generation for features visualization
if args.default_feat_visualizations:
    variance(df, df_clean)
    describe(df)
    groups_means(df_clean)
    feature_mean_var_by_label(df_clean)
    feature_correlation(df, df_clean)
    pairplot_correlations(df_clean)
    tweets_virality_features(groups)
    count_tweets_per_creation_date(df_clean)












