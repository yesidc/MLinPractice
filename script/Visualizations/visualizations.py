import argparse, csv, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from script.feature_extraction.character_length import CharacterLength
from script.feature_extraction.feature_collector import FeatureCollector
from script.util import COLUMN_TWEET, COLUMN_LABEL


# setting up CLI
parser = argparse.ArgumentParser(description = "Visualization")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output folder where images are saved")
# parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
# parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
parser.add_argument("-d", "--default_feat_visualizations", action = "store_true", help = "this generates all features visualization graphs")
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# clean and group data features
df_clean = df[["likes_count", "replies_count", "retweets_count", "language", "video", "label", "date", "time"]]
df_clean["photos"] = df["photos"].map(lambda x: len(x[1:-1].split(', ')))
df_clean["urls"] = df["urls"].map(lambda x: len(x[1:-1].split(', ')))
df_clean["hashtags"] = df["hashtags"].map(lambda x: len(x[1:-1].split(', ')))


groups = df_clean.groupby('label')


def scatterviral(groups, x, y, xlog = False, ylog = False, plot_title = ""):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    for name, group in groups:
        ax.plot(group[x], group[y], marker='o', linestyle='', ms=1, label=name, alpha = 0.3)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.title.set_text(plot_title)
    L = ax.legend()
    L.get_texts()[0].set_text("Not-viral")
    L.get_texts()[1].set_text("Viral")

    # save images to the visualization folder as indicated by the parser
    plt.savefig(args.output_file + "/retweets_count_as_function_of_likes_count.png")
    plt.show()

if args.default_feat_visualizations:
    scatterviral(groups, 'likes_count', 'retweets_count', True, True, plot_title = "Feature selection for viral: retweets as a function of likes")

