import argparse, csv, pickle
import pandas as pd
import numpy as np
from script.feature_extraction.character_length import CharacterLength
from script.feature_extraction.feature_collector import FeatureCollector
from script.util import COLUMN_TWEET, COLUMN_LABEL


# setting up CLI
parser = argparse.ArgumentParser(description = "Visualization")
parser.add_argument("input_file", help = "path to the input csv file")
# parser.add_argument("output_file", help = "path to the output pickle file")
# parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
# parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
# parser.add_argument("-c", "--char_length", action = "store_true", help = "compute the number of characters in the tweet")
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")


print("Hey")






# ====== In case the back-up jupyter notebook gets ignored ========
# #%%
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# #%%
#
# df = pd.read_csv("/Users/johnmadrid/GitHub/MLinPractice/data/preprocessing/labeled.csv")
#
# # df.head()
# # df.columns
# df.describe()
#
# #%%
# scatter = df[["likes_count", "replies_count", "retweets_count", "language", "video", "label"]]
# scatter["photos"] = df["photos"].map(lambda x: len(x[1:-1].split(', ')))
# scatter["urls"] = df["urls"].map(lambda x: len(x[1:-1].split(', ')))
# scatter["hashtags"] = df["hashtags"].map(lambda x: len(x[1:-1].split(', ')))
#
# scatter['photos']
# #%%
# groups = scatter.groupby('label')
# print(groups)
#
# #%%
#
# def scatterviral(groups, x, y, xlog = False, ylog = False, plot_title = ""):
#     fig, ax = plt.subplots(figsize=(7,5))
#     ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
#     if xlog:
#         ax.set_xscale('log')
#     if ylog:
#         ax.set_yscale('log')
#
#     for name, group in groups:
#         ax.plot(group[x], group[y], marker='o', linestyle='', ms=1, label=name, alpha = 0.3)
#     ax.set_xlabel(x)
#     ax.set_ylabel(y)
#     ax.title.set_text(plot_title)
#     L = ax.legend()
#     L.get_texts()[0].set_text("Not-viral")
#     L.get_texts()[1].set_text("Viral")
#
#
#     plt.savefig("retweets_count_as_function_of_likes_count.png")
#     plt.show()
#
# #%%
# scatterviral(groups, 'likes_count', 'retweets_count', True, True, plot_title = "Feature selection for viral: retweets as a function of likes")
#
#
# #%%
#
# scatterviral(groups, 'likes_count', 'replies_count', True, True)
#
# #%%
# likes_to_viral = df[["likes_count", "label"]]
# retweets_to_viral = df[["retweets_count", "label"]]
# replies_to_viral = df[["replies_count", "label"]]
#
# likes_groups = likes_to_viral.groupby('label')
# retweets_groups = retweets_to_viral.groupby('label')
# replies_groups = replies_to_viral.groupby('label')
#
#
#
# #%%
# likes_groups.describe()
# # Learning: Likely not viral if likes < 50
#
# #%%
# retweets_groups.describe()
# # Learning: Likely not viral if retweets < 47
#
# #%%
# replies_groups.describe()
# # Learning: Does not explain virality as well, given percentile distributions are flat between true and false labelled tweets
#
# #%%
#
# scatterviral(groups, 'likes_count', 'language', True, False)
# # Learning: much more likely to be viral with < 50 likes if language is 'en'
#
# #%%
#
# scatterviral(groups, 'likes_count', 'hashtags', True, False)
#
#
# #%%
#
# scatterviral(groups, 'likes_count', 'photos', True, False)
# #%%
# # filter by viral tweets
# viral = likes_to_viral[likes_to_viral["label"] == 1]
#
# viral["likes_count"].mean()
#
# #%%
# count = viral["likes_count"].value_counts()
#
#
# count
# #%%
#
# viral.plot.scatter(pd.DataFrame(['likes_count']), pd.DataFrame(['retweets_count']))
#
# #%%
