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





