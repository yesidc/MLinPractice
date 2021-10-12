#!/bin/bash

echo "loading data"
script/load_data.sh
echo "preprocessing"
script/preprocessing.sh
echo "feature extraction"
script/feature_extraction.sh
echo "dimensionality reduction"
script/dimensionality_reduction.sh
echo "classification"
script/classification.sh