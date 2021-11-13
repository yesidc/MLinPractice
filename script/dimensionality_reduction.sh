#!/bin/bash

# create directory if not yet existing
mkdir -p data/dimensionality_reduction/

# run dimensionality reduction on training set to fit the parameters
echo "  training set"
python -m script.dimensionality_reduction.reduce_dimensionality data/feature_extraction/training.pickle data/dimensionality_reduction/training.pickle -f data/dimensionality_reduction -e data/dimensionality_reduction/pipeline.pickle -p 2

# run feature extraction on validation set and test set (with pre-fit parameters)
echo "  validation set"
python -m script.dimensionality_reduction.reduce_dimensionality data/feature_extraction/validation.pickle data/dimensionality_reduction/validation.pickle -f data/dimensionality_reduction -i data/dimensionality_reduction/pipeline.pickle
echo "  test set"
python -m script.dimensionality_reduction.reduce_dimensionality data/feature_extraction/test.pickle data/dimensionality_reduction/test.pickle -f data/dimensionality_reduction -i data/dimensionality_reduction/pipeline.pickle