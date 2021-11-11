#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/

# run feature extraction on training set (may need to fit extractors)
echo "  training set"
python -m script.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle --majority --accuracy --kappa --logLoss --roc_auc

# run feature extraction on validation set (with pre-fit extractors)
echo "  validation set"
python -m script.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle --accuracy --kappa --logLoss --roc_auc

# don't touch the test set, yet, because that would ruin the final generalization experiment!