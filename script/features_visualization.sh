#!/bin/bash

# create directory if not yet existing
mkdir -p data/features_visualization/

# run visualizations script
echo "  Folder for visualizations created"
python -m script.Visualizations.visualizations data/preprocessing/labeled.csv data/features_visualization -d
echo "  Plots for features visualization created"
