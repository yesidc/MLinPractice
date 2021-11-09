# ML in Practice
Source code for the practical Seminar "Machine Learning in Practice", taught at Osnabrück University in the winter term 2021/2022.

As data source, we use the "Data Science Tweets 2010-2021" data set (version 3) by Ruchi Bhatia from [Kaggle](https://www.kaggle.com/ruchi798/data-science-tweets). The goal of our example project is to predict which tweets will go viral, i.e., receive many likes and retweets.



## Virtual Environment

In order to install all dependencies, please make sure that you have a local [Conda](https://docs.conda.io/en/latest/) distribution (e.g., Anaconda or miniconda) installed. Begin by creating a new environment called "MLinPractice" that has Python 3.6 installed:

```conda create -y -q --name MLinPractice python=3.6```

You can enter this environment with `conda activate MLinPractice` (or `source activate MLinPractice`, if the former does not work). You can leave it with `conda deactivate` (or `source deactivate`, if the former does not work). Enter the environment and execute the following commands in order to install the necessary dependencies (this may take a while):

```
conda install -y -q -c conda-forge scikit-learn=0.24.2
conda install -y -q -c conda-forge matplotlib=3.3.4
conda install -y -q -c conda-forge nltk=3.6.3
conda install -y -q -c conda-forge gensim=4.1.2
conda install -y -q -c conda-forge spyder=5.1.5
conda install -y -q -c conda-forge pandas=1.1.5
conda install -y -q -c conda-forge dataframe_image 
```

You can double-check that all of these packages have been installed by running `conda list` inside of your virtual environment. The Spyder IDE can be started by typing `~/miniconda/envs/MLinPractice/bin/spyder` in your terminal window (assuming you use miniconda, which is installed right in your home directory).

In order to save some space on your local machine, you can run `conda clean -y -q --all` afterwards to remove any temporary files.

The installed libraries used for machine learning are (`scikit-learn`), visualizations (`matplotlib`), NLP (`nltk`), word embeddings (`gensim`), and IDE (`spyder`), and data handling (`pandas`)

## Overall Pipeline

The overall pipeline can be executed with the script `script/pipeline.sh`, which executes all of the following shell scripts:
- The script `script/load_data.sh` downloads the raw csv files containing the tweets and their metadata. They are stored in the folder `data/raw/` (which will be created if it does not yet exist).
- The script `script/preprocessing.sh` executes all necessary preprocessing steps, including a creation of labels and splitting the data set.
- The script `script/feature_extraction.sh` takes care of feature extraction.
- The script `script/dimensionality_reduction.sh` takes care of dimensionality reduction.
- The script `script/classification.sh` takes care of training and evaluating a classifier.
- The script `script/application.sh` launches the application example.

## Features visualization

The data features can be visualized by running the script `visualizations.py` which creates a `features_visualization` folder to store the images. This script allows you to explore the relations between different data features and to do feature selection accordingly.

The script's optional parameters are as follows:
- `-d` or `--default_feat_visualizations` creates all the default visualizations and saves the images in the `features_visualization` folder.
- `-v` or `--features_variance` generates the csv data variance as png.
- `des` or `--data_description` generates the csv description of the data as png.
- `-mg` or `--group_means` generates png image of the features group means.
- `-fmv` or `--feature_mean_var` generates png image of the features mean and variance.
- `-fc` or `--feature_correlations` generates png images of the features correlations before and after feature selection.
- `-p` or `--pairplot_correlations` generates png image of the pairwise relationship of the features distributions in the dataset.
- `-vir` or `--tweets_virality` generates png images of the virality of tweets by feature.
- `-t` or `--time_virality` generates png images of the virality of tweets by time features.
- `-dv` or `--describe_virality` generates png images of the virality description of tweets by features.

## Preprocessing

All python scripts and classes for the preprocessing of the input data can be found in `script/preprocessing/`.

### Creating Labels

The script `create_labels.py` assigns labels to the raw data points based on a threshold on a linear combination of the number of likes and retweets. It is executed as follows:
```python -m script.preprocessing.create_labels path/to/input_dir path/to/output.csv```
Here, `input_dir` is the directory containing the original raw csv files, while `output.csv` is the single csv file where the output will be written.
The script takes the following optional parameters:
- `-l` or `--likes_weight` determines the relative weight of the number of likes a tweet has received. Defaults to 1.
- `-r` or `--retweet_weight` determines the relative weight of the number of retweets a tweet has received. Defaults to 1.
- `-t` or `--threshold` determines the threshold a data point needs to surpass in order to count as a "viral" tweet. Defaults to 50.

### Classical Preprocessing

The script `run_preprocessing.py` is used to run various preprocessing steps on the raw data, producing additional columns in the csv file. It is executed as follows:
```python -m script.preprocessing.run_preprocessing path/to/input.csv path/to/output.csv```
Here, `input.csv` is a csv file (ideally the output of `create_labels.py`), while `output.csv` is the csv file where the output will be written.
The preprocessing steps to take can be configured with the following flags:
- `-p` or `--punctuation`: A new column "tweet_no_punctuation" is created, where all punctuation is removed from the original tweet. (See `script/preprocessing/punctuation_remover.py` for more details)
- '-t' or '--tokenize' : Tokenize the given column and performs stop word removal, punctuation removal stemmzitation (can be spe;cified by "--tokenize_imput"", defaut = "tweet"), and create a new column cleaned_tweet which is preprocessed and cleaned data.
Moreover, the script accepts the following optional parameters:
- `-e` or `--export` gives the path to a pickle file where an sklearn pipeline of the different preprocessing steps will be stored for later usage.

### Splitting the Data Set

The script `split_data.py` splits the overall preprocessed data into training, validation, and test set. It can be invoked as follows:
```python -m script.preprocessing.split_data path/to/input.csv path/to/output_dir```
Here, `input.csv` is the input csv file to split (containing a column "label" with the label information, i.e., `create_labels.py` needs to be run beforehand) and `output_dir` is the directory where three individual csv files `training.csv`, `validation.csv`, and `test.csv` will be stored.
The script takes the following optional parameters:
- `-t` or `--test_size` determines the relative size of the test set and defaults to 0.2 (i.e., 20 % of the data).
- `-v` or `--validation_size` determines the relative size of the validation set and defaults to 0.2 (i.e., 20 % of the data).
- `-s` or `--seed` determines the seed for intializing the random number generator used for creating the randomized split. Using the same seed across multiple runs ensures that the same split is generated. If no seed is set, the current system time will be used.

### Grid
The preprocessing steps described on this section can be also executed on the grid. To do so use `qsub preprocessing.sge`


## Feature Extraction

All python scripts and classes for feature extraction can be found in `script/feature_extraction/`.

The script `extract_features.py` takes care of the overall feature extraction process and can be invoked as follows:
```python -m script.feature_extraction.extract_features path/to/input.csv path/to/output.pickle```
Here, `input.csv` is the respective training, validation, or test set file created by `split_data.py`. The file `output.pickle` will be used to store the results of the feature extraction process, namely a dictionary with the following entries:
- `"features"`: a numpy array with the raw feature values (rows are training examples, columns are features)
- `"feature_names"`: a list of feature names for the columns of the numpy array
- `"labels"`: a numpy array containing the target labels for the feature vectors (rows are training examples, only column is the label)

The features to be extracted can be configured with the following optional parameters:
- `-c` or `--char_length`: Count the number of characters in the "tweet" column of the data frame. (see script/feature_extraction/character_length.py)
- `-m` or `--month_tweet`: Extracts from the "date" column the month the tweet was posted. (see script/feature_extraction/month_tweet.py)
- `-p` or `--contain_photo`: Extracts the number of photos included in the tweet. (see script/feature_extraction/contain_photo.py)
- `-w` or `--contain_website`: Extracts the number of websites included in the tweet. (see script/feature_extraction/contain_website.py)
- `-t` or `--tfidf_vector`: Extracts tfidf for the tweets. (see script/feature_extraction/tfidf_features.py)
- `-d` or `--time_hour`: Extracts the hour tweet was posted. (see script/feature_extraction/feature_hour.py)
- `-n` or `--num_hashtags`: Retrieves the number of hashtags per tweet. (see script/feature_extraction/feature_hashtag.py)

Moreover, the script support importing and exporting fitted feature extractors with the following optional arguments:
- `-i` or `--import_file`: Load a configured and fitted feature extraction from the given pickle file. Ignore all parameters that configure the features to extract.
- `-e` or `--export_file`: Export the configured and fitted feature extraction into the given pickle file.

### Grid
To perform the feature extraction process on the Grid, use `qsub feature_extraction.sge`

## Dimensionality Reduction

All python scripts and classes for dimensionality reduction can be found in `script/dimensionality_reduction/`.

The script `reduce_dimensionality.py` takes care of the overall dimensionality reduction procedure and can be invoked as follows:

```python -m script.dimensionality_reduction.reduce_dimensionality path/to/input.pickle path/to/output.pickle```
Here, `input.pickle` is the respective training, validation, or test set file created by `extract_features.py`. 
The file `output.pickle` will be used to store the results of the dimensionality reduction process, containing `"features"` (which are the selected/projected ones) and `"labels"` (same as in the input file).

### Mutual Information

The dimensionality reduction method to be applied can be configured with the following optional parameters:
- `-m` or `--mutual_information`: Select the `k` best features (where `k` is given as argument) with the Mutual Information criterion

Moreover, the script support importing and exporting fitted dimensionality reduction techniques with the following optional arguments:
- `-i` or `--import_file`: Load a configured and fitted dimensionality reduction technique from the given pickle file. Ignore all parameters that configure the dimensionality reduction technique.
- `-e` or `--export_file`: Export the configured and fitted dimensionality reduction technique into the given pickle file.

Finally, if the flag `--verbose` is set, the script outputs some additional information about the dimensionality reduction process.

### PCA
To use this dimensionality reduction method execute the following:
- `-p` or `--pca`: Executes Principal Component Analysis given a number of components. 
- `-f` or `--export_plot`: Path to save PCA plots.

### Grid
To perform dimensionality reduction on the grid use `qsub dimensionality_reduction.sge`

## Classification

All python scripts and classes for classification can be found in `script/classification/`.

### Train and Evaluate a Single Classifier

The script `run_classifier.py` can be used to train and/or evaluate a given classifier. It can be executed as follows:
```python -m script.classification.run_classifier path/to/input.pickle```
Here, `input.pickle` is a pickle file of the respective data subset, produced by either `extract_features.py` or `reduce_dimensionality.py`. 

By default, this data is used to train a classifier, which is specified by one of the following optional arguments:
- `-m` or `--majority`: Majority vote classifier that always predicts the majority class.
- `-b` or `--m_naive_Bayes`: Executes Multinomial Naive Bayes. 

The classifier is then evaluated, using the evaluation metrics as specified through the following optional arguments:
- `-a`or `--accuracy`: Classification accurracy (i.e., percentage of correctly classified examples).


Moreover, the script support importing and exporting trained classifiers with the following optional arguments:
- `-i` or `--import_file`: Load a trained classifier from the given pickle file. Ignore all parameters that configure the classifier to use and don't retrain the classifier.
- `-e` or `--export_file`: Export the trained classifier into the given pickle file.

Finally, the optional argument `-s` or `--seed` determines the seed for intializing the random number generator (which may be important for some classifiers). 
Using the same seed across multiple runs ensures reproducibility of the results. If no seed is set, the current system time will be used.

## Application

All python script for the application demo can be found in `script/application/`.

The script `application.py` provides a simple command line interface, where the user is asked to type in their prospective tweet, which is then analyzed using the trained ML pipeline.
The script can be invoked as follows:
```python -m script.application.application path/to/preprocessing.pickle path/to/feature_extraction.pickle path/to/dimensionality_reduction.pickle path/to/classifier.pickle```
The four pickle files correspond to the exported versions for the different pipeline steps as created by `run_preprocessing.py`, `extract_features.py`, `reduce_dimensionality.py`, and `run_classifier.py`, respectively, with the `-e` option.

