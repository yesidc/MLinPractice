# Documentation for predicting tweets virality using ML

In this project we worked with a data set composed of tweets about data analysis, data science, and data visualization.
The main goal of this project was to analyse the data features and predict tweets virality.
For easy applicability, a console-based application for tweet classification is provided. This allows anyone to type a 
certain tweets into the console and have the model predict whether the tweet will become viral or not.

In the following documentation we provide specifics about the processes involved during the model implementation.


## Evaluation

### Design Decisions

Which evaluation metrics did you use and why? 
Which baselines did you use and why?

### Results

How do the baselines perform with respect to the evaluation metrics?

### Interpretation

Is there anything we can learn from these results?

## Preprocessing

I'm following the "Design Decisions - Results - Interpretation" structure here,
but you can also just use one subheading per preprocessing step to organize
things (depending on what you do, that may be better structured).

### Design Decisions

Which kind of preprocessing steps did you implement? Why are they necessary
and/or useful down the road?

### Results

Maybe show a short example what your preprocessing does.

### Interpretation

Probably, no real interpretation possible, so feel free to leave this section out.

## Feature Extraction

Again, either structure among decision-result-interpretation or based on feature,
up to you.

### Design Decisions

Which features did you implement? What's their motivation and how are they computed?

### Results

Can you say something about how the feature values are distributed? Maybe show some plots?

### Interpretation

Can we already guess which features may be more useful than others?

## Dimensionality Reduction

If you didn't use any because you have only few features, just state that here.
In that case, you can nevertheless apply some dimensionality reduction in order
to analyze how helpful the individual features are during classification

### Design Decisions

Which dimensionality reduction technique(s) did you pick and why?

### Results

Which features were selected / created? Do you have any scores to report?

### Interpretation

Can we somehow make sense of the dimensionality reduction results?
Which features are the most important ones and why may that be the case?

## Classification

### Design Decisions

Which classifier(s) did you use? Which hyperparameter(s) (with their respective
candidate values) did you look at? What were your reasons for this?

#### Multinomial Naive Bayes:
We implemented Multinomial Naive Bayes (which implements the Naive Bayes algorithm) since, according to <a href="https://scikit-learn.org/stable/modules/naive_bayes.html">sklearn</a> documentation, besides it being a classic naive Bayes variant used in text classification; it also performes well with tf-idf vectors (which is one of the features we extracted)

### Results

The big finale begins: What are the evaluation results you obtained with your
classifiers in the different setups? Do you overfit or underfit? For the best
selected setup: How well does it generalize to the test set?

### Interpretation

Which hyperparameter settings are how important for the results?
How good are we? Can this be used in practice or are we still too bad?
Anything else we may have learned?


