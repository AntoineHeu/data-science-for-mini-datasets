# Branch Prediction on skewed datasets
This branch contains the functions and code used in the prediction of skewed datasets.
This is based on the prediction of continuous data such as concentrations of metals in soils for example

# How to
You need python.
- You can just save the .py file from the master branch, import your data as csv and the code will work as you launch it. 
- elements will be stored on the directory you stored the .py file in.

# Architecture
Edited on spyder via Anaconda.
Based on the importation of csv files -> as DataFrame objects.
- anomaly detection on the dataset to train 3 different types of anomaly detection algorithm
- reading through the file containing the data - prediction on samples if inliers or outliers
- if sample is an inlier specific model (ElasticNet) fitted and used for the prediction
- if sample is an outlier the global model (ELasticNet) is used for the prediction
- final predicted value is a weighted average (weights are a probability of being either an outlier or an inlier given the results of the 3 anomaly detection models fitted)
- storing data in new csv - samples identified with here "drillhole_ID"

> steps of the entire program has been fragmented here for a better visibility. predict_csv.py however needs to be related to the other fragments of the code to work.

# Philosophy
The idea behind the code is that:
- we have a limited dataset with very few points we can base our model one
- we won't "sacrifice" data with a train/test set - except for the fine hyper parameters tuning
- multiplying the number of fitted anomaly-detection models might just avoid drifting away from coherent predictions
- because we are manipulating small datsets and classic metrics are not that representative - checks on the prediction and models are made with cross-validations (stored on directory).
- prediction of the target value for outliers is the linear model (ElasticNet) fitted to the entire dataset
- prediction of the target value for inliers is the lineat model (ElasticNet) fitted to the inliers elements

> In the context of the prediction of concentrations in soils I usually consider the result coherent if the histogram of the full set of values (raw + modelized) is similar to the histogram of the raw values
