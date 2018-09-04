# data-science-for-mini-datasets
Applying data-science and ML concepts to small datasets is highly discutable but I would argue that even the small one can benefit from it - if the user keeps in mind that the results are to be taken with a grain of salt.
This repository aims to share with you some ML application I have been using as part of my job in the environmental industry. The code is oriented toward really small datasets: from 15 to 50 observations.

# Code
Edited on spyder via Anaconda.
Mostly based on the importation of csv files -> as DataFrame objects.
Because datasets are small a few check steps consist on having a direct observation of the data throughout the process.

# Disclaimer
This is a first approach on solving the problem of small dataset. 
- Codes here are to be taken as draft so do not hesitate to point out incoherency and improvements points

# Programs proposed
the codes presented here are:

- prediction_skewed
  - predicts continuous values
  - values to predict are separated between inliers, which have there own ElasticNet (scikitlearn) model and outliers, which have a different Elasticnet (scikitlearn) model
  - First steps is to predict if the value is either an outlier or an inlier (modelized as a probability: 3 anomaly detection algorithms are fitted so probabilities ranges in O, O.33, O.66 or 1)
  - the predicted value is a weighted average of the models for outliers and inliers - weights being the probability of being an outlier or an inlier

- Basic_visualisation
  - this codes aims to propose various function that might be usefull for exploratory data analysis and basic statistics comparisons
  
- prediction_classes
  - predicts continuous values that can be classified from thresholds (lasso)
  - direct prediction of classes (logit regression)
  - compares the result between the two. It is expected that the end purpose is to make a classification as precise as possible
  - in the logit regression a part of the script creates a .csv files that contains all the predictions for a dataset

- to be continued...

For more additionnal elements on the programs, please visit the branch associated to the program of interest
