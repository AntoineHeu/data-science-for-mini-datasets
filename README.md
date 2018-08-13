# Branch Prediction on skewed datasets
This branch contains the functions and code used in the prediction of skewed datasets.
This is based on the prediction of continuous data such as concentrations of metals in soils for example

# Architecture
Edited on spyder via Anaconda.
Based on the importation of csv files -> as DataFrame objects.

1.anomaly detection on the dataset to train 3 different types of anomaly detection algorithm
2. reading through the file containing the data - prediction on samples if inliers or outliers
3. if sample is an inlier specific model (ElastiicNet) fitted and used for the prediction
4. if sample is an outlier the global model (ELasticNet) is used for the prediction
5. storing data in new csv - samples identified with here "drillhole_ID"
