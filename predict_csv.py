# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:01:10 2018

@author: a.heude
"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
print(sys.executable)
import warnings

import pickle

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import matplotlib.font_manager
matplotlib.rcParams.update({'font.size': 20})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)


def scale_dataset(data,scaler):
    
    #function that scale a dataset from a scaler object
    #return the scaled dataset
    
    l_col=list(data.columns)
    data=scaler.transform(data)
    data=pd.DataFrame(data)
    data.columns=l_col
    
    return data



def predict_skewed_dist(class_targ,elem,data,pred=[]):
    
    ######
    #function that performs the modelisation of continues variables from a training set skewed with numerous positive outliers
    #model used here: ElasticNet but can easily be adapted
    #No test set used here - this script was made in the context of very small datasets
    #Input data needs to be skewed
    ######
    
    ######
    #make prediction from dataset and stores the result in a new csv file
    #

    #will store the result of predictions
    df_results_outliers=pd.DataFrame()
    
    #run the anomaly detection function and save parameters for it to be used later
    classifiers,list_def_pred,scaler,data_train,clean_up,target_outliers = anomaly_detection(class_targ,elem,data,pred,pca=False)
    
    #Construction of data to predict
    a_predire=pd.DataFrame()
    for i in pred:
        a_predire=pd.concat([a_predire,data[i]],axis=1)
    l_pred=list(a_predire.columns)
    
    #clean_up and scaling of the data
    a_predire.dropna(inplace=True)
    a_predire=scaler.transform(a_predire)
    a_predire=pd.DataFrame(a_predire)
    a_predire.columns=l_pred
    
    a_predire_def=pd.DataFrame()
    for i in pred:
        if i in list_def_pred:
            a_predire_def=pd.concat([a_predire_def,a_predire[i]],axis=1)
    
    
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        name="saved_model_%s"%(clf_name + '_' + class_targ)
        file="%s.pkl"%name
        
        model=pickle.load(open(file, 'rb'))
        
        prediction=model.predict(a_predire_def)
        
        df_results_outliers=pd.concat([df_results_outliers,pd.DataFrame(prediction)],axis=1)
    print(df_results_outliers)
    
    #df_results_outliers gives the classification between outliers and inliers in the dataset to predict
    model_outliers,scale_no_use,predictors_no_use,target_no_use=ELasticNet_Eval(class_targ,elem,data,pred,'outliers')
    model_inliers,scale_no_use,predictors_no_use,target_no_use=ELasticNet_Eval(class_targ,elem,data_train,pred,'inliers')
    predictors_train=pd.DataFrame()
    predictors=pd.DataFrame()
    for i in pred:
        predictors_train=pd.concat([predictors_train,data_train[i]],axis=1)
        predictors=pd.concat([predictors,clean_up[i]],axis=1)
        
    predictors_train=scale_dataset(predictors_train,scaler)
    predictors=scale_dataset(predictors,scaler)
    
    figure1= plt.figure(figsize = (10,10))
    print(predictors_train)
    print(predictors)
    predict_outliers=model_outliers.predict(predictors)
    print(predict_outliers)
    predict_inliers=model_inliers.predict(predictors_train)
    target_inliers=data_train[class_targ]
    
    figure1= plt.figure(figsize = (10,10))
    plt.scatter(np.array(target_inliers),predict_inliers)
    plt.xlabel('concentrations mesurées')
    plt.ylabel('concentrations prédites')
    plt.title(elem)
    figure1.savefig("scatter_modelisation_%s.png" %(elem+'_inliers'))
    plt.clf()

    figure2= plt.figure(figsize = (10,10))
    plt.scatter(np.array(target_outliers),predict_outliers)
    plt.xlabel('concentrations mesurées')
    plt.ylabel('concentrations prédites')
    plt.title(elem)
    figure2.savefig("scatter_modelisation_%s.png" %(elem+'_outliers'))
    plt.clf()
    
    #Make the final prediction considering a 'probability' to be an outlier
    #Prediction on samples
    filename='prediction_%s.csv' %(elem)
    with open(filename, 'w',newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for lignes in range(0,len(data)):
            
            if data[class_targ].isnull()[lignes]==True:
                predictors_csv=[]
                predictors_csv_cat=[]
                for columns in list(data.columns):
                    if columns in pred:
                        predictors_csv.append(data[columns][lignes])

                if True in np.isnan(predictors_csv):
                    print('ligne sautée')
                else:
                    predictors_csv=scaler.transform(np.array(predictors_csv).reshape(1, -1))
                    predictors_csv=pd.DataFrame(predictors_csv)
                    predictors_csv.columns=l_pred

                    for columns in pred:
                        if columns in list_def_pred:
                            predictors_csv_cat.append(predictors_csv[columns][0])
                    categorie=[]
                    for i, (clf_name, clf) in enumerate(classifiers.items()):
                        name="saved_model_%s"%(clf_name + '_' + class_targ)
                        file="%s.pkl"%name
                        model=pickle.load(open(file, 'rb'))
                        categorie.append(model.predict(np.array(predictors_csv_cat).reshape(1,-1)))
                    count_outliers=categorie.count(-1)
                    count_inliers=categorie.count(1)

                    tot=count_outliers+count_inliers
                        
                    prediction = count_outliers/tot * model_outliers.predict(np.array(predictors_csv).reshape(1,-1)) + count_inliers/tot* model_inliers.predict(np.array(predictors_csv).reshape(1,-1))
                    
                    #prediction are written on the csv files according to the ID
                    filewriter.writerow([data['Drillhole ID'][lignes], prediction])
        
    
    return 