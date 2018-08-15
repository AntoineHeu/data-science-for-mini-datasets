# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:01:10 2018

@author: a.heude
"""

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


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM



import matplotlib.font_manager
matplotlib.rcParams.update({'font.size': 20})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

def selection_pred_pos_correl(dataset,class_target):
    
    #function that allows to select predictors positively corrolated to the target
    #returns a dictionnary with predictors and their corrolation coefficient
    ##############################################################################
    #dataset = dataframe with predictors and the target
    #class_targ= name of the target in the dataset
    
    dict_result={}
    list_correlations=[]
    for elem in list(dataset.columns):
        dict_result[elem]=np.corrcoef(np.array(dataset[elem]), np.array(class_target).reshape(1,-1))[0,1]
        list_correlations.append(np.corrcoef(np.array(dataset[elem]), np.array(class_target).reshape(1,-1)))
    
    return dict_result


def anomaly_detection(class_targ,elem,data,pred=[]):
    
    ######
    #function that trains 3 different types of onomaly detection models: classes 1 as inliers and -1 as outliers
    #models used here: Empirical Covariance, Robust covariance and One Class SVM
    #No test set used here - this script was made in the context of very small datasets
    #Input data needs to be skewed - might be worth a check with basic boxplot for example
    ######
    #####################################################################################
    #class_targ gives the target - parameter to predict
    #elem corresponds to the name of the target - will appear on graphs and saved files
    #data is the dataframe used
    #pred is a list of the predictors used to build the model
    ####################################################################################
    
    #the threshold defining outliers and inliers is calculated as the reference in box-plot: upper-whiskers
    df_reference=data[class_targ].dropna(inplace=False)    
    upper_quartile = np.percentile(np.array(df_reference), 75)
    lower_quartile = np.percentile(np.array(df_reference), 25)
    iqr = upper_quartile - lower_quartile
    upper_whisker = df_reference[df_reference<=upper_quartile+1.5*iqr].max()
    
    #upper_whisker of the distribution is the threshold retained here - can be adapted with a more extensive study of the distribution
    print(upper_whisker)
    
    #proportion of outliers in the dataset
    prop=0
    for i in range(len(df_reference)):
        if df_reference.iloc[i]>upper_whisker:
            print(df_reference.iloc[i])
            prop=prop+1
    proportion=prop/len(df_reference)
        
    #construction of predictors
    #clean_up only retains fully informed rows
    clean_up=pd.DataFrame()
    for i in pred:
        clean_up=pd.concat([clean_up,data[i]],axis=1)
    clean_up_scale=clean_up
    clean_up=pd.concat([clean_up,data[class_targ]],axis=1)
    
    clean_up.dropna(inplace=True)
    clean_up_scale.dropna(inplace=True)
    
    #reset the index to avoid confusion
    clean_up.index = pd.RangeIndex(len(clean_up.index))
    clean_up_scale.index = pd.RangeIndex(len(clean_up_scale.index))
    
    #separation of outliers from the dataset - this will matter as some methods need a trainset free from outliers
    clean_up_train=pd.DataFrame(columns=list(clean_up.columns),index=range(len(clean_up)-prop))
    clean_up_target=pd.DataFrame(columns=list(clean_up.columns),index=range(prop))
    target_class=pd.DataFrame(columns=['target'],index=range(len(clean_up)))

    
    i=0
    j=0
    l=0
    for rows in range(len(clean_up)):
        if clean_up.iloc[rows,-1]<=upper_whisker:
            clean_up_train.loc[i]=np.array(clean_up.iloc[rows,:])
            target_class.loc[l]=1
            i=i+1
            l=l+1
        else:
            clean_up_target.loc[j]=np.array(clean_up.iloc[rows,:])
            target_class.loc[l]=-1
            j=j+1
            l=l+1
    
    #final built-in of predictors
    predictors=pd.DataFrame()
    predictors_train=pd.DataFrame()
    predictors_targ=pd.DataFrame()
    for i in pred:
        predictors=pd.concat([predictors,clean_up[i]],axis=1)
        predictors_train=pd.concat([predictors_train,clean_up_train[i]],axis=1)
        predictors_targ=pd.concat([predictors_targ,clean_up_target[i]],axis=1)
    list_pred=list(predictors.columns)
    
    #fit the scaler on all the auxilliary data available
    scaler=StandardScaler()
    scaler.fit(clean_up_scale)
    
    #transform the data used to train the model with the scaler
    predictors=scaler.transform(predictors)
    predictors=pd.DataFrame(predictors)
    predictors.columns=list_pred
    print(predictors)
    
    predictors_train=scaler.transform(predictors_train)
    predictors_train=pd.DataFrame(predictors_train)
    predictors_train.columns=list_pred

    predictors_targ=scaler.transform(predictors_targ)
    predictors_targ=pd.DataFrame(predictors_targ)
    predictors_targ.columns=list_pred
    
    #built-in of the target
    target=pd.DataFrame()
    target=pd.concat([target,clean_up[class_targ]],axis=1)
    target = target.reset_index(drop=True)
    print(clean_up[class_targ])

    target_class=target_class.reset_index(drop=True)
        
    
    #retain only positively corrolated predictors to the target
    #a threshold of 0.3 is arbitrarly selected here but can be adapted to the situation
    list_correlation=selection_pred_pos_correl(predictors,target)
    print(list_correlation)
    list_def_pred=[]
    for keys in list_correlation.keys():
        if list_correlation[keys]>=0.3:
            list_def_pred.append(keys)
            
    for elements in list(predictors_train.columns):
        if elements not in list_def_pred:
            predictors_train.drop(elements, axis=1,inplace=True)
            predictors_targ.drop(elements, axis=1,inplace=True)
            predictors.drop(elements, axis=1,inplace=True)
            
    principalDf=predictors_train
    targetDf=predictors_targ


    # Define "classifiers" to be used
    # Again this can be adapted
    classifiers = {
        "Empirical Covariance": EllipticEnvelope(support_fraction=1.,
                                                 contamination=0),
        "Robust Covariance (Minimum Covariance Determinant)":
        EllipticEnvelope(contamination=0),
        "OCSVM": OneClassSVM(nu=proportion)}

    gammas_cv = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,8,10]
    
    ####
    #fitting of the models
    ####
    df_results=pd.DataFrame()
    df_result_bis=pd.DataFrame()
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        if clf_name=="OCSVM":
            
            #Gride search for SVM as there is one hyperparameter to fine-tune
            stratified=RepeatedStratifiedKFold(n_splits=3,n_repeats=10)
            search_grid={'gamma':gammas_cv}
            grid_s_cv=GridSearchCV(classifiers['OCSVM'],search_grid,cv=stratified,scoring='f1_macro',n_jobs=1)
            grid_s_cv.fit(predictors,target_class)
            
            CV_best_params_=grid_s_cv.best_params_
            print(CV_best_params_)
            print(grid_s_cv.best_score_)
            
            #optimal parameters are preserved and used to fit the model
            clf=OneClassSVM(nu=proportion,gamma=CV_best_params_['gamma'])
            clf.fit(predictors,target_class)
            

            
            #store fitted values
            df_results=pd.concat([df_results,pd.DataFrame(clf.predict(targetDf))],axis=1)
            df_result_bis=pd.concat([df_result_bis,pd.DataFrame(clf.predict(predictors))],axis=1)
                            
        else:
            #fitting the data not corrumpted by outliers
            clf.fit(principalDf)
            
            #returns fitted values
            df_results=pd.concat([df_results,pd.DataFrame(clf.predict(targetDf))],axis=1)
            df_result_bis=pd.concat([df_result_bis,pd.DataFrame(clf.predict(predictors))],axis=1)
            
        #saving the models
        name="saved_model_%s"%(clf_name + '_' + class_targ)
        file="%s.pkl"%name
        pickle.dump(clf, open(file,'wb'))
            
    #check on the results
    print(df_results)
    print(df_result_bis)
    data_train=clean_up_train
    
    #returns various variables and elements that will be used in the prediction for skewed datasets
    return classifiers,list_def_pred,scaler,data_train,clean_up,target