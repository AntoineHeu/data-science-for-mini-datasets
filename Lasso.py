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
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.linear_model import Lasso



import matplotlib.font_manager
matplotlib.rcParams.update({'font.size': 20})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)



def Lasso_Eval(class_targ,elem,data,pred=[],name=''):
    
    #function that performs an Lasso modelisation from data as a pd.Dataframe()
    #####################################################################################
    #class_targ gives the target - parameter to predict
    #elem corresponds to the name of the target - will appear on graphs and saved files
    #pred is a list of the predictors used to build the model
    #name = accessory parameter to further personalize names of saved files / graphs
    
    'built-in of predictors'
    #clean_up cleans up the dataframe of predictors + target
    #erases rows with NaN values
    clean_up=pd.DataFrame()
    for i in pred:
        clean_up=pd.concat([clean_up,data[i]],axis=1)
        clean_up_scale=clean_up
    clean_up=pd.concat([clean_up,data[class_targ]],axis=1)
    
    clean_up.dropna(inplace=True)
    clean_up_scale.dropna(inplace=True)

    #predictors is a dataframe of predictors
    predictors=pd.DataFrame()
    for i in pred:
        predictors=pd.concat([predictors,clean_up[i]],axis=1)
    list_pred=list(predictors.columns)
    predictors.index = pd.RangeIndex(len(predictors.index))
    
    #scale dataset - scaling on the predictors - all rows considered
    scaler=StandardScaler()
    scaler.fit(clean_up_scale)
    predictors=scaler.transform(predictors)
    predictors=pd.DataFrame(predictors)
    predictors.columns=list_pred       
            
    'built-in of the target'
    target=pd.DataFrame()
    target=pd.concat([target,clean_up[class_targ]],axis=1)
    target = target.reset_index(drop=True)
    
    'check on the data'
    print(target)
    print(predictors)

    'define stratification used for cross-validation of hyper parameters'
    stratified=RepeatedKFold(n_splits=3,n_repeats=15)
    
    'grid search for lasso'
    param_lass={'alpha': [1,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0001, 0.00005, 0.00001]}
    lasso=Lasso(normalize=False)
    grid_s_lass=GridSearchCV(lasso,param_lass,cv=stratified, scoring='neg_median_absolute_error')
    grid_s_lass.fit(predictors, target)
    
    #Best params are kept to build the model
    lasso_opt=Lasso(alpha=grid_s_lass.best_params_['alpha'],normalize=False)
    model=lasso_opt.fit(predictors,target)
    
    #saving the model for re-use
    flag='finalized_model_Lasso_%s' % (elem+'_'+name)
    filename = '%s.sav' %flag
    pickle.dump(model, open(filename,'wb'))
    
    #coefficient plot saved to directory
    lasso_coef=model.coef_
    figure1= plt.figure(figsize = (10,10))
    print("Tuned Lasso Parameters: {}".format(grid_s_lass.best_params_))
    print("Best score is {}".format(grid_s_lass.best_score_))
    print("Lasso coefficients per predictor: {}".format(lasso_coef))
    plt.plot(range(len(predictors.columns)), lasso_coef)
    plt.xticks(range(len(predictors.columns)), predictors.columns.values)
    plt.xticks(rotation=90)
    plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
    #plt.margins(0.02)
    plt.xlabel('prédicteurs')
    plt.ylabel('importance des prédicteurs')
    plt.title(elem)
    figure1.savefig("coefficients_%s.png" %(elem+'_'+name))
    plt.clf()
    
    loaded_model = pickle.load(open(filename, 'rb'))    
    
    return loaded_model#,scaler,predictors,target