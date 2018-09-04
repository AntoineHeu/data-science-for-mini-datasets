# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 17:01:10 2018

@author: a.heude
"""

import csv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import sys
print(sys.executable)
import warnings

import pickle

if not sys.warnoptions:
    warnings.simplefilter("ignore")


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix,accuracy_score


import matplotlib.font_manager
matplotlib.rcParams.update({'font.size': 20})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

data_field=pd.read_csv('data_remscan.csv',sep=';',encoding = "ISO-8859-1")


def feature_generation_stat(dataframe,ID,target,list_elem,name=''):
    
    #function that generates basic stats on a series of measurements- these stats are used as features for a predictive model
    ##########
    #PARAMETERS
    #dataframe is contains the data
    #ID is the name of the column that reference the samples
    #target is the target we'll be trying to predict
    #list_elem is a list of the name of the rows that basic stats will be applied on
    #name personnalize the eventual saving of figures
    
    features=pd.DataFrame()
    for elem in list_elem:
        features=pd.concat([features,dataframe[elem]],axis=1)
        
    features[ID]=dataframe[ID]
    
    for row in range(len(dataframe)):
        features['mean']=features[list_elem].mean(axis=1)
        features['max']=features[list_elem].max(axis=1)
        features['min']=features[list_elem].min(axis=1)
        features['median']=features[list_elem].median(axis=1)
        #More features can be used if the dataset allows it
    
    features.drop(list_elem,axis=1,inplace=True)
    features[target]=dataframe[target]
    
    #features contains predictors and the target - as well as the ID of samples fo ease of interpretation
    return features


def Lasso_eval(class_targ,elem,data,pred=[],name='',multinomial=None):
    
    #This function computes a Lasso Regression in order to find out the dominants parameters in a linear prediction of a target
    #Also performs the prediction on the train set and saves it in order to compare with classification algorithms

        ###############
        #PARAMETERS
        #class_targ = the target to predict = name of a column of data
        #elem = string of the name of the target to predict
        #data = pd.Dataframe of the tabular input data, containing target and predictors
        #pred = list of predictors to test
        #multinomial is used when the function is called from the Logit_Reg function
        ###############
    
    #graphs will be saved on the directory this script is launched from
    
    'built in of predictors'
    clean_up=pd.DataFrame()
    for i in pred:
        clean_up=pd.concat([clean_up,data[i]],axis=1)
        clean_up_scale=clean_up
    clean_up=pd.concat([clean_up,data[class_targ]],axis=1)
    if multinomial!=None:
        clean_up=pd.concat([clean_up,data[multinomial]],axis=1)
    
    clean_up.dropna(inplace=True)
    clean_up_scale.dropna(inplace=True)
    

    
    predictors=pd.DataFrame()
    for i in pred:
        predictors=pd.concat([predictors,clean_up[i]],axis=1)
    list_pred=list(predictors.columns)
    
    'transform lithologie to code'
    if 'litho' in predictors:
        dum_lithos=pd.get_dummies(clean_up['litho'])
        predictors=pd.concat([predictors,dum_lithos],axis=1)
        clean_up["litho"] = clean_up["litho"].astype('category')
        predictors['litho'] = clean_up['litho'].cat.codes
        predictors.pop('litho')
    
    predictors.index = pd.RangeIndex(len(predictors.index))
    print(predictors)
    scaler=StandardScaler()
    scaler.fit(clean_up_scale)
    predictors=scaler.transform(predictors)
    predictors=pd.DataFrame(predictors)
    predictors.columns=list_pred       
            
    'built in of the target'
    target=pd.DataFrame()
    target=pd.concat([target,clean_up[class_targ]],axis=1)
    target = target.reset_index(drop=True)
    print(target)
    print(predictors)
    
    if multinomial != None:
        target_multinomial=pd.DataFrame()
        target_multinomial=pd.concat([target_multinomial,clean_up[multinomial]],axis=1)
        target_multinomial = target_multinomial.reset_index(drop=True)
    else:
        target_multinomial=None

    'on peut imposer un random_state ici'
    stratified=RepeatedKFold(n_splits=3,n_repeats=15)

    
    'Function that evaluate the importance of predictors'
    global lasso_coef
    
    'grid search for lasso'
    param_lass={'alpha': [1,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0001, 0.00005, 0.00001]}
    lasso=Lasso(normalize=False)
    grid_s_lass=GridSearchCV(lasso,param_lass,cv=stratified, scoring='neg_median_absolute_error')
    grid_s_lass.fit(predictors, target)

    lasso_opt=Lasso(alpha=grid_s_lass.best_params_['alpha'],normalize=False)
    model=lasso_opt.fit(predictors,target)
    
    flag='finalized_model_Lasso_%s' % (elem+'_'+name)
    filename = '%s.sav' %flag
    pickle.dump(model, open(filename,'wb'))
    
    #Obtention des coefficients
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
    
    #Returns additionnal elements that could be useful for further investigations
    return loaded_model,scaler,predictors,target,target_multinomial



def Logit_reg(targ,elem,data,pred=[],multinomial=None):
    
    'Function that performs Logit regression models as well as continious predictions (specific to problematics where continious target can also make up classes)'
    '___'
    'This function is associated to a lasso_regression that performs a continious prediction of concentrations'
    'continous predictions are compared to the classification / in order to choose the best models'
    '___'
    
        ##############
        #PARAMETERS#
        #targ = the target continious variable
        #elem = name of the element predicted / used for names of saved figures and files
        #data = pd.DataFrame() of the data
        #pred = list of predictors (name of columns for data)
        #multinomial = name of the column (for data) containing the class target used to fit the classification model - classes are based on target continious variables
        ##############
    
    #graphs will be saved on the directory this script is launched from
    
    #Use Lasso fonction to assess the importance of predictors and built up continuous predictions
    lasso_model,scaler,predictors,target,target_multinomial=Lasso_eval(targ,elem,data,pred,'',multinomial)
    print(target_multinomial)
    
    #no need to work with scaled data for logit regression
    predictors=scaler.inverse_transform(predictors)
    stratified=RepeatedKFold(n_splits=3,n_repeats=15)
    
    print('the predictors used to fit the model are the following:')
    print(predictors)
        
        
    '____________LOGIT____________'
    
    'Cross validation and hyperparameter tuning'

    'preprocessing the data'
    logit_reg=LogisticRegression()

    'setting hyper parameters grid'
    param_grid = {'C': [1e-25,1e-12,1e-08,0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.01,0.1,1,5,10,30,50,100,200,300,400,500,600,700,800,900,1000]}

    
    'metrics used is accuracy / should be modified to be adapted to peculiar situations'
    grid_s_cv=GridSearchCV(logit_reg,param_grid,cv=stratified,scoring='accuracy')
    grid_s_cv.fit(predictors,target_multinomial)

    print("Tuned Logistic Regression Parameters: {}".format(grid_s_cv.best_params_)) 
    print("Best score is {}".format(grid_s_cv.best_score_))
    
    if any( [grid_s_cv.best_params_['C']==max(param_grid['C']), grid_s_cv.best_params_['C']==min(param_grid['C'])]):
        print("WARNING: It may be advised to extend the research perimeter for the parameters as max or min range have been reached")
    
    'Built up the pipeline'
    steps = [
        ('logit', LogisticRegression(multi_class='multinomial', solver='newton-cg' ,C=grid_s_cv.best_params_['C']))]

    pipeline=Pipeline(steps)
            

    
    def CV_optim_and_ROC():
        'cross validation automated - can"t get parameters so had to do it by hand'
        #Exceptionaly no train and test sets are used here because of a limited dataset
        #With enough data (50 ?) - train and test sets procedures are recommended

        l_precision=[]
        l_recall=[]
        l_fscore=[]
        l_accuracy=[]
        
        global Saved_param, Saved_intercept
        
        LogReg_model=pipeline.fit(predictors,target_multinomial)
        y_pred=pipeline.predict(predictors)
        
        flag='finalized_model_LogitReg_%s' % (elem)
        filename = '%s.sav' %flag
        pickle.dump(LogReg_model, open(filename,'wb'))
    
        print(y_pred)
        y_real=target_multinomial[multinomial].tolist()
        print(y_real)
        precision,recall,fscore,support=precision_recall_fscore_support(y_real,y_pred,average='macro')
         
        l_precision.append(precision)
        l_recall.append(recall)
        l_fscore.append(fscore)
        l_accuracy.append(accuracy_score(y_real, y_pred))

                
        list_classification=pd.Series((v for v in y_pred))

        plt.clf()        
        
        print("classifaction on the dataset: {}".format(classification_report(y_real, y_pred)))
        print("confusion matrix on the dataset: {}".format(confusion_matrix(y_real, y_pred)))
        cm=pd.DataFrame(confusion_matrix(y_real, y_pred), columns=list(list_classification.astype('category').cat.categories),index=list(list_classification.astype('category').cat.categories))
        fig=sns.heatmap(cm,annot=True,robust=True)
        fig_=fig.get_figure()
        fig_.savefig('classification for the dataset - LOGIT.png')
        plt.clf()
        print("score F1 maximal obtenu: {}".format(max(l_fscore)))
        
        df=pd.DataFrame()
        df=pd.concat([df,pd.DataFrame(y_real),pd.DataFrame(y_pred)],axis=1)
        df.columns=['y_real','y_pred']
        print(df)

        return df,LogReg_model
    
    df,LogReg_model=CV_optim_and_ROC()
    
    
    #Conduct the classification prediction on the entire dataset and store it in a .csv file
    filename='prediction_LogReg_%s.csv' %(elem)
    with open(filename, 'w',newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for lignes in range(0,len(data)):
            
            if data[targ].isnull()[lignes]==True:
                predictors_csv=[]
                predictors_csv_cat=[]
                for columns in list(data.columns):
                    if columns in pred:
                        predictors_csv.append(data[columns][lignes])

                if True in np.isnan(predictors_csv):
                    print('ligne sautée')
                else:
                    predictors_csv=np.array(predictors_csv).reshape(1, -1)
                    predictors_csv=pd.DataFrame(predictors_csv)
                    predictors_csv.columns=pred

                    for columns in pred:
                        predictors_csv_cat.append(predictors_csv[columns][0])
                        
                    prediction = LogReg_model.predict(np.array(predictors_csv).reshape(1,-1))
                    
                    filewriter.writerow([data['echantillon'][lignes], prediction])
    
    #Returns additionnal elements that could be useful for further investigations
    return df,lasso_model,predictors,target,target_multinomial,LogReg_model


def data_preprocessing(data,cat='ISDI'):
    
    
    "Exemple of classification from values of a specific component"
    #########
    #PARAMETERS
    #data is the pd.DataFrame object containing the data
    #Cat is the type of classification
    #########
    
    #Here, the example used is the classification of soils according to the French legislation
    #For Hydrocarbons (hct) limits are 500 (ISDI) and 5000 (ISDND)

    TOT=data                    
                    
    if cat=='hct':
        
        "Input of the classification - creation of a dictionnary with names and ISDI/ISDND values"
        Lixi_elements=np.array(['agrolab_hct'])
        ISDI_values=np.array([500])
        ISDND_values=np.array([5000])
    
        _=zip(ISDI_values,ISDND_values)
        __=zip(Lixi_elements,_)
        ___=list(__)
        print(___)
    
        Vseuil=dict(___)
    
        "Input of the classification - classification function"
        def classification(input_data,elem):
            Class=[]
            for col in input_data.columns.values:
                if col==elem:
                    for row in input_data[col]:
                        if math.isnan(row) ==True:
                            Class.append('ISDI')
                        if row<=Vseuil[elem][0]:
                            Class.append('ISDI')
                        if row>Vseuil[elem][0] and row<=Vseuil[elem][1]:
                            Class.append('ISDND')
                        if row>Vseuil[elem][1]:
                            Class.append('biocentre')

                            
                    
            input_data['class-'+elem]=Class
            return(input_data['class-'+elem])
        
        classification(TOT,'agrolab_hct')
        
    #Returns the dataframe with additionnal columns for the classification
    return TOT


def check_regression_vs_classification(model,predicteur,target_reg,df_classification,cat):
    
    #Function that combine the prediction from a direct classification algorithm and a regression which results are classified
    ###########
    #PARAMETERS
    ###########
    #model=regression model
    #predicteur=predictors
    #target_reg=real_values
    #df_classification=dataframe containing the classification from the classification algorithm
    #cat is used for the data_processing
    
    ###########
    
    reg_prediction=model.predict(predicteur)
    df=pd.DataFrame()
    df=pd.concat([df,target_reg,pd.DataFrame(reg_prediction)],axis=1)
    df.columns=['reg_real','agrolab_hct']
    print(df)

    df_reg_class=data_preprocessing(df, cat)
    df_reg_class.columns=['reg_real','reg_prediction','reg_prediction_class']
    print(df_reg_class)
    
    df_classification=pd.concat([df_classification,df_reg_class['reg_prediction_class']],axis=1)
    
    #Returns the dataframe
    return df_classification
    

# =============================================================================
# df_classification,Lasso_model,predictors,target_reg,target_multinomial,LogReg_model = Logit_reg('agrolab_hct','HCT',features,['mean','max','min','median'],'class-agrolab_hct')
# bilan_final=check_regression_vs_classification(Lasso_model,predictors,target_reg,df_classification,'hct')
# 
# =============================================================================
