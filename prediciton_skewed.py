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


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import ElasticNet

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM



import matplotlib.font_manager
matplotlib.rcParams.update({'font.size': 20})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)



def ELasticNet_Eval(class_targ,elem,data,pred=[],name=''):
    
    #function that performs an ElasticNet modelisation from data as a pd.Dataframe()
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
 
    'grid search for ELasticNet'
    param_ElasticNet={'alpha': [1,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0001, 0.00005, 0.00001],'l1_ratio':list(np.arange(0.0, 1.0, 0.1))}
    ElaNet=ElasticNet()
    grid_s_lass=GridSearchCV(ElaNet,param_ElasticNet,cv=stratified, scoring='neg_median_absolute_error')
    grid_s_lass.fit(predictors, target)

    #Best params are kept to build the model
    ElaNet_opt=ElasticNet(alpha=grid_s_lass.best_params_['alpha'],l1_ratio=grid_s_lass.best_params_['l1_ratio'])
    model=ElaNet_opt.fit(predictors,target)
    
    #saving the model for re-use
    flag='finalized_model_ElasticNet_%s' % (elem+'_'+name)
    filename = '%s.sav' %flag
    pickle.dump(model, open(filename,'wb'))
    
    #coefficient plot saved to directory
    lasso_coef=model.coef_
    figure1= plt.figure(figsize = (10,10))
    print("Tuned ElasticNet Parameters: {}".format(grid_s_lass.best_params_))
    print("Best score is {}".format(grid_s_lass.best_score_))
    print("ElasticNet coefficients per predictor: {}".format(lasso_coef))
    plt.plot(range(len(predictors.columns)), lasso_coef)
    plt.xticks(range(len(predictors.columns)), predictors.columns.values)
    plt.xticks(rotation=90)
    plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
    #plt.margins(0.02)
    plt.xlabel('predictors')
    plt.ylabel('relative importance of the predictors used')
    plt.title(elem)
    figure1.savefig("coefficients_%s.png" %(elem+'_'+name))
    plt.clf()
    
    loaded_model = pickle.load(open(filename, 'rb'))
    
    
    return loaded_model#,scaler,predictors,target



def scale_dataset(data,scaler):
    
    #function that scale a dataset from a scaler object
    #return the scaled dataset
    
    l_col=list(data.columns)
    data=scaler.transform(data)
    data=pd.DataFrame(data)
    data.columns=l_col
    
    return data

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


def predict_skewed_dist(class_targ,elem,data,pred=[]):
    
    ######
    #function that performs the modelisation of continues variables from a training set skewed with numerous positive outliers
    #model used here: ElasticNet but can easily be adapted
    #No test set used here - this script was made in the context of very small datasets
    #Input data needs to be skewed
    ######
    
    #class_targ gives the target - parameter to predict
    #elem corresponds to the name of the target - will appear on graphs and saved files
    #data is the dataframe the program works on
    #pred is a list of the predictors used to build the model
    
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
