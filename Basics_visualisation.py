# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:05:51 2018

@author: a.heude
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.linear_model import Lasso

import pickle


import matplotlib.font_manager
matplotlib.rcParams.update({'font.size': 20})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)


def basic_stat_comparaison(dataset1,dataset2,list_elem,nom_save,scale=False):

    # This function gives a basic visual representation of two datasets through boxplots
    # Outliers are not displayed to avoid shrinking the graphs - this can be easily changed if necessary
        
    ##################
        #parameters
        #dataset1, dataset2 are the two datasets to compare
        #list_elem is a list containing all the variable the user wants to compare the two datasets on
        #nom_save is a param used when graphs are saved
        #scale is a used to compare scaled distributions
    ##################

    #The purpose is to compare the same variables from two datasets
        
    for predictors in list(dataset1.columns):
        if predictors in list_elem:
            name=predictors
            serie1=dataset1[predictors]
            serie2=dataset2[predictors]
            
            serie1.dropna(inplace=True)
            serie2.dropna(inplace=True)
            
            if scale==True:
                try:
                    serie1=StandardScaler().fit_transform(np.array(serie1).reshape(-1,1))
                    serie2=StandardScaler().fit_transform(np.array(serie2).reshape(-1,1))
                except ValueError:
                    print("not enough occurrences to compare distributions")
            
            serie1=pd.DataFrame(serie1).assign(Trial='Data1')
            serie2=pd.DataFrame(serie2).assign(Trial='Data2')
            
            data_to_plot=pd.concat([serie1,serie2],axis=0)
            print(data_to_plot)
            mdf=pd.melt(data_to_plot,id_vars=['Trial'],var_name=['Number'])
            print(mdf)
            
            # Create a figure instance
            fig = plt.figure(1, figsize=(9, 6))
            
            # Create an axes instance
            ax = fig.add_subplot(111)
            
            # Create the boxplot
            ax = sns.boxplot(x="Trial", y="value", hue="Number", data=mdf,palette='Set3',showfliers=False)
            # Save the figure
            nom='box plot_%s.png'%(name+'_'+nom_save)
            fig.savefig(nom)
            plt.clf()

    return
        
def box_plot(dataframe,nom_classifier,nom_elem):
    #Function that saves two figures: one box plot of a data distribution and one swarm plot of the same data according to a classifier
    #Example run on that code: Mercury concentration according to the lithology
        
    ##################
    #parameters:
    #dataframe is the tabular dataset
    #nom_classifier is column name of the data used to classify
    #nom_elem is the column name of the data you want to characterize the distribution
    ##################

    #what to write on the legend - to be adapted
    legend={'ASG': 'Alluvions sablo-graveleux','DB':'dalles béton - enrobé','SGN':'sables et graviers noirs','MC':'Marnes et calcaires','SGG': 'sables et graviers gris','AS':'Argiles et silex','ASG_o':'Alluvions sablo-graveleux avec odeurs','CC':'Charbon concassé'}

    codes=list(dataframe[nom_classifier].astype('category').cat.categories)
    global df_bp,dict_prop
    df_bp=pd.concat([dataframe[nom_classifier],dataframe[nom_elem]],axis=1)
    
    #Swarm plot: plotting the distribution of a variable according to a classifier
    ident=nom_elem
    figure2 = plt.figure(figsize = (10,10))
    splot=sns.swarmplot(df_bp.iloc[:,0],df_bp.iloc[:,1])
    splot.set_xlim(-1,len(codes))
    plt.ylabel('distribution en fonction de la géologie décrite sur le terrain')
    plt.title(ident,size=20)
    plt.xlabel(nom_classifier,size=20)
    plt.ylabel(nom_elem,size=20)
    plt.legend(legend.items())
    plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
    figure2.savefig("%s_beeswarm.png"%(nom_classifier + '_' + ident))
    plt.clf()
    
    #boxplot: plotting the distribution of a variable
    df_box=pd.DataFrame()
    df_box=pd.concat([df_box,dataframe[nom_elem]],axis=1)
    df_box.dropna(inplace=True)
    np_box=np.array(df_box)
    print(df_box)
    figure3 = plt.figure(figsize = (10,10))
    sns.boxplot(np_box,sym='k')
    plt.xlabel(nom_elem,size=20)    
    plt.ylabel('distribution de %s'%ident,size=20)
    plt.title(ident,size=20)
    plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
    figure3.savefig("%s_boxplot.png"%ident)
    plt.clf()
    

    return


def Lasso_eval(class_targ,elem,data,pred=[]):
    #This function computes a Lasso Regression in order to find out the dominants parameters in a linear prediction of a target

        ###############
        #PARAMETERS
        #class_targ = the target to predict = name of a column of data
        #elem = string of the name of the target to predict
        #data = pd.Dataframe of the tabular input data, containing target and predictors
        #pred = list of predictors to test
        ###############
    
    #graphs will be saved on the directory this script is launched from

    global predictors, target,scaler
    
    #built in of predictors'
    clean_up=pd.DataFrame()
    for i in pred:
        clean_up=pd.concat([clean_up,data[i]],axis=1)
        clean_up_scale=clean_up
    clean_up=pd.concat([clean_up,data[class_targ]],axis=1)
    
    clean_up.dropna(inplace=True)
    clean_up_scale.dropna(inplace=True)

    
    predictors=pd.DataFrame()
    for i in pred:
        predictors=pd.concat([predictors,clean_up[i]],axis=1)
    list_pred=list(predictors.columns)
    
    predictors.index = pd.RangeIndex(len(predictors.index))
    print(predictors)
    scaler=StandardScaler()
    scaler.fit(clean_up_scale)
    predictors=scaler.transform(predictors)
    predictors=pd.DataFrame(predictors)
    predictors.columns=list_pred       
            
    #built in of the target'
    target=pd.DataFrame()
    target=pd.concat([target,clean_up[class_targ]],axis=1)
    target = target.reset_index(drop=True)
    
    dum_targ=pd.get_dummies(target[class_targ])
    target[class_targ]=dum_targ
    print(target)
    print(predictors)

    #A random state can be implemented here for reproductivity of results
    stratified=RepeatedKFold(n_splits=3,n_repeats=15)

    
    #Function that evaluate the importance of predictors
    global lasso_coef
    
    #grid search for lasso
    param_lass={'alpha': [1,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0001, 0.00005, 0.00001]}
    lasso=Lasso(normalize=True)
    grid_s_lass=GridSearchCV(lasso,param_lass,cv=stratified, scoring='neg_median_absolute_error')
    grid_s_lass.fit(predictors, target)

    lasso_opt=Lasso(alpha=grid_s_lass.best_params_['alpha'],normalize=True)
    model=lasso_opt.fit(predictors,target)
    
    flag='finalized_model_%s' % elem
    filename = '%s.sav' %flag
    pickle.dump(model, open(filename,'wb'))
    
    #Getting the predictors coefficients
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
    figure1.savefig("coefficients_%s.png" %elem)
    plt.clf()
    
    loaded_model = pickle.load(open(filename, 'rb'))
    
    
    return loaded_model
