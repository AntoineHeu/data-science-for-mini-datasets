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

import Codeffekt_prediction
from sklearn.preprocessing import StandardScaler

import sklearn
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold, cross_validate, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import shuffle

from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

import pickle


import matplotlib.font_manager
matplotlib.rcParams.update({'font.size': 20})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)


tunnel=pd.read_csv('tunnelier.csv',sep=';')
pre_tunnel=pd.read_csv('pre_tunnelier.csv',sep=';')

list_full_pred=['che_fl','che_elec','che_sulfate','che_ph','xra_s','xra_mo','xra_cr','xra_ni','xra_cu','xra_zn','xra_as','xra_sb','xra_ba','xra_pb']
list_full_target=['class-lab_sulfate','class-lab_fl','class-lab_fs','class-lab_cl','class-lab_mo','class-lab_se']
list_full_elem=['sulfate','lab_fl','lab_fs','lab_cl','lab_mo','lab_se']
dct_predicteurs={'Mo':['geo_p2','geo_p3','geo_p4','geo_p5','che_elec','che_ph','che_sulfate','xra_s','xra_mo'],
                 'F':['geo_p2','geo_p3','geo_p4','geo_p5','che_fl','che_elec','che_sulfate','che_ph','xra_s','xra_mo'],
                 'Se':['geo_p2','geo_p3','geo_p4','geo_p5','che_elec','xra_se','xra_s','xra_mo','che_ph','che_sulfate'],
                 'SO4':['geo_p2','geo_p3','geo_p4','geo_p5','che_elec','che_sulfate','xra_s','che_ph'],
                 'FS':['geo_p2','geo_p3','geo_p4','geo_p5','che_elec','che_sulfate','xra_s']}

def basic_stat_comparaison(dataset1,dataset2,list_elem,nom_save,scale=False):

    #comparaison des données brutes entre deux jeux de données à l'aide de boxplots (ne trace pas les outliers par soucis de lisibilité)
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
                    print("pas suffisamment d'éléments pour valider les distributions")
            
            serie1=pd.DataFrame(serie1).assign(Trial='tunnel')
            serie2=pd.DataFrame(serie2).assign(Trial='pre-tunnel')
            
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
    

#basic_stat_comparaison(tunnel,pre_tunnel,list_full_pred,'brut',scale=False)

def biased_target_stat_comparison(dataset1,dataset2,list_target,list_elem):
    pre_tun,tun=Codeffekt_prediction.data_preprocessing()
    
# =============================================================================
#     for columns in list(pre_tun):
#         if columns in list_elem:
#             pre_tun[columns]=StandardScaler().fit_transform(np.array(pre_tun[columns]).reshape(-1,1))
#             tun[columns]=StandardScaler().fit_transform(np.array(tun[columns]).reshape(-1,1))
# =============================================================================
            
    for targets in list(tun.columns):
        if targets in list_target:

            distribution_inf_isdi_1=tun.loc[tun[targets]=='ISDI']
            distribution_inf_isdi_2=pre_tun.loc[pre_tun[targets]=='ISDI']
            distribution_sup_isdi_1=tun.loc[tun[targets]!='ISDI']
            distribution_sup_isdi_2=pre_tun.loc[pre_tun[targets]!='ISDI']
            
            nom_inf=targets + '_' + 'INF'
            nom_sup=targets + '_' + 'SUP'
            
            basic_stat_comparaison(distribution_inf_isdi_1,distribution_inf_isdi_2,list_full_pred,nom_inf,scale=False)
            basic_stat_comparaison(distribution_sup_isdi_1,distribution_sup_isdi_2,list_full_pred,nom_sup,scale=False)
            

#biased_target_stat_comparison(tunnel,pre_tunnel,list_full_target,list_full_pred)


def Lasso_eval(class_targ,elem,data,pred=[]):
    
    global predictors, target,scaler
    
    'built in of predictors'
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
    
    dum_targ=pd.get_dummies(target[class_targ])
    target[class_targ]=dum_targ
    print(target)
    print(predictors)

    'on peut imposer un random_state ici'
    stratified=RepeatedKFold(n_splits=3,n_repeats=15)

    
    'Function that evaluate the importance of predictors'
    global lasso_coef
    
    'grid search for lasso'
    param_lass={'alpha': [1,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0001, 0.00005, 0.00001]}
    lasso=Lasso(normalize=True)
    grid_s_lass=GridSearchCV(lasso,param_lass,cv=stratified, scoring='neg_median_absolute_error')
    grid_s_lass.fit(predictors, target)

    lasso_opt=Lasso(alpha=grid_s_lass.best_params_['alpha'],normalize=True)
    model=lasso_opt.fit(predictors,target)
    
    flag='finalized_model_%s' % elem
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
    figure1.savefig("coefficients_%s.png" %elem)
    plt.clf()
    
    loaded_model = pickle.load(open(filename, 'rb'))
    
    
    return loaded_model

pre_tun,tun=Codeffekt_prediction.data_preprocessing()
indice=0
for targ in list_full_target:
    Lasso_eval(targ,list_full_elem[indice],tun,list_full_pred)
    indice=indice+1    
    
    
    
