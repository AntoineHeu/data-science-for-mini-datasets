# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 09:18:08 2018

@author: a.heude
"""

#Objectif du script: cross-valider le modèle géologique à l'aide d'une reconnaissance de la litho à partir des parmètres physico-chimiques

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from collections import Counter
import sys
print(sys.executable)
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import sklearn.mixture as mix

from sklearn.decomposition import PCA


import matplotlib.font_manager
matplotlib.rcParams.update({'font.size': 20})
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

pre_tun=pd.read_csv('file_1.csv',sep=';',encoding = "ISO-8859-1")
tun=pd.read_csv('file_2.csv',sep=';',encoding = "ISO-8859-1")


def select_dominant_litho(data,list_lithos,nom_ref):
    
    #function that selects the dominant lithology in the training dataset
    #return a dataset with the samples and their dominant lithology
    
    #####
    #data is the dataframe containing the data
    #list_lithos is the list of columns names for the lithologies
    #nom_ref is the name of the column containing sample references
    #####
    
    final_df=pd.DataFrame()
    final_df=pd.concat([final_df,data[nom_ref]],axis=1)
    list_=[]
    for rows in range(len(data)):
        i=0
        for lithos in list_lithos:
            if data[lithos][rows]>i:
                i=data[lithos][rows]
                main_litho=lithos
        list_.append(main_litho)
    final_df=pd.concat([final_df,pd.DataFrame(list_)],axis=1)
    
    return final_df

def clust_model_training(data,pred,nom_ref,list_lithos,n_clusters,cv_type='full',pca=False,n_comp=2):
    
    #function that trains a clustering algorithm
    #return a dataset with the samples and their dominant lithology
    
    #####
    #data is the dataframe containing the training data
    #pred is the list of names for the predictors
    #list_lithos is the list of columns names for the lithologies
    #nom_ref is the name of the column containing sample references
    #n_clusters is the number of clusters to compute
    #cv_type is the type of covariance (for GM algo)
    #pca=True or False if a PCA is to be computed on the predictors (allows the visualisation of clusters and data, etc..)
    #n_comp is the number of components to consider if pca is activated
    #####
    
    final_df=pd.DataFrame()
    
    'built in of predictors'
    clean_up=pd.DataFrame()
    for i in pred:
        clean_up=pd.concat([clean_up,data[i]],axis=1)
        clean_up_scale=clean_up
    for j in list_lithos:
        clean_up=pd.concat([clean_up,data[j]],axis=1)
    
    clean_up=pd.concat([clean_up,data[nom_ref]],axis=1)
    
    clean_up.dropna(inplace=True)
    clean_up_scale.dropna(inplace=True)
    clean_up.index = pd.RangeIndex(len(clean_up.index))
    clean_up_scale.index = pd.RangeIndex(len(clean_up_scale.index))

    predictors=pd.DataFrame()
    for i in pred:
        predictors=pd.concat([predictors,clean_up[i]],axis=1)
    list_pred=list(predictors.columns)
    
    predictors.index = pd.RangeIndex(len(predictors.index))

    scaler=StandardScaler()
    scaler.fit(clean_up_scale)
    predictors=scaler.transform(predictors)
    predictors=pd.DataFrame(predictors)
    predictors.columns=list_pred       
    
    if pca==True:
        predictors=dim_reduction_pca(predictors,n_comp)

    #trying other clustering algorithms can be interesting
    
    #kmeans = KMeans(n_clusters=n_clusters)
    #kmeans=SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',assign_labels='kmeans')
    
    #Here GM has shown good results
    kmeans=mix.GaussianMixture(n_components=n_clusters, covariance_type = cv_type)

    try:
        y_kmeans = kmeans.fit_predict(predictors)
    except:
        y_kmeans = kmeans.fit(predictors).predict(predictors)
    
    dataframe_litho=select_dominant_litho(clean_up,list_lithos,nom_ref)
    final_df=pd.concat([final_df,pd.DataFrame(y_kmeans),dataframe_litho],axis=1)
    final_df.columns=['clusters','sam_ref','lithos']
    
    return final_df,predictors,kmeans, y_kmeans,scaler


def assess_cluster_models(range_n_clusters,data,pred,nom_ref,list_lithos,cv_type,pca=False,n_comp=2):
    
    #function that assess a clustering model through different number of clusters with silhouettes
    #code largely extracted from: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    
    #####
    #range_n_clusters is the range of clusters to test
    #data is the dataframe containing the training data
    #pred is the list of names for the predictors
    #list_lithos is the list of columns names for the lithologies
    #nom_ref is the name of the column containing sample references
    #n_clusters is the number of clusters to compute
    #cv_type is the type of covariance (for GM algo)
    #pca=True or False if a PCA is to be computed on the predictors (allows the visualisation of clusters and data, etc..)
    #n_comp is the number of components to consider if pca is activated
    #####
    
    #Figures are saved in a path
    
    for n_clusters in range_n_clusters:
        
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        final_df,predictors,clusterer,cluster_labels,scaler = clust_model_training(data,pred,nom_ref,list_lithos,n_clusters,cv_type,pca,n_comp)
        
        # Create a subplot with 1 row and 2 columns
        if pca==True:
            fig, (ax1, ax2) = plt.subplots(1, 2)
        if pca==False:
            fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(predictors) + (n_clusters + 1) * 10])
    

    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(predictors, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(predictors, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        if pca==True:
            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(predictors.iloc[:, 0], predictors.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')
        
            # Labeling the clusters
            try: 
                centers = clusterer.cluster_centers_
                # Draw white circles at cluster centers
                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')
            
                for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')
            except:
                print("les centres ne peuvent pas être affichés")
        
            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")
        
            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')
            
        figure=plt
        
        name_figure="PATH_%s"%(str(n_clusters) + '_clusters')
        filename_lc="%s.png"%name_figure
        figure.savefig(filename_lc)
        plt.clf()
        
    return
    

def swarm_plot(dataframe,nom_classifier,nom_elem):
    
    #function that performs a swarm plot on the repartition of lithologies within each cluster
    
    #####
    #dataframe is the dataframe containing the data
    #nom_classifier is the x-axis (lithologies)
    #nom_elem is the y-axis (clusters)
    #####
    
    #Figures are saved in a path    
    
    lithos={'geo_p2': '','geo_p3':'','geo_p4':'','geo_p5':''}

    codes=list(dataframe[nom_classifier].astype('category').cat.categories)
    global df_bp,dict_prop
    df_bp=pd.concat([dataframe[nom_classifier],dataframe[nom_elem]],axis=1)
    
    #Swarm plot
    ident=nom_elem
    figure2 = plt.figure(figsize = (10,10))
    splot=sns.swarmplot(df_bp.iloc[:,0],df_bp.iloc[:,1])
    splot.set_xlim(-1,len(codes))
    plt.ylabel('distribution en fonction de la géologie décrite sur le terrain')
    plt.title(ident,size=20)
    plt.xlabel(nom_classifier,size=20)
    plt.ylabel(nom_elem,size=20)
    plt.legend(lithos.items())
    plt.tight_layout(pad=1.5, w_pad=1.5, h_pad=1.5)
    figure2.savefig("PATH_beeswarm_%s.png"%(nom_classifier + '_' + ident))
    plt.clf() 

    return

def dim_reduction_pca(data,n_digits):
    
    #function that performs a PCA dimension reduction
    #####
    #data is a dataframe
    #n_digits is the number of components to keep
    #####
    
    pca=PCA(n_components=n_digits)
    reduced_data = pca.fit_transform(data)
    reduced_data=pd.DataFrame(reduced_data)

    diag=pca.explained_variance_ratio_
    print(diag)
    
    return reduced_data

def home_weighted_scoring(dataframe,target,clusters):
    
    #function that performs a scoring according to how well clusters can delimitate lithologies
    #####
    #dataframe is a dataframe of the data
    #target is the classification we want to be well delimited
    #clusters are the clusers assigned to each sample
    #####
    
    #dictionnary containing the results
    dct=dict()
    n_samples=len(dataframe)
    for rows in range(len(dataframe)):
        if dataframe[clusters][rows] not in dct.keys():
            dct[dataframe[clusters][rows]]=[dataframe[target][rows]]

        else:
            dct[dataframe[clusters][rows]].append(dataframe[target][rows])
    
    #find the dominant litho in each cluster
    def dominant_litho(dictionnary,key):
        liste = dictionnary[key]
        count=Counter(liste)
        maxi=0
        for keys, values in count.items():
            if values > maxi:
                maxi=values
                dominant=keys
        return dominant,maxi
    
    #calculate scoring for a cluster as the proportion of dominant over the other classes
    def score(dictionnary, key):
        liste = dictionnary[key]
        dominant,maximum=dominant_litho(dictionnary,key)
        score = maximum/len(liste)
        n_sample_cluster=len(liste)

        return score,n_sample_cluster
    
    final_score=0
    
    for keys in dct.keys():
        scoring,n_sample_cluster=score(dct,keys)
        final_score=final_score+ scoring*(n_sample_cluster/n_samples)
    
    #returns final score
    return final_score

def assign_values_clusters(dataframe,target,clusters):
    
    #function that assigns, for each cluster its proportions of classes from the classification in target
    #####
    #dataframe is a dataframe of the data
    #target is the classification we want to be well delimited
    #clusters are the clusers assigned to each sample
    #####
    
    #dictionnary containing the results
    dct=dict()
    for rows in range(len(dataframe)):
        if dataframe[clusters][rows] not in dct.keys():
            dct[dataframe[clusters][rows]]=[dataframe[target][rows]]

        else:
            dct[dataframe[clusters][rows]].append(dataframe[target][rows])
    
    def proportions_litho(dictionnary,key):
        liste = dictionnary[key]
        count=Counter(liste)
        proportions=[(i, count[i] / len(liste)) for i in count]
        
        return proportions
    
    for keys in dct.keys():
        proportions=proportions_litho(dct,keys)
        dct[keys]=proportions
    
    #return a dictionnary of the results
    return dct


def predict_only_clusters(data,pred,nom_ref,model_clusters,scaler):
    
    #function that predicts clusters from a clustering model - on a new dataset, with writing the results in a csv file
    
    #####
    #data is the dataframe containing the training data
    #pred is the list of names for the predictors
    #nom_ref is the name of the column containing sample references
    #model_clusters is the clustering model fitted on a training set
    #scaler is the scaling step used for the training set
    #####
    
    #built in of predictors
    clean_up=pd.DataFrame()
    for i in pred:
        clean_up=pd.concat([clean_up,data[i]],axis=1)
        clean_up_scale=clean_up
    
    clean_up=pd.concat([clean_up,data[nom_ref]],axis=1)
    
    clean_up.dropna(inplace=True)
    clean_up_scale.dropna(inplace=True)
    clean_up.index = pd.RangeIndex(len(clean_up.index))
    clean_up_scale.index = pd.RangeIndex(len(clean_up_scale.index))

    predictors=pd.DataFrame()
    for i in pred:
        predictors=pd.concat([predictors,clean_up[i]],axis=1)
    list_pred=list(predictors.columns)
    
    predictors.index = pd.RangeIndex(len(predictors.index))

    predictors=scaler.transform(predictors)
    predictors=pd.DataFrame(predictors)
    predictors.columns=list_pred
    
    #prediction on the entire new dataset
    prediction_tot=model_clusters.predict(predictors)
    
    #inwrite of prediction row by row
    filename='prediction_clusters_lithos.csv'
    with open(filename, 'w',newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        for lignes in range(0,len(data)):
            
            predicteurs=[predictors[i][j] for i in list_pred for j in range(len(clean_up)) if data[nom_ref][lignes] ==clean_up[nom_ref][j]]
            print(predicteurs)

            try:            
                prediction = model_clusters.predict(np.array(predicteurs).reshape(1,-1))
                    
                filewriter.writerow([data[nom_ref][lignes], prediction])
                
            except:
                print('ligne sautée')
    
    #Returns additionnal elements that could be useful for further investigations
    return prediction_tot

#################################################################################
#################################################################################
    
#Visualizing MGMM models
#all credits goes to:
#https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

from matplotlib.patches import Ellipse 

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
        
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, 
                            angle, **kwargs))
        
    return
        
def plot_gmm(gmm, X, label=True, ax=None):
    
    fig, ax = plt.subplots(figsize=(9,7))      
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=50, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, ax=ax, alpha=w * w_factor)
        
    return

#################################################################################
#################################################################################



#################################################################################
#################################################################################

def test_BIC_validation_with_scoring(score=True):
    
    #BIC testing of different covariance definition and number of clusters
    #BIC scoring has been tuned to include the home_made_scoring if parameter score==True
    
    #largely extracted from scikit libraries - online examples
    #all credits goes to: http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
    #cehck the page for more infos
    
    import numpy as np
    import itertools
    
    from scipy import linalg
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    from sklearn import mixture
    
    plt.clf()
    print(__doc__)
        
    # Generate random sample, two components
    np.random.seed(0)    
    
    lowest_bic = np.infty
    bic = []
    n_components_range = range(2, 20)
    # 
    cv_types = ['spherical', 'tied', 'diag','full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            #house made prediction
            final_df,predictors,kmeans, cluster_model,scaler=clust_model_training(pre_tun,['xra_s','xra_cr','xra_ni','xra_cu','xra_zn','xra_as','xra_mo','xra_sb','xra_ba','xra_pb'],'sam_ref',['geo_p0','geo_p1','geo_p2','geo_p3','geo_p4','geo_p5','geo_p6','geo_p7','geo_p8','geo_p9'],n_components,cv_type,pca=False,n_comp=2)
            scoring=home_weighted_scoring(final_df,'lithos','clusters')
            
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(predictors)
            if score==True:
                bic.append(gmm.bic(predictors)/scoring)
            else:
                bic.append(gmm.bic(predictors))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                saved_cv=cv_type
                saved_component=n_components
    
    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []
    
    # Plot the BIC scores
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)
    
    if len(predictors.columns)==2:
        try:
            # Plot the winner
            splot = plt.subplot(2, 1, 2)
            Y_ = clf.predict(predictors)
    
            for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                                       color_iter)):
    
                v, w = linalg.eigh(cov)
                
                if not np.any(Y_ == i):
                    continue
                plt.scatter(np.array(predictors)[Y_ == i, 0], np.array(predictors)[Y_ == i, 1], .8, color=color)
    
                # Plot an ellipse to show the Gaussian component
                angle = np.arctan2(w[0][1], w[0][0])
                angle = 180. * angle / np.pi  # convert to degrees
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(.5)
                splot.add_artist(ell)
        
        except:
            print("le graphe scatter n'a pas pu être tracé")
    
    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()


    return saved_cv, saved_component,predictors

#################################################################################
#################################################################################
        
def plot_scorings (data,pred,nom_ref,list_lithos,n_clusters,cv_type='full',pca=False,n_comp=2):
    
    #function that plot the scoring value according to the number of clusters
    
    #####
    #data is the dataframe containing the training data
    #pred is the list of names for the predictors
    #list_lithos is the list of columns names for the lithologies
    #nom_ref is the name of the column containing sample references
    #n_clusters is the number of clusters to compute
    #cv_type is the type of covariance (for GM algo)
    #pca=True or False if a PCA is to be computed on the predictors (allows the visualisation of clusters and data, etc..)
    #n_comp is the number of components to consider if pca is activated
    #####
    
    scores=[]
    n_comp=[]
    for i in range(2,50):
        final_df,predictors,kmeans, cluster_model,scaler=clust_model_training(data,pred,nom_ref,list_lithos,i,cv_type='full',pca=False,n_comp=2)
        final_score=home_weighted_scoring(final_df,'lithos','clusters')
        scores.append(final_score)
        n_comp.append(i)
    
    figure = plt.figure(figsize = (10,10))
    plt.plot(n_comp, scores, color='g')
    plt.xlabel('number of clusters')
    plt.ylabel('scores')
    plt.title('scores according to the number of clusters')
    figure.savefig("%s_scoring.png"%(nom_ref))
    plt.clf()
    
    return
    
#################################################################################
#################################################################################

#Finding the best parameters: number of clusters, covariance type
assess_cluster_models([2,3,4,5,6,7,8],pre_tun,['xra_s','xra_cr','xra_ni','xra_cu','xra_zn','xra_as','xra_mo','xra_sb','xra_ba','xra_pb'],'sam_ref',['geo_p0','geo_p1','geo_p2','geo_p3','geo_p4','geo_p5','geo_p6','geo_p7','geo_p8','geo_p9'],pca=False,n_comp=2)
saved_cv, saved_component,predictors=test_BIC_validation_with_scoring(score=False)
plot_scorings (pre_tun,['xra_s','xra_cr','xra_ni','xra_cu','xra_zn','xra_as','xra_mo','xra_sb','xra_ba','xra_pb'],'sam_ref',['geo_p0','geo_p1','geo_p2','geo_p3','geo_p4','geo_p5','geo_p6','geo_p7','geo_p8','geo_p9'],'full',pca=False,n_comp=2)

#Extracting info and using the selected model
final_df,predictors,kmeans, cluster_model,scaler=clust_model_training(pre_tun,['xra_s','xra_cr','xra_ni','xra_cu','xra_zn','xra_as','xra_mo','xra_sb','xra_ba','xra_pb'],'sam_ref',['geo_p0','geo_p1','geo_p2','geo_p3','geo_p4','geo_p5','geo_p6','geo_p7','geo_p8','geo_p9'],17,'full',pca=False,n_comp=2)
swarm_plot(final_df,'lithos','clusters')
final_score=home_weighted_scoring(final_df,'lithos','clusters')
print(final_score)
clusters_values=assign_values_clusters(final_df,'lithos','clusters')
print(clusters_values)
Prediction=predict_only_clusters(tun,['xra_s','xra_cr','xra_ni','xra_cu','xra_zn','xra_as','xra_mo','xra_sb','xra_ba','xra_pb'],'sam_ref',kmeans,scaler)

#################################################################################
#################################################################################