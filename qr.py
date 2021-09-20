# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:15:35 2021

@author: albardan
"""
import os
import sys
import configparser

import numpy as np
import pandas as pd
from l1qr import L1QR
from collections import Counter
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve,auc

from ast import literal_eval
from ensemble import RandomForestQuantileRegressor
import warnings
warnings.filterwarnings("ignore") 


def select(df,qnlevel,dictionary):
    """
    function that selects important features 
    from the features space using lasso quantile regression

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing the features,
        It should be corrected and normalized.
    qnlevel : float
        quantile level.
    dictionary: dict,
         dictionary containing the selected features 
        their coefficients for each quantile level 

    Returns
    -------
    dictionary : dict,
        dictionary containing the selected features 
        their coefficients for each quantile level

    """
    features = list(df.columns[:-4])
    l = int(len(df)*0.33)
    X =  df[features].iloc[:l]
    y = pd.Series(df["score"].values).iloc[:l]
    mdl = L1QR(y=y, x=X, alpha=qnlevel)
    mdl.fit(s_max=np.inf)
    list_features = []
    for i in range(len(mdl.beta)):
        coeffs = mdl.beta[i]
        selected_features = np.array(features)[np.where(coeffs!=0)[0]].tolist()
        list_features.append(selected_features)
    flat_list_features = Counter([item for sublist in list_features for item in sublist]).keys()
    dictionary[qnlevel] = [list(flat_list_features),mdl.beta]                   
    return dictionary

def continuous2discrete(s,thr=2.6):
    """
    discretize a single score using a threshold in a binary manner
    
    
    Parameters:
    -----------
    s: float, score
    thr:float, threshold (default=2.6)
    
    Return:
    ------
    1 if greater than threshold and 0 if less than threshold
    """
    if s<=thr:
        return 0
    else:
        return 1

        
def discretize_labels(y,thr):
    y = [continuous2discrete(s, thr) for s in y]
    return np.array(y)

def get_output(list_of_predictions,list_of_labels,thr):
    """
    Function to evaluate the model

    Parameters
    ----------
    list_of_predictions : np.array of floats
        predictions of quantile regression.
    list_of_labels : np.array of ints
        true binary labels
        .
    thr : float
        threshold.

    Returns:
    -------
    output : list of outputs containing 5 elements
        kt: kendal tau correlation coefficient
        area: area under the ROC curve
        thr: threshold
        fpr: false positive rate
        tpr: true positive rate

    """
    y_true = discretize_labels(y=list_of_labels,thr=thr) 
    fpr, tpr, thresholds = roc_curve(y_true,list_of_predictions)
    area = auc(fpr,tpr).round(2)  
    kt = stats.kendalltau(list_of_predictions,list_of_labels)
    output = [kt,area,thr,fpr,tpr]
    return output


def evaluate(model,
             df,
             selected_features,
             train_size,
             thr=2.6,
             pqnlevel=0.95):
    
    X =  df[selected_features]
    y = pd.Series(df["score"].values)
    k = np.random.choice(np.arange(0,100000,1),size=1, replace=True)[0]
    Xn,  yn = shuffle(X.values,  y.values, random_state=k)

    X_train,y_train=Xn[:int(train_size*len(X))],yn[:int(train_size*len(X))]
    X_test,y_test = Xn[int(train_size*len(X)):],yn[int(train_size*len(X)):]
    model.fit(X_train,y_train)
    predictions = model.predict(X_test,quantile=pqnlevel*100)
    output = get_output(predictions,y_test,thr)
    return output
    
   
def get_results(model,
                df,
                selected_features,
                n_iterations,
                train_size,
                thr,
                pqnlevel):
    """
    print out result about AUC and plot the ROC curve
    
    
    Parameters:
    ----------
    model: random forest quantile regression model
    df: pd.DafaFrame, feature dataframe
    selected_features: list of strings, selected features (output of select function)
    train_size: float, train_size (0.1 to 0.9)
    thr: float, thresholds to binary label scores
    pqnlevel: float, prediction quantile level used for the random forest quantile regression model
    
    
    Return:
    ------
    mean_auc: float, mean of auc across iterations
    std_auc: float, std of auc across iterations
    fprs: list of fpr across all iterations (used to plot ROC)
    tprs: list of tpr across all iterations (used to plot ROC)
    """
    aucs = []
    fprs = []
    tprs = []
    for k in range(n_iterations):
        output = evaluate(model,df,selected_features,train_size,thr,pqnlevel)
        aucs.append(output[1])
        fprs.append(output[3])
        tprs.append(output[4])
    
    mean_auc = np.nanmean(aucs)
    std_auc  = np.nanstd(aucs)
    
    
    print ("The mean of AUC is ", mean_auc)
    print ("The std of AUC is ", std_auc)
    plot_roc(fprs,tprs,mean_auc)
    return mean_auc,std_auc,fprs,tprs
    
        
def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [1]*(target_len - len(some_list))

def plot_roc(fprs,tprs,mean_auc):
    target_len=67
    
    new_fprs,new_tprs = [],[]
    for i in range(len(fprs)):
        new_fprs.append(pad_or_truncate(fprs[i].tolist(),target_len))
        new_tprs.append(pad_or_truncate(tprs[i].tolist(),target_len))
    
    new_fprs_m = np.mean(np.array(new_fprs),axis=0)
    new_tprs_m = np.mean(np.array(new_tprs),axis=0)
    
    for j,ele in enumerate(fprs):
        plt.plot(new_fprs[j],new_tprs[j],color="black",alpha=0.05)
    
    plt.plot(new_fprs_m,
             new_tprs_m,
             color="orange",
             label="Auc = {m}".format(m=round(mean_auc,2)),
             linewidth=2,
             )
    plt.plot(np.array([0,0.5,1]),np.array([0,0.5,1]),color='red')
    plt.legend()

     
def get_config(configfile):
    """
    read config file
    
    Parameters:
    ----------
    configfile:str, configuration file
    
    Returns
    -------
    config : ConfigParser object
        configuration file.

    """
    config = configparser.ConfigParser()
    config.read(configfile)
    return config


def main(config):
    """
    main function
    """
    df_path = config["feature selection"]["dfpath"]
    d = int(config["feature selection"]["nb_features"])
    n_iterations = int(config["modeling"]["n_iterations"])
    train_size = float(config["modeling"]["train_size"])
    pqnlevel = float(config["modeling"]["pqnlevel"])
    thr = float(config["modeling"]["thr"])
    evaluation = literal_eval(config["modeling"]["evaluation"])
    
    print ("feature selection for different quantile levels ...")
    dictionary = {}
    features = [] 
    coefficients = []
    df = pd.read_pickle(df_path)
    grid_alpha = np.arange(0.01,0.99,0.05)
    for qnlevel in grid_alpha:
        try:
            #if round(qnlevel,2)*100%5==0:
            print ("quantile level = ",qnlevel)
            dictionary = select(df,qnlevel,dictionary)
            features.append(dictionary[qnlevel][0])
            coefficients.append(dictionary[qnlevel][1])
        except:
            continue
    pd.DataFrame(coefficients).to_pickle(os.path.join("./","coeffs"))
    flat_features = [item for sublist in features for item in sublist]
    selected_features = [e[0] for e in Counter(flat_features).most_common()[:d]]
    name = "selected_features"
    pd.DataFrame(selected_features).to_pickle(os.path.join("./",name))
    print ("Selected features are saved")
    if evaluation:
        print ("Model evaluation begins ...")
        model = RandomForestQuantileRegressor(random_state=0,n_estimators=100) 
        mean_auc,std_auc, fprs, tprs = get_results(model,
                                       df,
                                       selected_features,
                                       n_iterations,
                                       train_size,
                                       thr,
                                       pqnlevel)
    
    
    
    
if __name__ == "__main__":
    configfile = "D:/scripts/config.ini"
    config = get_config(configfile)
    main(config) 
    
    sys.exit()
    







# article
# tableau 
grid_alpha = np.arange(0.01,0.99,0.05)
df_path = config["feature selection"]["dfpath"]
features = pd.read_pickle(df_path).columns[:-4]
path = 'D:/scripts/coeffs'
list_of_features = []
newcoefs = []
coefdf = pd.read_pickle(path)
for i in range(coefdf.shape[0]):
    z = coefdf.iloc[i].values[0][-1]
    wh = np.where(z!=0)[0]
    list_of_features.append(list(features[wh]))
    newcoefs.append(z)

    
flat_features = [item for sublist in list_of_features for item in sublist]
selected_features = [e[0] for e in Counter(flat_features).most_common()[:28]]
newcoefs = np.vstack(newcoefs)
newcoefsdf = pd.DataFrame(newcoefs,columns=features,index=grid_alpha)
newcoefsdf = newcoefsdf[selected_features]




