# -*- coding: utf-8 -*-
"""
@author: albardan
"""
import sys
import os 

from scipy.signal import find_peaks
import neurokit2 as nk
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import configparser

from sklearn.ensemble import IsolationForest
from ecg_explorer import time_domain as td


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

def fill_empty(dataset):
    """
    fill empty lists in dataset
    """
    for j, row in enumerate(dataset):
        if len(row)!=0:
            to_replace=dataset[j]
            break
        
    for j,row in enumerate(dataset):
        if len(row)==0:
            dataset[j]=to_replace
        if len(row)!=0:
            to_replace=dataset[j]
    return np.array(dataset)
            


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
  
        
def getStats(array):
    """
    Get some statistical features from an array

    Parameters
    ----------
    array : np.array.
        data on which we want to compute statistical measures

    Returns
    -------
    stat: dictionary.
        dictionary where keys are measures names 

    """
    n5 = np.nanpercentile(array, 5)
    n50 = np.nanpercentile(array, 50)
    n25 = np.nanpercentile(array, 25)
    n75 = np.nanpercentile(array, 75)
    n95 = np.nanpercentile(array, 95)
    maxi = np.nanmax(array)
    mean = np.nanmean(array)
    mini = np.nanmin(array)
    median = np.nanmedian(array)
    std = np.nanstd(array)
    zero_crossing = ((array[:-1] * array[1:]) < 0).sum()
    rms = np.sqrt(np.nanmean(array**2))
    mav = np.nanmean(np.abs(array))

    
    stat = {"n5":n5,"n25":n25,"n50":n50,"n75":n75,"n95":n95,
            "max":maxi,"mean":mean,"min":mini,
            "median":median,"std":std,"zc":zero_crossing,
            "rms":rms,"mav":mav}

    
    return stat

        
def get_dataset(df,window_size,sampling_rate,signal_type):
    """
    constructing the dataset for anomaly detection 
    using the temporal features
    
    Parameters:
    -----------
    df: pd.DataFrame, dataframe containing ecg signals
    window_size: int, window size in seconds
    sampling_rate: int, 1000Hz
    signal_type: str, ECG or RSP
    
    Returns
    -------
    dataset:list of features extracted from each chunk
    list_of_signals:list of chunks of ecg signal of size window_size

    """
    dataset = []
    if signal_type=="ECG":
        fsignal = df[signal_type].values
        list_of_signals = [chunk for chunk in chunks(fsignal,window_size*sampling_rate)]
        total = len(list_of_signals)
        for j,sig in enumerate(list_of_signals):
            per = int(round(len(dataset)/total*100))
            print (per,"% of dataset construction is completed")
            try:
                hrv_results = td.time_domain(signal=sig,plot=False)
                hrv_df = pd.DataFrame(hrv_results).T
                hrv_df.columns = hrv_results.keys()
                hrv_features = hrv_df.values[0]
                dataset.append(hrv_features)
            except:
                dataset.append([])
           
        return list_of_signals,dataset


def anomaly_detection(df,
                      list_of_signals,
                      dataset,
                      sampleR,
                      threshold,
                      sindex,
                      stype,
                      cols,
                      signal_type):
    """
    Function that detects anomalies and replace abnormal parts of the signal

    Parameters
    ----------
    df : pd.DataFram object
        data frames.
    list_of_signals : TYPE
        DESCRIPTION.
    dataset : numpy.array
        dataset containing temporal features used for anomaly detection.
    sampleR: numpy.array,
        sample of R peak to insert if the interval when concatenating is too large
    threshold: float,
        anomaly threshold.
    sindex : int,
        subject index (used to plot).
    stype : str
        slalom type (used to plot).
    cols : list of strings,
        feature names.
    signal_type : str,
        ECG .

    Returns
    -------
    df : pd.DataFrame
        dataframe containing the corrected signal.

    """
    
    dataset = pd.DataFrame(np.vstack(dataset),columns=cols)
    dataset.dropna(axis=1, how='any', inplace=True)
    new_dataset = dataset.values
    clf = IsolationForest(random_state=0).fit(new_dataset)
    print (new_dataset,"new_dataset")
    scoring = clf.decision_function(new_dataset)
    print (scoring, "scoring")
    #breakpoint()
    # replace abnormal parts with the closest clean part
    wh = np.where(scoring<=threshold)[0]
    wh1 = np.where(scoring>threshold)[0]

    for w in wh[:-1]:   
        j = np.argmin(np.abs(wh1-w))
        list_of_signals[w] = list_of_signals[j]
        dataset.iloc[w] = dataset.iloc[j]
    
        pbefore, _ = find_peaks(list_of_signals[w-1], height=0.5,distance=150)
        pafter, _ = find_peaks(list_of_signals[w+1], height=0.5,distance=150)
        pcurrent, _ = find_peaks(list_of_signals[j], height=0.5,distance=150)
    
        try:
            right = len(list_of_signals[j])-pcurrent[-1]+pafter[0]
        except:
            right=600
        try:
            left = len(list_of_signals[w-1])-pbefore[-1]+pcurrent[0]
        except:
            left=600
                        
        print (len(sampleR),right, left)
        if right > 1100:
            list_of_signals[w+1][100:200] = sampleR
            print ("right large interval")
        if left > 1100 :
            list_of_signals[w-1][-200:-100]=sampleR
            print ("left large interval")
            
        if right < 400:
            list_of_signals[w+1][pafter[0]-50:pafter[0]+50] = 0
            print ("right small interval")
        
        if left < 400:
             list_of_signals[j][pcurrent[0]-50:pcurrent[0]+50] = 0
             print ("left small interval")
    
                
    new_scoring = []
    for j,score in enumerate(scoring):
        new_scoring = new_scoring + np.tile(score,len(list_of_signals[j])).tolist()
    new_scoring = np.array(new_scoring)
    
    
    length = df.shape[0]
    df["normal_ECG"] = np.concatenate(list_of_signals).ravel()[:length]
    df["anormality_ecg"] = new_scoring[:length]
        
    return df[["Time","normal_ECG","anormality_ecg"]]
    
   
# def combine(config):
#     """
#     Combine corrected EDA with corrected ECG and RSP

#     Parameters
#     ----------
#     config : config file
#     """
#     dir1 = config["EDA processing"]["destinationdir"]
#     dir2 = config["ECG processing"]["destinationdir"]
#     paths1 = [os.path.join(dir1,file) for file in os.listdir(dir1) if file.startswith("subject")]
#     paths2 = [os.path.join(dir2,file) for file in os.listdir(dir2) if file.startswith("subject")]
#     inds = []
#     for j,path1 in enumerate(paths1):
#         try:
#             sindex = path1.split("_")[-2]
#             stype  = path1.split("_")[-1]
#             print (sindex,stype)
#             df1 = pd.read_pickle(paths1[j])
#             df2 = pd.read_pickle(paths2[j])[["Time","normal_ECG","anormality_ecg"]]

#             assert (df1["Time"].values ==  df2["Time"].values).all()
#             print ("combination asserted for subject ",sindex)
#             df1["normal_ECG"]     = df2["normal_ECG"].values
#             df1["anormality_ecg"] = df2["anormality_ecg"].values


#             name = "subject_"+sindex+"_"+stype
#             path=os.path.join(dir2,name)
#             df1.to_pickle(path)
#         except:
#             inds.append(j)
            
            
def main(config):
    """
    main function
    """    
   
    localdir = config["handling files"]["localdir"]
    eda_destinationdir = config["EDA processing"]["destinationdir"]
    ecg_destinationdir = config["ECG processing"]["destinationdir"]
    signal_type = config["ECG processing"]["signal_type"] # ECG
    sampling_rate = int(config["ECG processing"]["sampling_rate"]) # 1000 Hz
    window_size = int(config["ECG processing"]["window_size"]) # ecg window size 
    threshold = float(config["ECG processing"]["anomaly_threshold"]) 
    sampleR = pd.read_pickle(config["ECG processing"]["sampleR"]).values.reshape(1,-1)[0]
    ecg_cols = pd.read_pickle(config["ECG processing"]["ecg_cols"]).values.reshape(1,-1)[0].tolist()

    
    paths = [os.path.join(localdir,file) for file in os.listdir(localdir) if file.startswith("subject")][-21:]

    for k,path in enumerate(paths):
        try:
            sindex = path.split("_")[-2]
            stype = path.split("_")[-1]
            print (sindex,stype)
            df = pd.read_pickle(path)
            print ("Constructing the dataset for subject ", sindex, " condition ",stype)
            list_of_signals,dataset = get_dataset(df,
                                                  window_size,
                                                  sampling_rate,
                                                  signal_type)
            filled_dataset = fill_empty(dataset)
            print ("Detection and correction of bad parts ...")
            df = anomaly_detection(df,
                                   list_of_signals,
                                   filled_dataset,
                                   sampleR,
                                   threshold,
                                   sindex,
                                   stype,
                                   ecg_cols,
                                   signal_type)
            name = "subject_"+sindex+"_"+stype
            path = os.path.join(eda_destinationdir,name)
            new_df = pd.read_pickle(path)
            new_df["normal_ECG"]     = df["normal_ECG"].values
            new_df["anormality_ecg"] = df["anormality_ecg"].values
            assert (df["Time"].values ==  new_df["Time"].values).all()
            print ("combination asserted for subject ",sindex)
            path = os.path.join(ecg_destinationdir,name)
            new_df.to_pickle(path)
            print ("saved with modifications")
        except:
            sindex = path.split("_")[-2]
            stype = path.split("_")[-1]
            print ("no modif for subject", sindex, " for condition ", stype)
            df = pd.read_pickle(path)
            name = "subject_"+sindex+"_"+stype
            name = "subject_"+sindex+"_"+stype
            path = os.path.join(eda_destinationdir,name)
            new_df = pd.read_pickle(path)
            new_df["normal_ECG"] =df["ECG"]
            new_df["anormality_ecg"] = 0
            path = os.path.join(ecg_destinationdir,name)
            new_df.to_pickle(name)
    
    


    

    
if __name__ == "__main__":  
    configfile = "D:/scripts/config.ini"
    config = get_config(configfile)
    main(config)
    
    



# dir1 = "F:/data/backups/maniputac/dataframes/all_slaloms/eda_d2_med_mit/"
# dir2 = "F:/data/backups/maniputac/dataframes/all_slaloms/ecg_temporal_iso/"
# #dir3 = "F:/data/backups/maniputac/dataframes/all_slaloms/rsp_temporal_iso/"

# paths1 = [os.path.join(dir1,file) for file in os.listdir(dir1) if file.startswith("subject")]
# paths2 = [os.path.join(dir2,file) for file in os.listdir(dir2) if file.startswith("subject")]
# #paths3 = [os.path.join(dir3,file) for file in os.listdir(dir3) if file.startswith("subject")]
    
# inds = []
# for j,path1 in enumerate(paths1):
#     try:
#         print (j)
#         sindex = path1.split("_")[-2]
#         stype  = path1.split("_")[-1]
#         print (sindex,stype)
#         df1 = pd.read_pickle(paths1[j])
#         df2 = pd.read_pickle(paths2[j])[["Time","normal_ECG","anormality_ecg"]]
#         #df3 = pd.read_pickle(paths3[j])[["Time","normal_RSP","anormality_rsp"]]
#         print (df1.shape,df2.shape)#,df3.shape)
#         assert (df1["Time"].values ==  df2["Time"].values).all()
#         #assert (df1["Time"].values ==  df3["Time"].values).all()
#         print ("asserted")
#         df1["normal_ECG"]     = df2["normal_ECG"].values
#         df1["anormality_ecg"] = df2["anormality_ecg"].values
        
#         # df1["normal_RSP"]     = df3["normal_RSP"].values
#         # df1["anormality_rsp"] = df3["anormality_rsp"].values
        
#         name = "F:/data/backups/maniputac/dataframes/all_slaloms/combined_dataframes/" +"subject_"+sindex+"_"+stype
#         df1.to_pickle(name)
#     except:
#         inds.append(j)


    

    # dataset["anomaly_score"] = scoring
    # namen = 'D:/data_1/dataframes/all_slaloms/ecg_temporal_iso/features/' + "bc_subject_"+sindex+"_"+stype
    # dataset.to_pickle(namen)
    
    # threshold = -0.1 # for -0.15 ecg 
    # thrs = np.tile(threshold,len(new_scoring))
    
    # scaler = MinMaxScaler(feature_range=(-1,1))
    # new_scoring = scaler.fit_transform(new_scoring.reshape(-1, 1))

    
    
    # path = 'D:/data_1/dataframes/all_slaloms/ecg_temporal_iso/figures'
    # fig = plt.figure(figsize=(13,7))
    # plt.plot(signal,color="b")
    # plt.scatter(range(len(new_scoring)),np.array(new_scoring)-2,color="k",s=0.1,label="anomaly scores")
    # plt.plot(thrs-2,color="r",label="anomaly threshold = {thr}".format(thr=threshold),linestyle='dashed')
    # plt.ylabel("Anomaly score (the lower the more abnormal)")
    # plt.title("Anomaly detection based on ECG temporal features")
    # plt.legend()
    # plt.show()
    # fig.savefig(os.path.join(path,"subject_"+sindex+"_"+stype))
    
    
    
        # if sindex in ['14', '15', '16', '17', '18',
    #               '19', '1', '20', '21', '22',
    #               '23']:
            # signal  = [sig for j,sig in enumerate(list_signal) if len(dataset[j])!=0]
        # dataset = [data for j,data in enumerate(dataset)  if  len(dataset[j])!=0 ]
    
    




# dir_ = 'D:/data_1/dataframes/all_slaloms/ecg_temporal_iso/features'
# files  = os.listdir(dir_)
# files = [os.path.join(dir_,file) for file in files if file.startswith("subject")]
# abnormals = []
# for j,file in enumerate(files):
#     try:
#         si = file.split("\\")[1].split("_")[1]
#         st = file.split("\\")[1].split("_")[2]
#         df = pd.read_pickle(file)
#         df["Subject"] = si
#         df["Slalomtype"] = st
#         abnormal = df[df["anomaly_score"]<-0.1][['hr_mean','hr_min', 'hr_max',
#                       'hr_std', 'hr_p5', 'hr_p25', 'hr_p50',
#                       'hr_p75', 'hr_p95',
#                       'Subject','Slalomtype','anomaly_score']]
#         print (abnormal.shape,abnormal.head())
#         abnormals.append(abnormal)
#     except:
#         continue
    
# abnormal_df = pd.concat(abnormals)
  
# hrcols = ['hr_mean','hr_min', 'hr_max',
#           'hr_std', 'hr_p5', 'hr_p25', 'hr_p50',
#           'hr_p75', 'hr_p95',
#       'anomaly_score']
# old_hrcols = ["old_"+c for c in hrcols]


# dir_ = 'D:/data_1/dataframes/all_slaloms/ecg_temporal_iso/features'
# files  = os.listdir(dir_)
# bc_files = [os.path.join(dir_,file) for file in files if file.startswith("bc_")]
# list_d  = []
# for bc_file in bc_files:
#     si = bc_file.split("\\")[1].split("_")[2]
#     st = bc_file.split("\\")[1].split("_")[3]
#     d = pd.read_pickle(bc_file)
#     list_d.append([si,st,d])

# cols_to_add = []        
# for i in range(abnormal_df.shape[0]):
#     subject = abnormal_df.iloc[i]["Subject"]
#     stype = abnormal_df.iloc[i]["Slalomtype"]
#     score = abnormal_df.iloc[i]["anomaly_score"]
    
#     for d in list_d:
#         if d[0]==subject and d[1]==stype:            
#             bc_df = d[2]
#     w = np.argmin(np.abs((bc_df["anomaly_score"] - score).values))
#     print (w,bc_df[hrcols].iloc[w])
#     cols_to_add.append(bc_df[hrcols].iloc[w].values.reshape(1,-1))
# cols_to_add = np.vstack(cols_to_add)
# abnormal_df[old_hrcols] = cols_to_add
# abnormal_df.to_pickle('D:/data_1/dataframes/all_slaloms/ecg_temporal_iso/features/abnormal_df')
    
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
#import matplotlib
#matplotlib.use("Agg")
# new_dataset = dataset.T[~np.isnan(dataset.T).any(axis=1)].T
# clf = IsolationForest(random_state=0).fit(new_dataset)
# scoring = clf.decision_function(new_dataset)

# wh = np.where(scoring<-0.1)[0]
# wh1 = np.where(scoring>-0.1)[0]
# for w in wh:
#     j = np.argmin(np.abs(wh1-w))
#     signal[w] = signal[j]

# signal = np.vstack(signal).flatten()
# s,d = get_dataset(pd.DataFrame(signal,columns=["ECG"]),30,1000)
# d = [data for j,data in enumerate(d) if len(d[j])!=0]
# d = pd.DataFrame(np.vstack(d),columns=cols)
# print (d.head())
    

# data = []
# list_of_signals = []
# for j,dataset in enumerate(datasets):
#     signal = signals[j]
#     signal = [sig for j,sig in enumerate(signal) if len(dataset[j])!=0]
#     dataset = [data for j,data in enumerate(dataset) if len(dataset[j])!=0]
#     dataset = np.vstack(dataset)
#     signal = [item for sublist in signal for item in sublist]
#     dataset = np.vstack(dataset)
#     data.append(dataset)
#     list_of_signals.append(signal)
# data = np.vstack(data)

# for i in range(len(paths)):
#     signal = np.array(list_of_signals[i])
#     anomaly_detection(signal, data,data[i])


# directory = "D:/data_1/dataframes/features/plot_old_new_ecg"
# images = ["".join(img.split("_")[1:3]) for img in os.listdir(directory)]
# d = {item:images.count(item)*3 for item in images}

# import pandas as pd

# s1 = pd.read_pickle('D:/data_1/dataframes/features/slaloms_a1/subject_1_AleatoireSerre')
# s1na = pd.read_pickle('D:/data_1/dataframes/features/slaloms_a1/normalized/subject_1_nbv_AleatoireSerre')
# s1nv = pd.read_pickle('D:/data_1/dataframes/features/slaloms_a1/normalized/subject_1_nba_AleatoireSerre')





