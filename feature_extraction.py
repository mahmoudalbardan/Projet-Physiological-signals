# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:28:07 2021

@author: albardan
"""
import sys
import os 
import configparser

import numpy as np
import pandas as pd
import pywt
import neurokit2 as nk

from ecg_explorer import hrv

pd.options.mode.chained_assignment = None  # default='warn'


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


def get_hrv(fsignal,df,rootdir):
    """
    Get heart rate variabiliyty features related to the ECG signal
    
    Parameters
    ----------
    fsignal : np.array
        ecg signal.
    df : pd.DataFrame
    rootdir:string.
        root directory containing the dataframes 
        
    Returns
    -------
    hrv_features : list, list of normalized ecg features 
    hrv_names : list of strings, features names

    """
    allowedtypes = [int,float,np.float64,np.int32,tuple]
    hrv_results = hrv.hrv(signal=fsignal, show=False,df=df)
    hrv_df = pd.DataFrame(hrv_results).T
    hrv_df.columns = hrv_results.keys()
    hkeys = list(hrv_results.keys())
    hrv_features = []
    hrv_names = []
    for j,ele in enumerate(hrv_df.values[0]):
        if type(ele) in allowedtypes:
            if type(ele)!=tuple:
                hrv_features.append(np.float64(ele))
                hrv_names.append(hkeys[j])
            else:
                hrv_features = hrv_features + list(ele)
                y = [hkeys[j]+str(i) for i,e in enumerate(ele)]
                hrv_names = hrv_names + y
    "hrv_features = ecg_baseline_correction(hrv_features,df,rootdir)"
    return hrv_features,hrv_names
            
            
def ecg_baseline_correction(hrv_features,df,rootdir):
    """
    Baseline correction of ecg features using divisive method
    
    Parameters:
    ----------
    hrv_features:
    subject_index:
    slalomtype:
    
    Return:
    ------
    hrv_features: corrected feature vector
    """
    subject_index = df["Subject"].iloc[0]
    slalomtype = df["Slalomtype"].iloc[0]
    name = "Subject_" + subject_index + "_" + slalomtype
    path = os.path.join(rootdir,"dataframes/all_slaloms",name)
    baseline = pd.read_pickle(path)
    baseline = baseline[baseline["trigger"].isin(["Baseline_Arrêt_Start",
                                                  "Baseline_Arrêt_Stop",
                                                  "Baseline_Véhicule_Start",
                                                  "Baseline_Véhicule_Stop"])]
    
    ecg_baseline = baseline["ECG"].values
    hrv_baseline = hrv.hrv(signal=ecg_baseline, show=False,df=df)
    for i in range(28):
        try:
            r= hrv_features[i]/hrv_baseline[i]
            r = np.nan if r==np.inf else r
            hrv_features[i] = r
        except:
            hrv_features[i] = np.nan
    
    return hrv_features
    
        
        
def get_scr(fsignal,sampling_rate):
    """
    Get skin conductance response features
    
    Parameters:
    ----------
    fsignal: np.array.
        filtered eda signal with baseline correction
    sampling_rate:int.
        sampling rate
        
        
    Return:
    -------
    scr_features:list of scr features
    scr_names: liste of strings, features names related to skin conductance
    """
    
    signal_type = "EDA"
    scr_features = []
    scr_names = []
    df_output, info = nk.bio_process(eda=fsignal,
                                     sampling_rate=sampling_rate)
    features_names = ["SCR_Amplitude",
                      "SCR_Height",
                      "SCR_RiseTime",
                      "SCR_RecoveryTime"]
           
    for feature in features_names:
        statdict = getStats(info[feature])
        scr_features = scr_features + list(statdict.values()) 
        name = [signal_type+"_"+feature+"_"+ele for ele in list(statdict.keys())]
        scr_names = scr_names + name 


    scr_features = scr_features+ [len(info["SCR_Peaks"])]
    scr_names = scr_names + ["SCR_npeaks"]
    return scr_features,scr_names
                

def rsp_baseline_correction(df,rootdir):
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        dataframe.
    rootdir : string,
        root directory.

    Returns
    -------
    rsp_features:list of floats.
        respiration features in the baseline phase

    """
    rsp_features = []
    rsp_names = []
    subject_index = df["Subject"].iloc[0]
    slalomtype = df["Slalomtype"].iloc[0]
    name = "Subject_" + subject_index + "_" + slalomtype
    path = os.path.join(rootdir,"dataframes/all_slaloms",name)
    baseline = pd.read_pickle(path)
    baseline = baseline[baseline["trigger"].isin(["Baseline_Arrêt_Start",
                                                  "Baseline_Arrêt_Stop",
                                                  "Baseline_Véhicule_Start",
                                                  "Baseline_Véhicule_Stop"])]
    rsp_baseline = baseline["RSP"].values
    df_output,info = nk.rsp_process(rsp_baseline,sampling_rate=1000)
    l = min(len(info["RSP_Peaks"]),len(info["RSP_Troughs"]))
    peaks = info["RSP_Peaks"][:l]
    troughs =  info["RSP_Troughs"][:l]
    cleansignal = df_output["RSP_Clean"].values
    outbreath = cleansignal[peaks]-cleansignal[troughs]
    inbreath  = cleansignal[peaks[1:]]-cleansignal[troughs[:-1]]
    info["RSP_InBreath"] = inbreath.astype(int)
    info["RSP_OutBreath"] = outbreath.astype(int)
    features_names = [  "RSP_Amplitude",
                        "RSP_Rate",
                        "RSP_Phase",
                        "RSP_Phase_Completion",
                        "RSP_InBreath",
                        "RSP_OutBreath"]
    
    for feature in features_names:
        if feature not in ["RSP_InBreath","RSP_OutBreath"]:
            statdict = getStats(df_output[feature].values)
        else:
            statdict = getStats(info[feature])
        rsp_features = rsp_features + list(statdict.values()) 
        name = [feature+"_"+ele for ele in list(statdict.keys())]
        rsp_names = rsp_names + name 
            
        
    rsp_features = rsp_features+ [len(info["RSP_Peaks"])] + [len(info["RSP_Troughs"])]
    rsp_names = rsp_names + ["RSP_npeaks"] + ["ntroughs"]
    return rsp_features
    
    
def get_rsp(fsignal,df,rootdir):
    """
    Get respiration features
    
    Parameters:
    ----------
    fsignal:
    df:
    rootdir:
        
    Return:
    -------
    rsp_features:
    rsp_names:
    
    """
    signal_type = "RSP"
    rsp_features = []
    rsp_names = []
    df_output, info = nk.rsp_process(fsignal,sampling_rate=1000)
    l = min(len(info["RSP_Peaks"]),len(info["RSP_Troughs"]))
    peaks = info["RSP_Peaks"][:l]
    troughs =  info["RSP_Troughs"][:l]
    cleansignal = df_output["RSP_Clean"].values
    outbreath = cleansignal[peaks]-cleansignal[troughs]
    inbreath  = cleansignal[peaks[1:]]-cleansignal[troughs[:-1]]
    info["RSP_InBreath"] = inbreath.astype(int)
    info["RSP_OutBreath"] = outbreath.astype(int)
    features_names = [  "RSP_Amplitude",
                        "RSP_Rate",
                        "RSP_Phase",
                        "RSP_Phase_Completion",
                        "RSP_InBreath",
                        "RSP_OutBreath"]
    
    for feature in features_names:
        if feature not in ["RSP_InBreath","RSP_OutBreath"]:
            statdict = getStats(df_output[feature].values)
        else:
            statdict = getStats(info[feature])
        rsp_features = rsp_features + list(statdict.values()) 
        name = [feature+"_"+ele for ele in list(statdict.keys())]
        rsp_names = rsp_names + name 
            
        
    rsp_features = rsp_features+ [len(info["RSP_Peaks"])] + [len(info["RSP_Troughs"])]
    rsp_names = rsp_names + ["RSP_npeaks"] + ["ntroughs"]
    """
    rsp_features_baseline = rsp_baseline_correction(df,rootdir)
    for j,feat in enumerate(rsp_features_baseline):
        try:
            r= rsp_features[j]/feat
            r = np.nan if r==np.inf else r
            rsp_features[j] = r
        except:
            rsp_features[j] = np.nan"""
            
    return rsp_features,rsp_names
    
    
    
    
def get_time_features(signal_types,
                      fsignals,
                      sampling_rate,
                      df,
                      rootdir):
    """
    Process the physiological signal and extract time features from it.
    
    EDA signal main event indicator is the Skin Conductance Response SCR. 
    EDA filtering: d2-correc --> medfilt 31ms --> MIT labeling --> neurokit2 (lowpass 3hz-o4)
    The function computes:
        SCR_Onsets: the samples at which the onsets of the peaks occur (1 or 0)
        SCR_Peaks: the samples at which the peaks occur (1 or 0)
        SCR_Height: the SCR amplitude of the signal including the Tonic component
        SCR_Amplitude: the SCR amplitude of the signal excluding the Tonic component
        SCR_RiseTime: the time taken for SCR onset to reach peak amplitude within the SCR
        SCR_Recovery: the samples at which SCR peaks recover (decline) to half amplitude (1 or 0)
        SCR_RecoveryTime: the time taken for recovery from the event
        

    for ECG signal:
        check bands http://ems12lead.com/2014/03/10/understanding-ecg-filtering/#gref
        filtering is done as follow (from neurokit2)
           1- High pass filter 0.5Hz order 5
           2- moving avg with 20 ms = 20 pts window size (to remove power line 50Hz noise) 
        
        
    for RSP signal:
        filtering: butter band pass [0.05Hz,3Hz] order 2  (neurokit2)

    Parameters
    ----------
    signal_types : list of string, types of signal in capital
    fsignals: list of np.array or pd.DaFrame or list (artifact removed) signal
    it should have the same order as signal_types
    sampling_rate: int, sampling frequency (here 1000Hz)
    df: pd.DataFrame, to get identification information (slalomtyoe,
                                                         subject,slalom index)
    
    Returns
    -------
    time_features: np.array, time related features 

    """
    time_features = []
    columns = []
    for i,fsignal in enumerate(fsignals):
        signal_type = signal_types[i]
        if  signal_type in ["EDA","EDA".lower()]:
            print ("time feature extraction eda")
            fsignal = fsignal.values.astype(np.float64)
            scr_features,scr_names = get_scr(fsignal,sampling_rate)
            time_features = time_features + scr_features
            columns = columns + scr_names

        if signal_type in [ "ECG","ECG".lower()]:
            print ("time feature extraction ecg")
            hrv_features,hrv_names = get_hrv(fsignal,df,rootdir)
            time_features = time_features + hrv_features
            columns = columns  + hrv_names
            
        if signal_type in ["RSP","RSP".lower()]:
            print ("time feature extraction rsp")
            rsp_features,rsp_names = get_rsp(fsignal,df,rootdir)
            time_features = time_features + rsp_features
            columns = columns  + rsp_names
        
    
    time_features = np.array(time_features)
    time_features = pd.DataFrame(time_features).T
    time_features.columns = columns
    
    return time_features
        
  
    
    
def calculate_wavelet_features(list_coeffs):
    
    """
    Compute a feature vector resulting from wavelets decomposition
    at different level; We  only take details coefficients when 
    computing features
    
    Parameters:
    ----------
    list_coeffs: list of np.arrays, wavelets coefficients 
    its length is n+1. 
    
    """
    wavelets_features = []
    for coeff in list_coeffs[1:]:
        wavelets_features.append(getStats(coeff))
    return wavelets_features




def get_time_frequency_features(signal_types,
                                fsignals,
                                wtypes,
                                levels):
    """
    Apply DWT on the signal. The sampling frequency fs of the signal is 1000Hz.
    at level j, DWT computes coeffients in the interval [fs/2^j, fs/(2^j+1)].
    DWT applies two filters at each decomposition level. 
    A low pass one that lets only low frequency passes, it captures approximation
    coefficients and has a cutoff frequency of fs/(2^j+1) at level j of decomposition.
    The second one is a high pass frequency filter that captures details 
    coefficients and has a cutoff frequency fs/2^j at level j of decomposition.

    Parameters
    ----------
    signal_types: list of string, signal types eda,ecg ...
    fsignals : list of np.array, filtered signals. Same order as signal_types
    wtypes : list of list of string, each element is a list of wavelet types to be applied on signal
    levels : list of list of int, decomposition levels. Same order as wtypes

    Returns
    -------
    wavelets_features : np.array, wavelet feature vector for all signals
                        signal_type+wtype+level+stat
    """
    wavelets_features = []
    cols = []
    for i,fsignal in  enumerate(fsignals):
        signal_type=signal_types[i]
        for j,_ in enumerate(wtypes[i]):
            wtype = wtypes[i][j]
            level=levels[i][j]
            list_coeffs = pywt.wavedec(fsignal,
                                       wtype,
                                       level=level)
            list_wave_dict = calculate_wavelet_features(list_coeffs)
            wavelets_features = wavelets_features + list_wave_dict
            name = signal_type+"_"+wtype+"_l"
            z = [[name+str(level-j)+"_"+e for e in list(d_.keys())] for j,d_ in enumerate(list_wave_dict)]
            cols.append(z)
            
    # aggregate features values
    wavelets_features = [list(d.values()) for d in wavelets_features]
    wavelets_features = [item for sublist in wavelets_features for item in sublist]
    
    # aggregate features names
    cols = [item for sublist in cols for item in sublist]
    cols = [item for sublist in cols for item in sublist]
    
    # create dataframe 
    wavelets_features = pd.DataFrame(wavelets_features).T
    wavelets_features.columns = cols
    
    return wavelets_features
    

def get_all_features(feature_dict,
                     sampling_rate,
                     df,rootdir):
    """
    Get all features wavelets + temporal
    
    Parameters:
    ----------
    feature_dict: dictionary.
        Contains signal types, signal values,wavelet types and levels to be used
    sampling_rate: int, sampling rate of the signal (here 1000Hz)
    df:pd.DataFrame.
        To get information about identification information (subject, slalom,..)
    rootdir:string.
        to get path of the dataframe used for ecg baseline correction

    Return:
    ------
    all_features: np.array, one dimensional feature array
    """
    tfeatures = get_time_features(feature_dict["signal_types"],
                                  feature_dict["fsignals"],
                                  sampling_rate,
                                  df,
                                  rootdir)
    tffeatures = get_time_frequency_features(feature_dict["signal_types"],
                                            feature_dict["fsignals"],
                                            feature_dict["wtypes"],
                                            feature_dict["levels"])
    
    features = pd.concat([tfeatures, tffeatures], axis=1)
    return features
    
def dataset_correction(df,method="median",alpha=1,beta=3):
    """
    function to detect and correct anomalies in the final dataframe
    It can be applied to initial and normalized dataframes. It depends on
    either the median or mean. Default is median.
    A value is considered abnormal if it lies outside this following
    interval : [alpha*median -beta*std, alpha*median +beta*std]
    
    It is a personalized approach, median and std are computed for each subject.
    Abnormal values are corrected via a linear interpolation given discrete data 
    for ordered slaloms.
    
    Parameters:
    ----------
    df: pd.DataFrame, dataframe to which corrections are applied
    method: string, specify the measure on which outlier detection is based
    alpha: float, median (or mean) coeffcient. Default to 1. 
    beta: float, std coefficient. Default to 3.
    Returns
    -------
    new_df: pd.DataFrame, dataframe with the corrected values
    info: dictionary, contains information about the modifications
    
    """
    k=0
    info = {}
    one_df_modification = []
    df.reset_index(inplace=True,drop=True)
    new_df = df.copy(deep=True)
    gps_stype = [g[1] for g in df.groupby(by="Slalomtype")]
    for gp_stype in gps_stype:
        gps_stype_sindex = [g[1] for g in gp_stype.groupby(by="Subject")]
        cols = gps_stype_sindex[0].columns
        for gp_stype_sindex in gps_stype_sindex:
            for col in cols[:-4]:          
                gp_stype_sindex["slalom"]=gp_stype_sindex["slalom"].astype(int)
                gp_stype_sindex.sort_values(by="slalom",inplace=True)
                ids = np.array(list(gp_stype_sindex[col].index))
                array = gp_stype_sindex[col].values
                median = gp_stype_sindex[col].median()
                mean =  gp_stype_sindex[col].mean()
                std = gp_stype_sindex[col].std()
                
                if method == "median":
                    sup=alpha*median+beta*std
                    inf=alpha*median-beta*std
                if method == "mean":
                    sup=alpha*mean+beta*std
                    inf=alpha*mean-beta*std
                    
                         
                js=[]
                xp=[]
                fp=[]
                
                for j,ele in enumerate(array):
                    if inf <=array[j]<=sup: # detection of outliers
                        xp.append(j)
                        fp.append(array[j])
                    else:
                        js.append(j)
                js=np.array(js)       
                if len(js)!=0: # replace outliers
                    new_df.loc[ids[js],col] = np.interp(js,xp,fp) 
                    mod_df = new_df.loc[ids[js],[col,"Subject","slalom"]]
                    mod_df.columns = [col + " (after modification)","Subject","slalom"]
                    mod_df[col+" (before modification)"] = array[j]
                    one_df_modification.append(mod_df)
                    k+=1
    info["#modifications"] = k
    info["wh_modifications"] = one_df_modification
    
    return new_df, info


def normalize(dataset_baseline, dataset_slaloms):
    """
    Normalize slaloms values w.r.t. baseline

    Parameters
    ----------
    dataset_baseline : pd.DataFrame, 
        dataframe of the baseline (vehicule or arrêt) depending on the parameter 
        <<normalize_wrt>> in the config file .
    dataset_slaloms : pd.DataFrame,
        dataframe of the slaloms.

    Returns
    -------
    normalized : pd.DataFrame,
        normalized dataFrame.

    """
    
    normalized = pd.DataFrame().reindex_like(dataset_slaloms)
    slcols = list(dataset_slaloms.columns)

    for j,row in dataset_slaloms.iterrows():
        for k,slcol in enumerate(slcols[:-4]):
            slalomvalue = dataset_slaloms[slcol].iloc[j]
            si,st = dataset_slaloms.iloc[j]["Subject"], dataset_slaloms.iloc[j]["Slalomtype"]
            baselinevalue = dataset_baseline[slcol][(dataset_baseline["Subject"]== si)& (dataset_baseline["Slalomtype"]==st)].values[0]
            try:
                normalized_value = slalomvalue/baselinevalue
            except ZeroDivisionError:
                normalized_value=np.inf
            normalized[slcol].iloc[j] = normalized_value
            
    normalized["score"] = dataset_slaloms["score"]
    normalized["Subject"] = dataset_slaloms["Subject"]
    normalized["Slalomtype"] = dataset_slaloms["Slalomtype"]
    normalized["slalom"] = dataset_slaloms["slalom"]
    normalized.replace([np.inf, -np.inf], np.nan, inplace=True)
    normalized.dropna(how="any",axis=1,inplace=True)
    return normalized


def save_dataframe(ph,featuresdir,dataframe):
    name = "dataset_" + ph.split("_")[1]
    path = os.path.join(featuresdir,name)
    dataframe.replace([np.inf, -np.inf], np.nan,inplace=True)
    dataframe.dropna(axis=1,how="any",inplace=True)
    assert dataframe.isnull().sum().max()==0
    dataframe.to_pickle(path)
    
    
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
    job = config["feature extraction"]["job"]
    ph = config["feature extraction"]["ph"]
    method = config["feature extraction"]["method"]
    alpha = float(config["feature extraction"]["alpha"])
    beta = float(config["feature extraction"]["beta"])
    featuresdir = config["feature extraction"]["featuresdir"]
    normalize_wrt = config["feature extraction"]["normalize_wrt"]
    
    if job == "bare":
        di = config["handling files"]["baredir"]
    if job == "split":
        di = config["handling files"]["splitdir"]
    
    
    files = [os.path.join(di,file) for file in os.listdir(di) if file.startswith("subject") and ph in file]
    list_of_features = []
    for i in range(len(files)):
       try:
           df = pd.read_pickle(files[i])
           feature_dict = {"signal_types":["EDA","ECG","RSP"],
                           "fsignals":[df["artReplaced_eda"],df["normal_ECG"],df["RSP"]],
                           "wtypes":[["haar","db4","db10"],["coif5","db4"],["db4"]],
                           "levels":[[4,4,8],[14,8],[8]]}
           #print (df)
           features = get_all_features(feature_dict,1000,df,"./")
           #print (features)
           if ph not in ["_re_","_ba_","_bv_"]:
               id_features = df[["Subject","Slalomtype","slalom","score"]].iloc[0]
               id_features = pd.DataFrame(id_features.values.T).T
               id_features.columns = ["Subject","Slalomtype","slalom","score"]
           else:
               id_features = df[["Subject","Slalomtype","score"]].iloc[0]
               id_features = pd.DataFrame(id_features.values.T).T
               id_features.columns = ["Subject","Slalomtype","score"]
               
           features = pd.concat([features, id_features],axis=1)
           try:
               features.drop(columns =["dfa_alpha1","dfa_alpha2"],inplace=True,axis=1)
           except:
               pass
           list_of_features.append(features)
           print ("added",i)
           dataframe = pd.concat(list_of_features)
           save_dataframe(ph,featuresdir,dataframe)
       except:
           continue
       
    if job == "split":
        ph = "_corrected_"
        print ("detecting outliers and correcting them")
        dataframe,info = dataset_correction(df=dataframe,
                                  method=method,
                                  alpha=alpha,
                                  beta=beta)
        save_dataframe(ph,featuresdir,dataframe)
        path_ba = os.path.join(featuresdir,"dataset_"+normalize_wrt)
        dataset_baseline = pd.read_pickle(path_ba)
        normalized = normalize(dataset_baseline, dataframe)
        ph = "_normalized-" +normalize_wrt+"_"
        save_dataframe(ph,featuresdir,normalized)

    
    
if __name__ == "__main__":
    configfile = "D:/scripts/config.ini"
    config = get_config(configfile)
    main(config) 
    
    sys.exit() 
    