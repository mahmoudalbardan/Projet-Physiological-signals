# -*- coding: utf-8 -*-
"""
@author: albardan
"""
import os
import configparser

import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.signal import medfilt

import warnings
warnings.filterwarnings("ignore")

def add_columns(df,filename):
    """
    Add columns, mainly subject index and Slalom type extracted from
    file path
    
    Parameters
    ----------
    df: pd.DataFrame, dataframe 
    filename: string, filename
    
    Returns
    -------
    df: pd.DataFrame, dataframe to which columns were added
    """
    df["Subject"] = filename.split('\\')[1].split('_')[1]
    df["Slalomtype"] = filename.split('\\')[2].split('_')[1]
    return df

def sync(df,canfile):
    """
    Function to synchronize CAN and BIOPAC files

    Parameters
    ----------
    df : pd.DataFrame, dataframe read from the biopac file.
    canfile : string, path to the corresponding can file.

    Returns
    -------
    df: pd.DataFrame, synchronized biopac dataframe
    """
    df_can = pd.read_csv(canfile,sep = " ")
    df_can.rename(columns={"Event": "trigger", "Score": "score"},inplace=True)
    time_can = df_can["Time"][df_can["trigger"] =="Aller_Début"].iloc[0]
    
    time_bio = df["Time"][df["trigger"]=="Aller_Début"].iloc[0]
    delta = time_can - time_bio 
    tvec = np.arange(0,delta-1/1000,1/1000) # time vector to add
    to_add = df.iloc[0].to_frame().T
    to_add = pd.concat([to_add]*tvec.shape[0]) # tiling df.iloc[0] above df 
    to_add["Time"] = tvec
    df = to_add.append(df)
    df.reset_index(drop=True,inplace=True)
    return df
    
def read_file(biopacfile,canfile):
    """
    Reads .txt file and transform it to a pandas dataframe
    
    Parameters
    ----------
    biopacfile : string, biopac filename to read
    canfile: string, can filename to read

    Returns
    -------
    df : pd.DataFrame, dataframe computed from the file
    """
    columns = [ "Time", "X1", "Y1", "Z1",
                "X2", "Y2" ,"Z2" ,"ECG" ,
                "PPG", "RSP", "EDA", "SKT1" ,
                "EGG1","score","trigger"]
    
    df = pd.read_csv(biopacfile,sep = " ")
    new_files = ["17_Large.txt","8_AleatoireSerre.txt","20_AleatoireLarge.txt"]
    booleans = np.array([biopacfile.endswith(f) for f in new_files])
    if df.shape[0]!=0:
        df.reset_index(inplace=True)
        if booleans.any():
            cols = list(df.columns)
            df.drop(columns=cols[-1],inplace=True)
        if not booleans.any():
            try:
                df = df.drop(index=1,columns="index")
            except:
                pass
        df.columns = columns
        df = add_columns(df,biopacfile)
        
    else:
        df = pd.DataFrame(np.nan, index=[0], columns=columns)
        df = add_columns(df,biopacfile)
    if not booleans.any():
        df = sync(df,canfile)
    return df

def get_filenames(rootdir,starts,ends,contains):
    """
    Gets files names from the directory where the data is saved

    Parameters
    ----------
    rootdir : string, root directory where data can be found
    starts: string, filename starting string
    ends: string, filename ending string
    contains: string, filename has to contain this string 


    Returns
    -------
    files: list of strings, list of paths of the files

    """
    list_of_files = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.startswith(starts) and file.endswith(ends) and contains in file:
               list_of_files.append(os.path.join(subdir, file)) 
    return list_of_files

def process(df):
    #check for  "_" and replace them with the latest trigger
    """
    Process the dataframe according to some conditions
    for example: rows where trigger is _ should be replaced

    Parameters
    ----------
    df : pd.DataFrame, original dataframe
        

    Returns
    -------
    df: pd.DataFrame, processed dataframe

    """
    df =replace(df)
    return df

def is_bound(re,df):
    return True if re[0]==0 or re[-1]==len(df)-1 else False

def replace(df):
    """
    Replace "_" from trigger column by the previous
    trigger. 
    If the "_" trigger have no valid previous (begining of the dataframe)
    or following (end of the dataframe) triggers, then the corresponding part of the dataframe
    is unchanged 
    """
    inds = df["trigger"][df["trigger"] == "_"].index.values
    res, last = [[]], None
    for x in inds:
        if last is None or abs(last - x) ==1:
            res[-1].append(x)
        else:
            res.append([x])
        last = x
      
        
    for re in res:
        if not is_bound(re,df):
            df.loc[df.index[re],"trigger"] = df["trigger"].iloc[re[0]-1]  
    df.reset_index(inplace=True)        
    return df
        
def extract_from_df(df,to_extract):
    """
    Extract some columns from dataframe

    Parameters
    ----------
    df : pd.DataFrame,
    to_extract : list of strings, list of indicators (EDA,RSP,..) to extract

    Returns
    -------
    Extracted columns if no error is raised, else empty dataframe

    """
    try:
        return df[to_extract]
    except:
        return pd.DataFrame([])

def construct_save_dataframes(rootdir,localdir):
    """
    Constructs and saves dataframes from BIOPAC.txt and 
    detection.mat files 
    
    Parameters
    ----------
    rootdir: string, root directory 
    
    """
    biopac_files = get_filenames(rootdir, "BIOPAC_2", ".txt", "")
    can_files = get_filenames(rootdir, "CAN_1_", ".txt", "")
    
    can_files.pop(26) # CAN_1_2018_0_0_16_57_6_Sujet_22_Large_Aléatoire.txt (old date apparantly)
    biopac_files.pop(26) # BIOPAC_2018_0_0_16_57_6_Sujet_22_AleatoireLarge.txt
    
    # biopac_files = [biopac_files[-4],biopac_files[15],biopac_files[22],biopac_files[2]]
    # can_files = [can_files[-4],can_files[15],can_files[22],can_files[2]]
    for i,filename in enumerate(biopac_files):
        try:
            df_subject = read_file(filename,can_files[i])
            df_subject = process(df_subject)
            subject_index = df_subject["Subject"].iloc[0]
            slalomtype = df_subject["Slalomtype"].iloc[0]
            print ("le type du slalom du sujet {suid} is {slid}".format(suid = subject_index, slid = slalomtype))
            name = os.path.join(localdir,"subject_" + subject_index + '_' + slalomtype)
            df_subject.to_pickle(name)
        except:
            continue

    detectionslalom_files = get_filenames(rootdir, "Detectionslaloms", "", "")
    for i,filename in enumerate(detectionslalom_files):
        df_time = pd.DataFrame(loadmat(filename)["time_startend"])
        df_time = add_columns(df_time, filename)
        df_time.columns = ["Starttime", "Endtime","Subject","Slalomtype"]
        subject_index = df_time["Subject"].iloc[0]
        slalomtype = df_time["Slalomtype"].iloc[0]
        name = os.path.join(localdir,"time_" + subject_index + '_' + slalomtype)
        df_time.to_pickle(name)
        
def get_score(df_subject,time_df,j):
    """
    Get the score for a specific slalom
    
    Parameters
    ----------
    df_subject:pd.DataFrame, dataframe of the subject for the whole experience
    time_df: pd.DataFrame, dataframe converted from .mat files
    j: int, slalom index, starts from 0
    
    Return
    ------
    score: float, score given for the slalom j
    
    """
    kernelsize = 1001
    st = time_df.iloc[j]["Endtime"]
    try:
        et = time_df.iloc[j+1]["Starttime"]
    except:
        et = st+15
    
    time = df_subject["Time"].values
    wh = np.where((st<=time) & (time<=et))[0]
    scores = df_subject.loc[wh,"score"].values
    
    scores = df_subject["score"][df_subject["Time"]>=st][df_subject["Time"]<=et]

    kernelsize=51
    filtered_pooled_score = np.max(medfilt(scores.tolist(),kernelsize))
    return filtered_pooled_score

    
def modify_score(df_subject,time_df,slalom_indexes,new_scores,set_to_max):
    """
    function to modify scores in a dataframe according to a list of slaloms 
    and a list of new scores. It was created to modify scores of subjects 1,2,3

    Parameters
    ----------
    df_subject : pd.DataFrame
            dataframe whose scores should be modified.
    time_df : pd.DataFrame
        time dataframe (from detect.mat) .
    slalom_indexes : list of ints
        slaloms whose scores should be modified.
    new_scores : list of floats
        list of modified scores.
    set_to_max: bool, which modification
        if True, set to 4 last slalom,
        if False, correct scores for subjects 1,2,3

    Returns
    -------
    df_subject: pd.DataFrame, initial dataframe with modified scores

    """
    for j in slalom_indexes:
        print (j)
        st = time_df.iloc[j]["Endtime"]
        try:
            et = time_df.iloc[j+1]["Starttime"]
        except:
            et = st+15
        
        time = df_subject["Time"].values
        wh = np.where((st<=time) & (time<=et))[0]
        if not set_to_max:
            df_subject.loc[wh,"score"] = new_scores[j]
        else:
            df_subject.loc[wh,"score"] = 4
    return df_subject


def split_by_slalom(source_directory,destination_directory):
     """
     Reads dataframes containing all slaloms for a single subject, 
     splits them into multiple dataframes each containing one slalom and 
     save them in another directory

     Parameters
     ----------
     
     source_directory: string, source directory where subjects 
     and detection time dataframes are saved. 
     They are saved as 'subject_1_Serre', 'time_1_Serre' ..
     
     destination_directory: string, directory in which slaloms dataframes are saved
     they are saved in this way 'subject_1_slalom_5_Serre'
     
     """
     times = [os.path.join(source_directory,file) for file in os.listdir(source_directory) if file.startswith("time")]
     for time_path in times:
        try:
            print (time_path)
            time_df = pd.read_pickle(time_path)
            subject_index = time_df["Subject"].iloc[0]
            slalomtype = time_df["Slalomtype"].iloc[0]
            df_subject = pd.read_pickle(os.path.join(source_directory,"subject_" + subject_index + "_" + slalomtype))
            for j,row in time_df.iterrows():
                starttime = time_df.iloc[j]["Starttime"]
                endtime = time_df.iloc[j]["Endtime"]
                time = df_subject["Time"].values
                wh = np.where((starttime<=time) & (time<=endtime))[0]
                slalom_index = str(j)
                slalomtype = df_subject["Slalomtype"].iloc[0]
                slalom_df = df_subject.loc[wh,:]
                slalom_df["slalom"] = str(j)
                slalom_df.loc[:,"score"] = get_score(df_subject,time_df,j)
                name = os.path.join(destination_directory,"subject_" + subject_index+
                                    "_slalom_" + slalom_index + "_" + slalomtype)
                slalom_df.to_pickle(name)
        except:
            continue
     
def get_by_subject_and_slalom(slaloms_destination_directory,
                              subject_index,
                              slalomtype,
                              slalom_index):
    """
    Extract and reads data for specific subject, slalom type and slalom index
    extracted according to "detection.mat" files

    Parameters
    ----------
    slaloms_destination_directory: string, directory where dataframes
    are saved. One dataframe per slalom
    subject_index : int, subject index
    slalomtype : string, slalom type
    slalom_index : int, slalom index

    Returns
    -------
    pd.DataFrame read from slaloms_destination_directory 

    """
    name = "subject_" + str(subject_index) + "_slalom_" + str(slalom_index) + "_" + slalomtype
    path = os.path.join(slaloms_destination_directory,name)
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        raise FileNotFoundError("""This file is not found in the data folder""")
    

def make_modifications(localdir):
    """
    Make modifications on the complete dataframes

    Parameters:
    ----------
    localdir: string, path to the directory where completed dataframes are saved

    """
    triggers = ['_','Baseline_Arrêt_Start', 'Baseline_Arrêt_Stop',
               'Baseline_Véhicule_Start','Baseline_Véhicule_Stop',
               'Aller_Début', 'Aller_Fin',
               'Retour_Début', 'Retour_Fin',
               'Récupération_Début','Récupération_Fin']
    triggers_indexes = list(range(len(triggers)))
    dictionary = dict(zip(triggers,triggers_indexes))

     # make modifications on the complete dataframes for missing or badly positioned triggers
    files = [os.path.join(localdir,file) for file in os.listdir(localdir) if file.startswith("subject")]
    for file in files:
        stype = file.split("_")[-1]
        subject_index = file.split("_")[-2]
        if stype=="Serre" and subject_index=="15":
            df = pd.read_pickle(file)
            ed = df["trigger"][df["trigger"]=="Récupération_Début"].index[0] 
            st = ed - 5*60*1000
            df["trigger"].loc[st:ed] = "Récupération_Début"
            df.to_pickle(file)
            print (stype,subject_index," bad positioned trigger done")
            # x = df["trigger"].map(dictionary).values
            # plt.plot(x)
        if stype=="AleatoireSerre" and subject_index=="1":
            df = pd.read_pickle(file)
            st = df["trigger"][df["trigger"]=="Récupération_Début"].index[0] 
            ed = 1774709
            df["trigger"].loc[st:ed] = "Aller_Début"
            df["trigger"][df["trigger"]=="Récupération_Fin"] = "Récupération_Début"
            df.to_pickle(file)
            print (stype,subject_index," bad positioned trigger done")
            # x = df["trigger"].map(dictionary).values
            # plt.plot(x)
        if stype=="AleatoireSerre" and subject_index=="2":
            df = pd.read_pickle(file)
            to_drop = np.arange(469150,805830,1)
            df.drop(axis=0,inplace=True,labels=to_drop)
            df["trigger"][df["trigger"]=="Récupération_Fin"] = "Récupération_Début"
            df.to_pickle(file)
            print (stype,subject_index," bad positioned trigger done")
            # x = df["trigger"].map(dictionary).values
            # plt.plot(x)
        if stype=="AleatoireSerre" and subject_index=="5":
            df = pd.read_pickle(file)
            st,ed=981090,987769
            df["trigger"].loc[st:ed] = "Retour_Fin"
            df.to_pickle(file)
            print (stype,subject_index,"done")
            # x = df["trigger"].map(dictionary).values
            # plt.plot(x)
            
        # make modifications on the complete dataframes when combining two files
        if stype=="Serre" and subject_index=="3":
            df = pd.read_pickle(file)
            bio_path1 = "F:/data/1/Sujet_3_300718/Session_Serre/BIOPAC_2018_0_0_15_28_1_Stéphanie.txt"
            columns = [ "Time", "X1", "Y1", "Z1",
                        "X2", "Y2" ,"Z2" ,"ECG" ,
                        "PPG", "RSP", "EDA", "SKT1" ,
                        "EGG1","score","trigger"]
            df1 = pd.read_csv(bio_path1,sep = " ")
            if df1.shape[0]!=0 :
                df1.reset_index(inplace=True)
                try:
                    df1 = df1.drop(index=1,columns = "index")
                except:
                    pass
                df1.columns = columns
                
            df1["Subject"] = "3"
            df1["Slalomtype"] = "Serre"
            
            ed1 = df1["Time"].iloc[-1]
            df["Time"] = df["Time"]  + ed1
            df = pd.concat([df1,df])
            df.to_pickle(file)
            
            time_path=os.path.join(localdir,"time_3_Serre") 
            time_df = pd.read_pickle(time_path)
            time_df[["Starttime","Endtime"]] = time_df[["Starttime","Endtime"]] + ed1 # change time_df 
            time_df.to_pickle(time_path)
            print (stype,subject_index,"combination of two files done")
                            
        if stype=="AleatoireLarge" and subject_index=="22":
            df = pd.read_pickle(file)
            bio_path1 = "F:/data/1/Sujet_22_040918/Session_AleatoireLarge/BIOPAC_2018_0_0_16_57_6_Sujet_22_AleatoireLarge.txt"
            columns = [ "Time", "X1", "Y1", "Z1",
                        "X2", "Y2" ,"Z2" ,"ECG" ,
                        "PPG", "RSP", "EDA", "SKT1" ,
                        "EGG1","score","trigger"]
            df1 = pd.read_csv(bio_path1,sep = " ")
            if df1.shape[0]!=0 :
                df1.reset_index(inplace=True)
                try:
                    df1 = df1.drop(index=1,columns = "index")
                except:
                    pass
                df1.columns = columns
             
            df1["Subject"] = "22"
            df1["Slalomtype"] = "AleatoireLarge"
            
            ed1 = df1["Time"].iloc[-1]
            df["Time"] = df["Time"]  + ed1
            df = pd.concat([df1,df])
            df.to_pickle(file)
            
            time_path=os.path.join(localdir,"time_22_AleatoireLarge") 
            time_df = pd.read_pickle(time_path)
            time_df[["Starttime","Endtime"]] = time_df[["Starttime","Endtime"]] + ed1 # change time_df 
            time_df.to_pickle(time_path)
            print (stype,subject_index,"combination of two files done")
            
        # # make modifications on slaloms dataframes of scores of recupération (email 15 JUIN)  
        recup_scores = {
    "Large": {"11":4,
              "12":2.7,
              "14":1.994,
              "16":2.83,
              "18":1.24,
              "20":1.884,
              "21":3.58,
              "22":1.476,
              "24":1.873,
              "7":0.487,
              "9":1.6,
              "17":1.978},
                        "Serre": {"10":0,
                                  "13":1,
                                  "15":2.936,
                                  "19":0.644,
                                  "1":0,
                                  "23":0.95,
                                  "2":2.75,
                                  "3":0,
                                  "4":0.251,
                                  "5":0,
                                  "6":3,
                                  "8":1.25},       
                                        "AleatoireLarge": {"11":4,
                                                           "12":3.339,
                                                           "14":2.167,
                                                           "16":3.297,
                                                           "17":2.418,
                                                           "18":4,
                                                           "21":2.81,
                                                           "22":0,
                                                           "24":2.072,
                                                           "7":1,
                                                           "9":0,
                                                           "20":2.5},
                                                                    "AleatoireSerre": {"10":0,
                                                                                      "13":0.706,
                                                                                      "15":2.983,
                                                                                      "19":1.837,
                                                                                      "1":1.5,
                                                                                      "23":0.251,
                                                                                      "2":0.5,
                                                                                      "3":0.5,
                                                                                      "4":0.55,
                                                                                      "5":1.0833,
                                                                                      "8":1.021}
}
        df = pd.read_pickle(file)
        df["score"][df["trigger"].isin(["Récupération_Début",
                                        "Récupération_Fin"])] = recup_scores[stype][subject_index]
        print (stype,subject_index,"modify recup scores done")

    # make modifications on slaloms dataframes of scores for subjects 1,2,3 (email 15 JUIN)
    subject123_dir = '//zfs-b232.enst.fr/albardan/Desktop/Scores corrigés sujet1-3'
    files = [os.path.join(subject123_dir,file) for file in os.listdir(subject123_dir)]
 
    for j,file in enumerate(files):
        sindex = file.split("_")[-2]
        stype = "AleatoireSerre" if file.split("_")[-1].startswith("Alea") else "Serre"
        print ("modifying the scores of subject", sindex," for condition ",stype)
        time_df = pd.read_pickle(os.path.join(localdir,"time_"+sindex+"_"+stype))
        
        if sindex=="3" and stype=="Serre": # to correct the concatenation effect
            time_df[["Starttime","Endtime"]] = time_df[["Starttime","Endtime"]]-460.8989868
        
        df_can = pd.read_excel(file, index_col=0)
        df_can["Time"] = df_can.index
        df_can["score"] = df_can["Score"]
        df_can.reset_index(inplace=True,drop=True)
        
        biopac_name = "subject_"+sindex+"_"+stype
        df_biopac = pd.read_pickle(os.path.join(localdir,biopac_name))
        slaloms_indexes = list(range(time_df.shape[0]))
        new_scores = [get_score(df_can,time_df,j) for j in slaloms_indexes]
        df_biopac = modify_score(df_biopac,time_df,slaloms_indexes,new_scores,set_to_max=False)
        df_biopac.to_pickle(os.path.join(localdir,biopac_name))
    print ("modify scores for subjects 1, 2 and 3")

           
    # make modification for early stoppings: make them 4(email 15 JUIN)
    # WARNING: slalom indexes starts from 0
    early_stoppings= ["subject_18_slalom_1_AleatoireLarge",
                     "subject_6_slalom_25_Serre",
                     "subject_14_slalom_13_Large",
                     "subject_18_slalom_2_Large",
                     "subject_4_slalom_21_AleatoireSerre"]
    for f in early_stoppings:
        filename = "_".join(np.array(f.split("_"))[np.array([0,1,4])].tolist())
        filename = os.path.join(localdir,filename)
        subject_index = f.split("_")[1]
        stype = f.split("_")[-1]
        print ("set the score of the last slalom for subject ", subject_index, " to 4")
        df=pd.read_pickle(filename)
        time_df = pd.read_pickle(os.path.join(localdir,"time_"+subject_index+"_"+stype))
        df = modify_score(df,time_df,[int(f.split("_")[-2])],[4],set_to_max=True)
        df.to_pickle(file)
    print ("set scores to 4 for early stoppings")


def get_ba(df):
    """
    get baseline arret file
    
    Parameters
    ----------
    df : pd.DataFrame
        complete dataframe.

    Returns
    -------
    ba : pd.DataFrame
    
    """
    indexes_arret = df[df["trigger"].isin(["Baseline_Arrêt_Start",
                                           "Baseline_Arrêt_Stop",
                                        ])].index.values         
    
    if len(indexes_arret)!=0:
        diff = np.diff(df["EDA"])
        wh = np.where(diff!=0)[0][0]
        indexes_arret = indexes_arret[indexes_arret > wh]
        ba = df.loc[indexes_arret,:].iloc[-120000:]
        
    return ba

def get_bv(df):
    """
    get baselinevéhicule file

    Parameters
    ----------
    df : pd.DataFrame
        complete dataframe.

    Returns
    -------
    bv : pd.DataFrame

    """
    indexes_vehicule = df[df["trigger"].isin(["Baseline_Véhicule_Start",
                                               "Baseline_Véhicule_Stop"
                                               ])].index.values
    if len(indexes_vehicule)!=0:
        bv = df.loc[indexes_vehicule,:].iloc[-120000:]
    return bv


def get_re(df):  
    """
    get recuperation file

    Parameters
    ----------
    df : pd.DataFrame
        complete dataframe.

    Returns
    -------
    re : pd.DataFrame
    """
    indexes_recup = df[df["trigger"].isin(["Récupération_Début",
                                           "Récupération_Fin"])].index.values
    if len(indexes_recup)!=0:
        re = df.loc[indexes_recup,:]
    return re


def save_baseline_recup(dir1,dir2):
    """
    save baseline and recuperation phases

    Parameters
    ----------
    dir1 : str,
        source directory of combined files.
    dir2 : str,
        destination directory of combined files.

    """
    files = [os.path.join(dir1,file) for file in os.listdir(dir1) if file.startswith("subject")]
    for j,file in enumerate(files):
        sindex = file.split("_")[-2]
        stype  = file.split("_")[-1]
        df = pd.read_pickle(file)
        ba = get_ba(df)
        bv = get_bv(df)
        re = get_re(df)

        path_ba = os.path.join(dir2,"subject_"+sindex+"_ba_"+stype)
        path_bv = os.path.join(dir2,"subject_"+sindex+"_bv_"+stype)
        path_re = os.path.join(dir2,"subject_"+sindex+"_re_"+stype)
        
        ba.to_pickle(path_ba)
        bv.to_pickle(path_bv)
        re.to_pickle(path_re)
    
    
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
    job = config["handling files"]["job"]
    rootdir = config["handling files"]["rootdir"]
    localdir = config["handling files"]["localdir"]
    splitdir = config["handling files"]["splitdir"]
    destinationdir = config["ECG processing"]["destinationdir"]
    if job == "construct":
        construct_save_dataframes(rootdir,localdir)
        print ("modification begins")
        make_modifications(localdir)
    if job == "split":        
        split_by_slalom(destinationdir,splitdir)
    if job == "bare":
        dir1 = config["ECG processing"]["destinationdir"]
        dir2 = config["handling files"]["baredir"]
        save_baseline_recup(dir1,dir2)
        
        
        
        
        
if __name__ == "__main__":  
    configfile = "D:/scripts/config.ini"
    config = get_config(configfile)
    main(config)
        
        
        
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
"""
# correction of scores. Set the score of the final slalom to 4 
files = ["subject_18_slalom_1_AleatoireLarge",
         "subject_6_slalom_25_Serre",
         "subject_14_slalom_13_Large",
         "subject_18_slalom_2_Large",
         "subject_4_slalom_21_AleatoireSerre"]
files = [os.path.join(slaloms_destination_directory,file) for file in files]
for file in files:
    df = pd.read_pickle(file)
    df["score"] = 4
    df.to_pickle(file)
"""
"""



dir_ = '//zfs-b232.enst.fr/albardan/Desktop/Scores corrigés sujet1-3'
source_directory = 'D:/data_1/dataframes/all_slaloms'
files = [os.path.join(dir_,file) for file in os.listdir(dir_)]
slaloms = [ file for file in os.listdir(slaloms_destination_directory) if "subject" in file]
slaloms = [file for file in slaloms if int(file.split("_")[1]) in [1,2,3]]


#files = [files[0]]
for j,file in enumerate(files):
    sindex = file.split("_")[-2]
    stype = "AleatoireSerre" if file.split("_")[-1].startswith("Alea") else "Serre"
    
    time_df = pd.read_pickle(os.path.join(source_directory,"time_"+sindex+"_"+stype))
    
    df_subject = pd.read_excel(file, index_col=0)
    df_subject["Time"] = df_subject.index
    df_subject["score"] = df_subject["Score"]
    df_subject.reset_index(inplace=True,drop=True)
    
    for slalomfile in slaloms:
        slalomsubject =  slalomfile.split("_")[1]
        slalomtype  = slalomfile.split("_")[-1]
        slalomindex = int(slalomfile.split("_")[-2])
        if slalomsubject==sindex and slalomtype==stype:
            print (slalomsubject,sindex,slalomtype,stype,slalomindex)
            newscore = get_score(df_subject,time_df,slalomindex)
            path = os.path.join(slaloms_destination_directory,slalomfile)
            sdf = pd.read_pickle(path)
            sdf["score"] = newscore
            sdf.to_pickle(path)
            #print (sdf["score"].unique(),newscore,"saved")
"""


