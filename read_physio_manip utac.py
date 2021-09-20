# -*- coding: utf-8 -*-
"""
@author: albardan
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import medfilt



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
    df["Subject"] = filename.split("\\")[1].split("t")[1] 
    df["Slalomtype"] = filename.split("\\")[2]
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
    
    
def read_file(biopacfile):
    """
    Reads .txt file and transform it to a pandas dataframe
    
    Parameters
    ----------
    biopacfile : string, biopac filename to read

    Returns
    -------
    df : pd.DataFrame, dataframe computed from the file
    """
    columns = [ "Time", "X1", "Y1", "Z1",
                 "X2", "Y2" ,"Z2","X3",
                 "Y3" ,"Z3" ,"RSP", "PPG",
                 "ECG","SKT1_B", "SKT2_B", "EDA",
                 "score","trigger"]
    
    df = pd.read_csv(biopacfile,sep = " ")
    if df.shape[0]!=0:
        df.reset_index(inplace=True)
        try:
            df = df.drop(index=1,columns = "index")
        except:
            pass
        df.columns = columns
        df = add_columns(df,biopacfile)
        
    else:
        df = pd.DataFrame(np.nan, index=[0], columns=columns)
        df = add_columns(df,biopacfile)
    #df = sync(df,canfile) # we do not need to synchronize in UTAC manip
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
    for i,filename in enumerate(biopac_files):
        try:
            print (filename)
            df_subject = read_file(filename)
            df_subject = process(df_subject)
            subject_index = df_subject["Subject"].iloc[0]
            slalomtype = df_subject["Slalomtype"].iloc[0]
            print ("le type du slalom du sujet {suid} is {slid}".format(suid = subject_index, slid = slalomtype))
            name = os.path.join(localdir,"dataframes/all_slaloms/subject_" + subject_index + '_' + slalomtype)
            df_subject.to_pickle(name)
        except:
            continue

    # detectionslalom_files = get_filenames(rootdir, "Detectionslaloms", "", "")
    # for i,filename in enumerate(detectionslalom_files):
    #     df_time = pd.DataFrame(loadmat(filename)["time_startend"])
    #     df_time = add_columns(df_time, filename)
    #     df_time.columns = ["Starttime", "Endtime","Subject","Slalomtype"]
    #     subject_index = df_time["Subject"].iloc[0]
    #     slalomtype = df_time["Slalomtype"].iloc[0]
    #     name = os.path.join(localdir,"dataframes/all_slaloms/time_" + subject_index + '_' + slalomtype)
    #     df_time.to_pickle(name)
        

def get_score(df_subject,time_df,j):
    """
    Get the score for a specific slalom
    
    Parameters
    ----------
    df_subject:pd.DataFrame, dataframe of the subject for the whole experience
    time_df: pd.DataFrame, dataframe converted from .mat files
    j: int, slalom index
    
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
    filtered_pooled_score = np.max(medfilt(scores.tolist(),kernelsize))
    return filtered_pooled_score
    
    
def split_by_slalom(source_directory,destination_directory):
     """
     Reads dataframes containing all slaloms for a single subject, 
     splits them into multiple dataframes each containing one slalom and 
     save them in another directory

     Parameters
     ----------
     
     source_directory: string, source directory where subjects 
     and detection time dataframes are saved. They are saved as 'subject_1_Serre', 'time_1_Serre' ..
     
     destination_directory: string, directory in which slaloms dataframes are saved
     they are saved in this way 'subject_1_slalom_5_Serre'
     
     """
     times = [os.path.join(source_directory,file) for file in os.listdir(source_directory) if file.startswith("time")]
     print (times)
     for time_path in times:
        try:
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
    



condition_df=pd.DataFrame([["C1",0.14 ,1.5],
                            ["C2",0.14 ,4.5],
                            ["C3",0.4 ,1.5],
                            ["C4",0.4, 4.5],
                            ["C5",0.4, 7],
                            ["C6",0.7, 4.5],
                            ["C7",0.7, 7]],columns = ["condition","frequency","acceleration"])


def write_time_files(localdir,condition_dict):
    """
    """
    dir_ = os.path.join(localdir,"Detections_slaloms")
    files = [file for file in os.listdir(dir_) if file.endswith("Detectionslaloms.mat")]
    for file in files:
        accel = float(file.split("_")[-2])
        freq = float(file.split("_")[-3])
        slalomtype = condition_df["condition"][(condition_df["frequency"]==freq) & 
                                  (condition_df["acceleration"]==accel)].values[0]
        subject_index = file.split("_")[-4][-2:]
        filename = os.path.join(localdir,file)
        name = os.path.join(localdir,"dataframes/all_slaloms/time_" + subject_index + '_' + slalomtype)
        df_time = pd.DataFrame(loadmat(filename)["time_startend"])
        df_time = add_columns(df_time, filename)
        df_time.columns = ["Starttime", "Endtime","Subject","Slalomtype"]
        df_time.to_pickle(name)
        
        
        
    # detectionslalom_files = get_filenames(rootdir, "Detectionslaloms", "", "")
    # for i,filename in enumerate(detectionslalom_files):
    #     df_time = pd.DataFrame(loadmat(filename)["time_startend"])
    #     df_time = add_columns(df_time, filename)
    #     df_time.columns = ["Starttime", "Endtime","Subject","Slalomtype"]
    #     subject_index = df_time["Subject"].iloc[0]
    #     slalomtype = df_time["Slalomtype"].iloc[0]
    #     name = os.path.join(localdir,"dataframes/all_slaloms/time_" + subject_index + '_' + slalomtype)
    #     df_time.to_pickle(name)

if __name__ == "__main__":   

    # Explarotary data analysis
    # reading the data and constructing pd.dataframes
    rootdir = 'F:/data/UTAC_Etude 2019_freq-accel'  
    localdir = 'F:/data/backups/maniputac'
    
    # for the split into slaloms
    slaloms_source_directory = os.path.join(localdir,"dataframes/all_slaloms/combined_dataframes")
    slaloms_destination_directory = os.path.join(localdir,"dataframes/splited_by_slaloms_using_detection.mat")
    
    construct_save_dataframes(rootdir,localdir)
    #split_by_slalom(slaloms_source_directory,slaloms_destination_directory)
    sys.exit()








