# -*- coding: utf-8 -*-
"""
@author: albardan
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from denoising import *
import matplotlib.pyplot as plt 


def add_columns(df,filename):
    """
    Add columns, mainly subject index and Slalom type extracted from
    file path
    
    Parameters
    ----------
    
    
    Returns
    -------
    df: pd.DataFrame, dataframe to which columns were added
    """
    df["Subject"] = filename.split('\\')[1].split('_')[1]
    df["Slalomtype"] = filename.split('\\')[2].split('_')[1]
    return df

def read_file(filename):
    """
    Reads .txt file and transform it to a pandas dataframe
    
    Parameters
    ----------
    filename : string, filename to read


    Returns
    -------
    df : pd.DataFrame, dataframe computed from the file
    """
    columns = [ "Time", "X1", "Y1", "Z1",
                "X2", "Y2" ,"Z2" ,"ECG" ,
                "PPG", "RSP", "EDA", "SKT1" ,
                "EGG1","score","trigger"]
    
    df = pd.read_csv(filename,sep = " ")
    if df.shape[0]!=0:
        df.reset_index(inplace=True)
        try:
            df = df.drop(index=1,columns = "index")
        except:
            pass
        df.columns = columns
        df = add_columns(df,filename)
        
    else:
        df = pd.DataFrame(np.nan, index=[0], columns=columns)
        df = add_columns(df,filename)
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
    """
    Process the dataframe according to some conditions
    for example: rows where trigger is _ should be removed

    Parameters
    ----------
    df : pd.DataFrame, original dataframe
        

    Returns
    -------
    df: pd.DataFrame, processed dataframe

    """
    df = df[df["trigger"] != "_"]
    df.reset_index(inplace=True)  
    return df
    

def extract_from_df(df,to_extract):
    """
    Extract some columns from dataframe

    Parameters
    ----------
    df : pd.DataFrame,
    to_extract : list of strings, list of indicators (EDA,Rsp,..) to extract

    Returns
    -------
    Extracted columns if no error is raised, else empty dataframe

    """
    try:
        return df[["Time","EDA","RSP","score","trigger"]]
    except:
        return pd.DataFrame([])













































if __name__ == "__main__":   

    # Explarotary data analysis
    # reading the data and constructing pd.dataframes
    rootdir = 'D:/data_1'
    biopac_files = get_filenames(rootdir, "BIOPAC_", ".txt", "")
    list_of_dataframes = []
    for i,filename in enumerate(biopac_files):
        df = read_file(filename)
        df = process(df)
        print (df.head())
        list_of_dataframes.append(df) 
        
        
        
    
    # Extract eda and respiration signals from dataframes
    to_extract = ["Time","EDA","RSP","score","trigger"]
    signals = [extract_from_df(df,to_extract) for df in list_of_dataframes]
    
    
    
    # split by slalloms  
    slaloms_start_end = []
    detectionslalom_files = get_filenames(rootdir, "Detectionslaloms", "", "")
    for i,filename in enumerate(detectionslalom_files):
        slaloms_start_end.append(loadmat(filename))






signal = signals[0]["EDA"]
time = signals[0]["Time"]
plt.figure()
plt.plot(time,signal,color='r')
plt.show()


signal = signal.values
time = time.values

fig,ax = plt.subplots(figsize=(15,6),nrows=2, ncols=2)
ax = ax.flatten()
denoised_signal1 = wavelet_denoising(signal,"db4",4,"VisuShrink","hard")
ax[0].plot(time,denoised_signal1,color="blue")
ax[0].plot(time,signal,color="red")


denoised_signal2 = wavelet_denoising(signal,"db4",4,"SureShrink","hard")
ax[1].plot(time,denoised_signal1,color="blue")
ax[1].plot(time,signal,color="red")


denoised_signal3 = wavelet_denoising(signal,"db4",4,"VisuShrink","soft")
ax[2].plot(time,denoised_signal1,color="blue")
ax[2].plot(time,signal,color="red")


denoised_signal4 = wavelet_denoising(signal,"db4",4,"SureShrink","soft")
ax[3].plot(time,denoised_signal1,color="blue")
ax[3].plot(time,signal,color="red")

plt.show()









# Sampling frequency 1000Hz
# time = df["Time"].values
# diff = np.diff(time)
# fig, ax = plt.subplots(nrows=1,ncols=1)
# ax.hist(diff,bins=20)
    
    
