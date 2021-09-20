import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pywt
import os
import neurokit2 as nk
import configparser

from scipy.signal import butter, lfilter, medfilt
from ArtifactClassifiers import predict_binary_classifier, predict_multiclass_classifier


def correct_eda(signal,n_max, show_trait=False):
    """
    correction of eda signals based on the second derivative
    fonction proposé par Eric 
    
    
    
    """
    d_eda = np.diff(signal)
    d2_eda = np.diff(d_eda)
    abs_d2 = np.abs(d2_eda)
    # fixation d'un seuil pour les valeurs de dérivée seconde trop forte
    seuil = np.quantile(abs_d2, 0.8) * 20
    #recherche de tous les points avec une dérivée seconde trop forte
    default = np.where(abs_d2 > seuil)[0]
    deb_def = []
    fin_def = []
    new_eda = np.copy(signal)
    # regroupement des points avec une dérivée trop forte pour constituer
    # un défaut
    for i_def in default:
        if len(fin_def) > 0:
            # on ne retraite pas les valeurs déjà agglomérée
            if i_def <= np.max(fin_def): continue
        fin = min(i_def + n_max, len(d2_eda)) # durée max du défaut 0.5s
        where_fin_def = np.where(abs_d2[i_def:fin] > seuil)[0]
        # la fin du défaut est le point le plus éloigné "mais pas trop"
        # avec une dérivée seconde trop forte
        i_fin_def = i_def + max(where_fin_def)
        deb_def = np.append(deb_def, i_def).astype(int)
        fin_def = np.append(fin_def, i_fin_def).astype(int)
    
    # pour tous les segments de défaut
    # on prend en référence les points situés avant de -600 à -200
    # après de 200 à 600
    # on construit un polynome d'interpolation dessus
    # les points à l'intéreiur du segment -200 200 sont remplacés
    # par l'interpolation
    for i_def in range(len(deb_def)):
        #traitement de l'excetion où la fin de correction
        # est après la fin des données
        if fin_def[i_def]+200 > len(signal):
            deb_def = np.delete(deb_def, i_def)
            fin_def = np.delete(fin_def, i_def)
            print('no correction')
            continue
        fin_corr = min(fin_def[i_def] + 600, len(signal)-1)
        pour_correct_avant = np.arange(deb_def[i_def]-600, deb_def[i_def]-200)
        pour_correct_apres = np.arange(fin_def[i_def]+200, fin_corr)
        pour_correct = np.append(pour_correct_avant,pour_correct_apres)
        poly_correct = np.polyfit(pour_correct, signal[pour_correct],2)
        to_correct = np.arange(deb_def[i_def]-200, fin_def[i_def]+200)
        corrected = np.polyval(poly_correct,to_correct)
        new_eda[to_correct] = corrected
        
    # en particulier pour le sujet 3 il n'y avait pas de défaut détecté
    # avec la dérivée seconde, le même schéma est appliqué avec la derivée première
    # il est également possible de décider de pratiquer les deux corrections
    # en séquance
    if len(deb_def) == 0:
        d_eda = np.diff(new_eda)
        abs_d = np.abs(d_eda)
        seuil = np.quantile(abs_d, 0.9) * 5
        default = np.where(abs_d > seuil)[0]

        for i_def in default:
            if len(fin_def) > 0:
                if i_def <= np.max(fin_def): continue
            fin = min(i_def + n_max, len(d_eda)) # durée max du défaut 0.5s
            where_fin_def = np.where(abs_d[i_def:fin] > seuil)[0]
            i_fin_def = i_def + max(where_fin_def)
            deb_def = np.append(deb_def, i_def).astype(int)
            fin_def = np.append(fin_def, i_fin_def).astype(int)
        
        for i_def in range(len(deb_def)):
            if fin_def[i_def]+200 > len(signal):
                deb_def = np.delete(deb_def, i_def)
                fin_def = np.delete(fin_def, i_def)
                print('no correction')
                continue
            fin_corr = min(fin_def[i_def] + 600, len(signal)-1)
            pour_correct_avant = np.arange(deb_def[i_def]-600, deb_def[i_def]-200)
            pour_correct_apres = np.arange(fin_def[i_def]+200, fin_corr)
            pour_correct = np.append(pour_correct_avant,pour_correct_apres)
            poly_correct = np.polyfit(pour_correct, signal[pour_correct],2)
            to_correct = np.arange(deb_def[i_def]-200, fin_def[i_def]+200)
            corrected = np.polyval(poly_correct,to_correct)
            new_eda[to_correct] = corrected            
        
    
    if show_trait:
        if len(default) >0:
            t = np.arange(len(signal)) / 1000
            plt.figure()
            plt.plot(t,signal)
            plt.plot(t[default],signal[default],'.')
            plt.plot(t[deb_def-200], signal[deb_def-200],'.')
            plt.plot(t[fin_def+200], signal[fin_def+200],'.')
            plt.plot(t,new_eda)
    
            # titre = test_name + ' slalom n :' + str(i)          
            # plt.title(titre)
            plt.show()
            plt.pause(1)
        
    return new_eda

def getWaveletData(data):
    '''
    This function computes the wavelet coefficients

    INPUT:
        data: DataFrame, index is a list of timestamps at 8Hz, columns include EDA, filtered_eda

    OUTPUT:
        wave1Second:    DateFrame, index is a list of timestamps at 1Hz, columns include OneSecond_feature1, OneSecond_feature2, OneSecond_feature3 
        waveHalfSecond: DateFrame, index is a list of timestamps at 2Hz, columns include HalfSecond_feature1, HalfSecond_feature2 
    '''
    startTime = data.index[0]

    # Create wavelet dataframes
    oneSecond = pd.date_range(start=startTime, periods=len(data), freq='1s')
    halfSecond = pd.date_range(start=startTime, periods=len(data), freq='500L')

    # Compute wavelets
    cA_n, cD_3, cD_2, cD_1 = pywt.wavedec(data['EDA'], 'Haar', level=3) #3 = 1Hz, 2 = 2Hz, 1=4Hz
    
    # Wavelet 1 second window
    N = int(len(data)/8)
    coeff1 = np.max(abs(np.reshape(cD_1[0:4*N],(N,4))), axis=1)
    coeff2 = np.max(abs(np.reshape(cD_2[0:2*N],(N,2))), axis=1)
    coeff3 = abs(cD_3[0:N])
    wave1Second = pd.DataFrame({'OneSecond_feature1':coeff1,'OneSecond_feature2':coeff2,'OneSecond_feature3':coeff3})
    wave1Second.index = oneSecond[:len(wave1Second)]
    
    # Wavelet Half second window
    N = int(np.floor((len(data)/8.0)*2))
    coeff1 = np.max(abs(np.reshape(cD_1[0:2*N],(N,2))),axis=1)
    coeff2 = abs(cD_2[0:N])
    waveHalfSecond = pd.DataFrame({'HalfSecond_feature1':coeff1,'HalfSecond_feature2':coeff2})
    waveHalfSecond.index = halfSecond[:len(waveHalfSecond)]

    return wave1Second,waveHalfSecond


def getDerivatives(eda):
    deriv = (eda[1:-1] + eda[2:])/ 2. - (eda[1:-1] + eda[:-2])/ 2.
    second_deriv = eda[2:] - 2*eda[1:-1] + eda[:-2]
    return deriv,second_deriv


def getDerivStats(eda):
    deriv, second_deriv = getDerivatives(eda)
    maxd = max(deriv)
    mind = min(deriv)
    maxabsd = max(abs(deriv))
    avgabsd = np.mean(abs(deriv))
    max2d = max(second_deriv)
    min2d = min(second_deriv)
    maxabs2d = max(abs(second_deriv))
    avgabs2d = np.mean(abs(second_deriv))
    
    return maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d


def getStats(data):
    eda = data['EDA'].values
    filt = data['filtered_eda'].values
    maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d = getDerivStats(eda)
    maxd_f,mind_f,maxabsd_f,avgabsd_f,max2d_f,min2d_f,maxabs2d_f,avgabs2d_f = getDerivStats(filt)
    amp = np.mean(eda)
    amp_f = np.mean(filt)
    return amp, maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d,amp_f,maxd_f,mind_f,maxabsd_f,avgabsd_f,max2d_f,min2d_f,maxabs2d_f,avgabs2d_f


def computeWaveletFeatures(waveDF):
    maxList = waveDF.max().tolist()
    meanList = waveDF.mean().tolist()
    stdList = waveDF.std().tolist()
    medianList = waveDF.median().tolist()
    aboveZeroList = (waveDF[waveDF>0]).count().tolist()

    return maxList,meanList,stdList,medianList,aboveZeroList


def getWavelet(wave1Second,waveHalfSecond):
    max_1,mean_1,std_1,median_1,aboveZero_1 = computeWaveletFeatures(wave1Second)
    max_H,mean_H,std_H,median_H,aboveZero_H = computeWaveletFeatures(waveHalfSecond)
    return max_1,mean_1,std_1,median_1,aboveZero_1,max_H,mean_H,std_H,median_H,aboveZero_H


def getFeatures(data,w1,wH):
    # Get DerivStats
    (amp,maxd,mind,maxabsd,avgabsd,
     max2d,min2d,maxabs2d,avgabs2d,
     amp_f,maxd_f,mind_f,maxabsd_f,
     avgabsd_f,max2d_f,min2d_f,
     maxabs2d_f,avgabs2d_f) = getStats(data)
    
    statFeat = np.hstack([amp,maxd,mind,maxabsd,avgabsd,
                          max2d,min2d,maxabs2d,avgabs2d,
                          amp_f,maxd_f,mind_f,maxabsd_f,
                          avgabsd_f,max2d_f,min2d_f,
                          maxabs2d_f,avgabs2d_f])

    # Get Wavelet Features
    (max_1,mean_1,std_1,median_1,
     aboveZero_1,max_H,mean_H,
     std_H,median_H,aboveZero_H) = getWavelet(w1,wH)
    waveletFeat = np.hstack([max_1,mean_1,std_1,median_1,
                             aboveZero_1,max_H,mean_H,
                             std_H,median_H,aboveZero_H])

    all_feat = np.hstack([statFeat,waveletFeat])
    
    if np.Inf in all_feat:
        print("Inf")
    
    if np.NaN in all_feat:
        print("NaN")

    return list(all_feat)


def createFeatureDF(data,window_size,step_size):
    '''
    INPUTS:
        filepath:           string, path to input file  
    OUTPUTS:
        features:           DataFrame, index is a list of timestamps for each 5 seconds, contains all the features
        data:               DataFrame, index is a list of timestamps at 8Hz, columns include AccelZ, AccelY, AccelX, Temp, EDA, filtered_eda
    '''
    # Load data from q sensor
    wave1sec,waveHalf = getWaveletData(data)
    
    # Create 5 second timestamp list
    timestampList = data.index.tolist()[0::8*step_size]
    #timestampList = data.index.tolist()[0::8*window_size]
    
    # feature names for DataFrame columns
    allFeatureNames = ['raw_amp','raw_maxd','raw_mind',
                       'raw_maxabsd','raw_avgabsd','raw_max2d',
                       'raw_min2d','raw_maxabs2d','raw_avgabs2d',
                       'filt_amp','filt_maxd','filt_mind',
                       'filt_maxabsd','filt_avgabsd','filt_max2d',
                       'filt_min2d','filt_maxabs2d','filt_avgabs2d',
                       'max_1s_1','max_1s_2','max_1s_3','mean_1s_1',
                       'mean_1s_2','mean_1s_3','std_1s_1','std_1s_2',
                       'std_1s_3','median_1s_1','median_1s_2',
                       'median_1s_3','aboveZero_1s_1','aboveZero_1s_2',
                       'aboveZero_1s_3','max_Hs_1','max_Hs_2',
                       'mean_Hs_1','mean_Hs_2','std_Hs_1','std_Hs_2',
                       'median_Hs_1','median_Hs_2','aboveZero_Hs_1',
                       'aboveZero_Hs_2']

    # Initialize Feature Data Frame
    features = pd.DataFrame(np.zeros((len(timestampList),
                                      len(allFeatureNames))),
                                    columns=allFeatureNames,index=timestampList)
    
    shrunks = []
    # Compute features for each 5 second epoch
    for i in range(len(features)-1):
        try:
            start = features.index[i]
            end = features.index[i+1+window_size-step_size]
            this_data = data[start:end]
            this_w1 = wave1sec[start:end]
            this_w2 = waveHalf[start:end]
            features.iloc[i] = getFeatures(this_data,this_w1,this_w2)
            shrunks.append(this_data)
        except:
            continue
      
    return features,shrunks


def classifyEpochs(features,featureNames,classifierName):
    '''
    This function takes the full features DataFrame and classifies each 5 second epoch into artifact, questionable, or clean

    INPUTS:
        features:           DataFrame, index is a list of timestamps for each 5 seconds, contains all the features
        featureNames:       list of Strings, subset of feature names needed for classification
        classifierName:     string, type of SVM (binary or multiclass)

    OUTPUTS:
        labels:             Series, index is a list of timestamps for each 5 seconds,
        values of -1, 0, or 1 for artifact, questionable, or clean
    '''
    # Only get relevant features
    features = features[featureNames]
    X = features[featureNames].values
    
    # Classify each 5 second epoch and put into DataFrame
    if 'Binary' in classifierName:
        featuresLabels = predict_binary_classifier(X)
    elif 'Multi' in classifierName:
        featuresLabels = predict_multiclass_classifier(X)

    return featuresLabels


def getSVMFeatures(key):
    '''
    This returns the list of relevant features

    INPUT:
        key:                string, either "Binary" or "Multiclass"

    OUTPUT:
        featureList:        list of Strings, subset of feature names needed for classification
    '''
    if key == "Binary":
        return ['raw_amp','raw_maxabsd','raw_max2d','raw_avgabs2d','filt_amp','filt_min2d','filt_maxabs2d','max_1s_1',
                                'mean_1s_1','std_1s_1','std_1s_2','std_1s_3','median_1s_3']
    elif key == "Multiclass":
        return ['filt_maxabs2d','filt_min2d','std_1s_1','raw_max2d','raw_amp','max_1s_1','raw_maxabs2d','raw_avgabs2d',
                                    'filt_max2d','filt_amp']
    else:
        print('Error!! Invalid key, choose "Binary" or "Multiclass"\n\n')
        return


def classify(data,window_size,step_size):
    '''
    This function wraps other functions in order to load, classify, and return the label for each 5 second epoch of Q sensor data.

    INPUT:
        classifierList:         list of strings, either "Binary" or "Multiclass"
    OUTPUT:
        featureLabels:          Series, index is a list of timestamps for each 5 seconds,
                                values of -1, 0, or 1 for artifact, questionable, or clean
        data:                   DataFrame, only output if fullFeatureOutput=1, index is a list of timestamps at 8Hz, columns include AccelZ, AccelY, AccelX, Temp, EDA, filtered_eda
    '''

    classifierList = ["Binary"]
    # Get pickle List and featureNames list
    featureNameList = [[]]*len(classifierList)
    for i in range(len(classifierList)):
        featureNames = getSVMFeatures(classifierList[i])
        featureNameList[i]=featureNames


    features,shrunks = createFeatureDF(data,window_size,step_size)
    for i in range(len(classifierList)):
        # Get correct feature names for classifier
        classifierName = classifierList[i]
        featureNames = featureNameList[i]        
        
        # Label each 5 second epoch
        temp_labels = classifyEpochs(features, featureNames, classifierName)

    return temp_labels,shrunks



def interpolateDataTo8Hz(data,sampling_rate):
    startTime=0
    if sampling_rate<8:
        # Upsample by linear interpolation
        if sampling_rate==2:
            data.index = pd.date_range(start=startTime, periods=len(data), freq='500L')
        elif sampling_rate==4:
            data.index = pd.date_range(start=startTime, periods=len(data), freq='250L')
        data = data.resample("125L").mean()
    else:
        if sampling_rate>8:
            # Downsample
            idx_range = list(range(0,len(data))) # TODO: double check this one
            data = data.iloc[idx_range[0::int(int(sampling_rate)/8)]]
        # Set the index to be 8Hz
        data.index = pd.date_range(start=startTime, periods=len(data), freq='125L')

    # Interpolate all empty values
    data = interpolateEmptyValues(data)
    return data

def myround(x, base=5):
    return base * round(x/base)

def interpolateEmptyValues(data):
    cols = data.columns.values
    for c in cols:
        data.loc[:, c] = data[c].interpolate()

    return data


def butter_lowpass(cutoff, resampling_rate, order=6):
    nyq = 0.5 * resampling_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, resampling_rate, order=6):
    b, a = butter_lowpass(cutoff, resampling_rate, order=order)
    y = lfilter(b, a, data)
    return y

    
def pooling(df,pooling_size):
    # Not used for the moment
    """
    Max pool the original signal because undetectable artifacts may occurs in a short time interval
    Sliding non-intersecting windows of size pooling_size(in sec)*1000(in Hz)
    
    Parameters
    ----------
    df:pd.DataFrame
    pooling_size: float, size of the window whose values are to 
    be replaced by the max
    """
    
    # Note: not used
    eda = df["EDA"].groupby(pd.cut(df["Time"],np.arange(0, df["Time"].iloc[-1], pooling_size))).max()
    eda_series = pd.Series(eda)
    eda_series = eda_series.repeat(pooling_size*1000)
    df = df.iloc[:len(eda_series)]
    df["pooled_eda"] = eda_series.values
    return df


def plot_filtering(data):
    """
    Plot to visualize low pass butterworth filtering 
    

    Parameters
    ----------
    data : pd.DataFrame, containing original eda signal
    and the filtered signal (at 1hz low pass butterworth)

    """
    plt.figure()
    time = data["Time"].values
    signal = data["EDA"].values
    filtered_signal = data["filtered_eda"].values
    plt.plot(time, signal, 'b-', label='data')
    plt.plot(time, filtered_signal, 'g-', label='filtered data')
    plt.xlabel('Time')
    plt.grid()
    plt.legend()
    plt.show()


def get_labels(df,sampling_rate,
               resampling_rate,cutoff,
               window_size,step_size,order):
    """
    Assign shrunks of 5 seconds to two labels: 1 clean, -1 artifact

    Parameters
    ----------
    df : pd.DataFrame, dataframe extracted from slaloms_source_directory
    sampling_rate : int, sampling rate of signals(=1000Hz)
    resampling_rate : int , resampling rate 8Hz to match with the work of Sara Taylor (MIT)
    cutoff : int, cutoff frequency for low pass filter (=1Hz)
    window_size : int, window size to detect artifact (=5s)
    order : int, order of butterworth low pass filter (=6)

    Returns
    -------
    shrunks_labels : np.array, list of labels
    shrunks : list of pd.DataFrame, list of dataframes to which we affect labels
    """
    
    segments=dict([*df.groupby(df['trigger'].ne(df['trigger'].shift()).cumsum())])
    corrected_segments= []
    for segment in segments.values():
        try:
            corrected_segments.append(correct_eda(segment["EDA"].values,700))
        except: 
            corrected_segments.append(segment["EDA"])
    
    df["d2_eda"] = np.concatenate(corrected_segments).ravel()
    df["medfilt_eda"] = medfilt(df["d2_eda"], kernel_size=51)
    #print ("eric done")
    resampled_df = interpolateDataTo8Hz(df,sampling_rate)
    #print ("resampling done ")
    resampled_df.loc[:, "filtered_eda"] = butter_lowpass_filter(resampled_df["medfilt_eda"].values,
                                                                cutoff,
                                                                resampling_rate,
                                                                order)    
    
    #print (len(resampled_df))
    temp_labels,shrunks = classify(resampled_df,window_size,step_size)
    shrunks_labels = temp_labels[:-1]
    #print (len(shrunks_labels))
    return shrunks_labels,shrunks,df


def add_labels(df,sampling_rate,
               resampling_rate,cutoff,
               window_size,step_size,order):
    """
    Add label column to the original dataframe
    
    Parameters
    ----------
    df : pd.DataFrame, dataframe extracted from slaloms_source_directory
    sampling_rate : int, sampling rate of signals(=1000Hz)
    resampling_rate : int , resampling rate 8Hz to match with the work of Sara Taylor
    cutoff : int, cutoff frequency for low pass filter (=1Hz)
    window_size : int, window size to detect artifact (=5s)
    order : int, order of butterworth low pass filter (=6)

    Returns
    -------
    df: pd.DataFrame, original dataframe with additional column 'labels'
    """
    
    shrunks_labels,shrunks,df = get_labels(df,sampling_rate,
                                        resampling_rate,cutoff,
                                        window_size,step_size,order)
    shrunks_labels = shrunks_labels[:len(shrunks)]
    df["labels"] = 1
    
    for j,t in enumerate(shrunks_labels):
        end = shrunks[j]["Time"].iloc[-1] 
        start = shrunks[j]["Time"].iloc[0]
        wh = np.where((df["Time"].values<=end) & (df["Time"].values>=start))[0]
        wh = wh + df.index[0]
        df.loc[wh,"labels"] = t
        #print (len(shrunks_labels),j,t,wh,df.shape[0])
    
    return df


def baseline_correction(rootdir,df):
    """
    References correction (empatica)
    https://support.empatica.com/hc/en-us/articles/203621955-What-should-I-know-to-use-EDA-data-in-my-experiment-
    https://www.lancaster.ac.uk/media/lancaster-university/content-assets/documents/psychology/ResearchNote_SCRs.pdf

    Parameters
    ----------
    rootdir: string, root directory
    df : pd.DataFrame, dataframe

    Returns
    -------
    df: pd.DataFrame with corrected 

    """
    if df.shape[0]>100*100:
        byslalom=False
    else:
        byslalom=True
        
    if byslalom:
        subject_index = df["Subject"].iloc[0]
        slalomtype = df["Slalomtype"].iloc[0]
        name="subject_"+str(subject_index)+"_"+slalomtype
        path = os.path.join(rootdir,"dataframes/all_slaloms/labeled_step_median_31ms",name)
        allslaloms_df = pd.read_pickle(path)
        signal = allslaloms_df["EDA"]
        indexes_arret = allslaloms_df[allslaloms_df["trigger"].isin(["Baseline_Arrêt_Start",
                                                               "Baseline_Arrêt_Stop",
                                                               ])].index.values
        indexes_vehicule = allslaloms_df[allslaloms_df["trigger"].isin(["Baseline_Véhicule_Start",
                                                                        "Baseline_Véhicule_Stop"
                                                                        ])].index.values
        if 'utac' not in rootdir:
            # to handle the sync effect
            diff = np.diff(allslaloms_df["EDA"])
            wh = np.where(diff!=0)[0][0]
            indexes_arret = indexes_arret[indexes_arret > wh]
            
        df_output,info = nk.bio_process(eda=signal,sampling_rate=1000)
        tonic = df_output["EDA_Tonic"].values
        phasic = df_output["EDA_Phasic"].values
        scl_min_arret = np.min(tonic[indexes_arret])
        scl_min_vehicule = np.min(tonic[indexes_vehicule])
        scl_max = np.max(np.array(sorted(tonic)[-5000:]))
        scr_max = np.max(phasic)
    else:
        indexes_arret = df[df["trigger"].isin(["Baseline_Arrêt_Start",
                                               "Baseline_Arrêt_Stop",
                                        ])].index.values
        indexes_vehicule = df[df["trigger"].isin(["Baseline_Véhicule_Start",
                                               "Baseline_Véhicule_Stop"
                                               ])].index.values
        
        if 'utac' not in rootdir:
            # to handle the sync effect
            diff = np.diff(df["EDA"])
            wh = np.where(diff!=0)[0][0]
            indexes_arret = indexes_arret[indexes_arret > wh]
            
    
            if len(indexes_arret)==0 and len(indexes_vehicule)==0:
                # to handle the sync effect
                diff = np.diff(df["EDA"])
                wh = np.where(diff!=0)[0][0]
                barret_start = wh
                bveh_start = wh + 5*60*1000
                bveh_end = bveh_start + 5*60*1000
                
                indexes_arret = np.arange(barret_start,bveh_start,1)
                indexes_vehicule = np.arange(bveh_start,bveh_end,1)
        
        signal = df["artReplaced_eda"].values
        df_output,info = nk.bio_process(eda=signal,sampling_rate=1000)
        tonic = df_output["EDA_Tonic"].values
        phasic = df_output["EDA_Phasic"].values
        scl_min_arret = np.min(tonic[indexes_arret])
        scl_min_vehicule = np.min(tonic[indexes_vehicule])
        scl_max = np.max(np.array(sorted(tonic)[-5000:]))
        scr_max = np.max(phasic)
    signal = df["artReplaced_eda"].values
    df_output,info = nk.bio_process(eda=signal,sampling_rate=1000)
    tonic = df_output["EDA_Tonic"].values
    phasic = df_output["EDA_Phasic"].values
    scl_arret    = (tonic-scl_min_arret)/(scl_max-scl_min_arret)
    scl_vehicule = (tonic-scl_min_vehicule)/(scl_max-scl_min_vehicule)
    scr = phasic/scr_max
    corrected_eda_arret = scl_arret+scr
    corrected_eda_vehicule = scl_vehicule+scr
    df.loc[:,"corrected_eda_arret"] = corrected_eda_arret
    df.loc[:,"corrected_eda_vehicule"] = corrected_eda_vehicule
    return df

def replace_artifact(rootdir,df):
    # RE-CHECK HOW TO REPLACE ARTIFACTS, ADD before and after components
    """
    Replaces artifacts with the mean value of EDA signals according to
    the trigger where the signal is considered clean (labels=1)
    After that it corrects the signal w.r.t to its baseline at BASELINE_ARRET_(START/STOP) triggers

    Parameters
    ----------
    rootdir: string, root directory
    df : pd.DataFrame, dataframe extracted from slaloms_source_directory

    Returns
    -------
    df : pd.DataFrame, dataframe with artifact values replaced

    """
    df.loc[:,"artReplaced_eda"] = df["medfilt_eda"].values
    df_artifacts = df[df["labels"]==-1]
    df_artifacts["Time"] = df_artifacts["Time"].apply(myround)
    gps = [g for g in df_artifacts.groupby(by="Time")]
    for g in gps:
        time = g[0] 
        start = time - 60
        end = time + 60
        indexes = g[1].index
        allowed_triggers = ["Aller_Début","Retour_Début"]
        try:
            r = df["EDA"][(df["Time"]>start) & 
                          (df["Time"]<end) &
                          (df["labels"]==1) & 
                          (df["trigger"].isin(allowed_triggers))].mean()
        except:
            r = df["EDA"].mean() 
        df.loc[indexes,["artReplaced_eda"]] = r
        
    df["artReplaced_eda"]= df["medfilt_eda"].values
    # Fill the rest of nan values
    df["artReplaced_eda"] =  df["artReplaced_eda"].fillna(method="bfill")
    df["artReplaced_eda"] =  df["artReplaced_eda"].fillna(method="ffill")
   
    # Baseline correction for slaloms to be added
    df = baseline_correction(rootdir,df)
    return df


def save_labeled_df(df,labeled_directory):
    """
    Save labeled dataframes

    Parameters
    ----------
    df : pd.DataFrame, dataframe with artifact values replaced
    labeled_directory : string, directory where new dataframes are saved
    """
    #name = "subject_" + df["Subject"].iloc[0] +"_slalom_"+df["slalom"].iloc[0]+ "_" + df["Slalomtype"].iloc[0]
    name = "subject_" + df["Subject"].iloc[0] + "_" + df["Slalomtype"].iloc[0]
    df.to_pickle(os.path.join(labeled_directory,name))
    #plot_artRemoval(df,labeled_directory)
    
    
def plot_artRemoval(df,labeled_directory):
    """
    Visualize the signal after artifact removal along with
    its original shape and the location of artifacts
    
    Parameters
    ----------
    df : pd.DataFrame, dataframe with artifact values replaced
    labeled_directory : string, directory where new dataframes are saved
    """
    matplotlib.use('Agg')
    dir_ = os.path.join(labeled_directory,"figures")
    colors = []
    for e in df["labels"].values:
        if e==1:
            colors.append("k")
        if e==0:
            colors.append("k")
        if e==-1:
            colors.append("r")
                
                
    fig = plt.figure(figsize=(15,7))
    x=7
    plt.scatter(df["Time"],df["labels"].values-5,s=0.1,c=colors)
    plt.plot(df["Time"],df["EDA"],color="b",label="original signal")
    plt.plot(df["Time"],df["d2_eda"]+x,color="r",label="d2-filtered signal")
    plt.plot(df["Time"],df["medfilt_eda"]+2*x,color="c",label="median-filtered signal")
    #plt.plot(df["Time"],df["pooled_eda"]+3*x,color="k",label="pooled signal")
    plt.plot(df["Time"],df["artReplaced_eda"]+4*x,color="g",label="shifted resultant signal")
    plt.plot(df["Time"],df["corrected_eda"]+5.5*x,color="m",label = "corrected resultant signal")
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    plt.title("""Experience for subject {subject_index} for slalom {slalom_index} ({slalomtype}).
                       black and red stands for clean and artifact 
                       signal of 5 seconds interval. The whole experience 
                       contains {artnum} artifacts""".format(
                                                     artnum = int(list(df["labels"]).count(-1)/5000),
                                                     subject_index = df["Subject"].iloc[0],
                                                     slalom_index = "all",#df["slalom"].iloc[0], 
                                                     slalomtype = df["Slalomtype"].iloc[0]))
    plt.xlabel("Time [sec]")
    subject_index = df["Subject"].iloc[0]
    slalomtype = df["Slalomtype"].iloc[0]
    
    # slalom_index = df["slalom"].iloc[0]
    # name = "subject_{suid}_{si}_{st}.png".format(suid = subject_index,
    #                                               si = slalom_index,
    #                                               st=slalomtype)
    
    name =  "subject_{suid}_{st}.png".format(suid = subject_index,
                                                    st=slalomtype)
    fig.savefig(os.path.join(dir_,name))
        
        
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

     # Artifact detection parameters (configuration of the MIT algorithm, DO NOT CHANGE)
    localdir = config["handling files"]["localdir"]
    destinationdir = config["EDA processing"]["destinationdir"]
    sampling_rate = int(config["EDA processing"]["sampling_rate"]) # 1000 Hz
    resampling_rate = int(config["EDA processing"]["resampling_rate"])# MIT algorithm is built for 8hz sampled signals
    cutoff = int(config["EDA processing"]["cutoff"]) # cutoff frequency
    window_size = int(config["EDA processing"]["window_size"]) # eda window size 
    step_size = int(config["EDA processing"]["step_size"]) # step size
    filter_order = int(config["EDA processing"]["filter_order"]) # filter order
    rootdir="./"
    
    files = [os.path.join(localdir,file) for file in os.listdir(localdir) if file.startswith("subject")]
    for i in range(len(files)):
       df = pd.read_pickle(files[i])
       try:
           print ("EDA artifacts detection in process for subject ",df["Subject"].iloc[0])
           df = add_labels(df,sampling_rate,
                           resampling_rate,cutoff,
                           window_size,step_size,
                           filter_order)
           print ("labels are added")
           df = replace_artifact(rootdir,df)
           save_labeled_df(df,destinationdir)
           print ("artifacts are detected and replaced .. saving data in process")
       except:
            continue
        
if __name__ == "__main__":
    configfile = "D:/scripts/config.ini"
    config = get_config(configfile)
    main(config)


    
           
            
            
            




