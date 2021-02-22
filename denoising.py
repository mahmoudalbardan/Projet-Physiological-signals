# -*- coding: utf-8 -*-
"""
@author: albardan
"""


import pywt 
import numpy as np


"""
Notes: it is known that VisuShrink universal threshold based wavelet denoising method
tends to over smooth the signal which may lead to loose some of the information in the signal
"""


def wavelet_denoising(signal,wavetype="db4",level=4,method="visushrink",thresholdtype = "hard"):
    """
    Denoise the signal
    
    Parameters
    ----------
    signal : numpy.array, numpy array containing the signal
    wavetype : string optional, wavelet type The default is "db4".
    level : int, optional, level of decomposition. defaulto 4
    method: string, thresholding method of wavelets coefficients

    Returns
    -------
    denoised_signal : numpy.array, denoised signal

    """
    thresholded_coeffs = []
    coeffs = pywt.wavedec(signal,wavetype, level)
    for j, coeff in enumerate(coeffs):
        thresholded_coeffs.append(threshold_coeff(coeff,method,thresholdtype))
    denoised_signal =  pywt.waverec(coeffs, wavetype)
    return denoised_signal



def threshold_coeff(coeff, method="VisuShrink",thresholdtype="hard"):
    """
    Perfrom a thresholding for a single level wavelets coefficients
    Parameters
    ----------
    coeff : numpy.array, wavelet coefficients for a single level
    method : string, VisuShrink or SureShrink. The first one uses a 
            universal  threshold for wavelets coefficients accross all levels.
            The second one uses a level-specific threshold
    thresholdtype: string, hard or soft thresholding

    Returns
    -------
    list of thresholded wavelets coeffcients 

    """
    if method == "VisuShrink":
        sigma = np.median(np.abs(coeff))/0.6745
        universal_threshold = sigma*np.sqrt(2*np.log(len(coeff)))
        print ("Universal threshold is",universal_threshold)
        return [thresholding(x,universal_threshold,thresholdtype) for x in coeff]
    
    if method == "SureShrink":
        sure = []
        values = np.arange(-1,1,0.1)
        for value in values:
            d =len(coeff)
            count = 2*list(coeff<value).count(True)
            sum_ = sum([min(x,value)**2 for x in coeff])
            sure.append(d-count+sum_)
            
        sure_threshold = np.mean(values[np.where(np.array(sure)==np.min(sure))[0]])  
        print ("SureShrink threshold is", sure_threshold)
        return [thresholding(x,sure_threshold,thresholdtype) for x in coeff]
        
        

def thresholding(x,threshold,thresholdtype):
    """
    Perform the thresholding operation
    """
    if thresholdtype=="hard":
        return x if np.abs(x)> threshold else 0
    if thresholdtype=="soft":
        return np.sign(x)*(np.abs(x-threshold)) if np.abs(x)> threshold else 0

    




