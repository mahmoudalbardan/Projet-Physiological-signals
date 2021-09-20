# -*- coding: utf-8 -*-
"""
pyHRV - Heart Rate Variability
------------------------------

This module contains the hrv() functions which allows a full HRV parameter computation using only a single line
function.

Notes
-----
..  Up to v.0.3 this work has been developed within the master thesis
    "Development of an Open-Source Python Toolbox for Heart Rate Variability (HRV)".
..    You find the API reference for this module here:
    https://pyhrv.readthedocs.io/en/latest/_pages/api/hrv.html
..     See 'references.txt' for a full detailed list of references

Author
------
..  Pedro Gomes, pgomes92@gmail.com

Contributors (and former Thesis Supervisors)
--------------------------------------------
..  Hugo Silva, PhD, Instituto de Telecomunicacoes & PLUX wireless biosignals S.A.
..  Prof. Dr. Petra Margaritoff, University of Applied Sciences Hamburg

Last Update
-----------
12-11-2019

:copyright: (c) 2019 by Pedro Gomes
:license: BSD 3-clause, see LICENSE for more details.

"""
# Compatibility
from __future__ import absolute_import

# Imports
import warnings
import numpy as np
import time
# BioSPPy import
import neurokit2 as nk
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz,filtfilt,iirnotch,iirfilter
import neurokit2 as nk
import pandas as pd

# Import toolbox functions
from ecg_explorer import tools
from ecg_explorer import utils
from ecg_explorer import time_domain as td
from ecg_explorer import frequency_domain as fd
from ecg_explorer import nonlinear as nl
try:
    from pyhrv import tools
except ImportError as e:
    pass

try:
    from pyhrv import utils
except ImportError as e:
    pass


def hrv(nni=None,
        rpeaks=None,
        signal=None,
        sampling_rate=1000.,
        interval=[0, 10],
        plot_ecg=False,
        plot_tachogram=False,
        show=False,
        fbands=None,
        kwargs_ecg_plot={},
        kwargs_tachogram=None,
        kwargs_time=None,
        kwargs_nonlinear=None,
        kwargs_welch=None,
        kwargs_lomb=None,
        kwargs_ar=None,
        df=None):
    """Computes all HRV parameters of the pyHRV toolkit (see list below).

    References:    See 'references.txt' for the full list of references

    Parameters
    ----------
    nni : array
        NN intervals in (ms) or (s).
    rpeaks : array
        R-peak times in (ms) or (s).
    signal : array
        ECG signal.
    sampling_rate : int, float
        Sampling rate used for the ECG acquisition in (Hz).
    plot_ecg : bool, optional
        If True, plots ECG signal with specified interval ('signal' must not be None).
    plot_tachogram : bool, optional
        If True, plots tachogram with specified interval.
    fbands : dict, optional
        Dictionary with frequency bands for frequency domain (tuples or list).
        Value format:    (lower_freq_band_boundary, upper_freq_band_boundary)
        Keys:    'ulf'    Ultra low frequency        (default: none) optional
                'vlf'    Very low frequency        (default: (0.003Hz, 0.04Hz))
                'lf'    Low frequency            (default: (0.04Hz - 0.15Hz))
                'hf'    High frequency            (default: (0.15Hz - 0.4Hz))
    show : bool, optional
        If true, shows all plots (default: True).

    kwargs_ecg_plot : dict, optional
        **kwargs for the plot_ecg() function (see 'tools.py' module):
            ..    rpeaks : bool, optional
                    If True, marks R-peaks in ECG signal (default: True).
            ..    title : str, optional
                    Plot figure title (default: None).

    kwargs_tachogram : dict, optional
        **kwargs for the plot_tachogram() function (see 'tools.py' module):
            ..    hr : bool, optional
                    If True, plots series of heart rate data in [bpm] (default: True).
            ..    title : str, optional
                    Plot figure title (default: None).

    kwargs_time : dict, optional
        **kwargs for the time_domain() function (see 'time_domain()' function)
            ..    threshold : int, optional
                    Custom threshold in [ms] for the NNXX and pNNXX parameters (default: None).
            ..    plot : bool
                    If True, creates histogram plot using matplotlib, else uses numpy (data only, no plot) -
                    (geometrical params).
            ..    binsize : int, float
                    Bin size in [ms] of the histogram bins - (geometrical params).

    kwargs_welch : dict, optional
        **kwargs for the 'welch_psd()' function:
            ..    nfft : int, optional
                    Number of points computed for the FFT result (default: 2**12).
            ..    detrend : bool optional
                    If True, detrend NNI series by subtracting the mean NNI (default: True).
            ..    window : scipy window function, optional
                    Window function used for PSD estimation (default: 'hamming').

    kwargs_lomb : dict, optional
        **kwargs for the 'lomb_psd()' function:
            ..    nfft : int, optional
                    Number of points computed for the Lomb result (default: 2**8).
            ..    ma_size : int, optional
                    Window size of the optional moving average filter (default: None).

    kwargs_ar : dict, optional
        **kwargs for the 'ar_psd()' function:
            ..    nfft : int, optional
                    Number of points computed for the entire AR result (default: 2**12).
            ..    order : int, optional
                    Autoregressive model order (default: 16).

    kwargs_nonlinear : dict, optional
        **kwargs for the nonlinear functions (poincare(), sample_enntropy(), dfa()):
            ..    ellipse : bool, optional
                    If true, shows fitted ellipse in plot (default: True).
            ..    vectors : bool, optional
                    If true, shows SD1 and SD2 vectors in plot (default: True).
            ..    legend : bool, optional
                    If True, adds legend to the Poincaré plot (default: True).
            ..    marker : character, optional
                    NNI marker in plot (default: 'o').
            ..    short : array, 2 elements
                    Interval limits of the short term fluctuations (default: None: [4, 16]).
            ..    long : array, 2 elements
                    Interval limits of the long term fluctuations (default: None: [17, 64]).
            ..    legend : bool
                    If True, adds legend with alpha1 and alpha2 values to the DFA plot (default: True).
            ..    dim : int, optional
                    Entropy embedding dimension (default: 2).
            ..    tolerance : int, float, optional
                    Tolerance distance for which the vectors to be considered equal (default: std(NNI) * 0.2).

    Returns
    -------
    results : biosppy.biosspy.utils.ReturnTuple object
        All time domain results.

    Returned Parameters - Time Domain
    ---------------------------------
    ..    NNI parameters (# of NNI, mean, min, max) in [count] and [ms] (keys: 'nni_counter', 'nni_mean', 'nni_min',
        'nni_max')
    ..    NNI differences (mean, min, max, standard deviation) in [ms] (keys: 'nni_diff_mean', 'nni_diff_min',
        'nn_diff_max')
    ..    HR parameters (mean, min, max, standard deviation) in [BPM] (keys: 'hr_mean', 'hr_min', 'hr_max', 'hr_std')
    ..    SDNN in [ms] (key: 'sdnn')
    ..    SDNN index in [ms] (key: 'sdnn_index')
    ..    SDANN in [ms] (key: 'sdann')
    ..    RMSSD in [ms] (key: 'rmssd')
    ..    SDSD in [ms] (key: 'sdsd')
    ..    nn50 in [count] & pNN50 in [%] (keys: 'nn50', 'pnn50')
    ..    nn20 in [count] & pNN20 in [%] (keys: 'nn20', 'pnn20')
    ..    nnXX (XX = custom threshold) if specified (keys: 'nnXX', 'pnnXX')
    ..    Triangular Index [-] (key: 'tri_index')
    ..     TINN in [ms] (key: 'tinn', 'tinn_n', 'tinn_m')

    Returned Parameters - Frequency Domain
    --------------------------------------
    (below, X = one of the methods 'fft' or 'lomb')
    ..    Peak frequencies of all frequency bands in [Hz] (key: 'X_peak')
    ..    Absolute powers of all frequency bands in [ms^2] (key: 'X_abs')
    ..    Relative powers of all frequency bands in [%] (key: 'X_rel')
    ..    Logarithmic powers of all frequency bands [-] (key: 'X_log')
    ..    Normalized powers of the LF and HF frequency bands [-] (key: 'X_norms')
    ..    LF/HF ratio [-] (key: 'X_ratio')
    ..    Total power over all frequency bands (key: 'X_total')
    ..    Interpolation method used for NNI interpolation (FFT/Welch's method only) (key: 'fft_interpolation')
    ..    Resampling frequency used for NNI interpolation (FFT/Welch's method only) (key: 'fft_resampling_frequency')
    ..    Spectral window used for PSD estimation of the Welch's method
        (key: 'fft_spectral_window)'

    Returned Parameters - Nonlinear
    -------------------------------
    ..    SD1    in [ms] (key: 'sd1')
    ..    SD2 in [ms] (key: 'sd2')
    ..    SD2/SD1 [-] (key: 'sd_ratio')
    ..    Area of the fitted ellipse in [ms^2] (key: 'ellipse_area')
    ..    Sample Entropy [-] (key: 'sampen')
    ..    Detrended Fluctuations Analysis [-] (short and long term fluctuations) (key: 'dfa_short', 'dfa_long')

    Returned Figures
    ----------------
    ..    ECG plot (key: 'ecg_plot') (only if ECG signal is provided)
    ..    Tachogram (key: 'tachogram_plot')
    ..    Poincaré plot (key: 'poincare_plot')
    ..    NNI Histogram (key: 'nn_histogram')
    ..    Welch PSD (key: 'fft_plot')
    ..    Lomb PSD (key: 'lomb_plot')
    ..    AR PSD (key: 'ar_plot')
    ..     Poincaré (key: 'pincare_plot')

    Notes
    -----
    ..    Results are stored in a biosppy.biosspy.utils.ReturnTuple object and need to be accessed with the respective keys as
        done with dictionaries (see list of parameters and keys above).
    ..    Provide at least one type of input data (ecg_signal, nn, or rpeaks).
    ..    Input data will be prioritized in the following order: 1. ecg_signal, 2. nn, 3. rpeaks.
    ..    SDNN Index and SDANN: In some cases, the NN interval may start in a segment (or time interval) N and end only
        in the successive segment N+1. In this case, use the 'overlap' parameter to select if the first element of the
        segment should be dropped or not:
        ..    If True: overlap allowed, returns all NNI but the cumulative sum of the NNI in a segment can be greater
            than the specified duration.
        ..    If False: no overlap allowed, first NNI will be dropped and the cumulative sum of the NNI in a segment
            will always be < specified duration.

    Raises
    ------
    TypeError
        If no input data for 'signal', 'nn' and 'rpeaks' provided.

    """
    
    # get information
    dir_ = 'D:/data_1/dataframes/splited_by_slaloms_using_detection.mat/figures'
    subject_index = str(10)#df["Subject"].iloc[0]
    slalom_index =  str(10)# df["slalom"].iloc[0]
    slalom_type =  "Large"# df["Slalomtype"].iloc[0]
    # Check input
    if signal is not None:
         df_output,info = nk.ecg.ecg_process(ecg_signal=signal,
                                             sampling_rate=1000,
                                             method="neurokit") # not neurokit2 by
                                                             #biosppy filtering method
                                                   # bandpass [3,45] 
         signal = df_output["ECG_Clean"].values
         signal = _signal_filter_powerline(signal,1000,50) # extracted from neurokit
         rpeaks = info["ECG_R_Peaks"]
       
        # signal, rpeaks = nk.ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)[1:3]
        # signal = df_output["ECG_Clean"].values
        # signal = _signal_filter_powerline(signal,1000,50)
    elif nni is None and rpeaks is None:
        raise TypeError('No input data provided. Please specify input data.')

    nn = utils.check_input(nni, rpeaks)


    # COMPUTE TIME DOMAIN PARAMETERS
    # Check for kwargs for the 'kwargs_time'
    if kwargs_time is not None:
        if type(kwargs_time) is not dict:
            raise TypeError("Expected <type 'dict'>, got %s: 'kwargs_time' must be a dictionary containing "
                            "parameters (keys) and values for the 'time_domain()' function." % type(kwargs_time))

        # Supported kwargs
        available_kwargs = ['threshold', 'binsize', 'plot']

        # Unwrwap kwargs dictionary
        threshold = kwargs_time['threshold'] if 'threshold' in kwargs_time.keys() else None
        binsize = kwargs_time['binsize'] if 'binsize' in kwargs_time.keys() else 7.8125
        plot = kwargs_time['plot'] if 'plot' in kwargs_time.keys() else True

        unsupported_kwargs = []
        for args in kwargs_time.keys():
            if args not in available_kwargs:
                unsupported_kwargs.append(args)

        # Throw warning if additional unsupported kwargs have been provided
        if unsupported_kwargs:
            warnings.warn("Unknown kwargs for 'time_domain()': %s. These kwargs have no effect."
                          % unsupported_kwargs, stacklevel=2)

        # Compute Time Domain Parameters
        t_results = td.time_domain(nni=nn, show=False, threshold=threshold, plot=plot)

    else:
        # Compute Welch's PSD with default values
        t_results = td.time_domain(nni=nn, show=False)
        
    # COMPUTE FREQUENCY DOMAIN RESULTS (kwargs are verified by the frequency_domain() function)
    f_results = fd.frequency_domain(nni=nn, fbands=fbands, kwargs_welch=kwargs_welch, kwargs_lomb=kwargs_lomb,
                                 kwargs_ar=kwargs_ar, show=False)

    
    # COMPUTE NONLINEAR PARAMETERS
    if kwargs_nonlinear is not None:
        if type(kwargs_nonlinear) is not dict:
            raise TypeError("Expected <type 'dict'>, got %s: 'kwargs_nonlinear' must be a dictionary containing "
                            "parameters (keys) and values for the 'nonlinear()' function." % type(kwargs_time))

        # Supported kwargs
        available_kwargs = ['ellipse', 'vectors', 'legend', 'marker', 'dim', 'tolerance', 'short', 'long', 'legend']
        kwargs_poincare = {}
        kwargs_sampen = {}
        kwargs_dfa = {}

        # Unwrwap kwargs dictionaries
        kwargs_poincare['ellipse'] = kwargs_nonlinear['ellipse'] if 'ellipse' in kwargs_nonlinear.keys() else True
        kwargs_poincare['vectors'] = kwargs_nonlinear['vectors'] if 'vectors' in kwargs_nonlinear.keys() else True
        kwargs_poincare['legend'] = kwargs_nonlinear['legend'] if 'legend' in kwargs_nonlinear.keys() else True
        kwargs_poincare['marker'] = kwargs_nonlinear['marker'] if 'marker' in kwargs_nonlinear.keys() else 'o'
        kwargs_sampen['dim'] = kwargs_nonlinear['dim'] if 'dim' in kwargs_nonlinear.keys() else 2
        kwargs_sampen['tolerance'] = kwargs_nonlinear['tolerance'] if 'tolerance' in kwargs_nonlinear.keys() else None
        kwargs_dfa['short'] = kwargs_nonlinear['short'] if 'short' in kwargs_nonlinear.keys() else None
        kwargs_dfa['long'] = kwargs_nonlinear['long'] if 'long' in kwargs_nonlinear.keys() else None
        kwargs_dfa['legend'] = kwargs_nonlinear['legend'] if 'legend' in kwargs_nonlinear.keys() else True

        unsupported_kwargs = []
        for args in kwargs_nonlinear.keys():
            if args not in available_kwargs:
                unsupported_kwargs.append(args)

        # Throw warning if additional unsupported kwargs have been provided
        if unsupported_kwargs:
            warnings.warn("Unknown kwargs for 'nonlinear()': %s. These kwargs have no effect."
                          % unsupported_kwargs, stacklevel=2)

        n_results = nl.nonlinear(nni=nn, show=False, kwargs_poincare=kwargs_poincare, kwargs_sampen=kwargs_sampen,
                                 kwargs_dfa=kwargs_dfa)
    else:
        n_results = nl.nonlinear(nni=nn, show=False)

    # Prepare output
    results = utils.join_tuples(t_results, f_results, n_results)

    # Plot ECG signal
    try:
        if plot_ecg and signal is not None:
            # COMPUTE NONLINEAR PARAMETERS
            if kwargs_ecg_plot is not None:
                if type(kwargs_ecg_plot) is not dict:
                    raise TypeError("Expected <type 'dict'>, got %s: 'kwargs_ecg_plot' must be a dictionary containing "
                                    "parameters (keys) and values for the 'plot_ecg()' function." % type(kwargs_ecg_plot))
    
                # Supported kwargs
                available_kwargs = ['rpeaks', 'title']
    
                # Unwrwap kwargs dictionaries
                show_rpeaks = kwargs_ecg_plot['rpeaks'] if 'rpeaks' in kwargs_ecg_plot.keys() else True
                title = kwargs_ecg_plot['title'] if 'title' in kwargs_ecg_plot.keys() else None
    
                unsupported_kwargs = []
                for args in kwargs_ecg_plot.keys():
                    if args not in available_kwargs:
                        unsupported_kwargs.append(args)
    
                # Throw warning if additional unsupported kwargs have been provided
                if unsupported_kwargs:
                    warnings.warn("Unknown kwargs for 'plot_ecg()': %s. These kwargs have no effect."
                                  % unsupported_kwargs, stacklevel=2)
    
                ecg_plot = tools.plot_ecg(dir_,
                                          subject_index = subject_index,
                                          slalom_index = slalom_index,
                                          slalom_type = slalom_type,
                                          signal=signal, show=False, 
                                          rpeaks=show_rpeaks, title=title, 
                                          interval=interval,
                                          sampling_rate=sampling_rate)
            else:
                ecg_plot = tools.plot_ecg(dir_,
                                          subject_index = subject_index,
                                          slalom_index = slalom_index,
                                          slalom_type = slalom_type,
                                          signal=signal, sampling_rate=sampling_rate,
                                          show=False, interval=interval)
    
            results = utils.join_tuples(results, ecg_plot)
    except:
        pass

    # Plot Tachogram
    try:
        if plot_tachogram:
            # COMPUTE NONLINEAR PARAMETERS
            if kwargs_tachogram is not None:
                if type(kwargs_tachogram) is not dict:
                    raise TypeError("Expected <type 'dict'>, got %s: 'kwargs_tachogram' must be a dictionary containing "
                                    "parameters (keys) and values for the 'tachogram()' function." % type(kwargs_tachogram))
    
                # Supported kwargs
                available_kwargs = ['hr', 'title']
    
                # Unwrwap kwargs dictionaries
                hr = kwargs_tachogram['hr'] if 'hr' in kwargs_tachogram.keys() else True
                title = kwargs_tachogram['title'] if 'title' in kwargs_tachogram.keys() else None
    
                unsupported_kwargs = []
                for args in kwargs_tachogram.keys():
                    if args not in available_kwargs:
                        unsupported_kwargs.append(args)
    
                # Throw warning if additional unsupported kwargs have been provided
                if unsupported_kwargs:
                    warnings.warn("Unknown kwargs for 'tachogram()': %s. These kwargs have no effect."
                                  % unsupported_kwargs, stacklevel=2)
    
                tachogram_plot = tools.tachogram( dir_,
                                                  subject_index = subject_index,
                                                  slalom_index = slalom_index,
                                                  slalom_type = slalom_type,
                                                  nni=nn, sampling_rate=sampling_rate, 
                                                  hr=hr, interval=interval,
                                                  title=title, show=False)
            else:
                tachogram_plot = tools.tachogram(dir_,
                                                  subject_index = subject_index,
                                                  slalom_index = slalom_index,
                                                  slalom_type = slalom_type,
                                                  nni=nn, show=False, interval=interval)
    
            results = utils.join_tuples(results, tachogram_plot)
    
        if show:
            plt.show()
    except:
        pass

    # Output
    return results

def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normalalized_high = cutoff / nyq
    b, a = butter(order,normalalized_high, btype='high', analog=False)
    return b, a

def butter_highs_filter(data, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b,a,data)
    return y


def notch_filter(data,fs, freq,band,order):
    #https://stackoverflow.com/questions/35565540/designing-an-fir-notch-filter-with-python/35662193
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], btype='bandstop', analog=False, ftype='butter')
    filtered_data = lfilter(b, a, data)
    return filtered_data


def filtering_ecg(ecg,
                  fs,
                  high_cutoff,
                  high_order,
                  notch_freq,
                  notch_band,
                  notch_order):
    """
    Filtering methods used for ecg artifact and noise removal. 
    We deal with 3 types of artifacts: drift (breathing), motion, PIL
    
    1st filtering:
        - High pass IIR filter with 5hz cut off frequency which
         removes motion and drift (breathing) artifacts. 
         drift occurs at <~ 1Hz, motion between 0-5Hz
    
    remarks: 
        **this filtering technique may leads to ECG waveforme distrotion 
    since the time delay between frequency components of ECG signal will change
    filtering because of the non linearity of phase response --> we apply zero phase filtering
    (forward and backward)
        
    
    2ed filtering:
        - Notch (stopband) IRR filtering order 2 at 50Hz which
        removes power line interference. 
        
        
    
    References:
        - https://sci-hub.se/10.1049/htl.2016.0020
        - https://sci-hub.se/10.1109/jsen.2012.2222572
        - https://sci-hub.se/10.3390/s20051468
        
        
    Parameters
    ----------
    ecg: np.array, ecg signal
    fs:int,sampling frequency
    high_cutoff:int, cutoff frequency of the high pass filter
    high_order: int, order of the high pass filter
    notch_freq: int, centerline frequency to be attenuated
    notch_band: int, length of the band around the centerline frequency
    notch_order: int, order of the notch filter
    
    
    Return
    ------
    filtered_ecg: np.array, filtered ecg signal
    """
    y1 = butter_highs_filter(ecg,high_cutoff,fs,high_order)
    filtered_ecg = notch_filter(y1,fs,notch_freq,notch_band,notch_order)
    return filtered_ecg
    

def _signal_filter_powerline(signal, sampling_rate, powerline=50):
    """Filter out 50 Hz powerline noise by smoothing the signal with a moving average kernel with the width of one
    period of 50Hz."""

    if sampling_rate >= 100:
        b = np.ones(int(sampling_rate / powerline))
    else:
        b = np.ones(2)
    a = [len(b)]
    y = filtfilt(b, a, signal, method="pad")
    return y