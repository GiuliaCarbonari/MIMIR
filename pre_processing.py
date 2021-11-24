import yasa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mne

from scipy import signal
from scipy.signal import find_peaks,freqs, iirfilter, butter, sosfilt, sosfreqz,filtfilt
from mne.filter import filter_data


sns.set(font_scale=1.2)


def _zerocrossings(x):
    """
    Find indices of zero-crossings in a 1D array
    """
    pos = x > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]

def butter_lowpass_filter(data, cutoff, fs, order):
    """
    Design band pass filter.

    Args:
        - low_cut  (float) : the low cutoff frequency of the filter.
        - high_cut (float) : the high cutoff frequency of the filter.
        - fs       (float) : the sampling rate.
        - order      (int) : order of the filter, by default defined to 5.
    """
    # Nyquist Frequency
    nyq = 0.5 * fs   
    normal_cutoff = cutoff / nyq

    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def bandpass_chebyshev(order,rs, Wn,data):
    """
    Complete IIR digital and analog filter design. Type: Chebyshev II
    """
    sos = signal.cheby2(order, rs, Wn, btype='bandpass', analog=False, output='sos', fs=100)
    #sos = signal.iirfilter(16, Wn, rs=40, btype='band', analog=False, ftype='cheby2', fs=100, output='sos')
    filtered = signal.sosfilt(sos, data)
    return filtered

def tononi_filter(segment,sf):
    #1 Downsampled (200HZ--> 100 Hz)
    data_downsampled = signal.decimate(segment, 2, n=None, ftype='iir', axis=-1,zero_phase=True)
    #2 Referenced Averag
    data_referenced = data_downsampled - np.mean(data_downsampled)
    #3 Low-pass 30 Hz
    data_lowpass = butter_lowpass_filter(data_referenced, 30, sf, 5)
    #4 Band-pas Cheby II 0.5-4 Hz
    data_chebyshev=bandpass_chebyshev(16,40,[0.01,6]  ,data_lowpass)
    #5 Over sampling 100 Hz --> 200 Hz 
    data_oversamp=signal.resample(data_chebyshev, len(segment), t=None, axis=0, window=None)
    return data_oversamp

def window_filter(data,sf,window):
    # Define window length (window seconds)
    win =int(window * sf)
    n=len(data)
    filter_data = np.zeros([1,n])  

    for i in range(0,n,win):
        if i + win < n-1 :  
            segment = data[i:i+win]
            filter_data[0,i:i+win] =tononi_filter(segment,sf)
        else: 
            segment = data[i:n-1]
            filter_data[0,i:n-1] = tononi_filter(segment,sf)

    filter_data.tolist()
    return filter_data

def FIR_bandpass_filtering(data,sf,freq_sw):
    # Slow-waves FIR bandpass filter : freq_sw = (0.3, 2)
    data_filt = filter_data(data, sf, freq_sw[0], freq_sw[1], method='fir', verbose=1, 
                            l_trans_bandwidth=0.2, h_trans_bandwidth=0.2)
    return data_filt

def yasa_filter(data,sf):
    # Slow-waves FIR bandpass filter
    freq_sw = (0.3, 2)
    data_filt = filter_data(data, sf, freq_sw[0], freq_sw[1], method='fir', verbose=1, 
                        l_trans_bandwidth=0.2, h_trans_bandwidth=0.2)
    return data_filt


def Siclari2014_filter(data,sf):  
    """ Siclari 2014 
    Downsampled to 128 Hz and band-pass filtered (0.5-4.0 Hz, stop-band at 0.1 and 10 Hz) using a Chebyshev Type II 
    """    
    freq_sw = (0.5, 4)
    
    data_dow=signal.decimate(data, 2, n=None, ftype='iir', axis=-1,zero_phase=True)
    data_filt = filter_data(data_dow, sf, freq_sw[0], freq_sw[1], method='fir', verbose=1, 
                       l_trans_bandwidth=0.1, h_trans_bandwidth=0.1)

    sos = signal.cheby2(16,0.1, [0.01,10], btype='bandpass', analog=False, output='sos', fs=100)
    data_filt = signal.sosfilt(sos, data_filt)
    data_filt=signal.resample(data_filt, len(data), t=None, axis=0, window=None)
    return data_filt


def Massimini2004_filter(data,sf):  
    """ Massimini 2004 
    Each local average (bandpass, 0.1–4 Hz) 
    a) negative-to-positive peak-to-peak amplitude 140 uV
    b) negative peak between the two zero crossings with voltage less than 80 uV
    c) negative-to-positive peak-to-peak amplitude 140 uV 
    """
    freq_sw = (0.1, 4)
    data_filt = filter_data(data, sf, freq_sw[0], freq_sw[1], method='fir', verbose=1, 
                       l_trans_bandwidth=0.1, h_trans_bandwidth=0.2)
    return data_filt


def peaks_detection(data): 
    """
    Returns the indices of the positive and negative peaks
    """
    # Negative peaks with value comprised between -30 to -300 uV
    idx_neg_peaks, _ = find_peaks(-1 * data, height=(30, 300))
    
    # Positive peaks with values comprised between 20 to 150 uV
    idx_pos_peaks, _ = find_peaks(data, height=(20, 150))

    # For each negative peak, we find the closest following positive peak
    pk_sorted = np.searchsorted(idx_pos_peaks, idx_neg_peaks)
    closest_pos_peaks = idx_pos_peaks[pk_sorted] - idx_neg_peaks
    closest_pos_peaks = closest_pos_peaks[np.nonzero(closest_pos_peaks)]
        
    idx_pos_peaks = idx_neg_peaks + closest_pos_peaks
    return data, idx_pos_peaks, idx_neg_peaks

def amplitude_duration_criteria(data, sf, idx_pos_peaks, idx_neg_peaks):
    """
    Returns a pandas DataFrame with all the detected slow-waves and their properties
    """
    n = len(data)  #samples
    times = np.arange(n) / sf    
    
    # Now we check that the total Peak-to-Peak(PTP) amplitude is within our bounds (75 to 400 uV)
    sw_ptp = np.abs(data[idx_neg_peaks]) + data[idx_pos_peaks]
    good_ptp = np.logical_and(sw_ptp > 75, sw_ptp < 400)   

    # Remove the slow-waves with peak-to-peak ampitude outside the bounds
    sw_ptp = sw_ptp[good_ptp]
    idx_neg_peaks = idx_neg_peaks[good_ptp]
    idx_pos_peaks = idx_pos_peaks[good_ptp]

    # Then we check the negative and positive phase duration. To do so,
    # we first need to compute the zero crossings of the filtered signal:
    zero_crossings = _zerocrossings(data)

    #plot_zero_and_picks()

    # Safety check: Make sure that there is a zero-crossing after the last detected peak
    if zero_crossings[-1] < max(idx_pos_peaks[-1], idx_neg_peaks[-1]):
        # If not, append the index of the last peak
        zero_crossings = np.append(zero_crossings,
                                max(idx_pos_peaks[-1], idx_neg_peaks[-1]))

    # For each negative peak, we find the previous and following zero-crossings
    neg_sorted = np.searchsorted(zero_crossings, idx_neg_peaks)
    previous_neg_zc = zero_crossings[neg_sorted - 1] - idx_neg_peaks
    following_neg_zc = zero_crossings[neg_sorted] - idx_neg_peaks

    # And from that we calculate the duration of the negative phase
    neg_phase_dur = (np.abs(previous_neg_zc) + following_neg_zc) / sf
    neg_phase_dur

    # For each positive peak, we find the previous and following zero-crossings
    pos_sorted = np.searchsorted(zero_crossings, idx_pos_peaks)
    previous_pos_zc = zero_crossings[pos_sorted - 1] - idx_pos_peaks
    following_pos_zc = zero_crossings[pos_sorted] - idx_pos_peaks

    # And from that we calculate the duration of the positive phase
    pos_phase_dur = (np.abs(previous_pos_zc) + following_pos_zc) / sf

    # Now we can start computing the properties of each detected slow-waves
    sw_start = times[idx_neg_peaks + previous_neg_zc]
    sw_end = times[idx_pos_peaks + following_pos_zc]
    sw_dur = sw_end - sw_start  # Same as pos_phase_dur + neg_phase_dur
    sw_midcrossing = times[idx_neg_peaks + following_neg_zc]
    sw_idx_neg, sw_idx_pos = times[idx_neg_peaks], times[idx_pos_peaks]
    sw_slope = sw_ptp / (sw_midcrossing - sw_idx_neg)  # Slope between peak trough and midcrossing

    # Finally we apply a set of logical thresholds to exclude "bad" slow waves
    good_sw = np.logical_and.reduce((
                                # Data edges
                                previous_neg_zc != 0,
                                following_neg_zc != 0,
                                previous_pos_zc != 0,
                                following_pos_zc != 0,
                                # Duration criteria
                                neg_phase_dur > 0.3,
                                neg_phase_dur < 1.5,
                                pos_phase_dur > 0.1,
                                pos_phase_dur < 1,
                                # Sanity checks
                                sw_midcrossing > sw_start,
                                sw_midcrossing < sw_end,
                                sw_slope > 0,
                                ))
    # Create the dataframe
    events = pd.DataFrame({'Start': sw_start,
                        'NegPeak': sw_idx_neg,
                        'MidCrossing': sw_midcrossing,
                        'PosPeak': sw_idx_pos,  
                        'End': sw_end, 
                        'Duration': sw_dur,
                        'ValNegPeak': data[idx_neg_peaks], 
                        'ValPosPeak': data[idx_pos_peaks], 
                        'PTP': sw_ptp, 
                        'Slope': sw_slope, 
                        'Frequency': 1 / sw_dur,
                            })[good_sw]

    return events

def clean_events(events):    
    """
    Remove all duplicates and reset index
    """
    events.drop_duplicates(subset=['Start'], inplace=True, keep=False)
    events.drop_duplicates(subset=['End'], inplace=True, keep=False)
    events.reset_index(drop=True, inplace=True)
    events.round(3)
    return events

def yasa_events(raw,yasa_sw_data,annot = 'no'):   
    """
    Add events as raw annotations and returns the new raw 
    """
    onset=yasa_sw_data[:,0]
    duration=yasa_sw_data[:,1]
    description=['sw yasa']*(yasa_sw_data.shape[0])

    yasa_annot = mne.Annotations(onset,duration,description, orig_time=raw.annotations.orig_time)    
    if (annot=='yes'):
        original_annot=raw.annotations  
        reraw = raw.copy().set_annotations(yasa_annot+original_annot)
    else:
        reraw = raw.copy().set_annotations(yasa_annot)
    return reraw, yasa_annot

def yasa_sw_detection(raw,annot = 'no'):    
    ### Extract data, sampling frequency and channels names
    data,sf=raw._data,raw.info['sfreq']
    data=data* 1e6      # Convert Volts to uV
    n = data.shape[1]   #samples    
    data=(data[1][:]).tolist()  #Channel C4

    # Define sampling frequency and time vector
    times = np.arange(n) / sf    
    # Slow-waves FIR bandpass filter
    freq_sw = (0.3, 2)
    data_filt = FIR_bandpass_filtering (data, sf,freq_sw)
    #data_filt = tononi_filter(data,sf)
    #data_filt=Massimini2004_filter(data,sf)
    #data_filt=window_filter(data,sf,4)

    _, idx_pos_peaks, idx_neg_peaks=peaks_detection(data_filt)
    events=amplitude_duration_criteria(data_filt, sf,idx_pos_peaks, idx_neg_peaks)
    events=clean_events(events)
    print(events)
    #DataFrame to Numpy
    #np.set_printoptions(suppress=True)
    yasa_sw_data=events.to_numpy()
    yasa_sw_data=yasa_sw_data[:,[0,5]]

    if (annot=='yes'):
        reraw,yasa_annot=yasa_events(raw,yasa_sw_data,annot='yes')
    else:
       reraw,yasa_annot=yasa_events(raw,yasa_sw_data)
    return reraw, yasa_annot


def findpeaks_MIMIR(data):
    z=(data-np.mean(data)) / np.std(data)
    """ Let’s find all peaks (local maxima) in x whose amplitude lies 30-300"""
    pos_peaks, _ = find_peaks(data, height=(-z, 500))

    """ Let’s find all peaks (local minima) in x whose amplitude lies 10-150"""
    neg_peaks, _ = find_peaks(-data,  height=(z, 450))

    pk_sorted = np.searchsorted(pos_peaks, neg_peaks)

    cont=0
    for i in neg_peaks:
        if i <(max(pos_peaks)):
            cont=cont+1
        #print('')

    next_peaks=pos_peaks[pk_sorted[:cont]]
    print(next_peaks)

    pk_sorted = np.searchsorted(pos_peaks, neg_peaks)   
    pk_sorted2=pk_sorted-1
    prev_peaks=pos_peaks[pk_sorted2]

    # Now we check that the total Peak-to-Peak(PTP) amplitude is within our bounds (75 to 400 uV)
    sw_ptp = np.abs(data[neg_peaks[:len(next_peaks)]]) + data[next_peaks]
    slow_wave=[]
    for i in range(cont):
        amplitude=abs(data[neg_peaks[i]])+ data[next_peaks[i]]
        if amplitude>75:
            slow_wave.append(neg_peaks[i])

    slowwave=[]
    swstart=[]
    swend=[]
    swdur=[]
    cont=0
    for i in range(len(next_peaks)):
        a=data[prev_peaks[i]]
        b=data[next_peaks[i]]
        ampl=min([data[prev_peaks[i]],data[next_peaks[i]]])+abs(data[neg_peaks[i]])
        dur=next_peaks[i]-prev_peaks[i]
        if ampl>75:
            if dur>100 and dur<800:    
                swstart.append(prev_peaks[i])
                swend.append(next_peaks[i])
                swdur.append(swend[cont]-swstart[cont])
                cont=cont+1
    
    # Define sampling frequency and time vector
    sf = 200.
    times = np.arange(data.size) / sf
    zero_crossings = _zerocrossings(data) 

    return pos_peaks, neg_peaks,next_peaks,prev_peaks,sw_ptp,slow_wave,swstart,swdur,swend


def MIMIR_detection(raw):
    data,sfreq =raw.get_data(),raw.info['sfreq']                   
    time_shape = data.shape[1]
    #print('time shape:',time_shape)
    c4_1= raw.get_data(picks='C4_1')* 1e6   
    c4_1=c4_1.ravel()
    c4= Massimini2004_filter(c4_1,200) 

    pos_peaks, neg_peaks,next_peaks,prev_peaks,sw_ptp,slow_wave,swstart,swdur,swend=findpeaks_MIMIR(c4)
    
    onset = np.asarray(swstart)
    onset=onset/200
    duration=np.asarray(swdur)/200
    description=np.asarray(['SO']*(len(swdur)))

    onset= onset.astype(np.float)
    duration=np.asarray(duration)
    duration= duration.astype(np.float)
    description=np.asarray(description)

    my_annot = mne.Annotations(onset,duration,description, orig_time=raw.annotations.orig_time)    
    reraw = raw.copy().set_annotations(my_annot)
    return reraw, my_annot
