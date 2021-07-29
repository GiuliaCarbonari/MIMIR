import numpy as np
import mne
import os.path
import sys
from mne import read_events, Epochs
from pathlib import Path
from numpy import loadtxt


def load_brainvision(path,annot = 'no'):
    """
    Import the BrainVision data into an MNE Raw object
    """
    mne.set_log_level("WARNING")
    
    print('Reading raw file...')
    print('')

    _, ext=os.path.splitext(path)
    if ext=='.fif':
        raw = mne.io.read_raw_fif(path)
    elif ext=='.vhdr':
            raw= mne.io.read_raw_brainvision(path, 
            preload=True, 
            eog=('EOG1_1','EOG2_1'),
            misc=('EMG1_1','EMG2_1'),
            verbose=True)
            # Specify this as the emg channel (channel type)
            raw.set_channel_types({'EMG1_1': 'emg','EMG2_1': 'emg'}) 
    # Strip channel names of "." characters
    raw.rename_channels(lambda s: s.strip(".")) 
    
    """
    If you want to see the original RAW labels. For ex, stimulus.
    By default they are not displayed
    """
    if (annot == 'no'):
        raw.set_annotations(None)   
    elif (annot=='yes'):
        raw.set_annotations(raw.annotations)    
    
    print('')
    print('Done!')
    return raw    


def show_info(raw):  
    """
    Show information of the RAW file 
    """
    print('Python v{}'.format(sys.version))
    print('MNE v{}'.format(mne.__version__))

    """ Show information from raw of Brainvision files"""
    print('\n --------------------------- Information --------------------------- \n')
    print('File location:', __file__)
    print('')
    _, times = raw[:, :] 
    print('Data type: {}\n {} \n '.format(type(raw), raw))

    # Give the size of the data matrix
    print('%s channels x %s samples' % (len(raw.info['ch_names']), len(raw.times)))
    # Give the sample rate
    print('Sample rate:', raw.info['sfreq'], 'Hz \n')
    
    #Give Channels    
    print('Channels: \n',raw.info['ch_names'])
    print('EEG: ', list(raw.copy().pick_types(eeg=True).info['ch_names']))
    print('EOG: ', raw.copy().pick_types(eog=True).info['ch_names'])
    #Originally the brainvision EMG channels are misc but we changed them to emg
    print('EMG: ', raw.copy().pick_types(emg=True).info['ch_names'])     

    print('Time min: %s seg. \nTime max: %s seg. \n' % (raw.times.min(), raw.times.max()))
    print('Filters:\nHighpass: %s hz. Lowpass: %s hz. \n' % (raw.info['highpass'], raw.info['lowpass']))
    if len(set(raw.annotations.description))==0:
        print('No annotations.')
    else:
        print('There are %s type of annotations: \n' % (len(set(raw.annotations.description))),set(raw.annotations.description))
    print('\n-------------------------------------------------------------------\n')


def get_data(raw, select=None,tovolt=True):
    """
    Extract data from various channels from RAW     
    By default select all channels and convert to microvolts 
    """
    if select==None:
        select=raw.ch_names
    data, ch_names,=raw.get_data(),raw.ch_names
    positions=[]

    for i in select:
        #i --> select channel names
        pos =(raw.ch_names).index(i)
        positions.append(pos)
    select_data = data[positions,:]

    if tovolt==True:
        select_data =select_data * 1e6
    return select_data 


def choose_window(data,sf=200,start=0,dur=5):
    """
    Choose same seconds of your data
    Default: frequency sample:200 ; start= 0 seconds ; duration= 5 seconds
    Ex: choose_window(data,200,0,10) choose  0-10 sec of  the data
    """
    start=start*sf
    end=start+sf*dur
    
    #Create time vector
    time=[start+i for i in range(end-start)]
    data=data[:,start:end]
    
    return data,time


def set_sleep_stages(raw,hypno_path,annot = 'no'):
    """ 
    Set Hypnpgram annotations
    Read raw + hypno (txt file) and Return raw, annot (onset, duration, description) 
    """  
    hypno_text = loadtxt(hypno_path)
    hypno= hypno_text[:,0]
    n=len(hypno)*30

    index=np.where(np.roll(hypno,1)!=hypno)[0]
    description=[]
    for i in index:
        v=hypno[i]
        if v==0:
            description.append('Wake')
        elif v==1:
            description.append('S1')
        elif v==2:
            description.append('S2')
        elif v==3:
            description.append('S3')
        elif v==4:
            description.append('S4')
        elif v==5:
            description.append('REM')
        elif v==8:
            description.append('Movement')

    onset=np.where(np.roll(hypno,1)!=hypno)[0]*30
    duration=np.diff(np.append(onset,n))

    hypno_annot = mne.Annotations(onset,duration,description, orig_time=raw.annotations.orig_time) 

    if (annot == 'no'):   
        reraw = raw.copy().set_annotations(hypno_annot)
    elif (annot=='yes'):
        original_annot=raw.annotations  
        reraw = raw.copy().set_annotations(hypno_annot+original_annot)

    return reraw, hypno_annot


def set_hypno_annot(raw,hypno_path,annot = 'no'):
    """ 
    Set Hypnpgram annotations
    Read raw + hypno (txt file) and Return raw, annot (onset, duration, description) 
    """
    if (annot == 'no'):
        original_annot=None
    elif (annot=='yes'):
        original_annot=raw.annotations    
    
    hypno_txt = loadtxt(hypno_path)
    hypno= hypno_txt[:,0]
    hypno_txt= np.loadtxt(hypno_path,delimiter =' ', usecols =(0))

    hypno=[]
    for i in hypno_txt:
        for x in range(30):
            hypno.append(i)        
    n=len(hypno)

    cont=1
    summary_hypno=np.zeros([n,1])
    duration=[]
    description=['Despierto']
    for i in range(n-1):
        if hypno[i] != hypno[i+1]:
            if hypno[i+1]==0:
                description.append('Despierto')
            elif hypno[i+1]==1:
                description.append('Fase 1')
            elif hypno[i+1]==2:
                description.append('Fase 2')
            elif hypno[i+1]==3:
                description.append('Fase 3')
            elif hypno[i+1]==4:
                description.append('Fase 4')
            elif hypno[i+1]==5:
                description.append('Fase 5')
            elif hypno[i+1]==8:
                description.append('Movimiento')
            summary_hypno[cont,0]=(i+1)
            cont=cont+1
    summary_hypno=summary_hypno[:cont]
    onset=summary_hypno[:,0]
    duration=np.append(onset,len(hypno)) 
    duration=np.diff(duration)
    description=np.array(description)


    hypno_annot = mne.Annotations(onset,duration,description, orig_time=raw.annotations.orig_time)    
    reraw = raw.copy().set_annotations(hypno_annot)
    return reraw, hypno_annot
    

def set_event_annot(raw,path, annot='no'):
    """ 
    Read hypno(txt file) and raw (vhdr-fif file) and Return (start, duration, description, end) 
    Return hypnogram plus scoring 
    """ 
    hfiletxt = open(path, 'r')
    file = hfiletxt
    start=[]
    duration=[]
    end=[]
    description=[]

    for line in file:
        row=line.strip().split(',') 
        if len(row) ==3:
          if (row[2]=='sw_I' or row[2]=='sw_II' or row[2]=='sw_dudosa'or row[2]==None):
                aux=row[0]+row[1]           
                start.append(row[0])
                duration.append(row[1])
                description.append(row[2])
                end.append(aux)        
    
          if row[2]=='sw' or row[2]=='kc' or row[2]=='ambiguo'or row[2]=='dudoso' or row[2]==None:
                aux=row[0]+row[1]           
                start.append(row[0])
                duration.append(row[1])
                description.append(row[2])
                end.append(aux)

          if row[2]=='M' or row[2]==None:
                aux=row[0]+row[1]           
                start.append(row[0])
                duration.append(row[1])
                description.append(row[2])
                end.append(aux)

          if row[2]=='KC' or row[2]=='kc' or row[2]==None:
                aux=row[0]+row[1]           
                start.append(row[0])
                duration.append(row[1])
                description.append(row[2])
                end.append(aux)

    start=np.asarray(start)
    start= start.astype(np.float)
    duration=np.asarray(duration)
    duration= duration.astype(np.float)
    description=np.asarray(description)

    my_annot = mne.Annotations(start,duration,description, orig_time=raw.annotations.orig_time)  
    if (annot == 'no'):
        reraw = raw.copy().set_annotations(my_annot) 
    elif (annot=='yes'):
        original_annot=raw.annotations    
        reraw = raw.copy().set_annotations(my_annot+original_annot)
    return reraw, my_annot


def delete_stage(raw,hypno_annot,n_stage,sw_annot):
    """
    Allow you delete annotations from one stage
    """
    stages=['Wake','S1','S2','S3','S4','REM','Movement']
    index=[]
    for i in range(len(hypno_annot.description)):
        if hypno_annot.description[i]==stages[n_stage]:
            index.append(hypno_annot.onset[i])
            index.append(hypno_annot.onset[i]+hypno_annot.duration[i])
    t_descrip=sw_annot.description
    t_idx = sw_annot.onset
    t_dur = sw_annot.duration
    condition=True

    for i in range(len(index)):
        if i%2== 0:
            aux_cond=np.logical_not(np.logical_and(t_idx>index[i],t_idx<index[i+1]))
            condition=(np.logical_and(condition,aux_cond))
    
    new_onset=sw_annot.onset[condition]
    new_duration=sw_annot.duration[condition]
    new_description=sw_annot.description[condition]

    annots = mne.Annotations(new_onset, new_duration, new_description)
    reraw = raw.copy()
    reraw = reraw.set_annotations(annots)
    return reraw, reraw.annotations


def show_info_annot(raw):
    """
    Show information of annotations of the RAW file 
    """
    i=0
    for ann in raw.annotations:
        i=i+1
        descr = ann['description']
        start = ann['onset']
        end = ann['onset'] + ann['duration']
        print("'{}' goes from {} sec to {} sec".format(descr, start, end)) 

