import numpy as np
import mne
import os
import matplotlib.pyplot as plt                    
import argparse
import easygui

from read_files import*
from pre_processing import*

from datetime import datetime    
from tkinter import messagebox
from scipy import signal
from pathlib import Path


#Print the system information
mne.sys_info()


def file_name(path):
    """ Extract the file name"""
    name=os.path.splitext(os.path.basename(path))[0]
    if '_2020'  in name:
        ind = str.index(name,'_2020')
        name=name[:ind]
    elif '_2021' in name:
        ind = str.index(name,'_2021')
        name=name[:ind]

    subject=name
    name=name + datetime.now().strftime("_%Y%B%d_%H-%M") 
    return name, subject


##New Channels
def upper_than(raw,name_channel,threshold):  
    """ Create a New Channel for mark amplitud> threshold  """
    ### Extract data, sampling frequency and channels names
    data,sf,chan=raw._data,raw.info['sfreq'], raw.info['ch_names']
    data=data*1e6       # Convert Volts to uV
    print(data.shape)
    print(type(data))
    n = data.shape[1]   #samples    

   
    channel = (raw.ch_names).index(name_channel)
    print('Channel choose:',raw.ch_names[channel])
    data=data[channel][:]    
    data=data.tolist()
    
    step=int(sf/4)
    win2=int(sf*2)
    win1=int(sf*1)
    win05=int(sf/2)
    
    dat=[0 for i in range(0,n)]
    
    ###Cycle every second
    for i in range(0,n,step):   
        eeg2=data[i:i+win2]       #Every 2 seconds
        eeg1=data[i:i+win1]       #Every second
        eeg05=data[i:i+win05]     #Every half second
        

        aux2=abs(max(data[i:i+win2])-min(data[i:i+win2]))
        aux1=abs(max(data[i:i+win1] )-min(data[i:i+win1] ))
        aux05=abs(max(data[i:i+win05])-min(data[i:i+win05]))
        for j in range(0,n,int(sf*30)):
            dat[j]=3  
        if aux05>threshold:
            ind_max=data[i:i+win1].index(max(data[i:i+win1]))
            ind_max=ind_max+i
            ind_min=data[i:i+win1].index(min(data[i:i+win1]))
            ind_min=ind_min+i
            dat[ind_min]=1.4
            dat[ind_max]=1.5
        else:
            if aux1>threshold:
                ind_max=data[i:i+win1].index(max(data[i:i+win1]))
                ind_max=ind_max+i
                ind_min=data[i:i+win1].index(min(data[i:i+win1]))
                ind_min=ind_min+i
                dat[ind_min]=0.9
                dat[ind_max]=1
            else:
                if aux2>threshold:
                    ind_max=data[i:i+win2].index(max(data[i:i+win2]))
                    ind_max=ind_max+i
                    ind_min=data[i:i+win2].index(min(data[i:i+win2]))
                    ind_min=ind_min+i
                    dat[ind_min]=0.4
                    dat[ind_max]=0.5
        #To see the progress
        if (i % (n//100))==0:
            print('Progress: ',(i/(n//100)))
    return dat

def upper_new(raw,name_channel,threshold,sf):  #STEP IS NOT APPLIED YET!!
    ### Extract data, sampling frequency and channels names
    data,sf,chan=raw._data,raw.info['sfreq'], raw.info['ch_names']
    data=data*1e6       # Convert Volts to uV
    print(data.shape)
    print(type(data))
    n = data.shape[1]   #samples    

   
    channel = (raw.ch_names).index(name_channel)
    print('Channel choose:',raw.ch_names[channel])
    data=data[channel][:]    
    data=data.tolist()
    
    step=int(sf/4)
    win2=int(sf*2)
    win1=int(sf*1)
    win05=int(sf/2)
    
    dat=[0 for i in range(0,n)]
    
    ###Cycle every second
    for i in range(0,n,step):   
        eeg2=data[i:i+win2]       #Every 2 seconds
        eeg1=data[i:i+win1]       #Every second
        eeg05=data[i:i+win05]     #Every half second
    
        aux2=abs(max(data[i:i+win2])-min(data[i:i+win2]))
        aux1=abs(max(data[i:i+win1] )-min(data[i:i+win1] ))
        aux05=abs(max(data[i:i+win05])-min(data[i:i+win05]))
        for j in range(0,n,int(sf*30)):
            dat[j]=3  
        if aux05>threshold:
            ind_max=data[i:i+win1].index(max(data[i:i+win1]))
            ind_max=ind_max+i
            ind_min=data[i:i+win1].index(min(data[i:i+win1]))
            ind_min=ind_min+i
            dat[ind_min]=1.4
            dat[ind_max]=1.5
        else:
            if aux1>threshold:
                ind_max=data[i:i+win1].index(max(data[i:i+win1]))
                ind_max=ind_max+i
                ind_min=data[i:i+win1].index(min(data[i:i+win1]))
                ind_min=ind_min+i
                dat[ind_min]=0.9
                dat[ind_max]=1
            else:
                if aux2>threshold:
                    ind_max=data[i:i+win2].index(max(data[i:i+win2]))
                    ind_max=ind_max+i
                    ind_min=data[i:i+win2].index(min(data[i:i+win2]))
                    ind_min=ind_min+i
                    dat[ind_min]=0.4
                    dat[ind_max]=0.5
        #To see the progress
        # if (i % (n//100))==0:
        #     print('Progress: ',(i/(n//100)))
    return dat

def pulse(time_shape,sfreq):
    t = np.linspace(1, round(time_shape/sfreq), time_shape, endpoint=False)    #Create artificial signal with a 0.5 sec pulse
    pulso = signal.square(2 * np.pi * 1 * t) # pulse signal
    return pulso

def subtraction_eog(raw):
    eog1= raw.get_data(picks='EOG1_1') 
    eog2= raw.get_data(picks='EOG2_1')  
    sub_eog = eog1-eog2
    return sub_eog

def subtraction_emg(raw):
    emg1= raw.get_data(picks='EMG1_1') 
    emg2= raw.get_data(picks='EMG2_1')   
    sub_emg = emg1-emg2
    return sub_emg

##Re-estructure data
def re_esctructure(raw, annot='yes'):
    data,sfreq =raw.get_data(),raw.info['sfreq']  
    time_shape = data.shape[1]
    
    sub_eog=subtraction_eog(raw)
    sub_emg =subtraction_emg(raw)

    pos_c3 = (raw.ch_names).index('C3_1')
    c3_1 = data[pos_c3,:]

    pos_c4 =(raw.ch_names).index('C4_1')
    c4_1 = data[pos_c4,:]

    new_data=data.copy()
    new_data[0]= sub_eog
    new_data[1]= yasa_filter(c4_1,sfreq)
    new_data[2]= c4_1
    new_data[3]= pulse(time_shape,sfreq)
    new_data[4]= sub_emg     
    new_data=new_data[[0,1,2,3,4], :]

    # new_data[0]= sub_eog
    # new_data[1]= sub_emg
    # new_data[2]= pulse(time_shape,sfreq)
    # new_data[3]= c3_1
    # new_data[4]= c4_1    
    # new_data[5]= upper_new(raw,'C4_1',75,sfreq)  
    # new_data=new_data[[0,1,2,3,4,5], :]

    new_ch_names= ['EOG_1-EOG_2','Filter YASA','C4_1','Grid','EMG_1-EMG_2']
    new_chtypes = ['eog'] + 2 *['eeg'] + ['emg']+ ['misc']  # Recompongo los canales.
    #new_ch_names = ['EOG', 'EMG', 'Pulse', 'C3', 'C4_1','Supera 75']  
    #new_chtypes = ['eog'] + ['emg']+ ['misc'] + 2 *['eeg'] + ['stim'] # Recompongo los canales.
    
    # Initialize an info structure      
    new_info = mne.create_info(new_ch_names, sfreq=sfreq, ch_types=new_chtypes)
    new_info['meas_date'] = raw.info['meas_date']      # Record timestamp for annotations
    
    new_raw=mne.io.RawArray(new_data, new_info)        # Build a new raw object 
    new_raw.set_annotations(raw.annotations)         
    
   
    if (annot=='yes'):
        original_annot=raw.annotations 
        new_raw.set_annotations(new_raw.annotations+original_annot) 

    return new_raw

def select_channels(raw,channels_names,annot='no'):
    data,sfreq =raw.get_data(),raw.info['sfreq']  
    sf =raw.info['sfreq']  
    time_shape = data.shape[1]
    n= data.shape[1] 

    # Specify this as the emg channel (channel type)
    d = {'C3_1':'eeg' ,'C4_1':'eeg','F3_1':'eeg','F4_1':'eeg','P4_1':'eeg','P3_1':'eeg','EOG_1-EOG_2':'eog','EOG1_1':'eog','EOG2_1':'eog', 'Grid':'misc',
    'EMG_1-EMG_2':'emg','EMG1_1': 'emg','EMG2_1': 'emg','Filter Siclari':'eeg','Filter YASA':'eeg','Filter MASSINI':'eeg','Threshold': 'stim'}
    
    print(raw.ch_names)
    pos_c4 =(raw.ch_names).index('C4_1')
    c4_1 = data[pos_c4,:]
    
    keys = []
    new_data=np.zeros((len(channels_names), n))
    cont=0
    #print(channels_names)
    for i in channels_names:
        if i=='EOG_1-EOG_2':
            new_data[cont,:]=subtraction_eog(raw)
        elif  i=='EMG_1-EMG_2':
            new_data[cont,:]=subtraction_emg(raw)
        elif i=='Grid':
            new_data[cont,:]=pulse(n,sf)
        elif i=='Threshold':
            new_data[cont,:]=upper_new(raw,'C4_1',75,sf)
        elif i=='Filter Siclari':
            new_data[cont,:]=Siclari2014_filter(c4_1,200)
        elif i=='Filter Tononi':
            new_data[cont,:]=filter(n,sf)
        elif i=='Filter Massini':
            new_data[cont,:]=ilter(n,sf)
        elif i=='Filter YASA':
            new_data[cont,:]=yasa_filter(c4_1,sf) #Filter in channel C4 
        elif i=='Wave Morlet':
            new_data[cont,:]=wave_morlet(1,sf) #Filter in channel C4   
        elif i=='Complex Detection':
            new_data[cont,:]=k_complex_detection(1,sf) #Filter in channel C4             
        else:
            pos=(raw.ch_names).index(str(i))
            new_data[cont,:]= data[pos,:]
        keys.append(i)
        cont+=1

    keys=tuple(keys)
    d1 = {k: d[k] for k in keys}
    new_info = mne.create_info(channels_names, sfreq=sf)
    new_info['meas_date'] = raw.info['meas_date']      # Record timestamp for annotationsù
    
    raw_plot=mne.io.RawArray(new_data, new_info)        # Build a new raw object 
    raw_plot.set_channel_types(d1)   
    raw_plot.set_annotations(raw.annotations)         

    if (annot == 'no'):
        pass
    elif (annot=='yes'):
        original_annot=raw.annotations 
        raw_plot.set_annotations(raw_plot.annotations+original_annot) 
    return raw_plot


def plot(raw,n_channels,scal,order,subject):

    """To visualize the data"""
    raw.plot(show_options=True,
    title='Etiquetado '+ str(subject),
    start=0,                        # initial time to show
    duration=30,                    # time window (sec) to plot in a given time
    n_channels=n_channels, 
    scalings=scal,                  # scaling factor for traces.
    block=True,
    order=order)




#Main function
def main():  # Wrapper function
    #messagebox.showinfo(message="This program allows you to tag a specific event.", title="Info")
    #Select the path file
    path = easygui.fileopenbox(title='Select RAW file (vdhr or fif).')

    #This line is if you want to ask, if not default is 'no'
    #stim_annot = messagebox.askquestion(message=" Do you want to see the original tags? (Stimulus)", title="Anotaciones")  
    #raw=load_brainvision(path, annot=stim_annot) 
    raw=load_brainvision(path) 
    path_states = easygui.fileopenbox(title='Select the hypnogram (file with extension txt).') #selecciono el txt de anotaciones anteriores
    #raw,hypno_annot= set_sleep_stages(raw,path_states)  
        
    raw,hypno_annot= set_sleep_stages(raw,path_states)  
    #raw,hypno_annot= set_sleep_stages(raw,path_states,annot=stim_annot)     
    
    #defa = messagebox.askquestion(message=" See default", title="Default")
    #if defa=='yes':
        #raw=re_esctructure(raw, annot=stim_annot)
    raw=re_esctructure(raw)


    anotaciones2 = messagebox.askquestion(message=" Do you want to see the YASA DETECTION tags? ", title="Anotaciones")
    if (anotaciones2 == 'yes'):
        _, yasa_annot=yasa_sw_detection(raw)
        my_annot=raw.annotations
        my_annot=my_annot+yasa_annot
        raw = raw.set_annotations(my_annot)
    
    anotaciones3 = messagebox.askquestion(message=" Do you want to see the MIMIR DETECTION tags? ", title="Anotaciones")
    if (anotaciones3 == 'yes'):
        _, MIMIR_annot=MIMIR_detection(raw)
        my_annot=raw.annotations   
        _,new_MIMIR_annot=delete_stage(raw,hypno_annot,0,MIMIR_annot)
        my_annot=my_annot+new_MIMIR_annot
        #my_annot=my_annot+MIMIR_annot
        raw = raw.set_annotations(my_annot)
   

    anotaciones = messagebox.askquestion(message=" Was a scoring done previously with this data?", title="Anotaciones")
    if (anotaciones == 'yes'):
        sw_path = easygui.fileopenbox(title='Select txt file with ANNOTATIONS.')#selecciono la carpeta vhdr
        _,sw_annot=set_event_annot(raw,sw_path)
        #_,sw_annot=set_event_annot(raw,sw_path, annot=stim_annot)
        my_annot=sw_annot+hypno_annot
        raw = raw.set_annotations(my_annot)

    #If yoy want to see information of the raw file
    #show_info(raw)
    

    my_annot=raw.annotations 
    raw = raw.set_annotations(my_annot)
    name,subject=file_name(path)

    
    # # Plot it!
    #For actual EEG/EOG/EMG/STIM data different scaling factors should be used.
    scal = dict(eeg=20e-5, eog=150e-5,emg=15e-4, misc=1e-3, stim=15)

    #if defa=='yes':
    n_channels=5
    order=[0,1,2,3,4]
    plot(raw,n_channels,scal,order,subject)
        #raw.annotations.save(name + ".txt")
        #print('Scoring was completed and the data was saved.')

    # elif defa=='no':
    #     ##Chose what you want!
    #     channels_names=['C3_1','C4_1','Filter YASA','P4_1']
    #     raw_plot=select_channels(raw,channels_names)#,annot='no')
    #     n_channels=len(channels_names)
    #     order=[i for i in range(n_channels)]
    #     plot(raw_plot,n_channels,scal,order,subject)

    #Save the tagged data
    anotaciones = messagebox.askquestion(message=" Do you want to save the annotations?", title="Saving")
    if (anotaciones == 'yes'):
        raw.annotations.save(name + ".txt")
        print('Scoring was completed and the data was saved.')
    else:
        print('The data was not saved.')

if __name__ == '__main__':
    main() 