import scipy, sklearn, librosa
import essentia, essentia.standard as ess
import numpy as np
import signal
from scipy.signal import butter, lfilter, freqz
import glob
from pathlib import Path, PurePath
import unicsv # unicode csv library (installed via pip install unicsv)
import csv # unicode csv library (installed via pip install unicsv)
import time

filelist = []

for filename in Path('music').glob('**/*.mp3'):
    filelist.append(filename)
    
for filename in Path('music').glob('**/*.wav'):
    filelist.append(filename)
    
#print(filelist[4])

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_mfcc(x, fs):
    mfccbands = 12
    mfcc = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=mfccbands)
    mean = np.mean(mfcc, axis=1)
    #var = numpy.var(mfcc, axis=1)

    #get only upper triangular matrix values to shorten length    
    #cov = numpy.cov(mfcc)
        
    iu1 = np.triu_indices(mfccbands)
    cov = np.cov(mfcc)[iu1]
    
    return mean, cov

def extract_chroma(x, fs):
    # Filter requirements.
    order = 6
    fs = fs
    cutoff_hp = 128  # desired cutoff frequency of the filter, Hz
    cutoff_lp = 4096  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_highpass(cutoff_hp, fs, order)
    data = x

    # Filter the data, and plot both the original and filtered signals.
    x_hp_filtered = butter_highpass_filter(data, cutoff_hp, fs, order)

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff_lp, fs, order)

    data = x_hp_filtered

    # Filter the data, and plot both the original and filtered signals.
    x_lp_filtered = butter_lowpass_filter(data, cutoff_lp, fs, order)

    #original audio signal - extract max chroma key only
    chroma = librosa.feature.chroma_stft(x_hp_filtered, fs)
    avg = 0
    
    chroma = chroma.transpose()
    m, n = chroma.shape

    for j in chroma:
        avg = avg + np.sum(j)

    avg = avg / m
    threshold = avg

    for i in chroma:
        if np.sum(i) > threshold:
            ind = np.where(i == np.max(i))
            max_val = i[ind]#is always 1!
            i[ind] = 0

            ind2 = np.where(i == np.max(i))
            i[ind] = 1

            if np.any(i[ind2] >= 0.8 * max_val):
                #i[ind2] = i[ind2]
                pass
            #low_values_flags = i < 1
            low_values_flags = i < 0.8

            i[low_values_flags] = 0
        else:
            i.fill(0)         

    tempo, beats = librosa.beat.beat_track(x, fs)#
    #print(tempo)
    onset_env = librosa.onset.onset_strength(x, fs, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env,sr=fs)
    hop_length = 512
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=fs, hop_length=hop_length)

    prev_beat = 0
    act_beat = 0
    sum_key = np.zeros(12)

    for i in beats:
        act_beat = i
        sum_key = sum(chroma[prev_beat:act_beat])
        #print(sum_key)
        #print(chroma[prev_beat:act_beat])

        ind = np.where(sum_key == np.max(sum_key))
        ind = ind[0]
        #print("debug")
        fill = np.zeros(len(j))
        if(np.all(chroma[prev_beat:act_beat] == 0)):
            fill[ind] = 0
        else:    
            fill[ind] = 1
        chroma[prev_beat:act_beat] = fill
        #print(chroma[prev_beat:act_beat])
        prev_beat = i
        #print("BEAT")

    notes = []
    for i in notes:
        del i

    for i in beats:
        act_beat = i
        sum_key = sum(chroma[prev_beat:act_beat])
        ind = np.where(sum_key == np.max(sum_key))
        prev_beat = i
        notes.append(ind[0][0])

    chroma = chroma.transpose()  
    
    return tempo, notes

# Store start time
start_time = time.time()

with open("features/out.mfcc", "w") as myfile:
    myfile.write("")
    myfile.close()
        
with open("features/out.mfcc", "a") as myfile:
    count = 1
    for file_name in filelist:
        path = str(PurePath(file_name))
        print ("MFCC - File " + path + " " + str(count) + " von " + str(len(filelist))) 
        x, fs = librosa.load(path)
        mean, cov = extract_mfcc(x, fs)
        mean = np.array2string(mean, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
        cov = np.array2string(cov, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
        line = (str(PurePath(file_name)) + "; " + mean + "; " + cov).replace('\n', '')
        myfile.write(line + '\n')       
        count = count + 1
    myfile.close()

with open("features/out.chroma", "w") as myfile:
    myfile.write("")
    myfile.close()

with open("features/out.chroma", "a") as myfile:
    count = 1
    for file_name in filelist: 
        path = str(PurePath(file_name))
        print ("Chroma - File " + path + " " + str(count) + " von " + str(len(filelist))) 
        x, fs = librosa.load(path)
        tempo, notes = extract_chroma(x, fs)
        tempo = str(tempo)
        notes = str(notes).replace('\n', '')
        line = (str(PurePath(file_name)) + "; " + tempo + "; " + notes).replace('\n', '')
        myfile.write(line + '\n')       
        count = count + 1
    myfile.close()

# Perform any action like print a string
print("Printing this string takes ...")

# Store end time
end_time = time.time()

# Calculate the execution time and print the result
print("%.10f seconds" % (end_time - start_time))

