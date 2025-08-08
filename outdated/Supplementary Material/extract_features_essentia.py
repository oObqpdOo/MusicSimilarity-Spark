import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, urllib, IPython.display
import numpy as np
import librosa.display
import signal

from scipy.signal import butter, lfilter, freqz

import glob
from pathlib import Path, PurePath
import essentia
import essentia.standard as es
import essentia.streaming as ess
from essentia.standard import *

import time

filelist = []

for filename in Path('music').glob('**/*.mp3'):
    filelist.append(filename)
    
for filename in Path('music').glob('**/*.wav'):
    filelist.append(filename)
    
fs = 44100
path = 'music/guitar.mp3'

def compute_bpm_hist(path):
    # Loading audio file
    audio = MonoLoader(filename=path, sampleRate=fs)()
    # Compute beat positions and BPM
    rhythm_extractor = RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    peak1_bpm, peak1_weight, peak1_spread, peak2_bpm, peak2_weight, peak2_spread, histogram = BpmHistogramDescriptors()(beats_intervals)
    return bpm, histogram

def compute_mfcc(path):
    # Loading audio file
    audio = MonoLoader(filename=path, sampleRate=fs)()
    #analysis sample rate (audio will be converted to it before analysis, recommended and default value is 44100.0)
    # Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
    features, features_frames = es.MusicExtractor(analysisSampleRate=44100, mfccStats=['mean', 'cov'])(path)
    # See all feature names in the pool in a sorted order
    #get only upper triangular matrix values to shorten length
    m, n = features['lowlevel.mfcc.cov'].shape
    #print m
    iu1 = np.triu_indices(m)
    cov = features['lowlevel.mfcc.cov'][iu1]
    #print(features['lowlevel.mfcc.cov'])
    return features['lowlevel.mfcc.mean'], cov

def compute_chroma_notes(path):
    # Loading audio file
    audio = MonoLoader(filename=path, sampleRate=fs)()
    # Initialize algorithms we will use
    frameSize = 4096#512
    hopSize = 2048#256
    #will resample if sampleRate is different!
    loader = ess.MonoLoader(filename=path, sampleRate=44100)
    framecutter = ess.FrameCutter(frameSize=frameSize, hopSize=hopSize, silentFrames='noise')
    windowing = ess.Windowing(type='blackmanharris62')
    spectrum = ess.Spectrum()
    spectralpeaks = ess.SpectralPeaks(orderBy='magnitude',
                                      magnitudeThreshold=0.00001,
                                      minFrequency=20,
                                      maxFrequency=3500,
                                      maxPeaks=60)
    # Use default HPCP parameters for plots, however we will need higher resolution
    # and custom parameters for better Key estimation
    hpcp = ess.HPCP()
    hpcp_key = ess.HPCP(size=36, # we will need higher resolution for Key estimation
                        referenceFrequency=440, # assume tuning frequency is 44100.
                        bandPreset=False,
                        minFrequency=20,
                        maxFrequency=3500,
                        weightType='cosine',
                        nonLinear=False,
                        windowSize=1.)
    key = ess.Key(profileType='edma', # Use profile for electronic music
                  numHarmonics=4,
                  pcpSize=36,
                  slope=0.6,
                  usePolyphony=True,
                  useThreeChords=True)
    # Use pool to store data
    pool = essentia.Pool()
    # Connect streaming algorithms
    loader.audio >> framecutter.signal
    framecutter.frame >> windowing.frame >> spectrum.frame
    spectrum.spectrum >> spectralpeaks.spectrum
    spectralpeaks.magnitudes >> hpcp.magnitudes
    spectralpeaks.frequencies >> hpcp.frequencies
    spectralpeaks.magnitudes >> hpcp_key.magnitudes
    spectralpeaks.frequencies >> hpcp_key.frequencies
    hpcp_key.hpcp >> key.pcp
    hpcp.hpcp >> (pool, 'tonal.hpcp')
    key.key >> (pool, 'tonal.key_key')
    key.scale >> (pool, 'tonal.key_scale')
    key.strength >> (pool, 'tonal.key_strength')
    # Run streaming network
    essentia.run(loader)
    #print("Estimated key and scale:", pool['tonal.key_key'] + " " + pool['tonal.key_scale'])
    #print(pool['tonal.hpcp'].T)
    chroma = pool['tonal.hpcp'].T
    #print(chroma.shape)
    m, n = chroma.shape
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
    chroma = chroma.transpose()
    # Compute beat positions and BPM
    rhythm_extractor = RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    tempo = bpm
    times = beats
    beats_frames = (beats * fs) / hopSize
    beats_frames = beats_frames.astype(int)
    prev_beat = 0
    act_beat = 0
    sum_key = np.zeros(12)
    chroma = chroma.transpose()  
    for i in beats_frames:
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
    for i in beats_frames:
        act_beat = i
        sum_key = sum(chroma[prev_beat:act_beat])
        ind = np.where(sum_key == np.max(sum_key))
        prev_beat = i
        notes.append(ind[0][0])
    chroma = chroma.transpose()  
    return pool['tonal.key_key'], pool['tonal.key_scale'], notes

def compute_chroma_aligned(path):
    # Loading audio file
    audio = MonoLoader(filename=path, sampleRate=fs)()
    # Initialize algorithms we will use
    frameSize = 4096#512
    hopSize = 2048#256
    #will resample if sampleRate is different!
    loader = ess.MonoLoader(filename=path, sampleRate=44100)
    framecutter = ess.FrameCutter(frameSize=frameSize, hopSize=hopSize, silentFrames='noise')
    windowing = ess.Windowing(type='blackmanharris62')
    spectrum = ess.Spectrum()
    spectralpeaks = ess.SpectralPeaks(orderBy='magnitude',
                                      magnitudeThreshold=0.00001,
                                      minFrequency=20,
                                      maxFrequency=3500,
                                      maxPeaks=60)
    # Use default HPCP parameters for plots, however we will need higher resolution
    # and custom parameters for better Key estimation
    hpcp = ess.HPCP()
    hpcp_key = ess.HPCP(size=36, # we will need higher resolution for Key estimation
                        referenceFrequency=440, # assume tuning frequency is 44100.
                        bandPreset=False,
                        minFrequency=20,
                        maxFrequency=3500,
                        weightType='cosine',
                        nonLinear=False,
                        windowSize=1.)
    key = ess.Key(profileType='edma', # Use profile for electronic music
                  numHarmonics=4,
                  pcpSize=36,
                  slope=0.6,
                  usePolyphony=True,
                  useThreeChords=True)
    # Use pool to store data
    pool = essentia.Pool()
    # Connect streaming algorithms
    loader.audio >> framecutter.signal
    framecutter.frame >> windowing.frame >> spectrum.frame
    spectrum.spectrum >> spectralpeaks.spectrum
    spectralpeaks.magnitudes >> hpcp.magnitudes
    spectralpeaks.frequencies >> hpcp.frequencies
    spectralpeaks.magnitudes >> hpcp_key.magnitudes
    spectralpeaks.frequencies >> hpcp_key.frequencies
    hpcp_key.hpcp >> key.pcp
    hpcp.hpcp >> (pool, 'tonal.hpcp')
    key.key >> (pool, 'tonal.key_key')
    key.scale >> (pool, 'tonal.key_scale')
    key.strength >> (pool, 'tonal.key_strength')
    # Run streaming network
    essentia.run(loader)
    #print("Estimated key and scale:", pool['tonal.key_key'] + " " + pool['tonal.key_scale'])
    chroma = pool['tonal.hpcp'].T
    threshold = 300
    m, n = chroma.shape
    # Compute beat positions and BPM
    rhythm_extractor = RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    tempo = bpm
    times = beats
    beats_frames = (beats * fs) / hopSize
    beats_frames = beats_frames.astype(int)
    prev_beat = 0
    act_beat = 0
    sum_key = np.zeros(12)
    chroma = chroma.transpose()  
    for i in beats_frames:
        act_beat = i
        chroma[prev_beat:act_beat] = sum(chroma[prev_beat:act_beat])/(act_beat-prev_beat)
        prev_beat = i
    chroma = chroma.transpose()  
    return pool['tonal.key_key'], pool['tonal.key_scale'], chroma


path = 'music/guitar.mp3'

#key, scale, notes = compute_chroma_notes(path)
#print key
#print scale
#print notes
#key, scale, notes = compute_chroma_aligned(path)
#print key
#print scale
#print chroma.shape
#bpmret, hist = compute_bpm_hist(path)
#print bpmret
#print hist
#mean, cov = compute_mfcc(path)
#print mean
#print cov


# Store start time
start_time = time.time()

a = 1

#a = 0
if a == 0:
    with open("features/out.mfcc", "w") as myfile:
        myfile.write("")
        myfile.close()
            
    with open("features/out.mfcc", "a") as myfile:
        count = 1
        for file_name in filelist:
            path = str(PurePath(file_name))
            print ("MFCC - File " + path + " " + str(count) + " von " + str(len(filelist))) 
            mean, cov = compute_mfcc(path)
            mean = np.array2string(mean, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
            cov = np.array2string(cov, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
            line = (str(PurePath(file_name)) + "; " + mean + "; " + cov).replace('\n', '')
            myfile.write(line + '\n')       
            count = count + 1
        myfile.close()

#a = 1
if a == 1:
    with open("features/out.chroma", "w") as myfile:
        myfile.write("")
        myfile.close()

    with open("features/out.chroma", "a") as myfile:
        count = 1
        for file_name in filelist: 
            path = str(PurePath(file_name))
            print ("Chroma Full - File " + path + " " + str(count) + " von " + str(len(filelist))) 
            key, scale, notes = compute_chroma_aligned(path)
            key = str(key)
            scale = str(scale).replace('\n', '')
            notes = str(notes).replace('\n', '')
            line = (str(PurePath(file_name)) + "; " + key + "; " + scale + "; " + notes).replace('\n', '')
            myfile.write(line + '\n')       
            count = count + 1
        myfile.close()

#a = 2
if a == 2:
    with open("features/out.bh", "w") as myfile:
        myfile.write("")
        myfile.close()

    with open("features/out.bh", "a") as myfile:
        count = 1
        for file_name in filelist: 
            path = str(PurePath(file_name))
            print ("Beat Histogram - File " + path + " " + str(count) + " von " + str(len(filelist))) 
            bpmret, hist = compute_bpm_hist(path)
            bpmret = str(bpmret)
            hist = str(hist).replace('\n', '')
            line = (str(PurePath(file_name)) + "; " + bpmret + "; " + hist).replace('\n', '')
            myfile.write(line + '\n')       
            count = count + 1
        myfile.close()

    with open("features/out.notes", "w") as myfile:
        myfile.write("")
        myfile.close()

#a = 3
if a == 3:
    with open("features/out.notes", "a") as myfile:
        count = 1
        for file_name in filelist: 
            path = str(PurePath(file_name))
            print ("Chroma Notes - File " + path + " " + str(count) + " von " + str(len(filelist))) 
            key, scale, notes = compute_chroma_notes(path)
            key = str(key)
            scale = str(scale).replace('\n', '')
            notes = str(notes).replace('\n', '')
            line = (str(PurePath(file_name)) + "; " + key + "; " + scale + "; " + notes).replace('\n', '')
            myfile.write(line + '\n')       
            count = count + 1
        myfile.close()


# Perform any action like print a string
print("calculating this takes ...")

# Store end time
end_time = time.time()

# Calculate the execution time and print the result
print("%.10f seconds" % (end_time - start_time))

