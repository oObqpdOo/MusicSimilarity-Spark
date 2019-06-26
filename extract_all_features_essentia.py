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

#full array conversion to text! no truncation
np.set_printoptions(threshold=np.inf)

filelist = []

for filename in Path('music').glob('**/*.mp3'):
    filelist.append(filename)
    
for filename in Path('music').glob('**/*.wav'):
    filelist.append(filename)
    
fs = 44100
path = 'music/guitar.mp3'

def compute_features(path):
    # Loading audio file
    #will resample if sampleRate is different!
    audio = es.MonoLoader(filename=path, sampleRate=fs)()
    #will resample if sampleRate is different!
    loader = ess.MonoLoader(filename=path, sampleRate=44100)
    # Initialize algorithms we will use
    frameSize = 4096#512
    hopSize = 2048#256

    #######################################
    # DO FILTERING ONLY FOR MFCC - not with essentia standard
    # below is just an example

    #HP = es.HighPass(cutoffFrequency=128)
    #LP = es.LowPass(cutoffFrequency=4096)
    #lp_f = LP(audio)
    #hp_f = HP(lp_f)
    #audio = hp_f
    #MonoWriter(filename='music/filtered.wav')(filtered_audio)

    HP = ess.HighPass(cutoffFrequency=128)
    LP = ess.LowPass(cutoffFrequency=4096)
    #loader = ess.MonoLoader(filename=path, sampleRate=44100)
    #writer = ess.MonoWriter(filename='music/filtered.wav')
    #frameCutter = FrameCutter(frameSize = 1024, hopSize = 512)
    #pool = essentia.Pool()
    # Connect streaming algorithms
    #loader.audio >> HP.signal
    #HP.signal >> LP.signal
    #LP.signal >> writer.audio
    # Run streaming network
    #essentia.run(loader)
    
    #####################################
    # extract mfcc
    #####################################

    #features, features_frames = es.MusicExtractor(analysisSampleRate=44100, mfccStats=['mean', 'cov'])(path)
    #m, n = features['lowlevel.mfcc.cov'].shape
    #print m
    #iu1 = np.triu_indices(m)
    #cov = features['lowlevel.mfcc.cov'][iu1]
    #mean = features['lowlevel.mfcc.mean']
    #print(features['lowlevel.mfcc.cov'])
    
    hamming_window = es.Windowing(type='hamming')
    spectrum = es.Spectrum()  # we just want the magnitude spectrum
    mfcc = es.MFCC(numberCoefficients=13)
    frame_sz = 2048#512
    hop_sz = 1024#256
    mfccs = numpy.array([mfcc(spectrum(hamming_window(frame)))[1]
                   for frame in es.FrameGenerator(audio, frameSize=frame_sz, hopSize=hop_sz)])

    #Let's scale the MFCCs such that each coefficient dimension has zero mean and unit variance:
    #mfccs = sklearn.preprocessing.scale(mfccs)
    #print mfccs.shape

    mean = numpy.mean(mfccs.T, axis=1)
    #print(mean)
    var = numpy.var(mfccs.T, axis=1)
    #print(var)
    cov = numpy.cov(mfccs.T)

    #get only upper triangular matrix values to shorten length
    iu1 = np.triu_indices(12)
    cov = cov[iu1]

    #plt.imshow(mfccs.T, origin='lower', aspect='auto', interpolation='nearest')
    #plt.ylabel('MFCC Coefficient Index')
    #plt.xlabel('Frame Index')    
    #plt.colorbar()
    
    #####################################
    # extract beat features and histogram
    #####################################

    # Compute beat positions and BPM
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    peak1_bpm, peak1_weight, peak1_spread, peak2_bpm, peak2_weight, peak2_spread, histogram = BpmHistogramDescriptors()(beats_intervals)

    tempo = bpm
    times = beats
    beats_frames = (beats * fs) / hopSize
    beats_frames = beats_frames.astype(int)
    
    #fig, ax = plt.subplots()
    #ax.bar(range(len(histogram)), histogram, width=1)
    #ax.set_xlabel('BPM')
    #ax.set_ylabel('Frequency')
    #plt.title("BPM histogram")
    #ax.set_xticks([20 * x + 0.5 for x in range(int(len(histogram) / 20))])
    #ax.set_xticklabels([str(20 * x) for x in range(int(len(histogram) / 20))])
    #plt.show()
    
    #####################################
    # extract full beat aligned chroma
    #####################################
    
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
    
    ###################################
    # USE FILTER - comment next lines in
    
    loader.audio >> HP.signal
    HP.signal >> LP.signal
    LP.signal >> framecutter.signal
    ###################################
    
    ###################################
    # NO FILTER - comment next line in
    #loader.audio >> framecutter.signal        
    ###################################
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

    # Plot HPCP
    #imshow(pool['tonal.hpcp'].T, aspect='auto', origin='lower', interpolation='none')
    #plt.title("HPCPs in frames (the 0-th HPCP coefficient corresponds to A)")
    #show()
    
    #print beats_frames.shape[0]
    chroma_matrix = np.zeros((beats_frames.shape[0], 12))
    
    prev_beat = 0
    act_beat = 0
    sum_key = np.zeros(12)
    chroma_align = chroma
    chroma_align = chroma_align.transpose()
    mat_index = 0
    for i in beats_frames:
        act_beat = i
        value = sum(chroma_align[prev_beat:act_beat])/(act_beat-prev_beat)
        chroma_align[prev_beat:act_beat] = value
        prev_beat = i
        chroma_matrix[mat_index] = value
        mat_index = mat_index + 1
        
    #chroma_align = chroma_align.transpose()   
    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(chroma_align, y_axis='chroma', x_axis='time')
    #plt.vlines(times, 0, 12, alpha=0.5, color='r', linestyle='--', label='Beats')
    #plt.colorbar()
    #plt.title('Chromagram')
    #plt.tight_layout()
    #chroma_align = chroma_align.transpose()
    
    
    #print(chroma_align[24:28])
    #####################################
    # extract full chroma text
    #####################################
    
    #print(chroma.shape)
    m, n = chroma.shape
    avg = 0
    chroma = chroma.transpose()
    m, n = chroma.shape
    for j in chroma:
        avg = avg + np.sum(j)
    avg = avg / m
    threshold = avg / 2
    for i in chroma:
        if np.sum(i) > threshold:
            ind = np.where(i == np.max(i))
            max_val = i[ind]#is always 1!
            i[ind] = 0
            
            ind2 = np.where(i == np.max(i))
            i[ind] = 1
            
            #if np.any(i[ind2][0] >= 0.8 * max_val):
                #i[ind2] = i[ind2]
                #pass
            #low_values_flags = i < 1
            low_values_flags = i < 0.8
            
            i[low_values_flags] = 0
        else:
            i.fill(0)     
    chroma = chroma.transpose()
    # Compute beat positions and BPM
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
        
    prev_beat = 0
    act_beat = 0    
    for i in beats_frames:
        act_beat = i
        sum_key = sum(chroma[prev_beat:act_beat])
        ind = np.where(sum_key == np.max(sum_key))
        prev_beat = i
        notes.append(ind[0][0])
        prev_beat = i 
    
    #chroma = chroma.transpose()  
    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    #plt.vlines(times, 0, 12, alpha=0.5, color='r', linestyle='--', label='Beats')
    #plt.colorbar()
    #plt.title('Chromagram')
    #plt.tight_layout()
    #chroma = chroma.transpose()  
    
    return bpm, histogram, pool['tonal.key_key'], pool['tonal.key_scale'], notes, chroma_matrix, mean, cov


with open("features/out.mfcc", "w") as myfile:
    myfile.write("")
    myfile.close()
with open("features/out.chroma", "w") as myfile:
    myfile.write("")
    myfile.close()       
with open("features/out.bh", "w") as myfile:
    myfile.write("")
    myfile.close()
with open("features/out.notes", "w") as myfile:
    myfile.write("")
    myfile.close()


# Store start time
start_time = time.time()
count = 1

for file_name in filelist:
    path = str(PurePath(file_name))
    print ("File " + path + " " + str(count) + " von " + str(len(filelist))) 
    bpmret, hist, key, scale, notes, chroma_matrix, mean, cov = compute_features(path)
    with open("features/out.mfcc", "a") as myfile:
        print ("MFCC File " + path + " " + str(count) + " von " + str(len(filelist))) 
        mean = np.array2string(mean, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
        cov = np.array2string(cov, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
        line = (str(PurePath(file_name)) + "; " + mean + "; " + cov).replace('\n', '')
        myfile.write(line + '\n')       
        myfile.close()

    with open("features/out.chroma", "a") as myfile:
        print ("Chroma Full - File " + path + " " + str(count) + " von " + str(len(filelist))) 
        chroma_str = str(chroma_matrix).replace('\n', '')
        line = (str(PurePath(file_name)) + "; " + chroma_str).replace('\n', '')
        myfile.write(line + '\n')       
        myfile.close()

    with open("features/out.bh", "a") as myfile:
        print ("Beat Histogram - File " + path + " " + str(count) + " von " + str(len(filelist))) 
        bpmret = str(bpmret)
        hist = str(hist).replace('\n', '')
        line = (str(PurePath(file_name)) + "; " + bpmret + "; " + hist).replace('\n', '')
        myfile.write(line + '\n')       
        myfile.close()

    with open("features/out.notes", "a") as myfile:
        print ("Chroma Notes - File " + path + " " + str(count) + " von " + str(len(filelist))) 
        key = str(key)
        scale = str(scale).replace('\n', '')
        notes = str(notes).replace('\n', '')
        line = (str(PurePath(file_name)) + "; " + key + "; " + scale + "; " + notes).replace('\n', '')
        myfile.write(line + '\n')       
        myfile.close()

    count = count + 1

# Perform any action like print a string
print("calculating this takes ...")

# Store end time
end_time = time.time()

# Calculate the execution time and print the result
print("%.10f seconds" % (end_time - start_time))


