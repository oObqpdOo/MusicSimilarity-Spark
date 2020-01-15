#!/usr/bin/env python
from __future__ import print_function
from mpi4py import MPI
import numpy as np
from pathlib import Path, PurePath
from time import time, sleep
import multiprocessing
import os
import argparse
import gc
import scipy
import signal
from scipy.signal import butter, lfilter, freqz, correlate2d
import glob
import essentia
import essentia.standard as es
import essentia.streaming as ess
from essentia.standard import *
from pathlib import Path, PurePath
from time import time, sleep
import time as time

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

gc.enable()
filelist = []
for filename in Path('/beegfs/ja62lel/fma_full').glob('**/*.mp3'):
    filelist.append(filename)
for filename in Path('/beegfs/ja62lel/fma_full').glob('**/*.wav'):
    filelist.append(filename)
for filename in Path('/beegfs/ja62lel/fma_full').glob('**/*.aiff'):
    filelist.append(filename) 
for filename in Path('/beegfs/ja62lel/fma_full').glob('**/*.flac'):
    filelist.append(filename) 
for filename in Path('/beegfs/ja62lel/fma_full').glob('**/*.ogg'):
    filelist.append(filename) 
print("length of filelist" + str(len(filelist)))

np.set_printoptions(threshold=np.inf)
fs = 44100
octave = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

def transpose_chroma_matrix(key, scale, chroma_param):
    if key == 'Ab':
        key = 'G#'
    if key == 'Gb':
        key = 'F#'
    if key == 'Eb':
        key = 'D#'
    if key == 'Db':
        key = 'C#'
    if key == 'Bb':
        key = 'A#'
    chroma_param = chroma_param.transpose()
    transposed_chroma = np.zeros(chroma_param.shape)
    if key != 'A':
        offs = 12 - octave.index(key)
        for ind in range(len(chroma_param)):
            index = (ind + offs)
            if(index >= 12):
                index = index - 12
            transposed_chroma[index] = chroma_param[ind]
    else:
        transposed_chroma = chroma_param
    transposed_chroma = transposed_chroma.transpose()
    return transposed_chroma

def transpose_chroma_notes(key, scale, notes):
    if key == 'Ab':
        key = 'G#'
    if key == 'Gb':
        key = 'F#'
    if key == 'Eb':
        key = 'D#'
    if key == 'Db':
        key = 'C#'
    if key == 'Bb':
        key = 'A#'
    transposed = notes
    if key != 'A':
        offs = 12 - octave.index(key)
        index = 0
        for i in notes:
            i = i + offs
            if(i >= 12):
                i = i - 12
            transposed[index] = i
            index = index + 1
    return transposed

def compute_features(path, f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh):
    gc.enable()
    try: 
	audio = es.MonoLoader(filename=path, sampleRate=fs)()
    except: 
        print("Erroneos File detected by essentia standard: skipping!")
	    return 0, [], 0, 0, [], [], [], [], [], []
    #will resample if sampleRate is different!
    try: 
        loader = ess.MonoLoader(filename=path, sampleRate=44100)
    except:
	    print("Erroneos File detected by essentia streaming: skipping!")
	    return 0, [], 0, 0, [], [], [], [], [], []
    frameSize = 4096
    hopSize = 2048
    HP = ess.HighPass(cutoffFrequency=128)
    LP = ess.LowPass(cutoffFrequency=4096)
    bpm = 0
    histogram = 0
    key = 0
    scale = 0
    notes = 0
    chroma_matrix = 0
    mean = 0
    cov = 0
    var = 0
    cov_kl = 0
    if f_mfcc_kl == 1 or f_mfcc_euclid == 1:
        hamming_window = es.Windowing(type='hamming')
        spectrum = es.Spectrum()
        mfcc = es.MFCC(numberCoefficients=13)
        frame_sz = 2048#512
        hop_sz = 1024#256
        mfccs = np.array([mfcc(spectrum(hamming_window(frame)))[1] for frame in es.FrameGenerator(audio, frameSize=frame_sz, hopSize=hop_sz)])
        mean = np.mean(mfccs.T, axis=1)
        var = np.var(mfccs.T, axis=1)
        cov = np.cov(mfccs.T)
        cov_kl = cov
        iu1 = np.triu_indices(13)
        cov = cov[iu1]
    if f_bh == 1 or f_chroma == 1 or f_notes == 1:
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        if f_bh == 1:
            peak1_bpm, peak1_weight, peak1_spread, peak2_bpm, peak2_weight, peak2_spread, histogram = es.BpmHistogramDescriptors()(beats_intervals)
        tempo = bpm
        times = beats
        beats_frames = (beats * fs) / hopSize
        beats_frames = beats_frames.astype(int)
    framecutter = ess.FrameCutter(frameSize=frameSize, hopSize=hopSize, silentFrames='noise')
    windowing = ess.Windowing(type='blackmanharris62')
    spectrum = ess.Spectrum()
    spectralpeaks = ess.SpectralPeaks(orderBy='magnitude',
                                      magnitudeThreshold=0.00001,
                                      minFrequency=20,
                                      maxFrequency=3500,
                                      maxPeaks=60)
    hpcp = ess.HPCP()
    hpcp_key = ess.HPCP(size=36, 
                        referenceFrequency=440, 
                        bandPreset=False,
                        minFrequency=20,
                        maxFrequency=3500,
                        weightType='cosine',
                        nonLinear=False,
                        windowSize=1.)
    key = ess.Key(profileType='edma', 
                  numHarmonics=4,
                  pcpSize=36,
                  slope=0.6,
                  usePolyphony=True,
                  useThreeChords=True)
    pool = essentia.Pool()
    loader.audio >> HP.signal
    HP.signal >> LP.signal
    LP.signal >> framecutter.signal
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
    essentia.run(loader)
    chroma = pool['tonal.hpcp'].T
    key = pool['tonal.key_key']
    scale = pool['tonal.key_scale']
    if f_chroma == 1:
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
            if np.linalg.norm(value, ord=1) != 0:
                value = value / np.linalg.norm(value, ord=1)
            chroma_matrix[mat_index] = value
            mat_index = mat_index + 1
    if f_notes == 1:
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
                low_values_flags = i < 0.8
                i[low_values_flags] = 0
            else:
                i.fill(0)
        chroma = chroma.transpose()
        prev_beat = 0
        act_beat = 0
        sum_key = np.zeros(12)
        chroma = chroma.transpose()
        for i in beats_frames:
            act_beat = i
            sum_key = sum(chroma[prev_beat:act_beat])
            ind = np.where(sum_key == np.max(sum_key))
            ind = ind[0]
            fill = np.zeros(len(j))
            if(np.all(chroma[prev_beat:act_beat] == 0)):
                fill[ind] = 0
            else:
                fill[ind] = 1
            chroma[prev_beat:act_beat] = fill
            prev_beat = i
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
    gc.collect()
    return bpm, histogram, key, scale, notes, chroma_matrix, mean, cov, var, cov_kl

def parallel_python_process(process_id, cpu_filelist, f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh):
    if f_mfcc_euclid == 1:
        with open("features0/out" + str(process_id) + ".mfcc", "w") as myfile:
            myfile.write("")
            myfile.close()
    if f_mfcc_kl == 1:
        with open("features0/out" + str(process_id) + ".mfcckl", "w") as myfile:
            myfile.write("")
            myfile.close()
    if f_chroma == 1:
        with open("features0/out" + str(process_id) + ".chroma", "w") as myfile:
            myfile.write("")
            myfile.close()
    if f_bh == 1:
        with open("features0/out" + str(process_id) + ".bh", "w") as myfile:
            myfile.write("")
            myfile.close()
    if f_notes == 1:
        with open("features0/out" + str(process_id) + ".notes", "w") as myfile:
            myfile.write("")
            myfile.close()
    count = 1
    for file_name in cpu_filelist:
        path = str(PurePath(file_name))
        print ("File " + path + " " + str(count) + " von " + str(len(cpu_filelist)))
        bpmret, hist, key, scale, notes, chroma_matrix, mean, cov, var, cov_kl = compute_features(path, f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh)
        if key == 0:
            continue
        filename = path.replace(".","").replace(";","").replace(",","").replace("mp3",".mp3").replace("aiff",".aiff").replace("aif",".aif").replace("au",".au").replace("m4a", ".m4a").replace("wav",".wav").replace("flac",".flac").replace("ogg",".ogg")  # rel. filename as from find_files
        if f_mfcc_euclid == 1:
            with open("features0/out" + str(process_id) + ".mfcc", "a") as myfile:
                print ("MFCC File " + path + " " + str(count) + " von " + str(len(cpu_filelist)))
                str_mean = np.array2string(mean, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
                str_var = np.array2string(var, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
                str_cov = np.array2string(cov, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
                line = (filename + "; " + str_mean + "; " + str_var + "; " + str_cov).replace('\n', '')
                myfile.write(line + '\n')
                myfile.close()
        if f_chroma == 1:
            with open("features0/out" + str(process_id) + ".chroma", "a") as myfile:
                print ("Chroma Full - File " + path + " " + str(count) + " von " + str(len(cpu_filelist)))
                transposed_chroma = np.zeros(chroma_matrix.shape)
                transposed_chroma = transpose_chroma_matrix(key, scale, chroma_matrix)
                chroma_str = np.array2string(transposed_chroma.transpose(), separator=',', suppress_small=True).replace('\n', '')
                line = (filename + "; " + chroma_str).replace('\n', '')
                myfile.write(line + '\n')
                myfile.close()
        if f_bh == 1:
            with open("features0/out" + str(process_id) + ".bh", "a") as myfile:
                print ("Beat Histogram - File " + path + " " + str(count) + " von " + str(len(cpu_filelist)))
                bpmret = str(bpmret)
                hist = np.array2string(hist, separator=',', suppress_small=True).replace('\n', '')
                line = (filename + "; " + bpmret + "; " + hist).replace('\n', '')
                myfile.write(line + '\n')
                myfile.close()
        if f_notes == 1:
            with open("features0/out" + str(process_id) + ".notes", "a") as myfile:
                print ("Chroma Notes - File " + path + " " + str(count) + " von " + str(len(cpu_filelist)))
                key = str(key)
                transposed_notes = []
                transposed_notes = transpose_chroma_notes(key, scale, notes)
                #print notes
                scale = str(scale).replace('\n', '')
                transposed_notes = str(transposed_notes).replace('\n', '')
                line = (filename + "; " + key + "; " + scale + "; " + transposed_notes).replace('\n', '')
                myfile.write(line + '\n')
                myfile.close()
        if f_mfcc_kl == 1:
            with open("features0/out" + str(process_id) + ".mfcckl", "a") as myfile:
                print ("MFCC Kullback-Leibler " + path + " " + str(count) + " von " + str(len(cpu_filelist)))
                str_mean = np.array2string(mean, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
                str_cov_kl = np.array2string(cov_kl, precision=8, separator=',', suppress_small=True).replace('\n', '')#.strip('[ ]')
                line = (filename + "; " + str_mean + "; " + str_cov_kl).replace('\n', '')
                myfile.write(line + '\n')
                myfile.close()
        count = count + 1
        del bpmret, hist, key, scale, notes, chroma_matrix, mean, cov, var, cov_kl
        gc.enable()
        gc.collect()
    gc.enable()
    gc.collect()
    return 1

def process_stuff(startjob, maxparts, batchsz, f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh):
    startjob = int(startjob)
    maxparts = int(maxparts) + 1
    files_per_part = int(batchsz)
    print("starting with: ")
    print(startjob)
    print("ending with: ")
    print(maxparts - 1)
    print("files per part: ")
    print(files_per_part)
    start = 0
    end = len(filelist)
    print("used cores: " + str(size))
    ncpus = size
    parts = (len(filelist) / files_per_part) + 1
    print("Split problem in parts: ")
    print(str(parts))
    step = (end - start) / parts + 1
    if maxparts > parts:
        maxparts = parts
    for index in xrange(startjob + rank, maxparts, size):
        if index < parts:
            starti = start+index*step
            endi = min(start+(index+1)*step, end)
            print("calling process  " + str(rank) + " index " + str(index) + " size " + str(size) + " starti " + str(starti) + " endi " + str(endi))
            parallel_python_process(index, filelist[starti:endi], f_mfcc_kl, f_mfcc_euclid, f_notes, f_chroma, f_bh)
            gc.collect()
    gc.enable()
    gc.collect()
do_mfcc_kl = 1
do_mfcc_euclid = 1
do_notes = 1
do_chroma = 1
do_bh = 1
startbatch = 0
endbatch = 1000000
batchsize = 25

time_dict = {}
tic1 = int(round(time.time() * 1000))
process_stuff(startbatch, endbatch, batchsize, do_mfcc_kl, do_mfcc_euclid, do_notes, do_chroma, do_bh)
tac1 = int(round(time.time() * 1000))
time_dict['MPI TIME FEATURE']= tac1 - tic1
#if rank == 0:
print("Process " + str(rank) + " time: " + str(time_dict)) 

