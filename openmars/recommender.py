#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyspark
import pyspark.ml.feature
import pyspark.mllib.linalg
import pyspark.ml.param
import pyspark.sql.functions
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from scipy.spatial import distance
#only version 2.1 upwards
#from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.mllib.linalg import Vectors
from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import numpy as np
#import org.apache.spark.sql.functions.typedLit
from pyspark.sql.functions import lit
from pyspark.sql.functions import levenshtein  
from pyspark.sql.functions import col
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
import scipy as sp
from scipy.signal import butter, lfilter, freqz, correlate2d, sosfilt
import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
import edlib
import openmars

class recommender:
	def __init__(self, df, sc, td, st):
		self.fullFeatureDF = df
		self.time_dict = td
		self.starttime = st
		#TODO: For debugging purposes, find a way to pass this to @staticmethod
		#self.negjs = sc.accumulator(0)
		#self.nanjs = sc.accumulator(0)
		#self.nonpdjs = sc.accumulator(0)
		#self.negskl = sc.accumulator(0)
		#self.nanskl = sc.accumulator(0)
		#self.noninskl = sc.accumulator(0)
		return

	@staticmethod
	def chroma_cross_correlate(chroma1_par, chroma2_par):
		length1 = int(chroma1_par.size/12)
		chroma1 = np.empty([12, length1])
		length2 = int(chroma2_par.size/12)
		chroma2 = np.empty([12, length2])
		if(length1 > length2):
		    chroma1 = chroma1_par.reshape(12, length1)
		    chroma2 = chroma2_par.reshape(12, length2)
		else:
		    chroma2 = chroma1_par.reshape(12, length1)
		    chroma1 = chroma2_par.reshape(12, length2) 
		corr = correlate2d(chroma1, chroma2, mode='same') 
		#print np.max(mean_line)
		return np.max(corr)

	@staticmethod
	def chroma_cross_correlate_full(chroma1_par, chroma2_par):
		length1 = int(chroma1_par.size/12)
		chroma1 = np.empty([length1,12])
		length2 = int(chroma2_par.size/12)
		chroma2 = np.empty([length2,12])
		if(length1 > length2):
		    chroma1 = chroma1_par.reshape(length1, 12)
		    chroma2 = chroma2_par.reshape(length2, 12)
		else:
		    chroma2 = chroma1_par.reshape(length1, 12)
		    chroma1 = chroma2_par.reshape(length2, 12)    
		corr = correlate2d(chroma1, chroma2, mode='full')
		transposed_chroma = corr.transpose()  
		#print "length1: " + str(length1)
		#print "length2: " + str(length2)
		#transposed_chroma = transposed_chroma / (min(length1, length2))
		index = np.where(transposed_chroma == np.amax(transposed_chroma))
		index = int(index[0])
		#print "index: " + str(index)
		transposed_chroma = transposed_chroma.transpose()
		transposed_chroma = np.transpose(transposed_chroma)
		mean_line = transposed_chroma[index]
		sos = butter(1, 0.1, 'high', analog=False, output='sos')
		mean_line = sosfilt(sos, mean_line)
		#print np.max(mean_line)
		return np.max(mean_line)

	@staticmethod
	def chroma_cross_correlate_valid(chroma1_par, chroma2_par):
		length1 = int(chroma1_par.size/12)
		chroma1 = np.empty([12, length1])
		length2 = int(chroma2_par.size/12)
		chroma2 = np.empty([12, length2])
		if(length1 > length2):
		    chroma1 = chroma1_par.reshape(12, length1)
		    chroma2 = chroma2_par.reshape(12, length2)
		else:
		    chroma2 = chroma1_par.reshape(12, length1)
		    chroma1 = chroma2_par.reshape(12, length2)      
		#full
		#correlation = np.zeros([length1 + length2 - 1])
		#valid
		#correlation = np.zeros([max(length1, length2) - min(length1, length2) + 1])
		#same
		correlation = np.zeros([max(length1, length2)])
		for i in range(12):
		    correlation = correlation + np.correlate(chroma1[i], chroma2[i], "same")    
		#remove offset to get rid of initial filter peak(highpass of jump from 0-20)
		correlation = correlation - correlation[0]
		sos = butter(1, 0.1, 'high', analog=False, output='sos')
		correlation = sosfilt(sos, correlation)[:]
		return np.max(correlation)

	@staticmethod
	def jensen_shannon(vec1, vec2):
		d = 13
		mean1 = np.empty([d, 1])
		mean1 = vec1[0:d]
		cov1 = np.empty([d,13])
		cov1 = vec1[d:].reshape(d, d)
		div = np.inf
		#div = float('NaN')
		try:
		    cov_1_logdet = 2*np.sum(np.log(np.linalg.cholesky(cov1).diagonal()))
		    issing1=1
		except np.linalg.LinAlgError as err:
			#TODO:
		    #self.nonpdjs.add(1)
		    #print("ERROR: NON POSITIVE DEFINITE MATRIX 1\n\n\n") 
		    return div    
		#print(cov_1_logdet)
		mean2 = np.empty([d, 1])
		mean2 = vec2[0:d]
		cov2 = np.empty([d,d])
		cov2 = vec2[d:].reshape(d, d)
		try:
		    cov_2_logdet = 2*np.sum(np.log(np.linalg.cholesky(cov2).diagonal()))
		    issing2=1
		except np.linalg.LinAlgError as err:
			#TODO:
		    #self.nonpdjs.add(1)
		    #print("ERROR: NON POSITIVE DEFINITE MATRIX 2\n\n\n") 
		    return div
		#print(cov_2_logdet)
		#==============================================
		if (issing1==1) and (issing2==1):
		    mean_m = 0.5 * mean1 +  0.5 * mean2
		    cov_m = 0.5 * (cov1 + np.outer(mean1, mean1)) + 0.5 * (cov2 + np.outer(mean2, mean2)) - np.outer(mean_m, mean_m)
		    cov_m_logdet = 2*np.sum(np.log(np.linalg.cholesky(cov_m).diagonal()))
		    #print(cov_m_logdet)
		    try:        
		        div = 0.5 * cov_m_logdet - 0.25 * cov_1_logdet - 0.25 * cov_2_logdet
		    except np.linalg.LinAlgError as err:
        		#TODO:
				#self.nonpdjs.add(1)
		        #print("ERROR: NON POSITIVE DEFINITE MATRIX M\n\n\n") 
		        return div
		    #print("JENSEN_SHANNON_DIVERGENCE")   
		if np.isnan(div):
		    div = np.inf
			#TODO:
		    #self.nanjs.add(1)
		    #div = None
		    pass
		if div <= 0:
		    div = 0
			#TODO:
		    #self.negjs.add(1)
		    pass
		#print(div)
		return div

	@staticmethod
	#get 13 mean and 13x13 cov as vectors
	def symmetric_kullback_leibler(vec1, vec2):
		d = 13
		mean1 = np.empty([d, 1])
		mean1 = vec1[0:d]
		cov1 = np.empty([d,d])
		cov1 = vec1[d:].reshape(d, d)
		mean2 = np.empty([d, 1])
		mean2 = vec2[0:d]
		cov2 = np.empty([d,d])
		cov2 = vec2[d:].reshape(d, d)
		div = np.inf
		try:
		    g_chol = np.linalg.cholesky(cov1)
		    g_ui   = np.linalg.solve(g_chol,np.eye(d))
		    icov1  = np.matmul(np.transpose(g_ui), g_ui)
		    isinv1=1
		except np.linalg.LinAlgError as err:
		    isinv1=0
		try:
		    g_chol = np.linalg.cholesky(cov2)
		    g_ui   = np.linalg.solve(g_chol,np.eye(d))
		    icov2  = np.matmul(np.transpose(g_ui), g_ui)
		    isinv2=1
		except np.linalg.LinAlgError as err:
		    isinv2=0
		#================================
		if (isinv1==1) and (isinv2==1):
		    temp_a = np.trace(np.matmul(cov1, icov2)) 
		    #temp_a = traceprod(cov1, icov2) 
		    #print(temp_a)
		    temp_b = np.trace(np.matmul(cov2, icov1))
		    #temp_b = traceprod(cov2, icov1)
		    #print(temp_b)
		    temp_c = np.trace(np.matmul((icov1 + icov2), np.outer((mean1 - mean2), (mean1 - mean2))))
		    #print(temp_c)        
		    div = 0.25 * (temp_a + temp_b + temp_c - 2*d)
		else: 
		    div = np.inf
			#TODO:
		    #self.noninskl.add(1)
		    #print("ERROR: NON INVERTIBLE SINGULAR COVARIANCE MATRIX \n\n\n")    
		if div <= 0:
		    #print("Temp_a: " + temp_a + "\n Temp_b: " + temp_b + "\n Temp_c: " + temp_c)
		    div = 0
			#TODO:
		    #self.negskl.add(1)
		if np.isnan(div):
		    div = np.inf
			#TODO:
		    #self.nanskl.add(1)
		    #div = None
		#print(div)
		return div

	@staticmethod
	#get 13 mean and 13x13 cov + var as vectors
	def get_euclidean_mfcc(vec1, vec2):
		mean1 = np.empty([13, 1])
		mean1 = vec1[0:13]
		cov1 = np.empty([13,13])
		cov1 = vec1[13:].reshape(13, 13)        
		mean2 = np.empty([13, 1])
		mean2 = vec2[0:13]
		cov2 = np.empty([13,13])
		cov2 = vec2[13:].reshape(13, 13)
		iu1 = np.triu_indices(13)
		#You need to pass the arrays as an iterable (a tuple or list), thus the correct syntax is np.concatenate((,),axis=None)
		div = distance.euclidean(np.concatenate((mean1, cov1[iu1]),axis=None), np.concatenate((mean2, cov2[iu1]),axis=None))
		return div

	@staticmethod
	#even faster than numpy version
	def naive_levenshtein(self, seq1, seq2):
		result = edlib.align(seq1, seq2)
		return(result["editDistance"])
	
	def get_neighbors_mfcc_skl(self, song, featureDF):
		comparator_value = song[0]["mfccSkl"]
		distance_udf = F.udf(lambda x: float(recommender.symmetric_kullback_leibler(x, comparator_value)), DoubleType())
		result = featureDF.withColumn('distances_skl', distance_udf(F.col('mfccSkl'))).select("id", "distances_skl")
		#thresholding 
		result = result.filter(result.distances_skl <= 10000)  
		result = result.filter(result.distances_skl != np.inf)        
		return result

	def get_neighbors_mfcc_js(self, song, featureDF):
		comparator_value = song[0]["mfccSkl"]
		distance_udf = F.udf(lambda x: float(recommender.jensen_shannon(x, comparator_value)), DoubleType())
		result = featureDF.withColumn('distances_js', distance_udf(F.col('mfccSkl'))).select("id", "distances_js")
		result = result.filter(result.distances_js != np.inf)    
		return result

	def get_neighbors_rp_euclidean(self, song, featureDF):
		comparator_value = song[0]["rp"]
		distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
		result = featureDF.withColumn('distances_rp', distance_udf(F.col('rp'))).select("id", "distances_rp")
		return result

	def get_neighbors_rh_euclidean(self, song, featureDF):
		comparator_value = song[0]["rh"]
		distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
		result = featureDF.withColumn('distances_rh', distance_udf(F.col('rh'))).select("id", "distances_rh")
		return result

	def get_neighbors_bh_euclidean(self, song, featureDF):
		comparator_value = song[0]["bh"]
		distance_udf = F.udf(lambda x: float(distance.euclidean(x, comparator_value)), FloatType())
		result = featureDF.withColumn('distances_bh', distance_udf(F.col('bh'))).select("id", "bpm", "distances_bh")
		return result

	def get_neighbors_mfcc_euclidean(self, song, featureDF):
		comparator_value = song[0]["mfccSkl"]
		distance_udf = F.udf(lambda x: float(recommender.get_euclidean_mfcc(x, comparator_value)), FloatType())
		result = featureDF.withColumn('distances_mfcc', distance_udf(F.col('mfccSkl'))).select("id", "distances_mfcc")
		return result

	def get_neighbors_notes(self, song, featureDF):
		comparator_value = song[0]["notes"]
		df_merged = featureDF.withColumn("compare", lit(comparator_value))
		df_levenshtein = df_merged.withColumn("distances_levenshtein", levenshtein(col("notes"), col("compare")))
		#df_levenshtein.sort(col("word1_word2_levenshtein").asc()).show()    
		result = df_levenshtein.select("id", "key", "scale", "distances_levenshtein")
		return result

	def get_neighbors_chroma_corr_valid(self, song, featureDF):
		comparator_value = song[0]["chroma"]
		distance_udf = F.udf(lambda x: float(recommender.chroma_cross_correlate_valid(x, comparator_value)), DoubleType())
		result = featureDF.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
		return result

	def perform_scaling(self, unscaled_df):
		aggregated = unscaled_df.agg(F.min(unscaled_df.distances_bh),F.max(unscaled_df.distances_bh),F.mean(unscaled_df.distances_bh),F.stddev(unscaled_df.distances_bh),
		    F.min(unscaled_df.distances_rh),F.max(unscaled_df.distances_rh),F.mean(unscaled_df.distances_rh),F.stddev(unscaled_df.distances_rh),
		    F.min(unscaled_df.distances_rp),F.max(unscaled_df.distances_rp),F.mean(unscaled_df.distances_rp),F.stddev(unscaled_df.distances_rp),
		    F.min(unscaled_df.distances_corr),F.max(unscaled_df.distances_corr),F.mean(unscaled_df.distances_corr),F.stddev(unscaled_df.distances_corr),
		    F.min(unscaled_df.distances_levenshtein),F.max(unscaled_df.distances_levenshtein),F.mean(unscaled_df.distances_levenshtein),F.stddev(unscaled_df.distances_levenshtein),
		    F.min(unscaled_df.distances_mfcc),F.max(unscaled_df.distances_mfcc),F.mean(unscaled_df.distances_mfcc),F.stddev(unscaled_df.distances_mfcc),
		    F.min(unscaled_df.distances_js),F.max(unscaled_df.distances_js),F.mean(unscaled_df.distances_js),F.stddev(unscaled_df.distances_js),
		    F.min(unscaled_df.distances_skl),F.max(unscaled_df.distances_skl),F.mean(unscaled_df.distances_skl),F.stddev(unscaled_df.distances_skl)).persist()
		##############################
		#var_val = aggregated.collect()[0]["stddev_samp(distances_bh)"]
		#mean_val = aggregated.collect()[0]["avg(distances_bh)"]
		##############################
		max_val = aggregated.collect()[0]["max(distances_rp)"]
		min_val = aggregated.collect()[0]["min(distances_rp)"]
		result = unscaled_df.withColumn('scaled_rp', (unscaled_df.distances_rp-min_val)/(max_val-min_val))
		##############################    
		max_val = aggregated.collect()[0]["max(distances_rh)"]
		min_val = aggregated.collect()[0]["min(distances_rh)"]
		result = result.withColumn('scaled_rh', (unscaled_df.distances_rh-min_val)/(max_val-min_val))
		##############################
		max_val = aggregated.collect()[0]["max(distances_bh)"]
		min_val = aggregated.collect()[0]["min(distances_bh)"]
		result = result.withColumn('scaled_bh', (unscaled_df.distances_bh-min_val)/(max_val-min_val))
		##############################
		max_val = aggregated.collect()[0]["max(distances_levenshtein)"]
		min_val = aggregated.collect()[0]["min(distances_levenshtein)"]
		result = result.withColumn('scaled_notes', (unscaled_df.distances_levenshtein-min_val)/(max_val-min_val))
		##############################
		max_val = aggregated.collect()[0]["max(distances_corr)"]
		min_val = aggregated.collect()[0]["min(distances_corr)"]
		result = result.withColumn('scaled_chroma', (1 - (unscaled_df.distances_corr-min_val)/(max_val-min_val)))
		##############################
		max_val = aggregated.collect()[0]["max(distances_skl)"]
		min_val = aggregated.collect()[0]["min(distances_skl)"]
		result = result.withColumn('scaled_skl', (unscaled_df.distances_skl-min_val)/(max_val-min_val))
		##############################
		max_val = aggregated.collect()[0]["max(distances_js)"]
		min_val = aggregated.collect()[0]["min(distances_js)"]
		result = result.withColumn('scaled_js', (unscaled_df.distances_js-min_val)/(max_val-min_val))
		##############################
		max_val = aggregated.collect()[0]["max(distances_mfcc)"]
		min_val = aggregated.collect()[0]["min(distances_mfcc)"]
		result = result.withColumn('scaled_mfcc', (unscaled_df.distances_mfcc-min_val)/(max_val-min_val)).select("id", "key", "scale", "bpm", "scaled_rp", "scaled_rh", "scaled_bh", "scaled_notes", "scaled_chroma", "scaled_skl", "scaled_js", "scaled_mfcc")
		##############################
		aggregated.unpersist()
		return result

	def get_nearest_neighbors(self, song, outname, fullFeatureDF):
		tic1 = int(round(time.time() * 1000))
		song = fullFeatureDF.filter(fullFeatureDF.id == song).collect()#
		tac1 = int(round(time.time() * 1000))
		self.time_dict['COMPARATOR: ']= tac1 - tic1

		tic1 = int(round(time.time() * 1000))
		neighbors_rp_euclidean = self.get_neighbors_rp_euclidean(song, fullFeatureDF).persist()
		#print(neighbors_rp_euclidean.count())
		tac1 = int(round(time.time() * 1000))
		self.time_dict['RP: ']= tac1 - tic1
		tic1 = int(round(time.time() * 1000))
		neighbors_rh_euclidean = self.get_neighbors_rh_euclidean(song, fullFeatureDF).persist()   
		#print(neighbors_rh_euclidean.count()) 
		tac1 = int(round(time.time() * 1000))
		self.time_dict['RH: ']= tac1 - tic1
		tic1 = int(round(time.time() * 1000))
		neighbors_notes = self.get_neighbors_notes(song, fullFeatureDF).persist()
		#print(neighbors_notes.count())
		tac1 = int(round(time.time() * 1000))
		self.time_dict['NOTE: ']= tac1 - tic1
		tic1 = int(round(time.time() * 1000))
		neighbors_mfcc_eucl = self.get_neighbors_mfcc_euclidean(song, fullFeatureDF).persist()
		#print(neighbors_mfcc_eucl.count())
		tac1 = int(round(time.time() * 1000))
		self.time_dict['MFCC: ']= tac1 - tic1
		tic1 = int(round(time.time() * 1000))
		neighbors_bh_euclidean = self.get_neighbors_bh_euclidean(song, fullFeatureDF).persist()
		#print(neighbors_bh_euclidean.count())
		tac1 = int(round(time.time() * 1000))
		self.time_dict['BH: ']= tac1 - tic1
		tic1 = int(round(time.time() * 1000))
		neighbors_mfcc_skl = self.get_neighbors_mfcc_skl(song, fullFeatureDF).persist()
		#print(neighbors_mfcc_skl.count())
		tac1 = int(round(time.time() * 1000))
		self.time_dict['SKL: ']= tac1 - tic1
		tic1 = int(round(time.time() * 1000))
		neighbors_mfcc_js = self.get_neighbors_mfcc_js(song, fullFeatureDF).persist()
		#print(neighbors_mfcc_js.count())
		tac1 = int(round(time.time() * 1000))
		self.time_dict['JS: ']= tac1 - tic1
		tic1 = int(round(time.time() * 1000))
		neighbors_chroma = self.get_neighbors_chroma_corr_valid(song, fullFeatureDF).persist()
		#print(neighbors_chroma.count())
		tac1 = int(round(time.time() * 1000))
		self.time_dict['CHROMA: ']= tac1 - tic1

		tic1 = int(round(time.time() * 1000))
		mergedSim = neighbors_mfcc_eucl.join(neighbors_rp_euclidean, on=['id'], how='inner').persist()
		mergedSim = mergedSim.join(neighbors_bh_euclidean, on=['id'], how='inner').persist()
		mergedSim = mergedSim.join(neighbors_rh_euclidean, on=['id'], how='inner').persist()
		mergedSim = mergedSim.join(neighbors_notes, on=['id'], how='inner').persist()
		mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner').persist()
		mergedSim = mergedSim.join(neighbors_mfcc_skl, on=['id'], how='inner').persist()
		mergedSim = mergedSim.join(neighbors_mfcc_js, on=['id'], how='inner').dropDuplicates().persist()
		#print(mergedSim.count())
		tac1 = int(round(time.time() * 1000))
		self.time_dict['JOIN: ']= tac1 - tic1

		tic1 = int(round(time.time() * 1000))
		scaledSim = self.perform_scaling(mergedSim).persist()
		tac1 = int(round(time.time() * 1000))
		self.time_dict['SCALE: ']= tac1 - tic1

		tic1 = int(round(time.time() * 1000))
		#scaledSim = scaledSim.withColumn('aggregated', (scaledSim.scaled_notes + scaledSim.scaled_rp + scaledSim.scaled_mfcc) / 3)
		scaledSim = scaledSim.withColumn('aggregated', (scaledSim.scaled_notes + scaledSim.scaled_mfcc + scaledSim.scaled_chroma + scaledSim.scaled_bh + scaledSim.scaled_rp + scaledSim.scaled_skl + scaledSim.scaled_js + scaledSim.scaled_rh) / 8)
		scaledSim = scaledSim.orderBy('aggregated', ascending=True).persist()#.rdd.flatMap(list).collect()
		scaledSim.show()
		#scaledSim.toPandas().to_csv(outname, encoding='utf-8')

		neighbors_rp_euclidean.unpersist()
		neighbors_rh_euclidean.unpersist()    
		neighbors_notes.unpersist()
		neighbors_mfcc_eucl.unpersist()
		neighbors_bh_euclidean.unpersist()
		neighbors_mfcc_skl.unpersist()
		neighbors_mfcc_js.unpersist()
		neighbors_chroma.unpersist()
		mergedSim.unpersist()
		scaledSim.unpersist()

		tac1 = int(round(time.time() * 1000))
		self.time_dict['AGG: ']= tac1 - tic1
		return scaledSim


	def recommend(self, song1, filename):
		#song1 = song1.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')
		#song2 = song2.replace(";","").replace(".","").replace(",","").replace(" ","")#.encode('utf-8','replace')
		#filename = song1 + ".csv"

		tic1 = int(round(time.time() * 1000))
		res1 = self.get_nearest_neighbors(song1, filename, self.fullFeatureDF).persist()
		tac1 = int(round(time.time() * 1000))
		self.time_dict['MERGED_FULL_SONG1: ']= tac1 - tic1

		endtime = int(round(time.time() * 1000))
		self.time_dict['MERGED_TOTAL: ']= endtime - self.starttime

		tic1 = int(round(time.time() * 1000))
		res1.toPandas().to_csv(filename, encoding='utf-8')
		res1.unpersist()
		tac1 = int(round(time.time() * 1000))
		self.time_dict['CSV1: ']= tac1 - tic1

		print(self.time_dict)
		print("\n\n")
	
		#TODO:
		#debug_dict = {}		
		#debug_dict['Negative JS: ']= self.negjs.value
		#debug_dict['Nan JS: ']= self.nanjs.value
		#debug_dict['Non Positive Definite JS: ']= self.nonpdjs.value
		#debug_dict['Negative SKL: ']= self.negskl.value
		#debug_dict['Nan SKL: ']= self.nanskl.value
		#debug_dict['Non Invertible SKL: ']= self.noninskl.value
		#print(debug_dict)

		self.fullFeatureDF.unpersist()

	
