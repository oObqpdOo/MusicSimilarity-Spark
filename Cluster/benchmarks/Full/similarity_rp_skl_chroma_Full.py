import builtins
import pyspark
import pyspark.ml.feature
import pyspark.mllib.linalg
import pyspark.ml.param
import pyspark.sql.functions
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, DoubleType, StringType, Row
from pyspark.sql.functions import * # col, array, lit, udf, min, max, round # lit is used for applying one scalar to every row in a whole column when using withColumn and creating a new column
from pyspark.sql.functions import udf
from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler

from scipy.spatial import distance
from pyspark.mllib.linalg import Vectors
from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler

import numpy as np
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
import sys

#==============================================================

#either do_rh or do_rp (or both) has to be set by default!
do_rh = 0
do_rp = 1
do_bh = 0
do_chroma = 1
do_notes = 0
do_mfcc = 0
do_skl = 1
do_js = 0

weight_rh = 1
weight_rp = 1
weight_bh = 1
weight_chroma = 1
weight_notes = 1
weight_mfcc = 1
weight_skl = 1
weight_js = 1

#==============================================================

executor_s = "100"
parts_s = "100"
parts = 400

conf = SparkConf().setAppName("MergeDatasets").set("yarn.nodemanager.resource.detect-hardware-capabilities" , "True") \
                                              .set("yarn.nodemanager.resource.memory-mb", "196608") \
					      .set("yarn.scheduler.maximum-allocation-vcores", "36") \
                                              .set("spark.executor.memory", "16g") \
                                              .set("spark.driver.memory", "16g") \
                                              .set("spark.driver.cores", "4") \
                                              .set("spark.executor.cores", "4") \
					      					  .set("spark.dynamicAllocation.enabled", "True") \
                                              .set("spark.dynamicAllocation.initialExecutors", executor_s) \
                                              .set("spark.dynamicAllocation.executorIdleTimeout", "30s") \
                                              .set("spark.dynamicAllocation.minExecutors", executor_s) \
                                              .set("spark.dynamicAllocation.maxExecutors", executor_s) \
                                              .set("spark.executor.instances", executor_s) \
                                              .set("spark.default.parallelism", parts_s) #\
                                            #.set("spark.driver.memoryOverhead", "1024") \
                                            #.set("spark.executor.memoryOverhead", "1024") \
                                            #.set("yarn.nodemanager.resource.memory-mb", "196608") \
                                            #.set("yarn.nodemanager.vmem-check-enabled", "false") \
                                            #.set("spark.yarn.executor.memoryOverhead", "8192") \
                                            #.set("spark.shuffle.service.enabled", "True") \ 
					    #.set("spark.dynamicAllocation.shuffleTracking.enabled", "True") \ 



# Fehleranfällig: https://spark.apache.org/docs/3.5.2/job-scheduling.html#configuration-and-setup
# Create a SparkSession object
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext
sqlContext= SQLContext(sc)

sc.setLogLevel("ERROR");

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

time_dict = {}


debug_dict = {}
negjs = sc.accumulator(0)
nanjs = sc.accumulator(0)
nonpdjs = sc.accumulator(0)
negskl = sc.accumulator(0)
nanskl = sc.accumulator(0)
noninskl = sc.accumulator(0)

total1 = int(time.time() * 1000)

#===========================================================================================================================================================
#===========================================================================================================================================================

def chroma_cross_correlate_valid(chroma1_par, chroma2_par):
    chroma1_par = np.array(chroma1_par)
    chroma2_par = np.array(chroma2_par)
    length1 = int(chroma1_par.size/12)
    chroma1 = np.empty([12, length1])
    length2 = int(chroma2_par.size/12)
    chroma2 = np.empty([12, length2])
    if(length1 > length2):
        chroma1 = np.array(chroma1_par).reshape(12, length1)
        chroma2 = np.array(chroma2_par).reshape(12, length2)
    else:
        chroma2 = np.array(chroma1_par).reshape(12, length1)
        chroma1 = np.array(chroma2_par).reshape(12, length2)      
    #full
    #correlation = np.zeros([length1 + length2 - 1])
    #valid
    #correlation = np.zeros([builtins.max(length1, length2) - min(length1, length2) + 1])
    #same
    correlation = np.zeros([builtins.max(length1, length2)])
    for i in range(12):
        correlation = correlation + np.correlate(chroma1[i], chroma2[i], "same")    
    #remove offset to get rid of initial filter peak(highpass of jump from 0-20)
    correlation = correlation - correlation[0]
    sos = butter(1, 0.1, 'high', analog=False, output='sos')
    correlation = sosfilt(sos, correlation)[:]
    return np.max(correlation)


def jensen_shannon(vec1, vec2):
    d = 13
    mean1 = np.empty([d, 1])
    mean1 = np.array(vec1[0:d])
    cov1 = np.empty([d,13])
    cov1 = np.array(vec1[d:]).reshape(d, d)
    div = np.inf
    #div = float('NaN')
    try:
        cov_1_logdet = 2*np.sum(np.log(np.linalg.cholesky(cov1).diagonal()))
        issing1=1
    except np.linalg.LinAlgError as err:
        nonpdjs.add(1)
        #print("ERROR: NON POSITIVE DEFINITE MATRIX 1\n\n\n") 
        return div    
    #print(cov_1_logdet)
    mean2 = np.empty([d, 1])
    mean2 =  np.array(vec2[0:d])
    cov2 = np.empty([d,d])
    cov2 =  np.array(vec2[d:]).reshape(d, d)
    try:
        cov_2_logdet = 2*np.sum(np.log(np.linalg.cholesky(cov2).diagonal()))
        issing2=1
    except np.linalg.LinAlgError as err:
        nonpdjs.add(1)
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
            nonpdjs.add(1)
            #print("ERROR: NON POSITIVE DEFINITE MATRIX M\n\n\n") 
            return div
        #print("JENSEN_SHANNON_DIVERGENCE")   
    if np.isnan(div):
        div = np.inf
        nanjs.add(1)
        #div = None
        pass
    if div <= 0:
        div = 0
        negjs.add(1)
        pass
    #print(div)
    return div

#get 13 mean and 13x13 cov as vectors
def symmetric_kullback_leibler(vec1, vec2):
    d = 13
    mean1 = np.empty([d, 1])
    mean1 = np.array(vec1[0:d])
    cov1 = np.empty([d,d])
    cov1 = np.array(vec1[d:]).reshape(d, d)
    mean2 = np.empty([d, 1])
    mean2 = np.array(vec2[0:d])
    cov2 = np.empty([d,d])
    cov2 = np.array(vec2[d:]).reshape(d, d)
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
        noninskl.add(1)
        #print("ERROR: NON INVERTIBLE SINGULAR COVARIANCE MATRIX \n\n\n")    
    if div <= 0:
        #print("Temp_a: " + temp_a + "\n Temp_b: " + temp_b + "\n Temp_c: " + temp_c)
        div = 0
        negskl.add(1)
    if np.isnan(div):
        div = np.inf
        nanskl.add(1)
        #div = None
    #print(div)
    return div

#get 13 mean and 13x13 cov + var as vectors
def get_euclidean_mfcc(vec1, vec2):
    print(vec1)
    print(vec2)
    mean1 = np.array(vec1[1][0:13])
    cov1 = np.array(vec1[1][13:]).reshape(13, 13)        
    mean2 = np.array(vec2[1][0:13])
    cov2 = np.array(vec2[1][13:]).reshape(13, 13)
    iu1 = np.triu_indices(13)
    #You need to pass the arrays as an iterable (a tuple or list), thus the correct syntax is np.concatenate((,),axis=None)
    div = distance.euclidean(np.concatenate((mean1, cov1[iu1]),axis=None), np.concatenate((mean2, cov2[iu1]),axis=None))
    return div

#===========================================================================================================================================================
#===========================================================================================================================================================

def get_neighbors_rh_euclidean(song, featureDF):
    comparator_value = song[0]["rh"]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x[1], comparator_value[1])), FloatType())
    result = featureDF.withColumn('distances_rh', distance_udf(F.col('rh'))).select("id", "distances_rh")#,"artist", "track", "album", "preview_url")
    return result

def get_neighbors_bh_euclidean(song, featureDF):
    comparator_value = song[0]["bh"]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x[1], comparator_value[1])), FloatType())
    result = featureDF.withColumn('distances_bh', distance_udf(F.col('bh'))).select("id", "bpm", "distances_bh")
    return result

def get_neighbors_rp_euclidean(song, featureDF):
    comparator_value = song[0]["rp"]
    distance_udf = F.udf(lambda x: float(distance.euclidean(x[1], comparator_value[1])), FloatType())
    result = featureDF.withColumn('distances_rp', distance_udf(F.col('rp'))).select("id", "distances_rp")# ,"artist", "track", "album", "preview_url")
    return result

def get_neighbors_notes(song, featureDF):
    comparator_value = song[0]["notes"]
    df_merged = featureDF.withColumn("compare", lit(comparator_value))
    df_levenshtein = df_merged.withColumn("distances_levenshtein", levenshtein(col("notes"), col("compare")))
    result = df_levenshtein.select("id", "key", "scale", "distances_levenshtein")
    return result

def get_neighbors_chroma_corr_valid(song, featureDF):
    comparator_value = song[0]["chroma"]
    distance_udf = F.udf(lambda x: float(chroma_cross_correlate_valid(x[1], comparator_value[1])), DoubleType())
    result = featureDF.withColumn('distances_corr', distance_udf(F.col('chroma'))).select("id", "distances_corr")
    return result

def get_neighbors_mfcc_skl(song, featureDF):
    comparator_value = song[0]["mfccSkl"]
    distance_udf = F.udf(lambda x: float(symmetric_kullback_leibler(x[1], comparator_value[1])), DoubleType())
    result = featureDF.withColumn('distances_skl', distance_udf(F.col('mfccSkl'))).select("id", "distances_skl")
    #thresholding 
    result = result.filter(result.distances_skl <= 10000)  
    result = result.filter(result.distances_skl != np.inf)        
    return result

def get_neighbors_mfcc_js(song, featureDF):
    comparator_value = song[0]["mfccSkl"]
    distance_udf = F.udf(lambda x: float(jensen_shannon(x[1], comparator_value[1])), DoubleType())
    result = featureDF.withColumn('distances_js', distance_udf(F.col('mfccSkl'))).select("id", "distances_js")
    result = result.filter(result.distances_js != np.inf)    
    return result

def get_neighbors_mfcc_euclidean(song, featureDF):
    comparator_value = song[0]["mfccSkl"]
    distance_udf = F.udf(lambda x: float(get_euclidean_mfcc(x, comparator_value)), FloatType())
    result = featureDF.withColumn('distances_mfcc', distance_udf(F.col('mfccSkl'))).select("id", "distances_mfcc")
    return result

#==============================================================

def perform_scaling(unscaled_df):
    #aggregated = unscaled_df.agg(F.min(unscaled_df.distances_bh),F.max(unscaled_df.distances_bh), \
    #    F.min(unscaled_df.distances_rh),F.max(unscaled_df.distances_rh), \
    #    F.min(unscaled_df.distances_rp),F.max(unscaled_df.distances_rp), \
    #    F.min(unscaled_df.distances_corr),F.max(unscaled_df.distances_corr), \
    #    F.min(unscaled_df.distances_levenshtein),F.max(unscaled_df.distances_levenshtein), \
    #    F.min(unscaled_df.distances_mfcc),F.max(unscaled_df.distances_mfcc), \
    #    F.min(unscaled_df.distances_js),F.max(unscaled_df.distances_js), \
    #    F.min(unscaled_df.distances_skl),F.max(unscaled_df.distances_skl)).persist()
	##############################
	#CHATGPT sagt: 
	# Basis: Leere Aggregationsliste
	aggregations = []
	# Hilfsfunktion für Min/Max-Aggregate eines Feldes
	def add_min_max(colname, do_flag):
		if do_flag == 1:
		    aggregations.append(F.min(colname))
		    aggregations.append(F.max(colname))
	# Bedingungen prüfen und hinzufügen
	add_min_max("distances_bh", do_bh)
	add_min_max("distances_rh", do_rh)
	add_min_max("distances_rp", do_rp)
	add_min_max("distances_corr", do_chroma)
	add_min_max("distances_levenshtein", do_notes)
	add_min_max("distances_mfcc", do_mfcc)
	add_min_max("distances_js", do_js)
	add_min_max("distances_skl", do_skl)
	# Aggregation durchführen, wenn etwas drin ist
	if aggregations:
		aggregated = unscaled_df.agg(*aggregations).persist()
	else:
		aggregated = None  # oder raise Exception("Keine Aggregationen aktiviert")
	#CHATGPT ENDE 
	##############################
    ##############################
    #var_val = aggregated.collect()[0]["stddev_samp(distances_bh)"]
    #mean_val = aggregated.collect()[0]["avg(distances_bh)"]
    ##############################    
	if do_rh == 1: 
		max_val = aggregated.collect()[0]["max(distances_rh)"]
		min_val = aggregated.collect()[0]["min(distances_rh)"]
		result = unscaled_df.withColumn('scaled_rh', ((unscaled_df.distances_rh-min_val)/(max_val-min_val))*weight_rh)
    ##############################
	elif do_rp == 1 and do_rh == 0: 
		max_val = aggregated.collect()[0]["max(distances_rp)"]
		min_val = aggregated.collect()[0]["min(distances_rp)"]
		result = unscaled_df.withColumn('scaled_rp', ((unscaled_df.distances_rp-min_val)/(max_val-min_val))*weight_rp)
    ##############################
	if do_rp == 1 and do_rh == 1:
		max_val = aggregated.collect()[0]["max(distances_rp)"]
		min_val = aggregated.collect()[0]["min(distances_rp)"]
		result = result.withColumn('scaled_rp', ((unscaled_df.distances_rp-min_val)/(max_val-min_val))*weight_rp)
	##############################
	if do_bh == 1: 
		max_val = aggregated.collect()[0]["max(distances_bh)"]
		min_val = aggregated.collect()[0]["min(distances_bh)"]
		result = result.withColumn('scaled_bh', ((unscaled_df.distances_bh-min_val)/(max_val-min_val))*weight_bh)
    ##############################
	if do_notes == 1: 
		max_val = aggregated.collect()[0]["max(distances_levenshtein)"]
		min_val = aggregated.collect()[0]["min(distances_levenshtein)"]
		result = result.withColumn('scaled_notes', ((unscaled_df.distances_levenshtein-min_val)/(max_val-min_val))*weight_notes)
    ##############################
	if do_chroma == 1: 
		max_val = aggregated.collect()[0]["max(distances_corr)"]
		min_val = aggregated.collect()[0]["min(distances_corr)"]
		result = result.withColumn('scaled_chroma', ((1 - (unscaled_df.distances_corr-min_val)/(max_val-min_val)))*weight_chroma)
    ##############################
	if do_skl == 1:
		max_val = aggregated.collect()[0]["max(distances_skl)"]
		min_val = aggregated.collect()[0]["min(distances_skl)"]
		result = result.withColumn('scaled_skl', ((unscaled_df.distances_skl-min_val)/(max_val-min_val))*weight_skl)
    ##############################
	if do_js == 1:	
		max_val = aggregated.collect()[0]["max(distances_js)"]
		min_val = aggregated.collect()[0]["min(distances_js)"]
		result = result.withColumn('scaled_js', ((unscaled_df.distances_js-min_val)/(max_val-min_val))*weight_js)
    ##############################
	if do_mfcc == 1:
		max_val = aggregated.collect()[0]["max(distances_mfcc)"]
		min_val = aggregated.collect()[0]["min(distances_mfcc)"]
		result = result.withColumn('scaled_mfcc', ((unscaled_df.distances_mfcc-min_val)/(max_val-min_val))*weight_mfcc)
	##############################
	##############################
	#CHATGPT sagt: 
	optional_columns = {
		"scaled_rh": do_rh,
		"scaled_notes": do_notes,
		"scaled_rp": do_rp,
		"scaled_bh": do_bh,
		"scaled_chroma": do_chroma,
		"scaled_skl": do_skl,
		"scaled_js": do_js,
		"scaled_mfcc": do_mfcc,
 		"artist": do_rp, 
		"track": do_rp, 
		"album": do_rp, 
		"preview_url": do_rp,
		"bpm" : do_bh,
		"key" : do_notes,
		"scale" : do_notes
	}
	# Kombinieren der Spalten
	columns = ["id"] + [col for col, flag in optional_columns.items() if flag == 1]
	# Anwenden
	result = result.select(*columns)#.persist()
	#result = result.select("id", "key", "scale", "bpm", "scaled_rp", "scaled_rh", "scaled_bh", "scaled_notes", "scaled_chroma", "scaled_skl", "scaled_js", "scaled_mfcc").persist()
	#CHATGPT ENDE 
	##############################
	aggregated.unpersist()
	return result

#==============================================================

def get_nearest_neighbors(song, outname):
	tic1 = int(time.time() * 1000)
	#GEMINI says: limit(1) lets Spark stop as soon as it got the first match! ID should be unique
	#song = fullFeatureDF.filter(fullFeatureDF.id == song).limit(1).collect()#
	song = fullFeatureDF.filter(fullFeatureDF.id == song).collect()#
	tac1 = int(time.time() * 1000)
	time_dict['COMPARATOR: ']= tac1 - tic1
	if do_rp == 1: 
		tic1 = int(time.time() * 1000)
		neighbors_rp_euclidean = get_neighbors_rp_euclidean(song, fullFeatureDF)#.persist()
		#print(neighbors_rp_euclidean.count())
		tac1 = int(time.time() * 1000)
		time_dict['RP: ']= tac1 - tic1
	if do_rh == 1: 
		tic1 = int(time.time() * 1000)
		neighbors_rh_euclidean = get_neighbors_rh_euclidean(song, fullFeatureDF)#.persist()   
		#print(neighbors_rh_euclidean.count()) 
		tac1 = int(time.time() * 1000)
		time_dict['RH: ']= tac1 - tic1
	if do_notes == 1:
		tic1 = int(time.time() * 1000)
		neighbors_notes = get_neighbors_notes(song, fullFeatureDF)#.persist()
		#print(neighbors_notes.count())
		tac1 = int(time.time() * 1000)
		time_dict['NOTE: ']= tac1 - tic1
	if do_mfcc == 1: 
		tic1 = int(time.time() * 1000)
		neighbors_mfcc_eucl = get_neighbors_mfcc_euclidean(song, fullFeatureDF)#.persist()
		#print(neighbors_mfcc_eucl.count())
		tac1 = int(time.time() * 1000)
		time_dict['MFCC: ']= tac1 - tic1
	if do_bh == 1: 
		tic1 = int(time.time() * 1000)
		neighbors_bh_euclidean = get_neighbors_bh_euclidean(song, fullFeatureDF)#.persist()
		#print(neighbors_bh_euclidean.count())
		tac1 = int(time.time() * 1000)
		time_dict['BH: ']= tac1 - tic1
	if do_skl == 1: 
		tic1 = int(time.time() * 1000)
		neighbors_mfcc_skl = get_neighbors_mfcc_skl(song, fullFeatureDF)#.persist()
		#print(neighbors_mfcc_skl.count())
		tac1 = int(time.time() * 1000)
		time_dict['SKL: ']= tac1 - tic1
	if do_js == 1: 
		tic1 = int(time.time() * 1000)
		neighbors_mfcc_js = get_neighbors_mfcc_js(song, fullFeatureDF)#.persist()
		#print(neighbors_mfcc_js.count())
		tac1 = int(time.time() * 1000)
		time_dict['JS: ']= tac1 - tic1
	if do_chroma == 1: 
		tic1 = int(time.time() * 1000)
		neighbors_chroma = get_neighbors_chroma_corr_valid(song, fullFeatureDF)#.persist()
		#print(neighbors_chroma.count())
		tac1 = int(time.time() * 1000)
		time_dict['CHROMA: ']= tac1 - tic1
	tic1 = int(time.time() * 1000)
	if do_rh == 0 and do_rp == 1: 
		mergedSim = neighbors_rp_euclidean#.persist()
	if do_rh == 1 and do_rp == 0: 
		mergedSim = neighbors_rh_euclidean#.persist()
	if do_rh == 1 and do_rp == 1:
		mergedSim = neighbors_rh_euclidean#.persist()
		mergedSim = mergedSim.join(neighbors_rp_euclidean, on=['id'], how='inner')#.persist()
	if do_bh == 1: 
		mergedSim = mergedSim.join(neighbors_bh_euclidean, on=['id'], how='inner')#.persist()
	if do_notes == 1: 
		mergedSim = mergedSim.join(neighbors_notes, on=['id'], how='inner')#.persist()
	if do_chroma == 1: 
		mergedSim = mergedSim.join(neighbors_chroma, on=['id'], how='inner')#.persist()
	if do_mfcc == 1: 
		mergedSim = mergedSim.join(neighbors_mfcc_eucl, on=['id'], how='inner')#.persist()
	if do_skl == 1: 	
		mergedSim = mergedSim.join(neighbors_mfcc_skl, on=['id'], how='inner')#.persist()
	if do_js == 1: 
		mergedSim = mergedSim.join(neighbors_mfcc_js, on=['id'], how='inner')#.persist()
	mergedSim = mergedSim.dropDuplicates().persist()
	#print(mergedSim.count())
	tac1 = int(time.time() * 1000)
	time_dict['JOIN: ']= tac1 - tic1
	tic1 = int(time.time() * 1000)
	scaledSim = perform_scaling(mergedSim).persist()
	tac1 = int(time.time() * 1000)
	time_dict['SCALE: ']= tac1 - tic1
	tic1 = int(time.time() * 1000)
	#scaledSim = scaledSim.withColumn('aggregated', (scaledSim.scaled_notes + scaledSim.scaled_rp + scaledSim.scaled_mfcc) / 3)
	#scaledSim = scaledSim.withColumn('aggregated', (scaledSim.scaled_notes + scaledSim.scaled_mfcc + scaledSim.scaled_chroma + scaledSim.scaled_bh + scaledSim.scaled_rp + scaledSim.scaled_skl + scaledSim.scaled_js + scaledSim.scaled_rh) / 8)
	##############################
	#GEMINI sagt: 
	# Liste der (Spaltenname, Flag)
	from pyspark.sql.functions import when, col
	# Initialisieren der Summe und des Zählers
	sum_expr = lit(0)
	count_expr = lit(0)
	if do_notes == 1:
		sum_expr += scaledSim.scaled_notes
		count_expr += 1
	if do_mfcc == 1:
		sum_expr += scaledSim.scaled_mfcc
		count_expr += 1
	if do_chroma == 1:
		sum_expr += scaledSim.scaled_chroma
		count_expr += 1
	if do_bh == 1:
		sum_expr += scaledSim.scaled_bh
		count_expr += 1
	if do_rp == 1:
		sum_expr += scaledSim.scaled_rp
		count_expr += 1
	if do_skl == 1:
		sum_expr += scaledSim.scaled_skl
		count_expr += 1
	if do_js == 1:
		sum_expr += scaledSim.scaled_js
		count_expr += 1
	if do_rh == 1:
		sum_expr += scaledSim.scaled_rh
		count_expr += 1
	# Verhindern einer Division durch Null, falls keine Flags gesetzt sind
	scaledSim = scaledSim.withColumn(
		'aggregated',
		when(count_expr > 0, sum_expr / count_expr).otherwise(0) # Oder null, je nachdem was sinnvoll ist
	)
	#GEMINI ENDE 
	##############################
	scaledSim = scaledSim.orderBy('aggregated', ascending=True)#.rdd.flatMap(list).collect()
	#scaledSim.show()
	scaledSim.write.csv(outname, header="true", mode="overwrite", sep="\t")
	#if do_rp == 1: 
	#	neighbors_rp_euclidean.unpersist()
	#if do_rh == 1: 
	#	neighbors_rh_euclidean.unpersist()    
	#if do_notes == 1: 
	#	neighbors_notes.unpersist()
	#if do_mfcc == 1: 
	#	neighbors_mfcc_eucl.unpersist()
	#if do_bh == 1: 
	#	neighbors_bh_euclidean.unpersist()
	#if do_skl == 1: 
	#	neighbors_mfcc_skl.unpersist()
	#if do_js == 1: 
	#	neighbors_mfcc_js.unpersist()
	#if do_chroma == 1: 
	#	neighbors_chroma.unpersist()
	mergedSim.unpersist()
	scaledSim.unpersist()
	tac1 = int(time.time() * 1000)
	time_dict['AGG: ']= tac1 - tic1
	return scaledSim

##################################
#ONLY TAKE SUBSET OF 5000
##################################
#fullFeatureDF = spark.read.format("parquet").option("header", True).option("inferSchema", True).load("PreProcessedPart5000.parquet").coalesce(parts).persist()
#fullFeatureDF = spark.read.format("parquet").option("header", True).option("inferSchema", True).load("PreProcessed100000.parquet").coalesce(parts).persist()
#fullFeatureDF = spark.read.format("parquet").option("header", True).option("inferSchema", True).load("PreProcessed500000.parquet").coalesce(parts).persist()
#fullFeatureDF = spark.read.format("parquet").option("header", True).option("inferSchema", True).load("PreProcessed1000000.parquet").coalesce(parts).persist()
fullFeatureDF = spark.read.format("parquet").option("header", True).option("inferSchema", True).load("PreProcessedFull.parquet").coalesce(parts).persist()

#fullFeatureDF = fullFeatureDF.limit(10000).persist()

#print(fullFeatureDF.rdd.getNumPartitions())


if len (sys.argv) < 2:
    #song1 = "68kd1pdIjmaJAeWYXpUZAs" #In Flames - The New World
    song1 = "1hNHWkHUiUs7etZwdK4NtD"
    song2 = "0Zi7FzlLQsjdRIxtR0Tdm3" 
    song1 = "3TzFRDzRIiM5FEJ8PPpO6j"
    song2 = "3U4tQ24WRJUf8hc7NsNfwf"

    #song3 = "001DpamjDdVjjeAHWCLju9"
    #song4 = "001LKjMxQcD7impp1Fxfsj"
    #song5 = "001cVtpyINACqUDTAPsPhU"
    #song6 = "001hrMFlJXL4aE2yC94sgu"
    #song7 = "002WLZpjJPMsL5x6MP0Q7w"
    #song8 = "003atLcuih7RZelFJDVixX"
    #song9 = "0043iyWPotmbwSNoJVIMEN"
    #song10 = "004nQH1btdhv8iyFER04qt"
else: 
    song1 = sys.argv[1]
    song2 = sys.argv[1]
    song1 = "1hNHWkHUiUs7etZwdK4NtD"
    song2 = "0Zi7FzlLQsjdRIxtR0Tdm3" 

#print(df.filter(df.id == "4heC6SnTBxTZZDef4AIBOk").collect())
#print(df.filter(df.id == "4heC6SnTBxTZZDef4AIBOk").collect())

tic1 = int(time.time() * 1000)
res1 = get_nearest_neighbors(song1, "Benchmark1.csv")
tac1 = int(time.time() * 1000)
time_dict['MERGED_FULL_SONG1: ']= tac1 - tic1

tic2 = int(time.time() * 1000)
res2 = get_nearest_neighbors(song2, "Benchmark2.csv")
tac2 = int(time.time() * 1000)
time_dict['MERGED_FULL_SONG2: ']= tac2 - tic2

#tic1 = int(time.time() * 1000)
#res1 = get_nearest_neighbors(song3, "Benchmark3.csv")
#tac1 = int(time.time() * 1000)
#time_dict['MERGED_FULL_SONG3: ']= tac1 - tic1

#tic2 = int(time.time() * 1000)
#res2 = get_nearest_neighbors(song4, "Benchmark4.csv")
#tac2 = int(time.time() * 1000)
#time_dict['MERGED_FULL_SONG4: ']= tac2 - tic2

#tic1 = int(time.time() * 1000)
#res1 = get_nearest_neighbors(song5, "Benchmark5.csv")
#tac1 = int(time.time() * 1000)
#time_dict['MERGED_FULL_SONG5: ']= tac1 - tic1

#tic2 = int(time.time() * 1000)
#res2 = get_nearest_neighbors(song6, "Benchmark6.csv")
#tac2 = int(time.time() * 1000)
#time_dict['MERGED_FULL_SONG6: ']= tac2 - tic2

#tic1 = int(time.time() * 1000)
#res1 = get_nearest_neighbors(song7, "Benchmark7.csv")
#tac1 = int(time.time() * 1000)
#time_dict['MERGED_FULL_SONG7: ']= tac1 - tic1

#tic2 = int(time.time() * 1000)
#res2 = get_nearest_neighbors(song8, "Benchmark8.csv")
#tac2 = int(time.time() * 1000)
#time_dict['MERGED_FULL_SONG8: ']= tac2 - tic2

#tic1 = int(time.time() * 1000)
#res1 = get_nearest_neighbors(song9, "Benchmark9.csv")
#tac1 = int(time.time() * 1000)
#time_dict['MERGED_FULL_SONG9: ']= tac1 - tic1

#tic2 = int(time.time() * 1000)
#res2 = get_nearest_neighbors(song10, "Benchmark10.csv")
#tac2 = int(time.time() * 1000)
#time_dict['MERGED_FULL_SONG10: ']= tac2 - tic2

total2 = int(time.time() * 1000)
time_dict['MERGED_TOTAL: ']= total2 - total1

print(time_dict)

print("\n\n")

debug_dict['Negative JS: ']= negjs.value
debug_dict['Nan JS: ']= nanjs.value
debug_dict['Non Positive Definite JS: ']= nonpdjs.value
debug_dict['Negative SKL: ']= negskl.value
debug_dict['Nan SKL: ']= nanskl.value
debug_dict['Non Invertible SKL: ']= noninskl.value

print(debug_dict)

print(fullFeatureDF.count())

#featureDF.unpersist()
