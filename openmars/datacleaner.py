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
import edlib

class datacleaner:
	def __init__(self, path):
		self.pathName = path
		#self.pathName = "features[0-9]*/out[0-9]*"
		self.total1 = int(round(time.time() * 1000))
		self.confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
		self.confCluster.set("spark.driver.memory", "4g")
		self.confCluster.set("spark.executor.memory", "4g")
		self.confCluster.set("spark.driver.memoryOverhead", "2g")
		self.confCluster.set("spark.executor.memoryOverhead", "2g")
			#Be sure that the sum of the driver or executor memory plus the driver or executor memory overhead is always less than the value of yarn.nodemanager.resource.memory-mb
			#confCluster.set("yarn.nodemanager.resource.memory-mb", "196608")
			#spark.driver/executor.memory + spark.driver/executor.memoryOverhead < yarn.nodemanager.resource.memory-mb
			#self.confCluster.set("spark.yarn.executor.memoryOverhead", "4096")
			#set cores of each executor and the driver -> less than avail -> more executors spawn
		self.confCluster.set("spark.driver.cores", "2")
		self.confCluster.set("spark.executor.cores", "2")
			#confCluster.set("spark.shuffle.service.enabled", "True")
		self.confCluster.set("spark.dynamicAllocation.enabled", "True")
		self.confCluster.set("spark.dynamicAllocation.initialExecutors", "16")
		self.confCluster.set("spark.dynamicAllocation.executorIdleTimeout", "30s")	
		self.confCluster.set("spark.dynamicAllocation.minExecutors", "15")
		self.confCluster.set("spark.dynamicAllocation.maxExecutors", "15")
		self.confCluster.set("yarn.nodemanager.vmem-check-enabled", "false")
		self.repartition_count = 32
		self.sc = SparkContext(conf=self.confCluster)
		self.sqlContext = SQLContext(self.sc)
		self.sc.setLogLevel("ERROR")
		self.time_dict = {}
		self.list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
		
	def clean_features(self):
		self.tic1 = int(round(time.time() * 1000))
		#########################################################
		#   Pre- Process RH and RP for Euclidean
		#
		rp = self.sc.textFile(self.pathName + ".rp")
		rp = rp.map(lambda x: x.replace("\"","").replace("b\'","").replace("'","")).map(lambda x: x.split(","))
		kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
		rp_df = self.sqlContext.createDataFrame(kv_rp, ["id", "rp"])
		rp_df = rp_df.select(rp_df["id"],self.list_to_vector_udf(rp_df["rp"]).alias("rp"))
		rh = self.sc.textFile(self.pathName + ".rh")
		rh = rh.map(lambda x: x.replace("\"","").replace("b\'","").replace("'","")).map(lambda x: x.split(","))
		kv_rh= rh.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
		rh_df = self.sqlContext.createDataFrame(kv_rh, ["id", "rh"])
		rh_df = rh_df.select(rh_df["id"],self.list_to_vector_udf(rh_df["rh"]).alias("rh"))
		#########################################################
		#   Pre- Process BH for Euclidean
		#
		bh = self.sc.textFile(self.pathName + ".bh")
		bh = bh.map(lambda x: x.split(";"))
		kv_bh = bh.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""), x[1], Vectors.dense(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
		bh_df = self.sqlContext.createDataFrame(kv_bh, ["id", "bpm", "bh"])
		#########################################################
		#   Pre- Process Notes for Levenshtein
		#
		notes = self.sc.textFile(self.pathName + ".notes")
		notes = notes.map(lambda x: x.split(';'))
		notes = notes.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
		notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))
		notesDf = self.sqlContext.createDataFrame(notes, ["id", "key", "scale", "notes"])
		#########################################################
		#   Pre- Process Chroma for cross-correlation
		#
		chroma = self.sc.textFile(self.pathName + ".chroma")
		chroma = chroma.map(lambda x: x.replace(' ', '').replace(';', ','))
		chroma = chroma.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
		chroma = chroma.map(lambda x: x.split(';'))
		#try to filter out empty elements
		chroma = chroma.filter(lambda x: (not x[1] == '[]') and (x[1].startswith("[[0.") or x[1].startswith("[[1.")))
		chromaRdd = chroma.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
		chromaVec = chromaRdd.map(lambda x: (x[0], Vectors.dense(x[1])))
		chromaDf = self.sqlContext.createDataFrame(chromaVec, ["id", "chroma"])
		#########################################################
		#   Pre- Process MFCC for SKL and JS and EUC
		#
		mfcc = self.sc.textFile(self.pathName + ".mfcckl")            
		mfcc = mfcc.map(lambda x: x.replace(' ', '').replace(';', ','))
		mfcc = mfcc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
		mfcc = mfcc.map(lambda x: x.split(';'))
		mfcc = mfcc.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""), x[1].replace('[', '').replace(']', '').split(',')))
		mfccVec = mfcc.map(lambda x: (x[0], Vectors.dense(x[1])))
		mfccDfMerged = self.sqlContext.createDataFrame(mfccVec, ["id", "mfccSkl"])
		#########################################################
		#   DEBUGPRINT
		#				
		chromaDf.persist().show()
		mfccDfMerged.persist().show()
		notesDf.persist().show()
		rp_df.persist().show()
		rh_df.persist().show()
		bh_df.persist().show()
		#########################################################
		#   Gather all features in one dataframe
		#					
		featureDF = chromaDf.join(mfccDfMerged, on=["id"], how='inner')
		featureDF = featureDF.join(notesDf, on=['id'], how='inner')
		featureDF = featureDF.join(rp_df, on=['id'], how='inner')
		featureDF = featureDF.join(rh_df, on=['id'], how='inner')
		featureDF = featureDF.join(bh_df, on=['id'], how='inner').dropDuplicates().persist()
		#Force lazy evaluation to evaluate with an action
		trans = featureDF.count()
		#print(featureDF.count())
		#########################################################
		#   DEBUGPRINT UNPERSIST
		#					
		chromaDf.unpersist()
		mfccDfMerged.unpersist()
		notesDf.unpersist()
		rp_df.unpersist()
		rh_df.unpersist()
		bh_df.unpersist()
		#########################################################
		#  16 Nodes, 192GB RAM each, 36 cores each (+ hyperthreading = 72)
		#   -> max 1152 executors
		fullFeatureDF = featureDF.repartition(self.repartition_count).persist()
		#print(fullFeatureDF.count())
		#fullFeatureDF.toPandas().to_csv("featureDF.csv", encoding='utf-8')
		self.tac1 = int(round(time.time() * 1000))
		self.time_dict['PREPROCESS: ']= self.tac1 - self.tic1
		return fullFeatureDF, self.sc, self.time_dict, self.total1
