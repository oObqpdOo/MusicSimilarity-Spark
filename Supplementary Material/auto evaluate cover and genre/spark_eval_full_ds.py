#!/usr/bin/python
# -*- coding: utf-8 -*-

import pyspark
import pyspark.ml.feature
import pyspark.ml.linalg
import pyspark.ml.param
import pyspark.sql.functions
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from scipy.spatial import distance
from pyspark.ml.feature import BucketedRandomProjectionLSH
#from pyspark.mllib.linalg import Vectors
from pyspark.ml.param.shared import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import numpy as np
#import org.apache.spark.sql.functions.typedLit
from pyspark.sql.functions import lit
from pyspark.sql.functions import levenshtein  
from pyspark.sql.functions import col
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
import scipy as sp
from scipy.signal import butter, lfilter, freqz, correlate2d
from itertools import islice
import glob

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql import SparkSession
confCluster = SparkConf().setAppName("MusicSimilarity Cluster")
confLocal = SparkConf().setMaster("local").setAppName("MusicSimilarity Local")
sc = SparkContext(conf=confCluster)
sqlContext = SQLContext(sc)
spark = SparkSession.builder.master("cluster").appName("MusicSimilarity").getOrCreate()

list1l = (glob.glob("results/testsetfull/*.csv"))
count = 0

count = 0
for i in list1l[:]: 
    #outname = "results/testset/" + i.replace('.mp3', '').replace('music/', '').replace('/', '_').replace('mp3', '') + ".csv"
    outname = i    
    #outname = outname.encode('ascii','ignore')    
    print outname 
    rdd = sc.textFile(outname)
    rdd = rdd.map(lambda x: x.replace('music/','').replace('/', ''))
    #drop csv header
    rdd = rdd.mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it)
    #clean id
    rdd = rdd.map(lambda x: x.split(','))
    rdd = rdd.map(lambda x: x[1:])
    rdd = rdd.map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12]))
    rddDF = spark.createDataFrame(rdd, ["id","key","scale","bpm","rp","rh","bh","notes","chroma","skl","js","mfcc","agg"])
    if count == 0:
        result = rddDF
    else:
        result = result.union(rddDF)
    count = count + 1

result.show()
result.toPandas().to_csv("__feature_eval.csv", encoding='utf-8')


