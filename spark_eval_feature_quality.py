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

songs = sc.textFile("features[0-9]*/out.files", use_unicode=True)
list1 = songs.map(lambda x: x.split(':'))
#DO NOT USE str(x[0]) USE x[0].encode('utf-8') instead!!
list1 = list1.map(lambda x: x[0])
list1 = list1.map(lambda x: x.replace(";","").replace(".","").replace(",","").replace(" ",""))
list1l = list1.collect()


list1l = (glob.glob("results/covers80/*.csv"))

################################################################################
#
#   rddDF.chroma + rddDF.notes + rddDF.rp
#
#

count = 0
for i in list1l[:]: 
    #outname = "results/testset/" + i.replace('.mp3', '').replace('music/', '').replace('/', '_').replace('mp3', '') + ".csv"
    outname = i    
    #outname = outname.encode('ascii','ignore')    
    print outname 
    rdd = sc.textFile(outname)
    rdd = rdd.map(lambda x: x.replace('music/','').split('/'))
    #drop csv header
    rdd = rdd.mapPartitionsWithIndex(lambda idx, it: islice(it, 1, None) if idx == 0 else it)
    #clean id
    rdd = rdd.map(lambda x: (x[0].split(','), x[1])).map(lambda x: (x[0][1], x[1]))
    #create DF
    rdd = rdd.map(lambda x: (x[0], x[1].split(',')))
    rdd = rdd.map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6], x[1][7], x[1][8], x[1][9], x[1][10]))
    rddDF = spark.createDataFrame(rdd, ["cover", "id", "rp", "key", "scale", "notes", "bpm", "bh", "mfcc", "chroma", "skl", "agg"])
    rddRes = rddDF.withColumn('newdist', (rddDF.notes + rddDF.notes + rddDF.rp) / 3).select("id", "cover", "newdist").orderBy('newdist', ascending=True).limit(2)
    #rddRes.show()
    originalSong = rddRes.select("cover").limit(1).collect()[0]["cover"]
    originalID = rddRes.select("id").limit(1).collect()[0]["id"]
    #then drop row
    rddRes = rddRes.filter(rddRes.id != originalID)
    countdf = rddRes.groupBy("cover").agg(F.count("cover")).withColumn('original', F.lit(originalSong))
    countdf = countdf.filter(countdf.cover == originalSong)
    #countdf.show()
    if count == 0:
        result = countdf
    else:
        result = result.union(countdf)
    count = count + 1

result.show()
#result.toPandas().to_csv("__genre_estimation.csv", encoding='utf-8')
resultreduce = result.rdd
#map (original -> detected), count
resultreducekey = resultreduce.map(lambda x: ((x[2], x[0]), (x[1])))
reduced = resultreducekey.reduceByKey(lambda x, y: x + y)
reducedDF = spark.createDataFrame(reduced, ["original vs detected", "count"])
reducedDF.toPandas().to_csv("__cover_chroma_notes_rp.csv", encoding='utf-8')


