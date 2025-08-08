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

executor_s = "100"
parts_s = "100"
parts = 200

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
                                              .set("spark.executor.instances", executor_s) #\
                                            #.set("spark.default.parallelism", parts_s) 
                                            #.set("spark.driver.memoryOverhead", "1024") \
                                            #.set("spark.executor.memoryOverhead", "1024") \
                                            #.set("yarn.nodemanager.resource.memory-mb", "196608") \
                                            #.set("yarn.nodemanager.vmem-check-enabled", "false") \
                                            #.set("spark.yarn.executor.memoryOverhead", "8192") \
                                            #.set("spark.shuffle.service.enabled", "True") \ 
					    #.set("spark.dynamicAllocation.shuffleTracking.enabled", "True") \ 

# Create a SparkSession object
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext
sqlContext= SQLContext(sc)


fullFeatureDF = spark.read.format("parquet").option("header", True).option("inferSchema", True).load("PreProcessed100000.parquet").persist()

#root
# |-- id: string (nullable = true)
# |-- chroma: vector (nullable = true)
# |-- mfccSkl: vector (nullable = true)
# |-- key: string (nullable = true)
# |-- scale: string (nullable = true)
# |-- notes: string (nullable = true)
# |-- rp: vector (nullable = true)
# |-- rh: vector (nullable = true)
# |-- bpm: string (nullable = true)
# |-- bh: vector (nullable = true)
# |-- _c0: string (nullable = true)
# |-- track: string (nullable = true)
# |-- preview_url: string (nullable = true)
# |-- album_id: string (nullable = true)
# |-- artist_id: string (nullable = true)
# |-- album: string (nullable = true)
# |-- artist: string (nullable = true)


print(fullFeatureDF.count())

selection = fullFeatureDF.select(['id','preview_url'])

selcpy = selection.withColumnRenamed("preview_url","other_url")

joined = selection.join(selcpy, "id", "inner")
joined = joined.dropDuplicates()

print(joined.count())
joined.write.mode("overwrite").csv("Duplicates.csv")
#artist = selection.filter(selection.artist == 'In Flames')

