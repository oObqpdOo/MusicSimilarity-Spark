import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType, Row
from pyspark.sql.functions import * # col, array, lit, udf, min, max, round # lit is used for applying one scalar to every row in a whole column when using withColumn and creating a new column
from pyspark.sql.functions import udf
from pyspark.ml.param.shared import *
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F


#if executed in PySpark Shell with pyspark --num-executors 27
#sc.stop()

# Create a SparkConf object
#Be sure that the sum of the driver or executor memory plus the driver or executor memory overhead is always less than the value of yarn.nodemanager.resource.memory-mb \
#spark.driver/executor.memory + spark.driver/executor.memoryOverhead < yarn.nodemanager.resource.memory-mb \

conf = SparkConf().setAppName("MergeDatasets").set("yarn.nodemanager.resource.detect-hardware-capabilities" , "True") \
                                              .set("yarn.nodemanager.resource.memory-mb", "196608") \
                                              .set("spark.executor.memory", "20g") \
                                              .set("spark.driver.memory", "20g") \
                                              .set("spark.driver.cores", "4") \
                                              .set("spark.executor.cores", "4") \
                                              .set("spark.executor.instances", "100") \
											  .set("spark.dynamicAllocation.enabled", "True") \
                                              .set("spark.dynamicAllocation.initialExecutors", "100") \
                                              .set("spark.dynamicAllocation.executorIdleTimeout", "30s") \
                                              .set("spark.dynamicAllocation.minExecutors", "100") \
                                              .set("spark.dynamicAllocation.maxExecutors", "100")
                                         
# Create a SparkSession object
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext
sqlContext= SQLContext(sc)

#################################################################################################################################################
#################################################################################################################################################
#MERGE BOTH DATASETS
#################################################################################################################################################
#################################################################################################################################################

#########################################################
# Merge both
#########################################################

dataframe1_7M = spark.read.json("AudioFeatures1_7MMerged.json").cache()
dataframe1_7M.printSchema()
print(dataframe1_7M.count())		# 1746425
dataframe1_7M = dataframe1_7M.dropDuplicates(['id']) 	
print(dataframe1_7M.count())		# 1746425
#NO DUPLICATE IDs FOUND! NICE!

dataframe6M = spark.read.json("AudioFeatures6MMerged.json").cache()
dataframe6M.printSchema()
print(dataframe6M.count())		# 4221784
dataframe6M = dataframe6M.dropDuplicates(['id']) 
print(dataframe6M.count())		# 4221784

dataframeAll = dataframe6M.union(dataframe1_7M).dropDuplicates().cache()
dataframeAll.printSchema()
print(dataframeAll.count())		# 
dataframeAll = dataframeAll.dropDuplicates() 
print(dataframeAll.count())		# 

#https://spark.apache.org/docs/latest/sql-data-sources-load-save-functions.html
#dataframeAll.write.save("AllFeaturesMerged.parquet")

fullFeatureDF = spark.read.format("parquet").option("header", True).option("inferSchema", True).load("AllFeaturesMerged.parquet").persist()

fullFeatureDF.printSchema()
print(fullFeatureDF.count()) #5967970
fullFeatureDFDrop = fullFeatureDF.dropDuplicates(['id']) 
print(fullFeatureDF.count()) #5306163
fullFeatureDFDrop = fullFeatureDF.dropDuplicates() 
print(fullFeatureDF.count()) #5967970

#================================================================================
# Find Duplicates in FullFeatureDF
#================================================================================

primary_key = ['id']
duplicated_keys = (fullFeatureDF.groupby(primary_key).count().filter(F.col('count') > 1).drop(F.col('count')))
duplicates = (fullFeatureDF.join(F.broadcast(duplicated_keys), primary_key)).persist()
duplicates.orderBy("id").show()   #WEIRD! DUPLICATES ARE EXTREMELY SIMILAR BUT STILL DIFFERENT! -> Reason: Different time of Download? Preview Samples changed? No Clue Otherwise
print(duplicates.count()) #1323614 - 661807*2 -> 5967970 - 5306163 = 661807

non_duplicated_keys = (fullFeatureDF.groupby(primary_key).count().filter(F.col('count') == 1).drop(F.col('count')))
non_duplicates = (fullFeatureDF.join(F.broadcast(non_duplicated_keys), primary_key)).persist()
non_duplicates.orderBy("id").show() 
print(non_duplicates.count()) #4644356 -> 5967970-1323614 = 4644356
#================================================================================

duplicates.write.mode("overwrite").save("Duplicates.parquet")
non_duplicates.write.mode("overwrite").save("NonDuplicates.parquet")

#================================================================================
# Find FULL Duplicates in FullFeatureDF
#================================================================================

primary_key = ['id', 'bh', 'bpm', 'chroma', 'key', 'mfccSkl', 'notes', 'rh', 'rp', 'scale']
full_duplicated_keys = (fullFeatureDF.groupby(primary_key).count().filter(F.col('count') > 1).drop(F.col('count')))
full_duplicates = (fullFeatureDF.join(F.broadcast(full_duplicated_keys), primary_key)).persist()
full_duplicates.orderBy("id").show()   #Full Duplicates are 100% identical
print(full_duplicates.count()) #

#================================================================================
# LIMIT
#================================================================================
#strippedFeatureDF = fullFeatureDF.limit(1000000).persist()
#strippedFeatureDF.count()
#strippedFeatureDF.write.mode("overwrite").save("PreProcessed1000000.parquet")
#selection = strippedFeatureDF.select(['id', 'artist', 'track', 'album', 'preview_url'])
#selection.write.csv("Tracklist1000000.csv")

#================================================================================
# SELF-JOIN
#================================================================================
#selection = fullFeatureDF.select(['id','preview_url'])
#selcpy = selection.withColumnRenamed("preview_url","other_url")
#joined = selection.join(selcpy, "id", "inner")
#joined = joined.dropDuplicates()
#print(joined.count())
#joined.write.mode("overwrite").csv("Duplicates.csv")

#================================================================================
# SPOTIFY_INFO SUCKS; IT IS A CSV; THIS WONT WORK... EVER!!! REDO THIS
#================================================================================
#df_info = spark.read.option("header",True).csv("aggregated_spotify_info.csv")
#df_info = df_info.withColumnRenamed("track_id", "id")
#df_joined = dataframeAll.join(df_info, ["id"], "inner")
#df_joined.exceptAll(df_joined.dropDuplicates(['id'])).persist()#.show()
#df_joined.write.save("PreProcessedFull.parquet")
#df_joined.show()

dataframe1_7M.unpersist()
dataframe6M.unpersist()
dataframeAll.unpersist()



