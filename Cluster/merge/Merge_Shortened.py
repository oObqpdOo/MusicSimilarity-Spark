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

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

#################################################################################################################################################
#################################################################################################################################################
#MERGE BOTH DATASETS
#################################################################################################################################################
#################################################################################################################################################

#########################################################
# Merge both
#########################################################

dataframe1_7M = spark.read.json("AudioFeatures1_7MMerged.json").cache()
#dataframe1_7M.printSchema()
dataframe6M = spark.read.json("AudioFeatures6MMerged.json").cache()
#dataframe6M.printSchema()

dataframeAll = dataframe6M.union(dataframe1_7M).dropDuplicates().cache()

#https://spark.apache.org/docs/latest/sql-data-sources-load-save-functions.html
dataframeAll.write.save("AllFeaturesMerged.parquet")

#df_merged = spark.read.format("parquet").option("header", True).option("inferSchema", True).load("AllFeaturesMerged.parquet")

df_info = spark.read.option("header",True).csv("aggregated_spotify_info.csv")
df_info = df_info.withColumnRenamed("track_id", "id")

df_joined = dataframeAll.join(df_info, ["id"], "inner")

df_joined.exceptAll(df_joined.dropDuplicates(['id'])).persist()#.show()

df_joined.write.save("PreProcessedFull.parquet")

df_joined.show()

dataframe1_7M.unpersist()
dataframe6M.unpersist()
dataframeAll.unpersist()



