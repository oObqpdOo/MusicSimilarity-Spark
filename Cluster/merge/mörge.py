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
                                              .set("spark.executor.memory", "8g") \
                                              .set("spark.driver.memory", "8g") \
                                              .set("spark.driver.cores", "2") \
                                              .set("spark.executor.cores", "2") \
                                              .set("spark.executor.instances", "27") \
											  .set("spark.dynamicAllocation.enabled", "True") \
                                              .set("spark.dynamicAllocation.initialExecutors", "27") \
                                              .set("spark.dynamicAllocation.executorIdleTimeout", "30s") \
                                              .set("spark.dynamicAllocation.minExecutors", "27") \
                                              .set("spark.dynamicAllocation.maxExecutors", "27")
                                            #.set("spark.driver.memoryOverhead", "1024") \
                                            #.set("spark.executor.memoryOverhead", "1024") \
                                            #.set("yarn.nodemanager.resource.memory-mb", "196608") \
                                            #.set("yarn.nodemanager.vmem-check-enabled", "false") \
                                            #.set("spark.yarn.executor.memoryOverhead", "8192") \
                                            #.set("spark.shuffle.service.enabled", "True") \ 
											#.set("spark.dynamicAllocation.shuffleTracking.enabled", "True") \ #Fehleranfällig: https://spark.apache.org/docs/3.5.2/job-scheduling.html#configuration-and-setup
# Create a SparkSession object
spark = SparkSession.builder.config(conf=conf).getOrCreate()
sc = spark.sparkContext
sqlContext= SQLContext(sc)

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

#repartition_count = 144*100;
for option in sc.getConf().getAll():
    print( "{0:>40} = {1:<20}".format(str(option[0]), str(option[1])) )

#################################################################################################################################################
#################################################################################################################################################
#FIRST DATASET:
#################################################################################################################################################
#################################################################################################################################################

pathName = "SpotiFeat6M/merged"

repartition_count = 128

#########################################################
#   Pre- Process RH and RP for Euclidean
#
rp = sc.textFile(pathName + ".rp")
rp = rp.map(lambda x: x.replace("\"","").replace("b\'","").replace("'","")).map(lambda x: x.split(","))
kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
rp_df = sqlContext.createDataFrame(kv_rp, ["id", "rp"])
rp_df = rp_df.select(rp_df["id"],list_to_vector_udf(rp_df["rp"]).alias("rp"))#.repartition(repartition_count)

rh = sc.textFile(pathName + ".rh")
rh = rh.map(lambda x: x.replace("\"","").replace("b\'","").replace("'","")).map(lambda x: x.split(","))
kv_rh= rh.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
rh_df = sqlContext.createDataFrame(kv_rh, ["id", "rh"])
rh_df = rh_df.select(rh_df["id"],list_to_vector_udf(rh_df["rh"]).alias("rh"))#.repartition(repartition_count)

#########################################################
#   Pre- Process BH for Euclidean
#
bh = sc.textFile(pathName + ".bh")
bh = bh.map(lambda x: x.split(";"))
kv_bh = bh.map(lambda x: (x[0].replace("/beegfs/ja62lel/6M/","").replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""), x[1], Vectors.dense(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
bh_df = sqlContext.createDataFrame(kv_bh, ["id", "bpm", "bh"])#.repartition(repartition_count)
#########################################################
#   Pre- Process Notes for Levenshtein
#
notes = sc.textFile(pathName + ".notes")
notes = notes.map(lambda x: x.split(';'))
notes = notes.map(lambda x: (x[0].replace("/beegfs/ja62lel/6M/","").replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))
notesDf = sqlContext.createDataFrame(notes, ["id", "key", "scale", "notes"])#.repartition(repartition_count)
#########################################################
#   Pre- Process Chroma for cross-correlation
#
chroma = sc.textFile(pathName + ".chroma")
chroma = chroma.map(lambda x: x.replace(' ', '').replace(';', ','))
chroma = chroma.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
chroma = chroma.map(lambda x: x.split(';'))
#try to filter out empty elements
chroma = chroma.filter(lambda x: (not x[1] == '[]') and (x[1].startswith("[[0.") or x[1].startswith("[[1.")))
chromaRdd = chroma.map(lambda x: (x[0].replace("/beegfs/ja62lel/6M/","").replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
chromaVec = chromaRdd.map(lambda x: (x[0], Vectors.dense(x[1])))
chromaDf = sqlContext.createDataFrame(chromaVec, ["id", "chroma"])#.repartition(repartition_count)
#########################################################
#   Pre- Process MFCC for SKL and JS and EUC
#
mfcc = sc.textFile(pathName + ".mfcckl")
mfcc = mfcc.map(lambda x: x.replace(' ', '').replace(';', ','))
mfcc = mfcc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
mfcc = mfcc.map(lambda x: x.split(';'))
mfcc = mfcc.map(lambda x: (x[0].replace("/beegfs/ja62lel/6M/","").replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""), x[1].replace('[', '').replace(']', '').split(',')))
mfccVec = mfcc.map(lambda x: (x[0], Vectors.dense(x[1])))
mfccDfMerged = sqlContext.createDataFrame(mfccVec, ["id", "mfccSkl"])#.repartition(repartition_count)

#########################################################
#   Gather all features in one dataframe
#

featureDF1 = chromaDf.join(mfccDfMerged, on=["id"], how='inner').dropDuplicates().cache() 
featureDF2 = featureDF1.join(notesDf, on=['id'], how='inner').dropDuplicates().cache() 
featureDF1.unpersist()
featureDF3 = featureDF2.join(rp_df, on=['id'], how='inner').dropDuplicates().cache() 
featureDF2.unpersist()
featureDF4 = featureDF3.join(rh_df, on=['id'], how='inner').dropDuplicates().cache() 
featureDF3.unpersist()
dataframe6M = featureDF4.join(bh_df, on=['id'], how='inner').dropDuplicates().cache()
featureDF4.unpersist()

dataframe6M.write.json("AudioFeatures6MMerged.json")
dataframe6M.write.csv("AudioFeatures6MMerged.csv")

#dataframe6M.printSchema()
#dataframe6M.show()

#Force lazy evaluation to evaluate with an action
#trans = featureDF.count()
#print(featureDF.count())

#########################################################

#   DEBUGPRINT UNPERSIST
#chromaDf.unpersist()
#mfccDfMerged.unpersist()
#notesDf.unpersist()
#rp_df.unpersist()
#rh_df.unpersist()
#bh_df.unpersist()

#########################################################
#  4 Nodes, 148GB RAM usable, 36 cores each (+ hyperthreading = 72)
#   -> max 1152 executors
#fullFeatureDF = featureDF.repartition(self.repartition_count).persist()
#print(fullFeatureDF.count())
#fullFeatureDF.toPandas().to_csv("featureDF.csv", encoding='utf-8')


#################################################################################################################################################
#################################################################################################################################################
#SECOND DATASET:
#################################################################################################################################################
#################################################################################################################################################

pathName = "SpotiFeat1_7M/merged"

#########################################################
#   Pre- Process RH and RP for Euclidean
#
rp = sc.textFile(pathName + ".rp")
rp = rp.map(lambda x: x.replace("\"","").replace("b\'","").replace("'","")).map(lambda x: x.split(","))
kv_rp= rp.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
rp_df = sqlContext.createDataFrame(kv_rp, ["id", "rp"])
rp_df = rp_df.select(rp_df["id"],list_to_vector_udf(rp_df["rp"]).alias("rp"))#.repartition(repartition_count)

rh = sc.textFile(pathName + ".rh")
rh = rh.map(lambda x: x.replace("\"","").replace("b\'","").replace("'","")).map(lambda x: x.split(","))
kv_rh= rh.map(lambda x: (x[0].replace(";","").replace(".","").replace(",","").replace(" ",""), list(x[1:])))
rh_df = sqlContext.createDataFrame(kv_rh, ["id", "rh"])
rh_df = rh_df.select(rh_df["id"],list_to_vector_udf(rh_df["rh"]).alias("rh"))#.repartition(repartition_count)

#########################################################
#   Pre- Process BH for Euclidean
#
bh = sc.textFile(pathName + ".bh")
bh = bh.map(lambda x: x.split(";"))
kv_bh = bh.map(lambda x: (x[0].replace('/beegfs/ja62lel/','').replace('audio/', '').replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""), x[1], Vectors.dense(x[2].replace(' ', '').replace('[', '').replace(']', '').split(','))))
bh_df = sqlContext.createDataFrame(kv_bh, ["id", "bpm", "bh"])#.repartition(repartition_count)
#########################################################
#   Pre- Process Notes for Levenshtein
#
notes = sc.textFile(pathName + ".notes")
notes = notes.map(lambda x: x.split(';'))
notes = notes.map(lambda x: (x[0].replace('/beegfs/ja62lel/','').replace('audio/', '').replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""), x[1], x[2], x[3].replace("10",'K').replace("11",'L').replace("0",'A').replace("1",'B').replace("2",'C').replace("3",'D').replace("4",'E').replace("5",'F').replace("6",'G').replace("7",'H').replace("8",'I').replace("9",'J')))
notes = notes.map(lambda x: (x[0], x[1], x[2], x[3].replace(',','').replace(' ','')))
notesDf = sqlContext.createDataFrame(notes, ["id", "key", "scale", "notes"])#.repartition(repartition_count)
#########################################################
#   Pre- Process Chroma for cross-correlation
#
chroma = sc.textFile(pathName + ".chroma")
chroma = chroma.map(lambda x: x.replace(' ', '').replace(';', ','))
chroma = chroma.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
chroma = chroma.map(lambda x: x.split(';'))
#try to filter out empty elements
chroma = chroma.filter(lambda x: (not x[1] == '[]') and (x[1].startswith("[[0.") or x[1].startswith("[[1.")))
chromaRdd = chroma.map(lambda x: (x[0].replace('/beegfs/ja62lel/','').replace('audio/', '').replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""),(x[1].replace(' ', '').replace('[', '').replace(']', '').split(','))))
chromaVec = chromaRdd.map(lambda x: (x[0], Vectors.dense(x[1])))
chromaDf = sqlContext.createDataFrame(chromaVec, ["id", "chroma"])#.repartition(repartition_count)
#########################################################
#   Pre- Process MFCC for SKL and JS and EUC
#
mfcc = sc.textFile(pathName + ".mfcckl")
mfcc = mfcc.map(lambda x: x.replace(' ', '').replace(';', ','))
mfcc = mfcc.map(lambda x: x.replace('.mp3,', '.mp3;').replace('.wav,', '.wav;').replace('.m4a,', '.m4a;').replace('.aiff,', '.aiff;').replace('.aif,', '.aif;').replace('.au,', '.au;').replace('.flac,', '.flac;').replace('.ogg,', '.ogg;'))
mfcc = mfcc.map(lambda x: x.split(';'))
mfcc = mfcc.map(lambda x: (x[0].replace('/beegfs/ja62lel/','').replace('audio/', '').replace(";","").replace(".","").replace(",","").replace(" ","").replace("b\'","").replace("'","").replace("\"",""), x[1].replace('[', '').replace(']', '').split(',')))
mfccVec = mfcc.map(lambda x: (x[0], Vectors.dense(x[1])))
mfccDfMerged = sqlContext.createDataFrame(mfccVec, ["id", "mfccSkl"])#.repartition(repartition_count)

#########################################################
# WRITE
#########################################################

featureDF1 = chromaDf.join(mfccDfMerged, on=["id"], how='inner').dropDuplicates().cache() 
featureDF2 = featureDF1.join(notesDf, on=['id'], how='inner').dropDuplicates().cache() 
featureDF1.unpersist()
featureDF3 = featureDF2.join(rp_df, on=['id'], how='inner').dropDuplicates().cache() 
featureDF2.unpersist()
featureDF4 = featureDF3.join(rh_df, on=['id'], how='inner').dropDuplicates().cache() 
featureDF3.unpersist()
dataframe1_7M = featureDF4.join(bh_df, on=['id'], how='inner').dropDuplicates().cache()
featureDF4.unpersist()
dataframe1_7M.write.json("AudioFeatures1_7MMerged.json")
#dataframe1_7M.printSchema()
#dataframe1_7M.show()

#################################################################################################################################################
#################################################################################################################################################
#MERGE BOTH DATASETS
#################################################################################################################################################
#################################################################################################################################################

#########################################################
# Merge both
#########################################################

#dataframe1_7M = spark.read.json("AudioFeatures1_7MMerged.json").cache()
#dataframe1_7M.printSchema()
#dataframe6M = spark.read.json("AudioFeatures6MMerged.json").cache()
#dataframe6M.printSchema()

#dataframe6M.count() 
#dataframe6M.show()
#dataframeAll = dataframe6M.join(dataframe1_7M, on=["id"], how='inner').dropDuplicates().cache()
#dataframe6M.printSchema()
#dataframe6M.show()

dataframeAll = dataframe6M.union(dataframe1_7M).dropDuplicates().cache()

dataframeAll.write.json("dataframeAll.json")
dataframeAll.write.csv("dataframeAll.csv")


#spark.read.csv("", schema=df.schema, nullValue="Hyukjin Kwon").show()
#dataframeAll.toPandas().to_csv("8M_Spotify.csv", encoding='utf-8')

dataframe1_7M.unpersist()
dataframe6M.unpersist()
dataframeAll.unpersist()



