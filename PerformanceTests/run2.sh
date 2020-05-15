#!/bin/sh

for i in 1 2 3 4 5
do
  spark-submit spark_ara_merged.py >> perf0123456.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_df.py >> perf0123456.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_rdd.py >> perf0123456.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_filter_refine.py >> perf0123456.txt
done

hdfs dfs -rm -r features6

for i in 1 2 3 4 5
do
  spark-submit spark_ara_merged.py >> perf012345.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_df.py >> perf012345.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_rdd.py >> perf012345.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_filter_refine.py >> perf012345.txt
done


hdfs dfs -rm -r features5

for i in 1 2 3 4 5
do
  spark-submit spark_ara_merged.py >> perf01234.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_df.py >> perf01234.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_rdd.py >> perf01234.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_filter_refine.py >> perf01234.txt
done

