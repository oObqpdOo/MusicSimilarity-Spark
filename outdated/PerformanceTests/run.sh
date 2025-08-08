#!/bin/sh

for i in 1 2 3 4 5
do
  spark-submit spark_ara_merged.py >> perf0123.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_df.py >> perf0123.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_rdd.py >> perf0123.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_filter_refine.py >> perf0123.txt
done

hdfs dfs -rm -r features3

for i in 1 2 3 4 5
do
  spark-submit spark_ara_merged.py >> perf012.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_df.py >> perf012.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_rdd.py >> perf012.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_filter_refine.py >> perf012.txt
done


hdfs dfs -rm -r features1

for i in 1 2 3 4 5
do
  spark-submit spark_ara_merged.py >> perf02.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_df.py >> perf02.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_rdd.py >> perf02.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_filter_refine.py >> perf02.txt
done


hdfs dfs -rm -r features0

for i in 1 2 3 4 5
do
  spark-submit spark_ara_merged.py >> perf0.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_df.py >> perf0.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_rdd.py >> perf0.txt
done

for i in 1 2 3 4 5
do
  spark-submit spark_ara_filter_refine.py >> perf0.txt
done


