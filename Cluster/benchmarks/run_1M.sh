#!/bin/bash
source pyspark_venv/bin/activate
export PYSPARK_PYTHON=./pyspark_venv/bin/python

outfilenum=0
count=1

for i in {1..10}; do
	echo "RUN_$1" >> 1M/similarity_rp_js_chroma_1M${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 1M/similarity_rp_js_chroma_1M.py >> 1M/similarity_rp_js_chroma_1M${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> 1M/similarity_rp_skl_chroma_1M${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 1M/similarity_rp_skl_chroma_1M.py >> 1M/similarity_rp_skl_chroma_1M${outfilenum}.txt  
	((count++))
done
