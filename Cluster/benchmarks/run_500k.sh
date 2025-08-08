#!/bin/bash
source pyspark_venv/bin/activate
export PYSPARK_PYTHON=./pyspark_venv/bin/python

outfilenum=0
count=1

for i in {1..10}; do
	echo "RUN_$1" >> 500k/similarity_rp_js_chroma_500k${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 500k/similarity_rp_js_chroma_500k.py >> 500k/similarity_rp_js_chroma_500k${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> 500k/similarity_rp_skl_chroma_500k${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 500k/similarity_rp_skl_chroma_500k.py >> 500k/similarity_rp_skl_chroma_500k${outfilenum}.txt  
	((count++))
done
