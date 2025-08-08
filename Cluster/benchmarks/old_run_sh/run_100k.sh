#!/bin/bash
source pyspark_venv/bin/activate
export PYSPARK_PYTHON=./pyspark_venv/bin/python

outfilenum=0
count=1
for i in {1..10}; do
	echo "RUN_$1" >> 100k/similarity_rh_bh_mfcc_100k${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 100k/similarity_rh_bh_mfcc_100k.py >> 100k/similarity_rh_bh_mfcc_100k${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> 100k/similarity_rh_mfcc_notes_100k${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 100k/similarity_rh_mfcc_notes_100k.py >> 100k/similarity_rh_mfcc_notes_100k${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> 100k/similarity_rp_js_chroma_100k${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 100k/similarity_rp_js_chroma_100k.py >> 100k/similarity_rp_js_chroma_100k${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> 100k/similarity_rp_skl_chroma_100k${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 100k/similarity_rp_skl_chroma_100k.py >> 100k/similarity_rp_skl_chroma_100k${outfilenum}.txt  
	((count++))
done
