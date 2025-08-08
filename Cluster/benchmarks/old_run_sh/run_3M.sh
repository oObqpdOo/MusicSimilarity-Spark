#!/bin/bash
source pyspark_venv/bin/activate
export PYSPARK_PYTHON=./pyspark_venv/bin/python

outfilenum=0
count=1
for i in {1..10}; do
	echo "RUN_$1" >> 3M/similarity_rh_bh_mfcc_3M${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 3M/similarity_rh_bh_mfcc_3M.py >> 3M/similarity_rh_bh_mfcc_3M${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> 3M/similarity_rh_mfcc_notes_3M${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 3M/similarity_rh_mfcc_notes_3M.py >> 3M/similarity_rh_mfcc_notes_3M${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> 3M/similarity_rp_js_chroma_3M${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 3M/similarity_rp_js_chroma_3M.py >> 3M/similarity_rp_js_chroma_3M${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> 3M/similarity_rp_skl_chroma_3M${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv 3M/similarity_rp_skl_chroma_3M.py >> 3M/similarity_rp_skl_chroma_3M${outfilenum}.txt  
	((count++))
done
