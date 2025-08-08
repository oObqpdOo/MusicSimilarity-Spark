#!/bin/bash
source pyspark_venv/bin/activate
export PYSPARK_PYTHON=./pyspark_venv/bin/python

outfilenum=0
count=1
for i in {1..10}; do
	echo "RUN_$1" >> Full/similarity_rh_bh_mfcc_Full${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv Full/similarity_rh_bh_mfcc_Full.py >> Full/similarity_rh_bh_mfcc_Full${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> Full/similarity_rh_mfcc_notes_Full${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv Full/similarity_rh_mfcc_notes_Full.py >> Full/similarity_rh_mfcc_notes_Full${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> Full/similarity_rp_js_chroma_Full${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv Full/similarity_rp_js_chroma_Full.py >> Full/similarity_rp_js_chroma_Full${outfilenum}.txt  
	((count++))
done

for i in {1..10}; do
	echo "RUN_$1" >> Full/similarity_rp_skl_chroma_Full${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv Full/similarity_rp_skl_chroma_Full.py >> Full/similarity_rp_skl_chroma_Full${outfilenum}.txt  
	((count++))
done
