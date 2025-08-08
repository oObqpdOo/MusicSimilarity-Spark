source pyspark_venv/bin/activate
export PYSPARK_PYTHON=./pyspark_venv/bin/python

outfilenum=0
count=1
for i in {1..10}; do
	echo "RUN$1" >> rh_mfcc_notes_${outfilenum}.txt
	spark-submit --archives pyspark_venv.tar.gz#pyspark_venv similarity.py 1 0 0 0 1 1 0 0 2 >> rh_mfcc_notes_${outfilenum}.txt  
	((count++))
done
