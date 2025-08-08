source pyspark_venv/bin/activate
export PYSPARK_PYTHON=./pyspark_venv/bin/python
spark-submit --archives pyspark_venv.tar.gz#pyspark_venv similarity.py 
