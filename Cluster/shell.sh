source pyspark_venv/bin/activate
export PYSPARK_PYTHON=./pyspark_venv/bin/python
pyspark --driver-memory 16g --driver-cores 4 --executor-memory 16g --executor-cores 4 --conf "yarn.scheduler.maximum-allocation-vcores=36" --conf "spark.executor.instances=100" --conf "spark.dynamicAllocation.enabled=True" --conf "spark.dynamicAllocation.initialExecutors=100" --conf "spark.dynamicAllocation.minExecutors=100" --conf "spark.dynamicAllocation.maxExecutors=100" --archives pyspark_venv.tar.gz#pyspark_venv
