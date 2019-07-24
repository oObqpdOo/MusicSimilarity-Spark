#!/bin/bash

python rp_extract_batch.py -rh ./ out

vpn connect vpn.uni-jena.de

#ja62lel@uni-jena.de
#jojo2012

scp out.rh ja62lel@ppc802.mirz.uni-jena.de:MA/data
#jojo2012

ssh -X ja62lel@ppc802.mirz.uni-jena.de
#jojo2012

cd /MA/data

hdfs dfs -rm -r out.rh 

hdfs dfs -copyFromLocal out.rh .

pyspark

