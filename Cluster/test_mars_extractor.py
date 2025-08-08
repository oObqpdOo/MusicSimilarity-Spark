## RUN WITH: mpiexec -n 4 python test_mars_extractor.py
from openmars import extractor

print("Hello World")
obj = extractor.extractor("/beegfs/ja62lel/6M/", "/beegfs/ja62lel/features6M/features").extract_features()
