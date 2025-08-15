## RUN WITH: mpiexec -n 4 python example_mars_extractor.py
from openmars import extractor

print("Extracting Audio Files")
obj = extractor.extractor("./audio/", "./features/").extract_features()
