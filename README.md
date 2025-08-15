# MusicSimilarity-Spark
Feature Extraction Code for Chroma features, Note estimation, MFCC statistics, Rhythm Histogram, Rhythm Patterns and Beat Histograms

Python Spark Code for estimating the similarity of songs

## Demo (non-cluster local variant):

- Install dependencies (see below)  
- Place audio files in the '''/audio/''' folder and create a target '''/features/''' folder  
- Run '''mpiexec -n 4 python example_mars_extractor.py'''  
	- Audio features are extracted into the '''/features/''' folder  
- Run '''python example\_mars\_preprocess.py'''  
	- A Spark dataframe containing the relevant features is stored to '''AudioFeaturesMerged.json'''  
- Run '''spark-submit example\_mars\_preprocess.py'''  
	- The first two songs are taken as exemplary song requests  

## Important Note:
15.Aug.2025: Repository is in the progress of being merged with a repository from a private Gitlab server.
Minimal running example for 1 node installation, see above.

## Third Party Libraries

Code for Rhythm Histogram and Rhythm Patterns from TU Wien: https://github.com/tuwien-musicir/rp_extract under GNU General Public License v3.0
(slightly adapted version)

Description and Documentation: https://github.com/oObqpdOo/MusicSimilarity

## Requirements: 
pyspark 1.6.0 or newer,  
essentia,  
numpy,  
scipy,  
matplotlib,  
urllib,  
ipython/ jupyter,  
pathlib,  
signal,
glob, 
edlib,   
ffmpeg or mpg123  
