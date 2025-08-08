import sqlite3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import os
import urllib.request
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

cnx = sqlite3.connect('spotify.sqlite')
df = pd.read_sql_query("SELECT id, preview_url FROM tracks", cnx)
df = df[df.preview_url !='']

audio_path = '/beegfs/ja62lel/6M/'

#def download():
for index, row in df.iterrows():
    if index % size == rank:
        uri = row['id'] 
        url = row['preview_url']
        fn = uri+'.mp3'
        subpath = uri[-2:]
        full_path = os.path.join(audio_path, subpath)
        full_fn = os.path.join(full_path, fn)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        if not os.path.exists(full_fn):
            for i in range(5):
                try:
                    print("downloading!")
                    print(row['id'] + "|" + row['preview_url'] + "\n")
                    urllib.request.urlretrieve(url, full_fn)
                except:
                    print("Failed to retrieve " + url)
                    continue
                else:
                    break
            else:
                print('url error. continue')
        else:
            print("Exists: skipping")
            
#download()
print("Done")

