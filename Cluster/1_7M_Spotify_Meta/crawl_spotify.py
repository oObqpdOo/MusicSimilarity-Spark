#!/usr/bin/python
# -*- coding: utf-8 -*-
import sqlite3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

os.environ["SPOTIPY_CLIENT_ID"] = ""
os.environ["SPOTIPY_CLIENT_SECRET"] = ""
os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost"
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

artistName = "In Flames"
trackName = "Colony"
albumName = "Colony"
result = sp.search("artist:" + artistName + " track: " + trackName + " album: " + albumName, limit = 1)
print(result)

print(result['tracks']['items'][0]['preview_url'])
print(result['tracks']['items'][0]['id'])
print(result['tracks']['items'][0]['popularity'])
print(result['tracks']['items'][0]['name'])
print(result['tracks']['items'][0]['artists'][0]['name'])
print(result['tracks']['items'][0]['album']['name'])

import time 
results = []
failed = 0
succeeded = 0

file1 = open("dump.txt", "a")

#df_part = df_private.tail(-5926) # 1480+2740-1 (-4219) (2740)
                                 # 4219+979+727-1 (-5924) (3690)

for i, row in df_private.iterrows():
    #print(row['track'], row['artist'])
    artistName = str(row['artist'])
    trackName = str(row['track'])
    albumName = str(row['album'])

    result = sp.search("artist:" + artistName + " track: " + trackName + " album: " + albumName, limit = 1)
    try: 
        url = result['tracks']['items'][0]['preview_url']
        tid = result['tracks']['items'][0]['id']
        popularity = result['tracks']['items'][0]['popularity']
        title = result['tracks']['items'][0]['name']
        artist = result['tracks']['items'][0]['artists'][0]['name']
        album = result['tracks']['items'][0]['album']['name']
        result_tuple = [tid, url, title, album, artist, popularity]
        file1.write(str(result_tuple) + "\n")
        succeeded = succeeded + 1
        print("Found: " + str(succeeded))
    except: 
        failed = failed + 1
        print("Not Found: " + str(failed))


