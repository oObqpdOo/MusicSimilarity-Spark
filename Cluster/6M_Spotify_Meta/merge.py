#!/usr/bin/python
# -*- coding: utf-8 -*-
import sqlite3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os

df_spotify = pd.read_csv("aggregated_spotify_info.csv")
df_spotify['artist'].str.lower()
df_spotify['album'].str.lower()
df_spotify['track'].str.lower()

df_spotify.info()
print(df_spotify)

df_private = pd.read_csv("tags.csv")
df_private = df_private[df_private['Artist'].notna()]
df_private = df_private[df_private['Album'].notna()]
df_private = df_private[df_private['Title'].notna()]
df_private = df_private.rename(columns={'Artist': 'artist', 'Album': 'album', 'Title': 'track'})

df_private['artist'].str.lower()
df_private['album'].str.lower()
df_private['track'].str.lower()
print(df_private)

df_merged = pd.merge(df_spotify, df_private,  how='inner', left_on=['track','artist','album'], right_on = ['track','artist','album'])
df_merged = df_merged.drop_duplicates(subset = ['track', 'artist','album'],keep = 'last').reset_index(drop = True)
df_merged.info()
print(df_merged)
df_merged.to_csv("merged_datasets.csv")

df_merged_at = pd.merge(df_spotify, df_private,  how='inner', left_on=['track','artist'], right_on = ['track','artist'])
#df_merged_at = df_merged_at.drop_duplicates(subset = ['track', 'artist'],keep = 'last').reset_index(drop = True)
df_merged_at.info()
print(df_merged_at)
df_merged_at.to_csv("merged_datasets_AT.csv")
