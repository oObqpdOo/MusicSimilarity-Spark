#!/usr/bin/python
# -*- coding: utf-8 -*-
import sqlite3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os

cnx = sqlite3.connect('spotify.sqlite')
#cnx.text_factory = bytes
cnx.text_factory = lambda b: b.decode(errors = 'ignore')

df_tracks = pd.read_sql_query("SELECT id, name, preview_url, popularity, track_number, explicit FROM tracks", cnx)
df_tracks = df_tracks[df_tracks.preview_url !='']
df_albums = pd.read_sql_query("SELECT id, name FROM albums", cnx)
df_artists = pd.read_sql_query("SELECT id, name FROM artists", cnx)
df_albums_tracks = pd.read_sql_query("SELECT track_id, album_id FROM r_albums_tracks", cnx)
df_track_artist = pd.read_sql_query("SELECT track_id, artist_id FROM r_track_artist", cnx)

#df_artist_genre = pd.read_sql_query("SELECT * FROM r_artist_genre", cnx)
#df_audio_features = pd.read_sql_query("SELECT * FROM audio_features", cnx)

df_tracks = df_tracks.rename(columns={'name': 'track', 'id': 'track_id'})
df_albums = df_albums.rename(columns={'name': 'album', 'id': 'album_id'})
df_artists = df_artists.rename(columns={'name': 'artist', 'id': 'artist_id'})

df_albums_tracks = df_albums_tracks.rename(columns={'album_id': 'album_id', 'track_id': 'track_id'})
df_track_artist = df_track_artist.rename(columns={'artist_id': 'artist_id', 'track_id': 'track_id'})
#df_audio_features = df_audio_features.rename(columns={'id': 'track_id'})

print(df_tracks.head(2))
print(df_albums.head(2))
print(df_artists.head(2))

print(df_albums_tracks.head(2))
print(df_track_artist.head(2))

#print(df_artist_genre.head(2))
#print(df_audio_features.head(2))

df_spotify = df_tracks.merge(right=df_albums_tracks, how='inner', left_on='track_id', right_on='track_id')
df_spotify = df_spotify.merge(right=df_track_artist, how='inner', left_on='track_id', right_on='track_id')
df_spotify = df_spotify.merge(right=df_albums, how='inner', left_on='album_id', right_on='album_id')
df_spotify = df_spotify.merge(right=df_artists, how='inner', left_on='artist_id', right_on='artist_id')

#df_spotify = df_spotify.merge(right=df_artist_genre, how='inner', left_on='artist_id', right_on='artist_id')
#df_spotify = df_spotify.merge(right=df_audio_features, how='inner', left_on='track_id', right_on='track_id')

df_spotify.info()
print(df_spotify)

df_spotify.to_csv("aggregated_spotify_info.csv")
