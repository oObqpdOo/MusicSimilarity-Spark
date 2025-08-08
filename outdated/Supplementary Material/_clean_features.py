#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path, PurePath
from time import time, sleep
import pp
import multiprocessing
import os
import argparse
import gc

gc.enable()

np.set_printoptions(threshold=np.inf)
filelist = []
for filename in Path('features').glob('**/*.files'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.bh'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.mfcc'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.mfcckl'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.rp'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.rh'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.chroma'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.notes'):
    filelist.append(filename)
for filename in Path('features').glob('**/*.files'):
    filelist.append(filename)


wrong_names = ["music/Metal & Rock/Compilations/Relapse Records - Relapse Sampler 2017/18 Ex Eye - Xenolith; The Anvil.mp3", "music/Metal & Rock/Folk Metal/Alestorm - Live At The End Of The World/10 - Alestorm - Wenches &Amp; Mead.mp3", "music/Metal & Rock/Symphonic Metal/Delain - Interlude/02 - Delain - Collars &Amp; Suits.mp3", "music/Modern & Soundtrack/Pop/Neo Magazin Royale - Live in Concert/08 - Baby Got Laugengeba&#x308;ck (Live).mp3", "music/Jazz & Klassik/Piano Collection/CD19 - Liszt/01 Waldesrauchen (Klára Würtz, piano; rec.2003).wav", "music/Jazz & Klassik/Piano Collection/CD19 - Liszt/06 Piano Sonata in B minor (Émil Gilels, piano; rec.1949).wav", "music/Jazz & Klassik/Piano Collection/CD4 - Brahms/04 Piano Sonata N°2 in F sharp minor, Op.2 - IV. Finale; Introduzione-allegro non troppo e rubato.wav", "music/Jazz & Klassik/Piano Collection/CD4 - Brahms/09 Piano Sonata N°3 in F minor, Op.5 - V. Finale; allegro moderato ma rubato.wav", "music/Jazz & Klassik/Mozart/CD2/06 String Quartet No. 14 in G Major ''Spring; Haydn Quartet No. 1'', KV 387 - 2. Menuetto- Allegro.wav", "music/Jazz & Klassik/Mozart/CD2/05 String Quartet No. 14 in G Major ''Spring; Haydn Quartet No. 1'', KV 387 - 1. Allegro vivace assai.wav", "music/Jazz & Klassik/Mozart/CD2/07 String Quartet No. 14 in G Major ''Spring; Haydn Quartet No. 1'', KV 387 - 3. Andante cantabile.wav", "music/Jazz & Klassik/Mozart/CD2/08 String Quartet No. 14 in G Major ''Spring; Haydn Quartet No. 1'', KV 387 - 4. Molto allegro.wav"]

right_names = ["music/Metal & Rock/Compilations/Relapse Records - Relapse Sampler 2017/18 Ex Eye - Xenolith The Anvil.mp3", "music/Metal & Rock/Folk Metal/Alestorm - Live At The End Of The World/10 - Alestorm - Wenches &Amp Mead.mp3", "music/Metal & Rock/Symphonic Metal/Delain - Interlude/02 - Delain - Collars &Amp Suits.mp3", "music/Modern & Soundtrack/Pop/Neo Magazin Royale - Live in Concert/08 - Baby Got Laugengeba&#x308ck (Live).mp3", "music/Jazz & Klassik/Piano Collection/CD19 - Liszt/01 Waldesrauchen (Klára Würtz, piano rec.2003).wav", "music/Jazz & Klassik/Piano Collection/CD19 - Liszt/06 Piano Sonata in B minor (Émil Gilels, piano rec.1949).wav", "music/Jazz & Klassik/Piano Collection/CD4 - Brahms/04 Piano Sonata N°2 in F sharp minor, Op.2 - IV. Finale Introduzione-allegro non troppo e rubato.wav", "music/Jazz & Klassik/Piano Collection/CD4 - Brahms/09 Piano Sonata N°3 in F minor, Op.5 - V. Finale allegro moderato ma rubato.wav", "music/Jazz & Klassik/Mozart/CD2/06 String Quartet No. 14 in G Major ''Spring Haydn Quartet No. 1'', KV 387 - 2. Menuetto- Allegro.wav", "music/Jazz & Klassik/Mozart/CD2/05 String Quartet No. 14 in G Major ''Spring Haydn Quartet No. 1'', KV 387 - 1. Allegro vivace assai.wav", "music/Jazz & Klassik/Mozart/CD2/07 String Quartet No. 14 in G Major ''Spring Haydn Quartet No. 1'', KV 387 - 3. Andante cantabile.wav", "music/Jazz & Klassik/Mozart/CD2/08 String Quartet No. 14 in G Major ''Spring Haydn Quartet No. 1'', KV 387 - 4. Molto allegro.wav"]

for myfile in filelist: 
    with open(str(PurePath(myfile)), 'r') as file :
        filedata = file.read()

    # Replace the target string
    index = 0
    for wrong in wrong_names:      
        filedata = filedata.replace(wrong, right_names[index])
        index = index + 1

    # Write the file out again
    with open(str(PurePath(myfile)), 'w') as file:
        file.write(filedata)

    print str(PurePath(myfile))
