#!/bin/bash
find fma_full -type f -size +25M -exec mv -nv {} /home/ja62lel/fma/large/ \;


find private -type f -size +25M -exec mv -nv {} /home/ja62lel/private/large/ \;

