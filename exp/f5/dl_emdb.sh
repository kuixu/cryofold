#!/bin/bash

# ./cryofold  -m tests/8638/8638.mrc -s tests/8638/8638.fasta -t tests/8638/8638.cif
j=$1
b=$2
d=/data/jobs/f5/jobs_24${b}_templ
out_dir=$d/$j
mapz=$out_dir/${j}.map.gz 
map=$out_dir/${j}.map
mrc=$out_dir/${j}.mrc
# seq=$out_dir/${j}.fasta
# tem=$out_dir/${j}.cif
# log=$out_dir/${j}.dlog
# python cryofold.py -m $map -s $seq -t $tem 
wget https://files.wwpdb.org/pub/emdb/structures/EMD-${j}/map/emd_${j}.map.gz -O $mapz
gunzip $mapz
mv $map $mrc