#!/bin/bash

# ./cryofold  -m tests/8638/8638.mrc -s tests/8638/8638.fasta -t tests/8638/8638.cif
j=$1
# b=$2
d=/data/jobs/f1/jobs_pab
out_dir=$d/$j
map=$out_dir/${j}.mrc
seq=$out_dir/${j}.fasta
tem=$out_dir/${j}.cif
log=$out_dir/${j}.log
# python cryofold.py -m $map -s $seq -t $tem 


if [ ! -f "$tem" ]; then
    exp/f1/dcif.sh $j

fi




date >$log;
python cryofold.py -m $map -s $seq -t $tem --msa_local --msa_wo;
date >> $log

