#!/bin/bash

# ./cryofold  -m tests/8638/8638.mrc -s tests/8638/8638.fasta -t tests/8638/8638.cif
j=$1
# b=$2
j1="${j:0:4}"
d=/data/jobs/f2/jobs_pab_te2
d1=/share01/hpc/zhangxiaoxing/data/jobs_pab
d2=/share01/hpc/zhangxiaoxing/data/jobs_pab_te2
out_dir=$d2/$j
out_dir1=$d1/$j1
map=$out_dir/${j1}.mrc
seq=$out_dir1/${j1}.fasta
tem=$out_dir1/${j1}.cif
log=$out_dir/${j1}.log
# python cryofold.py -m $map -s $seq -t $tem 


#if [ ! -f "$tem" ]; then
#    exp/f1/dcif.sh $j

#fi




date >$log;
python run_af3_d.py -m $map -s $seq -t $tem --msa_local --msa_wo;
date >> $log

