#!/bin/bash

# ./cryofold  -m tests/8638/8638.mrc -s tests/8638/8638.fasta -t tests/8638/8638.cif
j=$1
b=$2
d=/home/jsptgpu/data/xukui/data/result/bin${b}/

d=/data/jobs/f5/jobs_${b}_templ
out_dir=$d/$j
map=$out_dir/${j}.mrc
seq=$out_dir/${j}.fasta
tem=$out_dir/${j}.cif
log=$out_dir/${j}.log
# python cryofold.py -m $map -s $seq -t $tem 

date >$log;
CUDA_VISIBLE_DEVICES=0 python cryofold.py -m $map -s $seq -t $tem --msa_wo;
date >> $log

