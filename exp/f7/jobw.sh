#!/bin/bash

# ./cryofold  -m tests/8638/8638.mrc -s tests/8638/8638.fasta -t tests/8638/8638.cif
# b=$2


# if [ "$HOSTNAME" = "smtl01"  ]; then
#     d=/share01/hpc/zhangxiaoxing/data/f5/jobs_${b}_templ
# else
#     d=/home/jsptgpu/data/xukui/jobs/f5/jobs_${b}_templ

# fi
j=$1

d=/data1/jobs/$j
name=$(cat $d/name.list)
out_dir=$d/${name}
map=${out_dir}.mrc
seq=${out_dir}.fasta
tem=${out_dir}_aem0.pdb
out=${out_dir}_cf.pdb
log=${out_dir}.log
# python cryofold.py -m $map -s $seq -t $tem 
stg=$d/status

res=$(jq '.resolution' $stg)

date >$log;
CUDA_VISIBLE_DEVICES=0 python cryofold.py -m $map -s $seq -t $tem  --msa_wo;
date >> $log

