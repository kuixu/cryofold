#!/bin/bash

# ./cryofold  -m tests/8638/8638.mrc -s tests/8638/8638.fasta -t tests/8638/8638.cif
d=$1
# b=$2


# if [ "$HOSTNAME" = "smtl01"  ]; then
#     d=/share01/hpc/zhangxiaoxing/data/f5/jobs_${b}_templ
# else
#     d=/home/jsptgpu/data/xukui/jobs/f5/jobs_${b}_templ

# fi

out_dir=$d
map=${out_dir}.mrc
seq=${out_dir}.fasta
tem=${out_dir}.cif
log=${out_dir}.log
# python cryofold.py -m $map -s $seq -t $tem 

date >$log;
CUDA_VISIBLE_DEVICES=0 python cryofold.py -m $map -s $seq -t $tem --msa_wo;
date >> $log

