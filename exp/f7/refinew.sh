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
log=${out_dir}_cf_refine.log
tim=${out_dir}.tim
inp=${out_dir}_cf.pdb
out=${out_dir}_cf_refine.pdb
out2=${out_dir}_cryofold.pdb
stg=$d/status

vcx1=${out_dir}_cf_refine.vcx
vcx2=${out_dir}_cryofold.vcx

res=$(jq '.resolution' $stg)

# python cryofold.py -m $map -s $seq -t $tem 
if [ -f $inp ]; then
    date >$tim;
    # CUDA_VISIBLE_DEVICES=0 python cryofold.py -m $map -s $seq -t $tem --msa_wo;
    python exp/f5/post_refine.py $inp $map $res > $log
    cp $out $out2
    cp $vcx1 $vcx2
    # python exp/f5/vcx2json2.py $vcx2 $seq $stg
    python exp/f5/vcx2json2.py $vcx2 $stg
    date >> $tim
fi

