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
mkdir -p $out_dir
# 获取中间两个字母
pdb_id=$j
middle_letters=${pdb_id:1:2}

# job_dir="jobs_pab/$pdb_id"
# job_file="$job_dir/$pdb_id.cif"
db_dir="/data/colabfolddb/pdb/divided/$middle_letters"
db_file="$db_dir/$pdb_id.cif.gz"

if [ ! -f "$job_file" ]; then
    echo "File already exists in job directory: $job_file"
    if [ ! -f "$db_file" ]; then
        echo "File not found in database: $db_file"
        exit 1
    fi

    # mkdir -p "$job_dir"

    cp "$db_file" "$out_dir"
    gunzip "$out_dir/$pdb_id.cif.gz"

fi

if [ ! -f "$map" ]; then
    exp/f1/dl_f1_pdb.sh $j
    cp /home/jsptgpu/data/db/emdb/${j}.mrc $map
fi

if [ ! -f "$seq" ]; then
    grep -A1 ">${j}"  /home/jsptgpu/data/db/pdb/pdb_seqres.txt >  $seq
fi



