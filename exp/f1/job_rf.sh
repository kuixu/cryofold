#!/bin/bash
source ~/usr/phenix-1.20.1-4487/phenix_env.sh
# ./cryofold  -m tests/8638/8638.mrc -s tests/8638/8638.fasta -t tests/8638/8638.cif
j=$1
# b=$2
d=/data/jobs/f1/jobs_pab
out_dir=$d/$j
map=$out_dir/${j}.mrc
seq=$out_dir/${j}.fasta
tem=$out_dir/${j}.cif
cfr=$out_dir/${j}_cf.pdb
log=$out_dir/${j}.rsf
vpr=$out_dir/${j}_cf.vpd
cfrf0=$out_dir/${j}_cf_real_space_refined_000.pdb
cfrf=$out_dir/${j}_cfrf.pdb
vprf=$out_dir/${j}_cfrf.vpd

# python cryofold.py -m $map -s $seq -t $tem 

# 获取中间两个字母
# pdb_id=$j
# middle_letters=${pdb_id:1:2}

# job_dir="jobs_pab/$pdb_id"
# job_file="$job_dir/$pdb_id.cif"
# db_dir="/data/colabfolddb/pdb/divided/$middle_letters"
# db_file="$db_dir/$pdb_id.cif.gz"

# if [ ! -f "$job_file" ]; then
#     echo "File already exists in job directory: $job_file"
#     if [ ! -f "$db_file" ]; then
#         echo "File not found in database: $db_file"
#         exit 1
#     fi

#     mkdir -p "$job_dir"

#     cp "$db_file" "$out_dir"
#     gunzip "$out_dir/$pdb_id.cif.gz"

# fi

r=$(grep -i $j exp/emdb_list_info.csv|head -1|awk -F' ' '{print($4)}')
echo $j $r


date >$vpr;
phenix.validation_cryoem $cfr $map resolution=$r >> $vpr
date >> $vpr


cd $out_dir
date >$log;
/home/zhangqflab/exp/cryofold_dev/misc/real_refine_xz_v1.sh $cfr $map $r >> $log
date >> $log
mv  $cfrf0 $cfrf

date >$vprf;
phenix.validation_cryoem $cfrf $map resolution=$r >> $vprf
date >> $vprf

