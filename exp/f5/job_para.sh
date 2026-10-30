#!/bin/bash

# ./cryofold  -m tests/8638/8638.mrc -s tests/8638/8638.fasta -t tests/8638/8638.cif

#!/bin/bash
# Copyright (c) 2016, BarryKui.
# Email: xukui.cs@gmail.com
#
# bash profile configure
# Tsinghua University
# Modified at Aug 9 2016
#
# export PYTHONPATH=/home/qinghuadaxuezhangqiangfengketizu/xukui/exp/tmp/model_angelo:$PYTHONPATH
# export TORCH_HOME=/home/qinghuadaxuezhangqiangfengketizu/xukui/data/model_angelo

# ddir=/data1/jobs/f5/240904
# list=/data1/jobs/f5/240904/240904.list
# list=/data1/jobs/f5/jobs_240904_templ/240904_1.list
# odir=/data1/jobs/f5/jobs_240904_templ



batch=$1

list=/data/jobs/f5/jobs_${batch}_templ/all_seq.list
# list=/data/jobs/f5/jobs_${batch}_templ/all_seq_u.list
# list=/data1/jobs/f5/jobs_${1}_templ/rlxr.list
# out_dir=/data/jobs/f5/jobs_${batch}_templ
d=/data/jobs/f5/jobs_${batch}_templ


#### User Code Region 1 ###
# according to your computer
SEND_THREAD_NUM=3
###########################

tmp_fifofile="/tmp/$$.fifo"
mkfifo "$tmp_fifofile"
exec 6<>"$tmp_fifofile"
for ((i=0;i<$SEND_THREAD_NUM;i++));do
    echo                                                                                    
done >&6

# rm $log

i=1
# j=1

for chain in `cat $list`
do
    read -u6
    {   
        # echo "hh "$chain
        
        j=$chain
        out_dir=$d/$j
        map=$out_dir/${j}.mrc
        seq=$out_dir/${j}.fasta
        tem=$out_dir/${j}.cif
        pdb=$out_dir/${j}_cf.pdb
        log=$out_dir/${j}_cf.log
        # python cryofold.py -m $map -s $seq -t $tem -d 1 --msa_local
        if [ ! -f $pdb ]; then
            echo "processing $i $chain"
            rm -rf $out_dir/msa
            # sh exp/f5/job.sh $chain $batch
            sh exp/f5/job_nm.sh $chain $batch
            i=$((i+1))
        fi

        # &2>1 >  $out_dir/$chain/${chain}.o2
        sleep 1
        echo >&6
    } &
    pid=$!
    
    
    
done


wait
exec 6>&-
# tail -f $logfile
exit 0




