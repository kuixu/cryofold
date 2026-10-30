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



batch=241210

list=/data/jobs/f5/jobs_${batch}_syn/all0.list
# list=/data/jobs/f5/jobs_${batch}_templ/all_seq_u.list
# list=/data1/jobs/f5/jobs_${1}_templ/rlxr.list
# out_dir=/data/jobs/f5/jobs_${batch}_templ
d=/data/jobs/f5/jobs_${batch}_syn


#### User Code Region 1 ###
# according to your computer
SEND_THREAD_NUM=5
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

for chain in `tac $list`
do
    read -u6
    {   
        # echo "hh "$chain
        
        j=$chain
        out_dir=$d/$j
        
        seq=$out_dir/${j}.fasta
        tem=$out_dir/${j}.cif
        
        # python cryofold.py -m $map -s $seq -t $tem -d 1 --msa_local

        

        r=2
        map=$out_dir/${j}_${r}.mrc
        pdb=$out_dir/${j}_${r}_cf.pdb
        if [ ! -f $pdb ]; then
            echo "processing $i"
            # python cryofold.py -m $map -s $seq -t $tem --msa_local
            cmd="python cryofold.py -m $map -s $seq -t $tem -d 1 "
            echo $cmd
            eval $cmd
            i=$((i+1))
        fi

        r=3
        map=$out_dir/${j}_${r}.mrc
        pdb=$out_dir/${j}_${r}_cf.pdb
        if [ ! -f $pdb ]; then
            echo "processing $i"
            # python cryofold.py -m $map -s $seq -t $tem --msa_local
            python cryofold.py -m $map -s $seq -t $tem -d 1
            i=$((i+1))
        fi

        r=4
        map=$out_dir/${j}_${r}.mrc
        pdb=$out_dir/${j}_${r}_cf.pdb
        if [ ! -f $pdb ]; then
            echo "processing $i"
            # python cryofold.py -m $map -s $seq -t $tem --msa_local
            python cryofold.py -m $map -s $seq -t $tem -d 1
            i=$((i+1))
        fi

        r=5
        map=$out_dir/${j}_${r}.mrc
        pdb=$out_dir/${j}_${r}_cf.pdb
        if [ ! -f $pdb ]; then
            echo "processing $i"
            # python cryofold.py -m $map -s $seq -t $tem --msa_local
            python cryofold.py -m $map -s $seq -t $tem -d 1 
            i=$((i+1))
        fi

        r=6
        map=$out_dir/${j}_${r}.mrc
        pdb=$out_dir/${j}_${r}_cf.pdb
        if [ ! -f $pdb ]; then
            echo "processing $i"
            # python cryofold.py -m $map -s $seq -t $tem --msa_local
            python cryofold.py -m $map -s $seq -t $tem -d 1
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




