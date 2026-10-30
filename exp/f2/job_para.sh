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



# batch=$1

list=/data/jobs/f2/jobs_pab/te1_R.list
list=/data/jobs/f2/jobs_pab/missing_files.txt
# list=/data/jobs/f5/jobs_${batch}_templ/all_seq_u.list
# list=/data1/jobs/f5/jobs_${1}_templ/rlxr.list
# out_dir=/data/jobs/f5/jobs_${batch}_templ
#d=/data/jobs/f2/jobs_pab
ddir=/share01/hpc/zhangxiaoxing/data/jobs_pab_te2
list=$ddir/te2.list

#### User Code Region 1 ###
# according to your computer
SEND_THREAD_NUM=40
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
np=12
#for chain in `head -2 $list`
#for chain in `cat $list`
for chain in `tac $list`
do
    read -u6
    {   
        # echo "hh "$chain
        
        j=$chain
        #out_dir=$d/$j
        #log=$out_dir/${j}_cf.log
        log=$ddir/$chain/o1
        err=$ddir/$chain/e1
        # python cryofold.py -m $map -s $seq -t $tem -d 1 --msa_local
        #if [ ! -f $pdb ]; then
        if [ $((i % $np)) -eq 0 ]; then
            srun -p a100_q1_high   -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        elif [ $((i % $np)) -eq 1 ]; then
            srun -p a100_q1_medium    -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        elif [ $((i % $np)) -eq 2 ]; then
            srun -p a100_q1_low -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        elif [ $((i % $np)) -eq 3 ]; then
            srun -p a800_q2_high -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        elif [ $((i % $np)) -eq 4 ]; then
            srun -p a800_q2_medium -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        elif [ $((i % $np)) -eq 5 ]; then
            srun -p a800_q2_low  -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        elif [ $((i % $np)) -eq 6 ]; then
            srun -p l40_q1_medium -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        elif [ $((i % $np)) -eq 7 ]; then
            srun -p l40_q1_low -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        elif [ $((i % $np)) -eq 8 ]; then
            srun -p l40_q2_high -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        elif [ $((i % $np)) -eq 9 ]; then
            srun -p l40_q2_medium -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        else
            srun -p l40_q2_low  -c 8 --gres=gpu:1 -J af1 -o $log -e $err exp/f2/job.sh $chain
        fi

        #    echo "processing $i $chain"
            # rm -rf $out_dir/msa
            #srun  exp/f2/job.sh $chain
            # python cryofold.py -m $map -s $seq -t $tem --msa_local
            # date >$log;CUDA_VISIBLE_DEVICES=0 python cryofold.py -m $map -s $seq -t $tem >> $log;date >> $log
            i=$((i+1))
        #fi

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




