#!/bin/bash
# list=/data1/jobs/f5/all.list
# pdir=/data1/jobs/f5/vpd

pdir=/data/jobs/f1/jobs_pab
list=$pdir/te1.list



# for p in `head -2 $list`
for p in `cat $list`
do
   
   python  tools/map_model_cc_mp_stat.py $pdir/${p}/${p}_cf_rf.vc _cf_rf.vc

done
