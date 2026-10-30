#!/bin/bash

# ./cryofold  -m tests/8638/8638.mrc -s tests/8638/8638.fasta -t tests/8638/8638.cif
j=$1
b=241210
d=/data/jobs/f5/jobs_${b}_syn
out_dir=$d/$j

j0=${j}_2
map=$out_dir/${j0}.mrc
seq=$out_dir/${j}.fasta
tem=$out_dir/${j}.cif
log=$out_dir/${j0}.log
# python cryofold.py -m $map -s $seq -t $tem 

date >$log;
CUDA_VISIBLE_DEVICES=0 python cryofold.py -m $map -s $seq -t $tem --msa_local --msa_wo;
date >> $log


j0=${j}_3
map=$out_dir/${j0}.mrc
seq=$out_dir/${j}.fasta
tem=$out_dir/${j}.cif
log=$out_dir/${j0}.log
# python cryofold.py -m $map -s $seq -t $tem 

date >$log;
CUDA_VISIBLE_DEVICES=0 python cryofold.py -m $map -s $seq -t $tem --msa_local --msa_wo;
date >> $log


j0=${j}_4
map=$out_dir/${j0}.mrc
seq=$out_dir/${j}.fasta
tem=$out_dir/${j}.cif
log=$out_dir/${j0}.log
# python cryofold.py -m $map -s $seq -t $tem 

date >$log;
CUDA_VISIBLE_DEVICES=0 python cryofold.py -m $map -s $seq -t $tem --msa_local --msa_wo;
date >> $log



j0=${j}_5
map=$out_dir/${j0}.mrc
seq=$out_dir/${j}.fasta
tem=$out_dir/${j}.cif
log=$out_dir/${j0}.log
# python cryofold.py -m $map -s $seq -t $tem 

date >$log;
CUDA_VISIBLE_DEVICES=0 python cryofold.py -m $map -s $seq -t $tem --msa_local --msa_wo;
date >> $log



j0=${j}_6
map=$out_dir/${j0}.mrc
seq=$out_dir/${j}.fasta
tem=$out_dir/${j}.cif
log=$out_dir/${j0}.log
# python cryofold.py -m $map -s $seq -t $tem 

date >$log;
CUDA_VISIBLE_DEVICES=0 python cryofold.py -m $map -s $seq -t $tem --msa_local --msa_wo;
date >> $log

