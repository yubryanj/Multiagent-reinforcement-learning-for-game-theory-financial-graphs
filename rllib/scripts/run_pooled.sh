#!/bin/bash
source /itet-stor/bryayu/net_scratch/conda/etc/profile.d/conda.sh
conda activate pku

python trainer_pooled.py \
    --local-mode \
    --experiment-number $1