#!/bin/bash
source /itet-stor/bryayu/net_scratch/conda/etc/profile.d/conda.sh
conda activate pku

python trainer.py \
    --experiment-number $1
