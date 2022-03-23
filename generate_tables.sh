#!/bin/bash
source /itet-stor/bryayu/net_scratch/conda/etc/profile.d/conda.sh
conda activate pku

python data/generate_experiment_tables.py;

python data/generate_uniformly_mixed_tables.py;

# python data/genrate_pooled_experiment_tables.py;

