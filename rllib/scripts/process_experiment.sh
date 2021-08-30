#!/bin/bash
source /itet-stor/bryayu/net_scratch/conda/etc/profile.d/conda.sh
conda activate pku


# echo "python evaluator.py --experiment-number $1";
# python evaluator.py --experiment-number $1;

echo "python data/plot_results.py --experiment-number $1";
python data/plot_results.py --experiment-number $1;