#!/bin/bash

source /itet-stor/bryayu/net_scratch/conda/etc/profile.d/conda.sh
conda activate pku

GB32_EXPERIMENTS=()
GB128_EXPERIMENTS=(149 166 152 165 150 169 151 170 190 160 191 159 192 163 193 164)
GB128_EXPERIMENTS_POOLED=()


# Non-Pooled training
if ((${#GB32_EXPERIMENTS[@]}));
then
    for i in ${GB32_EXPERIMENTS[@]};
    do
        echo "sbatch run.sh $i";
        sbatch --output=./data/slurm_outputs/$i.log ./scripts/run.sh $i;
    done
fi


# Non-Pooled training
if ((${#GB128_EXPERIMENTS[@]})); 
then
    for i in ${GB128_EXPERIMENTS[@]};
    do
        echo "sbatch --mem=128G run.sh $i";
        sbatch --output=./data/slurm_outputs/$i.log --mem=128G ./scripts/run.sh $i;
    done
fi

if ((${#GB128_EXPERIMENTS_POOLED[@]}));
then
    for i in ${GB128_EXPERIMENTS_POOLED[@]};
    do
        echo "sbatch --mem=128G run_pooled.sh $j";
        sbatch --output=./data/slurm_outputs/$j.log --mem=128G ./scripts/run_pooled.sh $j;
    done
fi

echo "All jobs submitted";