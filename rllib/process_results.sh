#!/bin/bash
source /itet-stor/bryayu/net_scratch/conda/etc/profile.d/conda.sh
conda activate pku

POOLED_EXPERIMENTS=( 93 100 107 108 110 111 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 171 172 173 174 175 176 177 178)
EXPERIMENTS=( 86 94 99 101 106 149 150 151 152 159 160 163 164 165 166 169 170 181 182 183 184 185 188 190 191 192 193 194 195 196 197 198)


# Update the results dictionary containing paths to trained models
python populate_results_dictionary.py


# Evaluate pooled results
if ((${#POOLED_EXPERIMENTS[@]}));then
    for experiment_number in ${POOLED_EXPERIMENTS[@]};
    do
        sbatch --output=./data/process_results/$experiment_number.log ./scripts/process_pooled.sh $experiment_number;
    done
fi

# Evalute regular and uniform experiments
if ((${#EXPERIMENTS[@]}));then
    for experiment_number in ${EXPERIMENTS[@]};
    do
        sbatch --output=./data/process_results/$experiment_number.log ./scripts/process_experiment.sh $experiment_number;
    done
fi