from itertools import combinations_with_replacement
import pandas as pd
import numpy as np
import os


import sys
sys.path.insert(1, os.getcwd())

np.set_printoptions(precision=2)


if __name__=="__main__":
    


    observation_sets = {
        'no additional information':{
            'volunteers_dilemma_experiment_numbers':    [107,108,93,178],
            'coordination_game_experiment_numbers':     [110,111,100,174]
        },
        'reveal other agents prosociality':{
            'volunteers_dilemma_experiment_numbers':    [123,129,135,175],
            'coordination_game_experiment_numbers':     [120,126,132,172]
        },
        'reveal other agents identity':{
            'volunteers_dilemma_experiment_numbers':    [122,128,134,177],
            'coordination_game_experiment_numbers':     [119,125,131,171]
        },
        'reveal both':{
            'volunteers_dilemma_experiment_numbers':    [124,130,136,176],
            'coordination_game_experiment_numbers':     [121,127,133,173]
        }
    }

    ordering =  ['greedy', 'prosocial', 'mixed #1', 'mixed #2'] * 2 

    for observation_set in observation_sets.keys():
        
        # Storage for saving he percentage saved
        data = []

        # Insert the pairings
        pairings = [f'{i},{j}' for i, j in combinations_with_replacement(range(6),2)]
        data.append(pairings + ['Average'])
        
        for scenario in observation_sets.get(observation_set):

            for experiment_number in observation_sets.get(observation_set).get(scenario):
                
                # Read the dataframe
                df = pd.read_csv(f'./data/checkpoints/{experiment_number}/aggregated_statistics.csv')

                # Extract the percentage saved 
                percentage_saved = df[df['Description'] == 'Percentage Saved']['Statistic'].tolist()

                # Compute mean
                percentage_saved.append(np.mean(percentage_saved).round(2))

                # Collect it in the aggregator
                data.append(percentage_saved)

        # Transpose the data so the pairings are in the rows and the pools are in column 
        data = np.array(data).T

        # Prepare columns
        columns = ['Agent Pairing'] + ordering

        # Convert into dataframe
        tabular = pd.DataFrame(data, columns = columns)

        # Generate latex table from dataframe
        latex_table = tabular.to_latex(
            index=False,
            column_format='c'*data.shape[1],
        )

        file1 = open(f'./data/tables/pooled_experiment/{observation_set}.txt', 'w')
        file1.write(latex_table)
        file1.close()

        del data




