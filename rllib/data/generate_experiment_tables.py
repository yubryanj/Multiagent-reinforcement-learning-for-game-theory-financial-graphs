import pandas as pd
import numpy as np
import os

from plot_utils import save_table

import sys
sys.path.insert(1, os.getcwd())

np.set_printoptions(precision=2)

if __name__=="__main__":
    


    observation_sets = {
        'building_the_baseline':{
            'volunteers dilemma':       [185,158],
            'coordination game':        [184,86],
            'not in default':           [149,166],
            'not enough money together':[152,165],
            'only bank b can rescue':   [150,169],
            'only bank c can rescue':   [151,170],
            'columns_labels':           ['Scenario','baseline','embedded'],
            'row_labels':               ['VD','CG','ND','NE','BB','BC']
        },
        'full_information':{
            'volunteers dilemma':       [185,185,158,106],
            'coordination game':        [184,182,86,99],
            'not in default':           [149,190,166,160],
            'not enough money together':[152,191,165,159], 
            'only bank b can rescue':   [150,192,169,163], 
            'only bank c can rescue':   [151,193,170,164], 
            'columns_labels':           ['Scenario','NFI','FI','NFI','FI'],
            'row_labels':               ['VD','CG','ND','NE','BB','BC']
        },
        'pro_sociality':{
            'volunteers dilemma':       [185,194,106,101],
            'coordination game':        [182,195,99,94], 
            'columns_labels':           ['Scenario','NPS','PS','NPS','PS'],
            'row_labels':               ['VD','CG']
        },
    }

    scenarios = [
        'volunteers dilemma',
        'coordination game',
        'not in default',
        'not enough money together',
        'only bank b can rescue',
        'only bank c can rescue'
        ]

    for observation_set in observation_sets.keys():

        save_dir = f'./data/tables/experiments'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Storage for saving he percentage saved
        data = []
        
        for scenario in observation_sets.get(observation_set):

            if scenario in scenarios:
                
                aggregated_percentage_saved = []

                for experiment_number in observation_sets.get(observation_set).get(scenario):
                    
                    # Read the dataframe
                    df = pd.read_csv(f'./data/checkpoints/{experiment_number}/statistics.csv')

                    # Extract the percentage saved 
                    percentage_saved = pd.to_numeric(df[df['Description'] == 'Percentage Saved']['Statistic'])[0]

                    # Aggregate percentage saved
                    aggregated_percentage_saved.append(percentage_saved.round(2))

                # Collect it in the aggregator
                data.append(aggregated_percentage_saved)

        # Transpose the data so the pairings are in the rows and the pools are in column 
        save_table(
            data = np.array(data),
            row_labels = np.array(observation_sets.get(observation_set).get('row_labels')).reshape(-1,1),
            column_labels = observation_sets.get(observation_set).get('columns_labels'),
            save_dir = f'{save_dir}/{observation_set}.txt',
        )

        del data