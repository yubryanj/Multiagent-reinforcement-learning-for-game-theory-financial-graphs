import pandas as pd
import numpy as np
import os

from plot_utils import save_table

import sys
sys.path.insert(1, os.getcwd())

np.set_printoptions(precision=2)

if __name__=="__main__":
    
    uniformly_mixed_experiments = {
        'full_information':{
            'experiment_numbers':           [198,181,188,196],
            'columns_labels':               ['Scenario','NFI','FI','NFI','FI'],
            'row_labels':                   ['VD','CG','ND','NE','BB','BC']
        },
        'uniformly_mixed_prosocial':{
            'experiment_numbers':          [181,183,196,197],
            'columns_labels':               ['Scenario','NPS','PS','NPS','PS'],
            'row_labels':                   ['VD','CG','ND','NE','BB','BC']
        },
    }

    for experiment in uniformly_mixed_experiments.keys():

        save_dir = f'./data/tables/uniformly_mixed'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        # Storage for saving he percentage saved
        data = []
        
        for experiment_number in uniformly_mixed_experiments.get(experiment).get('experiment_numbers'):
            
            # Read the dataframe
            df = pd.read_csv(f'./data/checkpoints/{experiment_number}/subscenario_statistics.csv')
            
            # Extract the results
            percentage_saved = pd.to_numeric(df[df['Description'] == 'Percentage Saved']['Statistic'])\
                .round(2)\
                .tolist()    

            # Collect it in the aggregator
            data.append(percentage_saved)

        # Transpose the data so the pairings are in the rows and the pools are in column 
        save_table(
            data = np.array(data).T,
            row_labels = np.array(uniformly_mixed_experiments.get(experiment).get('row_labels')).reshape(-1,1),
            column_labels = uniformly_mixed_experiments.get(experiment).get('columns_labels'),
            save_dir = f'{save_dir}/{experiment}.txt'
        )


