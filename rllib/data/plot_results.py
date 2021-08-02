import numpy as np
import seaborn as sns
import pandas as pd
import argparse
import os
sns.set_theme()

from plot_utils import plot_confusion_matrix, plot_table

import sys
sys.path.insert(1, os.getcwd())

from trainer import setup
from utils import get_args



if __name__ == "__main__":
    args = get_args()
    config, stop = setup(args)

    data = pd.read_csv(f'./data/checkpoints/{args.experiment_number}/experimental_data.csv')
    root_dir = f'./data/checkpoints/{args.experiment_number}'

    # Allocate Storage
    aggregated_statistics = []

    for sub_scenario in sorted(data['sub_scenarios'].unique()):

        # Generate the directories for saving results
        if sub_scenario != 'not applicable':
            save_dir = f'{root_dir}/{sub_scenario}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = root_dir

        subscenario_dataset = data[data['sub_scenarios'] == sub_scenario]

        # Plot Agent 0's confusion matrix
        plot_confusion_matrix(
            allocations     = subscenario_dataset['agent 0 actions'], 
            rescue_amounts  = subscenario_dataset['rescue_amount'], 
            save_dir        = save_dir,
            title           = 'Agent 0 Confusion Matrix',
            n_rows          = args.maximum_rescue_amount,
            n_cols          = args.maximum_rescue_amount
        )

        # Plot Agent 1's confusion matrix
        plot_confusion_matrix(
            allocations     = subscenario_dataset['agent 1 actions'],
            rescue_amounts  = subscenario_dataset['rescue_amount'],
            save_dir        = save_dir,
            title           = 'Agent 1 Confusion Matrix',
            n_rows          = args.maximum_rescue_amount,
            n_cols          = args.maximum_rescue_amount

        )

        # Prepare the data for the table plotting
        percentage_of_rescue_amount_if_saved    = []
        dominant_contributions                  = []
        non_dominant_contributions              = []

        number_of_rescues                       = 0
        scenario                                = data['scenario'].unique()[0]
        number_of_samples                       = len(subscenario_dataset)
        agent_0_contribution                    = subscenario_dataset['agent 0 actions']
        agent_1_contribution                    = subscenario_dataset['agent 1 actions']
        total_contribution                      = agent_0_contribution + agent_1_contribution
        rescue_amount                           = subscenario_dataset['rescue_amount']
        # number_of_rescues                       = ( total_contribution >= rescue_amount ).sum()

        # Iterate through every row and calculate statistics
        for index, row in subscenario_dataset.iterrows():
            agent_0_contribution = row['agent 0 actions']
            agent_1_contribution = row['agent 1 actions']
            rescue_amount = row['rescue_amount']
            total_contribution = agent_0_contribution + agent_1_contribution
    
            # Total_contribution has to be greater than the rescue amount
            if  sub_scenario != 'not in default':
                if total_contribution >= rescue_amount:
                    number_of_rescues += 1
            else:
                # In 'not in default' subscenario, the rescue amount is always 0
                # thus, no rescue occurs unless the agents allocate larger than 0 assets
                if total_contribution > rescue_amount:
                    number_of_rescues += 1


            # TODO: This number breaks in 'not in default" as the rescue amount is always 0
            if  total_contribution >= rescue_amount and rescue_amount != 0:
                percentage_of_rescue_amount_if_saved.append( total_contribution / rescue_amount)
            
            # Calculate contributions
            if total_contribution == 0:
                dominant_contributions.append(0.50)
                non_dominant_contributions.append(0.50)
            elif agent_1_contribution > agent_0_contribution:
                dominant_contributions.append(agent_1_contribution / total_contribution)
                non_dominant_contributions.append(agent_0_contribution / total_contribution)
            else:
                dominant_contributions.append(agent_0_contribution / total_contribution)
                non_dominant_contributions.append(agent_1_contribution / total_contribution)

        # Prepare statistics
        table_data = [
            ["Percentage Saved", f'{number_of_rescues/ number_of_samples}'],
            ["Average percentage of rescue amount if rescued", f'{np.mean(percentage_of_rescue_amount_if_saved)}'],
            ['Average Dominant Contribution', f'{np.mean(dominant_contributions)}'],
            ['Average Non-Dominant Contribution', f'{np.mean(non_dominant_contributions)}'],
            ['Scenario', f'{scenario}'],
            ['Sub scenario', f'{sub_scenario}'],
            ['Beta', args.beta],
        ]

        plot_table(
            data            = table_data,
            save_dir        = save_dir,
            title           = 'Statistics'
        )

        # Aggregate statistics
        for name, statistic in table_data:
            aggregated_statistics.append([sub_scenario, name, statistic])
    
    # Plot aggregated statistics for uniformly mixed
    if len(data['sub_scenarios'].unique()) > 1:
        plot_table(
            data        = aggregated_statistics,
            save_dir    = root_dir,
            title       = 'Aggregated Statistics'
        )

        df = pd.DataFrame.from_records(aggregated_statistics)
        df.columns = ["Sub Scenario", "Description", "Statistic"]

        df.to_csv(
            f'{root_dir}/aggregated_statistics.csv', 
            index=False,
        )  



