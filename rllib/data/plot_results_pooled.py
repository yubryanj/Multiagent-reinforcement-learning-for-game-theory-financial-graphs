import numpy as np
import pandas as pd
import os

import sys
sys.path.insert(1, os.getcwd())

from trainer import setup
from utils import get_args

from plot_utils import plot_confusion_matrix, plot_table, plot_confusion_matrix_for_report
from itertools import combinations_with_replacement

if __name__ == "__main__":
    args = get_args()
    config, stop = setup(args)

    master_dataset = pd.read_csv(f'./data/checkpoints/{args.experiment_number}/experimental_data.csv')
    root_dir = f'./data/report_images/{args.experiment_number}'

    # Allocate Storage
    aggregated_statistics   = []
    policies                = pd.unique(master_dataset[['agent_0_policies','agent_1_policies']].values.ravel())

    for agent_0_policy, agent_1_policy in combinations_with_replacement(policies, 2):
        
        data = master_dataset[
            (master_dataset['agent_0_policies'] == agent_0_policy) &\
            (master_dataset['agent_1_policies'] == agent_1_policy) \
        ]

        agent_0_beta = data['agent 0 betas'].unique()[0]
        agent_1_beta = data['agent 1 betas'].unique()[0]

        # Generate the directories for saving results
        save_dir = f'{root_dir}/{agent_0_policy}-{agent_1_policy}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Plot Agent 0's confusion matrix
        plot_confusion_matrix_for_report(
            allocations     = data['agent 0 actions'], 
            rescue_amounts  = data['rescue_amount'], 
            save_dir        = save_dir,
            title           = f'Agent 0 Confusion Matrix - beta={agent_0_beta}',
            n_rows          = args.maximum_rescue_amount,
            n_cols          = args.maximum_rescue_amount
        )

        # Plot Agent 1's confusion matrix
        plot_confusion_matrix_for_report(
            allocations     = data['agent 1 actions'],
            rescue_amounts  = data['rescue_amount'],
            save_dir        = save_dir,
            title           = f'Agent 1 Confusion Matrix - beta={agent_1_beta}',
            n_rows          = args.maximum_rescue_amount,
            n_cols          = args.maximum_rescue_amount
        )

        # Prepare the data for the table plotting
        percentage_of_rescue_amount_if_saved    = []
        dominant_contributions                  = []
        non_dominant_contributions              = []
        percentage_saved_by_rescue_amount       = {}

        # Insert a key per rescue amount
        for rescue_amount in sorted(data['rescue_amount'].unique()):
            percentage_saved_by_rescue_amount[rescue_amount] = []


        number_of_samples = len(data)
        number_of_rescues = ((data['agent 0 actions'] + data['agent 1 actions']) >= data['rescue_amount']).sum()

        # Iterate through every row and calculate statistics
        for index, row in data.iterrows():
            agent_0_contribution = row['agent 0 actions']
            agent_1_contribution = row['agent 1 actions']
            rescue_amount = row['rescue_amount']
            total_contribution = agent_0_contribution + agent_1_contribution

            if  total_contribution >= rescue_amount and rescue_amount != 0:
                percentage_of_rescue_amount_if_saved.append( total_contribution / rescue_amount)
                percentage_saved_by_rescue_amount[rescue_amount].append(1.0)
            else:
                percentage_saved_by_rescue_amount[rescue_amount].append(0.0)

            
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

        # Prepare percentage saved by rescue amount
        percentage_saved_by_rescue_amount_table = []
        for rescue_amount in percentage_saved_by_rescue_amount.keys():
            percentage_saved_by_rescue_amount_table.append( 
                [f'percentage saved by rescue amount - {rescue_amount}',np.mean(percentage_saved_by_rescue_amount[rescue_amount])]
            )
        
        df = pd.DataFrame.from_records(percentage_saved_by_rescue_amount_table)
        df.columns = ["Percentage saved by rescue amount", "Percentage Saved"]

        df.to_csv(
            f'{save_dir}/percentage_saved_by_rescue_amount.csv', 
            index=False,
        ) 

        # Prepare statistics
        table_data = [
            ["Percentage Saved", f'{number_of_rescues/ number_of_samples}'],
            ["Average percentage of rescue amount if rescued", f'{np.mean(percentage_of_rescue_amount_if_saved)}'],
            ['Average Dominant Contribution', f'{np.mean(dominant_contributions)}'],
            ['Average Non-Dominant Contribution', f'{np.mean(non_dominant_contributions)}'],
            ['Agent 0 Beta', agent_0_beta],
            ['Agent 1 Beta', agent_1_beta],
        ]

        plot_table(
            data            = table_data,
            save_dir        = save_dir,
            title           = 'Statistics'
        )

        # Aggregate statistics
        for name, statistic in table_data:
            aggregated_statistics.append([
                f'{agent_0_policy} beta={agent_0_beta}-{agent_1_policy} beta={agent_1_beta}', 
                agent_0_policy,
                agent_1_policy,
                name, 
                statistic
            ])

        for name, statistic in percentage_saved_by_rescue_amount_table:
            aggregated_statistics.append([
            f'{agent_0_policy} beta={agent_0_beta}-{agent_1_policy} beta={agent_1_beta}', 
            agent_0_policy,
            agent_1_policy,
            name, 
            statistic
        ])


    df = pd.DataFrame.from_records(aggregated_statistics)
    df.columns = ["Agent 0 Policy - Agent 1 Policy", "agent 0 policy", "agent 1 policy", "Description", "Statistic"]

    df.to_csv(
        f'{root_dir}/aggregated_statistics.csv', 
        index=False,
    )  

    # Plot aggregated table
    plot_confusion_matrix_for_report(
        allocations     = master_dataset['agent 0 actions'], 
        rescue_amounts  = master_dataset['rescue_amount'], 
        save_dir        = root_dir,
        title           = 'Aggregated Agent 0 Confusion Matrix',
        n_rows          = args.maximum_rescue_amount,
        n_cols          = args.maximum_rescue_amount
    )

    plot_confusion_matrix_for_report(
        allocations     = master_dataset['agent 1 actions'], 
        rescue_amounts  = master_dataset['rescue_amount'], 
        save_dir        = root_dir,
        title           = 'Aggregated Agent 1 Confusion Matrix',
        n_rows          = args.maximum_rescue_amount,
        n_cols          = args.maximum_rescue_amount
    )
