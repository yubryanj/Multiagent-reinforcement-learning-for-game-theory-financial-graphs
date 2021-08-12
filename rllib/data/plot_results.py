import numpy as np
import seaborn as sns
import pandas as pd
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
    subscenario_statistics = []

    aggregated_percentage_of_rescue_amount_if_saved    = []
    aggregate_percentage_saved_by_rescue_amount        = []
    aggregated_dominant_contributions                  = []
    aggregated_non_dominant_contributions              = []
    aggregated_number_of_rescues                       = 0
    total_number_of_samples                            = len(data)

    

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

        # Plot confusion matrices per trial

        for trial in sorted(data['trials'].unique()):

            trial_subdataset = data[data['trials'] == trial]
            trial_save_dir = f'{save_dir}/{trial}'
            if not os.path.exists(trial_save_dir):
                os.makedirs(trial_save_dir)

            # Plot Agent 0's confusion matrix
            plot_confusion_matrix(
                allocations     = trial_subdataset['agent 0 actions'], 
                rescue_amounts  = trial_subdataset['rescue_amount'], 
                save_dir        = trial_save_dir,
                title           = 'Agent 0 Confusion Matrix',
                n_rows          = args.maximum_rescue_amount,
                n_cols          = args.maximum_rescue_amount
            )

            # Plot Agent 1's confusion matrix
            plot_confusion_matrix(
                allocations     = trial_subdataset['agent 1 actions'],
                rescue_amounts  = trial_subdataset['rescue_amount'],
                save_dir        = trial_save_dir,
                title           = 'Agent 1 Confusion Matrix',
                n_rows          = args.maximum_rescue_amount,
                n_cols          = args.maximum_rescue_amount

            )



        # Prepare the data for the table plotting
        percentage_of_rescue_amount_if_saved    = []
        dominant_contributions                  = []
        non_dominant_contributions              = []
        percentage_saved_by_rescue_amount       = {}

        # Insert a key per rescue amount
        for rescue_amount in sorted(subscenario_dataset['rescue_amount'].unique()):
            percentage_saved_by_rescue_amount[rescue_amount] = []

        number_of_rescues                       = 0
        scenario                                = data['scenario'].unique()[0]
        number_of_samples                       = len(subscenario_dataset)
        agent_0_contribution                    = subscenario_dataset['agent 0 actions']
        agent_1_contribution                    = subscenario_dataset['agent 1 actions']
        total_contribution                      = agent_0_contribution + agent_1_contribution
        

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

            pass    


        # Prepare percentage saved by rescue amount
        percentage_saved_by_rescue_amount_table = []
        for rescue_amount in percentage_saved_by_rescue_amount.keys():
            percentage_saved_by_rescue_amount_table.append( 
                [f'{sub_scenario}-{rescue_amount}',np.mean(percentage_saved_by_rescue_amount[rescue_amount])]
            )

        df = pd.DataFrame.from_records(percentage_saved_by_rescue_amount_table)
        df.columns = ["Sub scenario - rescue amount", "Percentage Saved"]

        df.to_csv(
            f'{save_dir}/percentage_saved_by_rescue_amount.csv', 
            index=False,
        ) 


        # Append to aggregated dataset
        aggregated_number_of_rescues += number_of_rescues
        aggregated_dominant_contributions.extend(dominant_contributions)
        aggregated_non_dominant_contributions.extend(non_dominant_contributions)
        aggregated_percentage_of_rescue_amount_if_saved.extend(percentage_of_rescue_amount_if_saved)


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
        
        df = pd.DataFrame.from_records(table_data)
        df.columns = ["Description", "Statistic"]

        df.to_csv(
            f'{save_dir}/statistics.csv', 
            index=False,
        ) 


        # Aggregate statistics
        for name, statistic in table_data:
            subscenario_statistics.append([sub_scenario, name, statistic])


        
    
    if len(data['sub_scenarios'].unique()) > 1:


        # Svae sub scenario statistics into a csv
        df = pd.DataFrame.from_records(subscenario_statistics)
        df.columns = ["Sub Scenario", "Description", "Statistic"]

        df.to_csv(
            f'{root_dir}/subscenario_statistics.csv', 
            index=False,
        ) 

        # Prepare, plot, and save aggregated statistics
        aggregated_table_data = [
            ["Aggregated Percentage Saved", f'{aggregated_number_of_rescues/ total_number_of_samples}'],
            ["Aggregated Average percentage of rescue amount if rescued", f'{np.mean(aggregated_percentage_of_rescue_amount_if_saved)}'],
            ['Aggregated Average Dominant Contribution', f'{np.mean(aggregated_dominant_contributions)}'],
            ['Aggregated Average Non-Dominant Contribution', f'{np.mean(aggregated_non_dominant_contributions)}'],
            ['Scenario', f'{scenario}'],
            ['Beta', args.beta],
        ]

        df = pd.DataFrame.from_records(aggregated_table_data)
        df.columns = ["Description", "Statistic"]

        df.to_csv(
            f'{root_dir}/aggregated_statistics.csv', 
            index=False,
        ) 


 



