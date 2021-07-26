import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
sns.set_theme()

from plot_utils import plot_confusion_matrix, plot_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-number", type=int, default=000)
    args = parser.parse_args()

    data = pd.read_csv(f'./data/checkpoints/{args.experiment_number}/experimental_data.csv')


    root_dir = f'./data/checkpoints/{args.experiment_number}'

    plot_confusion_matrix(
        allocations     = data['agent 0 actions'], 
        rescue_amounts  = data['rescue_amount'], 
        save_dir        = root_dir,
        title           = 'Agent 0 Confusion Matrix',
    )

    plot_confusion_matrix(
        allocations     = data['agent 1 actions'],
        rescue_amounts  = data['rescue_amount'],
        save_dir        = root_dir,
        title           = 'Agent 1 Confusion Matrix'
    )

    # Prepare the data for the table plotting
    percentage_of_rescue_amount_if_saved    = []
    dominant_contributions                  = []
    non_dominant_contributions              = []

    number_of_samples = len(data)
    number_of_rescues = ((data['agent 0 actions'] + data['agent 1 actions']) >= data['rescue_amount']).sum()


    for index, row in data.iterrows():
        agent_0_contribution = row['agent 0 actions']
        agent_1_contribution = row['agent 1 actions']
        rescue_amount = row['rescue_amount']
        total_contribution = agent_0_contribution + agent_1_contribution

        if total_contribution >= rescue_amount:
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


    table_data = [
        ["Percentage Saved", f'{number_of_rescues/ number_of_samples}'],
        ["Average percentage of rescue amount if rescued", f'{np.mean(percentage_of_rescue_amount_if_saved)}'],
        ['Average Dominant Contribution', f'{np.mean(dominant_contributions)}'],
        ['Average Non-Dominant Contribution', f'{np.mean(non_dominant_contributions)}']
    ]

    plot_table(
        data            = table_data,
        save_dir        = root_dir,
        title           = 'Statistics'
    )
