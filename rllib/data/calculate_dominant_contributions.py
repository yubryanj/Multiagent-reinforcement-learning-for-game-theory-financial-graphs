import pandas as pd
import os
from plot_utils import line_plot_with_variances

if __name__ == "__main__":

    # Define the games and their associated experiments
    dict = {
        # 'Volunteers Dilemma - Full Information':{
        'VD-FI':{
            'experiment_numbers':[101,102,103,104,105,106]
        },
        # 'Volunteers Dilemma':{
        'VD':{
            'experiment_numbers':[153,154,155,156,157,158]
        },
        # 'Coordination Game - Full Information':{
        'CG-FI':{
            'experiment_numbers':[94,95,96,97,98,99]
        },
        # 'Coordination Game':{
        'CG':{
            'experiment_numbers':[81,82,83,84,85,86]
        },
    }

    # Pre-allocate storage to collect results
    scenarios                               = []
    dominant_contributions                  = []
    percentage_of_rescue_amount             = []
    total_as_percentage_of_rescue_amount    = []
    successful_rescues                      = []
    experiment_betas                        = []

    # Iterate through each Game
    for scenario in dict.keys():
    
        # Iterate across the betas in the games
        for experiment_number in dict.get(scenario).get('experiment_numbers'):

            # Load the dataset
            df = pd.read_csv(f'./data/checkpoints/{experiment_number}/experimental_data.csv')

            # Iterate through the rows in the dataset and collect results
            for index, row in df.iterrows():
                agent_0_contribution = row['agent 0 actions']
                agent_1_contribution = row['agent 1 actions']
                rescue_amount = row['rescue_amount']
                total_contribution = agent_0_contribution + agent_1_contribution

                # Statistic: Calculate contributions
                if agent_1_contribution > agent_0_contribution:
                    if total_contribution != 0:
                        dominant_contributions.append(agent_1_contribution / total_contribution)
                    else:
                        dominant_contributions.append(0)

                    percentage_of_rescue_amount.append(agent_1_contribution/ rescue_amount)
                else:
                    if total_contribution != 0:
                        dominant_contributions.append(agent_0_contribution / total_contribution)
                    else:
                        dominant_contributions.append(0)
                    
                    percentage_of_rescue_amount.append(agent_0_contribution/ rescue_amount)
                
                # Statistic: Calculate betas
                experiment_betas.append(row['beta'])

                # Statistics: Calculate the rescue percentage
                if total_contribution >= rescue_amount: 
                    successful_rescues.append(1.0)
                else:
                    successful_rescues.append(0.0)
                
                total_as_percentage_of_rescue_amount.append(total_contribution / rescue_amount)

                # Statistic: Store the scenario
                scenarios.append(scenario)
            
    # Collect the statistics
    master_data = {
        'Scenario': scenarios,
        'Beta': experiment_betas,
        'Dominant Contributions': dominant_contributions,
        'Percentage of Rescue Amount': percentage_of_rescue_amount,
        'Successful Rescues': successful_rescues,
        'Total Contribution': total_as_percentage_of_rescue_amount,
        }        

    # Prepare directory to save results
    root_dir = f'./data/dominant_contributions'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Convert and save as CSV
    df = pd.DataFrame(data=master_data)
    df.to_csv(
        f'{root_dir}/contribution_dataset.csv', 
        index=False,
    )  

    # generate plots
    line_plot_with_variances(
        data = master_data,
        save_dir = root_dir
    )

    for scenario in df['Scenario'].unique():

        # Prepare directory to save subgame results
        save_dir = f'./data/dominant_contributions/{scenario}'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        # Filter the master dataset to subdataset
        sub_dataset = df[df['Scenario'] == scenario]

        sub_dataset.to_csv(
            f'{save_dir}/contribution_dataset.csv', 
            index=False,
        )  

        # generate plots
        line_plot_with_variances(
            data = sub_dataset,
            save_dir = save_dir
        )
