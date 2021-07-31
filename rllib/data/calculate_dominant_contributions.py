import pandas as pd
import os
from plot_utils import line_plot_with_variances

if __name__ == "__main__":

    # Define the games and their associated experiments
    dict = {
        'Volunteers Dilemma':{
            # TODO: List is longer
            'experiment_numbers':[81,82,83,84,85]
        },
        'Coordination Game':{
            'experiment_numbers':[94,95,96,97,98,99]
        }
    }

    # Pre-allocate storage to collect results
    dominant_contributions          = []
    percentage_of_rescue_amount     = []
    successful_rescues              = []
    experiment_betas                = []

    # Iterate through each Game
    for game in dict.keys():

        # Iterate across the betas in the games
        for experiment_number in dict.get(game).get('experiment_numbers'):

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
            
            # Collect the statistics
            data = {
                'Beta': experiment_betas,
                'Dominant Contributions': dominant_contributions,
                'Percentage of Rescue Amount': percentage_of_rescue_amount,
                'Successful Rescues': successful_rescues
                }        

            # Prepare directory to save results
            root_dir = f'./data/dominant_contributions'
            save_dir = f'./data/dominant_contributions/{game}'
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
          
            # Convert and save as CSV
            df = pd.DataFrame(data=data)
            df.to_csv(
                f'./data/dominant_contributions/contribution_dataset.csv', 
                index=False,
            )  

            # generate plots
            line_plot_with_variances(
                data = data,
                save_dir = save_dir
            )