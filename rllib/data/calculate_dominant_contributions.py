import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme()

from plot_utils import line_plot_with_variances



if __name__ == "__main__":
    # Volunteers Dilemma
    # experiment_numbers = [81,82,83,84,85,86]

    # Coordination Game
    experiment_numbers = [94,95,96,97,98,99]

    dominant_contributions = []
    percentage_of_rescue_amount = []
    successful_rescues = []
    experiment_betas = []

    for experiment_number in experiment_numbers:
        root_dir = f'./data/checkpoints/{experiment_number}'

        df = pd.read_csv(f'{root_dir}/experimental_data.csv')

        for index, row in df.iterrows():
            agent_0_contribution = row['agent 0 actions']
            agent_1_contribution = row['agent 1 actions']
            rescue_amount = row['rescue_amount']
            total_contribution = agent_0_contribution + agent_1_contribution

            # Calculate contributions
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
            
            # Calculate betas
            experiment_betas.append(row['beta'])

            # Calculate the rescue percentage
            if total_contribution >= rescue_amount: 
                successful_rescues.append(1.0)
            else:
                successful_rescues.append(0.0)
        
        data = {
            'beta': experiment_betas,
            'dominant_contributions': dominant_contributions,
            'percentage_of_rescue_amount': percentage_of_rescue_amount,
            'successful_rescues': successful_rescues
            }        
        
        df = pd.DataFrame(data=data)
        df.to_csv(
            f'./data/contribution_dataset.csv', 
            index=False,
        )  

        line_plot_with_variances(
            data = data,
            save_dir = f'./data/dominant_contributions'
        )