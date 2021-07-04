import ray
import json
from utils import get_args
from trainer import setup
from ray.rllib.agents.dqn import DQNTrainer
from env import Volunteers_Dilemma

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot(
    actual_allocations, 
    optimal_allocations, 
    save_dir,
    title="Single Agent allocation; 1e3 eval, 1e6 training episodes",
    maximum_allocation = 6,):

    n_rows = maximum_allocation + 1
    n_cols = 4 # Starts at 3...6
    n_cells = n_rows * n_cols
    confusion_matrix = np.zeros((n_rows, n_cols))
    for actual,optimal in zip(actual_allocations, optimal_allocations):
        if actual >=0 and actual <=6:
            confusion_matrix[int(actual),int(optimal-3)] += 1

    print(confusion_matrix)

    df_cm = pd.DataFrame(confusion_matrix, index = [i for i in range(n_rows)],
                      columns = [i for i in range(3, 3 + n_cols)])

    sn.set(font_scale=1.25)  # crazy big

    fig = plt.figure(figsize = (20,15))
    plt.title(title)
    ax1 = plt.subplot2grid((20,20), (0,0), colspan=17, rowspan=17)
    ax2 = plt.subplot2grid((20,20), (17,0), colspan=17, rowspan=2)
    ax3 = plt.subplot2grid((20,20), (0,17), colspan=2, rowspan=17)

    sn.heatmap(df_cm, ax=ax1, annot=True, fmt='g', cmap="Blues", cbar=False)
    
    ax1.xaxis.tick_top()
    ax1.set(xlabel='Rescue Amount', ylabel='Actual Allocation')
    ax1.xaxis.set_label_position('top')

    sn.heatmap((pd.DataFrame(df_cm.sum(axis=0))).transpose(), ax=ax2,  annot=True, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
    sn.heatmap(pd.DataFrame(df_cm.sum(axis=1)), ax=ax3,  annot=True, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)

    plt.savefig(f'{save_dir}/{title}.png')
    plt.clf()


    return confusion_matrix


def plot_hist(data, title, save_dir):
    sn.set_theme()
    sn.histplot(data=data, shrink =0.8, bins=30)
    plt.xlabel("Allocations")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(f'{save_dir}/{title}_hist.png')
    plt.clf()


def jointplot(x, y, title, save_dir):

    x_lim_max = np.max([np.max(y),10])
    y_lim_max = np.max([np.max(y),10])

    sn.set_theme()
    g = sn.jointplot(
        x = x.flatten(), 
        y = y.flatten(),
        xlim = [0, x_lim_max],
        ylim = [0, y_lim_max],
    )
    g.set_axis_labels(xlabel="agent 0", ylabel="agent 1")
    plt.savefig(f'{save_dir}/{title}_jointplot.png')
    plt.clf()


def plot_table(
        title, 
        data,
        save_dir
    ):
    fig, ax =plt.subplots(1,1)    
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data,loc="center")

    plt.savefig(f'{save_dir}/{title}.png')
    plt.clf()

    
def plot_equality_table(
    confusion_matrix_0,
    confusion_matrix_1,
    starting_column = 3,
    starting_row = 0
    ):

    def compute_weighted_contribution(confusion_matrix):
        n_rows, n_cols = confusion_matrix.shape
        contributions = []
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                # Compute the percentage of the allocation
                contribution = (row_idx / ( col_idx + starting_column)) * confusion_matrix[row_idx, col_idx]

                # Store the contribution
                contributions.append(contribution)

        weighted_contribution = np.sum(contributions)/confusion_matrix.sum() * 100
        return weighted_contribution


    # Compute each agent's weighted contribution
    weighted_contribution_0 = compute_weighted_contribution(confusion_matrix_0)
    weighted_contribution_1 = compute_weighted_contribution(confusion_matrix_1)

    # Plot the table
    fig, ax =plt.subplots(1,1)
    data=[["Agent 0", f'{weighted_contribution_0} %'],
        ["Agent 1", f'{weighted_contribution_1} %']]
    
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data,loc="center")
    # ax.set_title("Agent's contribution towards rescue amount")

    plt.savefig(f'{save_dir}/rescue_contributions.png')
    plt.clf()

    return weighted_contribution_0, weighted_contribution_1


if __name__ == "__main__":

    # Retrieve the configurations used for the experiment
    args = get_args()
    ray.init(local_mode = args.local_mode)
    config, env_config, stop = setup(args)

    # Remove episode greedy so that the agent acts deterministically
    config['explore'] = False

    # Remove the seed used in training
    if 'seed' in config.keys():
        config.pop('seed')

    # Only consider the latest checkpoint in the directory
    checkpoints = [200]

    # Conduct 100 episodes in the evaluation
    n_rounds = 100
    

    # Read the json file containing a dictionary
    # specifying where the trained agent is stored
    with open('results_dictionary.json') as f:
        data = f.read()
    dictionary = json.loads(data)
    runs = dictionary[str(args.experiment_number)]
    
    # Create directory to store evaluation results
    if not os.path.exists(f'./data/checkpoints/{args.experiment_number}'):
        os.makedirs(f'./data/checkpoints/{args.experiment_number}')

    average_agent_0_contributions = []
    average_agent_1_contributions = []

    # Begin evaluations
    for i, run in enumerate(runs):

        # If there are multiple runs in the experiments
        # make a subdirectory for each experiment
        if not os.path.exists(f'./data/checkpoints/{args.experiment_number}/{i}'):
            os.makedirs(f'./data/checkpoints/{args.experiment_number}/{i}')

        # Specify path to the stored agent
        path = f"/itet-stor/bryayu/net_scratch/results/{run}"

        # For each agent training checkpoint to evaluate
        for checkpoint in checkpoints:

            # Define placeholders for statistics
            saved_rounds = 0
            percentage_of_optimal_if_saved = []

            # Create placeholders for agent's decisions
            # Used for generating statistics
            agent_0_actions             = []
            agent_1_actions             = []
            inverted_agent_0_actions    = []
            inverted_agent_1_actions    = []
            agent_0_assets              = []
            agent_1_assets              = []
            distressed_bank_assets      = []
            debt_owed_agent_0           = []
            debt_owed_agent_1           = []
            optimal_allocation          = []
            scenarios                   = []
            rescue_amounts              = []

            # Create directory for storing results
            root_dir = f'./data/checkpoints/{args.experiment_number}'
            save_dir = f'./data/checkpoints/{args.experiment_number}/{i}/{checkpoint}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Initialize and load the agent
            agent = DQNTrainer(config=config, env=Volunteers_Dilemma)
            agent.restore(f"{path}/checkpoint_{checkpoint}/checkpoint-{checkpoint}")

            # instantiate env class
            env = Volunteers_Dilemma(env_config)

            """ Main Loop """
            for i in range(n_rounds):
                # Placeholder for agents actions in this round
                a_0     = []
                a_1     = []
                
                # for inverted actions
                if args.invert_actions:
                    inverted_a_0 = []
                    inverted_a_1 = []

                # Reset the environment
                obs = env.reset()

                # Log the setting
                agent_0_assets.append(env.position[0])
                agent_1_assets.append(env.position[1])              
                distressed_bank_assets.append(env.position[2])
                debt_owed_agent_0.append(env.adjacency_matrix[2,0])
                debt_owed_agent_1.append(env.adjacency_matrix[2,1])

                # For each round in the episode
                for round in range(env_config.get('number_of_negotiation_rounds')):

                    # Define the actions dictionary used
                    # to transition the environment
                    actions = {}
                    
                    # Agent 0 decides an action
                    action_0 = agent.compute_action(
                        obs[0], 
                        policy_id='policy_0'
                    )
                    actions[0] = action_0
                    
                    if env_config.get('n_agents') == 2:

                        # Agent 1 decides an action
                        action_1 = agent.compute_action(
                            obs[1], 
                            policy_id='policy_1'
                        )
                        actions[1] = action_1

                    # Conduct a transition in the environment
                    obs, _, _, info = env.step(actions)

                    # Collect some statistics for logging
                    rescue_amount = env.config.get('rescue_amount')

                    # If we are inverting the actions
                    if args.invert_actions:
                        inverted_actions = actions.copy()
                        for i in range(args.n_agents):
                            actions[i] = env.position[i] - actions[i]
                        actual_allocation = sum(actions.values())
                    else:
                        actual_allocation = sum(actions.values())

                    # Post inversion
                    a_0.append(actions[0])
                    a_1.append(actions[1]) if env_config.get('n_agents') == 2 else None

                    if args.invert_actions:
                        inverted_a_0.append(inverted_actions[0])
                        inverted_a_1.append(inverted_actions[1]) if env_config.get('n_agents') == 2 else None

                    # Log statistics
                    if actual_allocation >= rescue_amount:
                        saved_rounds += 1
                        percentage_of_optimal_if_saved.append(actual_allocation/rescue_amount)

                # store the actions of each agent for statistics
                agent_0_actions.append(a_0)
                agent_1_actions.append(a_1)
                optimal_allocation.append(-obs[0]['real_obs'][2])
                scenarios.append(env.config.get('scenario'))
                rescue_amounts.append(rescue_amount)
                if args.invert_actions:
                    inverted_agent_0_actions.append(inverted_a_0)
                    inverted_agent_1_actions.append(inverted_a_1)
            

            """ Generate plots and tables """
            agent_0_actions = np.array(agent_0_actions)
            agent_1_actions = np.array(agent_1_actions)
            optimal_allocations = np.array(optimal_allocation)

            for round in range(env_config.get('number_of_negotiation_rounds')):
                a0_actions = agent_0_actions[:,round]

                if args.invert_actions:

                    inverted_a0_actions = np.array(inverted_agent_0_actions)[:,round]
                    inverted_confusion_matrix_0 = plot(
                        inverted_a0_actions, 
                        optimal_allocation, 
                        save_dir=save_dir, 
                        title=f"Inverted actions - checkpoint {checkpoint}, Agent 0, Round {round}"
                    )
                confusion_matrix_0 = plot(
                    a0_actions, 
                    optimal_allocation, 
                    save_dir=save_dir, 
                    title=f"Checkpoint {checkpoint}, Agent 0, Round {round}"
                )
                # plot_hist(
                #     a0_actions, 
                #     save_dir=save_dir, 
                #     title=f"Checkpoint {checkpoint}, Agent 0, Round {round}"
                # )


                if env_config.get('n_agents') == 2 :
                    a1_actions = agent_1_actions[:,round]

                    if args.invert_actions:
                        inverted_a1_actions = np.array(inverted_agent_1_actions)[:,round]
                        inverted_confusion_matrix_1 = plot(
                            inverted_a1_actions, 
                            optimal_allocation, 
                            save_dir=save_dir, 
                            title=f"Inverted actions - checkpoint {checkpoint}, Agent 1, Round {round}"
                        )

                    confusion_matrix_1 = plot(
                        a1_actions, 
                        optimal_allocation, 
                        save_dir= save_dir, 
                        title=f"Checkpoint {checkpoint}, Agent 1, Round {round}"
                    )
                    # plot_hist(
                    #     a1_actions,
                    #     save_dir=save_dir, 
                    #     title=f"Checkpoint {checkpoint}, Agent 1, Round {round}"    
                    # )
                    contribution_0, contribution_1 = plot_equality_table(
                        confusion_matrix_0,
                        confusion_matrix_1
                    )
                    
                    average_agent_0_contributions.append(contribution_0)
                    average_agent_1_contributions.append(contribution_1)

            
            plot_table(
                title=f'Checkpoint {checkpoint} Statistics',
                data=[["Percentage Saved", f'{saved_rounds/n_rounds * 100} %'],
                ["Percentage of optimal allocation if saved", f'{np.mean(percentage_of_optimal_if_saved) * 100} %']],
                save_dir = save_dir
            )

            

        """ Store experimental data """
        data = {
            'experiment number': args.experiment_number,
            'round number': np.arange(n_rounds),
            'scenario': scenarios,
            'rescue_amount': rescue_amounts,
            'agent 0 actions': list(agent_0_actions.flatten()),
            'agent 1 actions': list(agent_1_actions.flatten()),
            'agent 0 assets': agent_0_assets,
            'agent 1 assets': agent_1_assets,
            'distressed bank assets': distressed_bank_assets,
            'debt owed agent 0': debt_owed_agent_0,
            'debt owed agent 1': debt_owed_agent_1,
            }        
        
        df = pd.DataFrame(data=data)
        df.to_csv(
            f'{save_dir}/experimental_data.csv', 
            index=False,
        )  

    # This table displays the allocation between dominant and non-dominant contributions across
    # trained with varied seeds
    dominant_contributions      = []
    non_dominant_contributions  = []

    for agent_0_contribution, agent_1_contribution in zip(average_agent_0_contributions, average_agent_1_contributions):
        if agent_0_contribution >= agent_1_contribution:
            dominant_contributions.append(agent_0_contribution)
            non_dominant_contributions.append(agent_1_contribution)
        else:
            dominant_contributions.append(agent_1_contribution)
            non_dominant_contributions.append(agent_0_contribution)
    
    plot_table(
        title='Dominant vs Non Dominant Contributions',
        data=[["average dominant contributions", f'{np.mean(dominant_contributions)} %'],
            ["average non-dominant contributions", f'{np.mean(non_dominant_contributions)} %']],
        save_dir = root_dir
    )        

            
    ray.shutdown()
