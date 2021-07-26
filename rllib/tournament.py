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
from evaluator import plot_equality_table, plot_table, plot

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
    with open('tournament_configs.json') as f:
        data = f.read()
    dictionary = json.loads(data)
    runs = dictionary[str(args.tournament_number)]

    # Specify path to the stored agent
    agent_0_path = f"/itet-stor/bryayu/net_scratch/results/{runs['agent 0']}"
    agent_1_path = f"/itet-stor/bryayu/net_scratch/results/{runs['agent 1']}"
    
    # Create directory to store evaluation results
    if not os.path.exists(f'./data/checkpoints/{args.experiment_number}'):
        os.makedirs(f'./data/checkpoints/{args.experiment_number}')

    average_agent_0_contributions = []
    average_agent_1_contributions = []


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
        save_dir = f'./data/checkpoints/{args.experiment_number}/{checkpoint}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Initialize and load the agent
        agent_0 = DQNTrainer(config=config, env=Volunteers_Dilemma)
        agent_0.restore(f"{agent_0_path}/checkpoint_{checkpoint}/checkpoint-{checkpoint}")

        agent_1 = DQNTrainer(config=config, env=Volunteers_Dilemma)
        agent_1.restore(f"{agent_1_path}/checkpoint_{checkpoint}/checkpoint-{checkpoint}")

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
                action_0_action = agent_0.compute_action(
                    obs[0], 
                    policy_id='policy_0'
                )
                actions[0] = action_0_action
            
                # Agent 1 decides an action
                action_1_action = agent_1.compute_action(
                    obs[1], 
                    policy_id='policy_0'
                )
                actions[1] = action_1_action

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

                contribution_0, contribution_1 = plot_equality_table(
                    confusion_matrix_0,
                    confusion_matrix_1,
                    save_dir=save_dir
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
