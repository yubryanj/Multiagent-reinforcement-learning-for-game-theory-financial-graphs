import ray
import json
from utils import get_args
from trainer_pooled import setup
from ray.rllib.agents.dqn import DQNTrainer
from env import Volunteers_Dilemma
from itertools import combinations

import pandas as pd
import os


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
    checkpoint = stop.get('training_iteration')

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

    # Begin evaluations
    for i, run in enumerate(runs):

        # Specify path to the stored agent
        path = f"/itet-stor/bryayu/net_scratch/results/{run}"

        # Create placeholders for agent's decisions
        # Used for generating statistics
        agent_0_policies            = []
        agent_1_policies            = []
        agent_0_actions             = []
        agent_1_actions             = []
        agent_0_assets              = []
        agent_1_assets              = []
        distressed_bank_assets      = []
        debt_owed_agent_0           = []
        debt_owed_agent_1           = []
        scenarios                   = []
        sub_scenarios               = []
        rescue_amounts              = []

        # Create directory for storing results
        root_dir = f'./data/checkpoints/{args.experiment_number}'

        # Initialize and load the agent
        agent = DQNTrainer(config=config, env=Volunteers_Dilemma)
        agent.restore(f"{path}/checkpoint_000{checkpoint}/checkpoint-{checkpoint}")

        # instantiate env class
        env = Volunteers_Dilemma(env_config)

        policies = config.get('multiagent').get('policies').keys()

        # Iterate through the combination of policies
        for agent_0_policy, agent_1_policy in combinations(policies, 2):

            """ Main Loop """
            for i in range(n_rounds):

                # Reset the environment
                obs = env.reset()

                # Define the actions dictionary used
                # to transition the environment
                actions = {}
                
                # Agent 0 decides an action
                action_0 = agent.compute_action(
                    obs[0], 
                    policy_id = agent_0_policy
                )
                actions[0] = action_0
                

                # Agent 1 decides an action
                action_1 = agent.compute_action(
                    obs[1], 
                    policy_id = agent_1_policy
                )
                actions[1] = action_1

                # Conduct a transition in the environment
                obs, _, _, info = env.step(actions)

                # Collect some statistics for logging
                rescue_amount = env.config.get('rescue_amount')

                # Logging
                agent_0_policies.append(agent_0_policy)
                agent_1_policies.append(agent_1_policy)
                agent_0_assets.append(env.position[0])
                agent_1_assets.append(env.position[1])              
                agent_0_actions.append(actions[0])
                agent_1_actions.append(actions[1])
                rescue_amounts.append(rescue_amount)            
                distressed_bank_assets.append(env.position[2])
                debt_owed_agent_0.append(env.adjacency_matrix[2,0])
                debt_owed_agent_1.append(env.adjacency_matrix[2,1])
                scenarios.append(env.config.get('scenario'))

                # Store the subenvironment; else None
                if env.config.get('scenario') == 'uniformly mixed':
                    sub_scenarios.append(env.generator.sub_scenario)
                else:
                    sub_scenarios.append("not applicable")


        """ Store experimental data """
        data = {
            # 'experiment number': args.experiment_number,
            'scenario': scenarios,
            'sub_scenarios': sub_scenarios,
            'rescue_amount': rescue_amounts,
            'agent 0 actions': agent_0_actions,
            'agent 1 actions': agent_1_actions,
            'agent_0_policies': agent_0_policies,
            'agent_1_policies': agent_1_policies,
            'agent 0 assets': agent_0_assets,
            'agent 1 assets': agent_1_assets,
            'distressed bank assets': distressed_bank_assets,
            'debt owed agent 0': debt_owed_agent_0,
            'debt owed agent 1': debt_owed_agent_1,
            }        
        
        df = pd.DataFrame(data=data)
        df.to_csv(
            f'{root_dir}/experimental_data.csv', 
            index=False,
        )  

    ray.shutdown()
