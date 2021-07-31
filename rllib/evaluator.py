import ray
import json
import pandas as pd
import os

from utils import get_args
from trainer import setup
from ray.rllib.agents.dqn import DQNTrainer
from env import Volunteers_Dilemma

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
    if env_config['scenario'] == 'uniformly mixed':
        n_rounds = 600
    else:
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

    experiment_number           = []
    betas                       = []
    trials                      = []
    run_identifiers             = []
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
    sub_scenarios               = []
    rescue_amounts              = []


    # Begin evaluations
    for i, run in enumerate(runs):

        # Specify path to the stored agent
        path = f"/itet-stor/bryayu/net_scratch/results/{run}"

        # For each agent training checkpoint to evaluate
        for checkpoint in checkpoints:

            # Create directory for storing results
            root_dir = f'./data/checkpoints/{args.experiment_number}'

            # Initialize and load the agent
            agent = DQNTrainer(config=config, env=Volunteers_Dilemma)

            # Naming convnetion changed in latest version of Ray
            if os.path.exists(f"{path}/checkpoint_{checkpoint}/checkpoint-{checkpoint}"):
                agent.restore(f"{path}/checkpoint_{checkpoint}/checkpoint-{checkpoint}")
            else:
                agent.restore(f"{path}/checkpoint_000{checkpoint}/checkpoint-{checkpoint}")

            # instantiate env class
            env = Volunteers_Dilemma(env_config)

            """ Main Loop """
            for _ in range(n_rounds):
                # Placeholder for agents actions in this round
                a_0     = []
                a_1     = []
                
                # for inverted actions
                if args.invert_actions:
                    inverted_a_0 = []
                    inverted_a_1 = []

                # Reset the environment
                obs = env.reset()

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

                # store the actions of each agent for statistics
                experiment_number.append(args.experiment_number)
                trials.append(i)
                betas.append(env_config.get('beta'))
                run_identifiers.append(run)
                agent_0_assets.append(env.position[0])
                agent_1_assets.append(env.position[1])              
                distressed_bank_assets.append(env.position[2])
                debt_owed_agent_0.append(env.adjacency_matrix[2,0])
                debt_owed_agent_1.append(env.adjacency_matrix[2,1])
                agent_0_actions.append(action_0)
                agent_1_actions.append(action_1)
                scenarios.append(env.config.get('scenario'))
                rescue_amounts.append(env.config.get('rescue_amount'))

                # Store the subenvironment; else None
                if env.config.get('scenario') == 'uniformly mixed':
                    sub_scenarios.append(env.generator.sub_scenario)
                else:
                    sub_scenarios.append("not applicable")
                

        """ Store experimental data """
        data = {
            'experiment_number': experiment_number,
            'trials':trials,
            'beta': betas,
            'scenario': scenarios,
            'sub_scenarios': sub_scenarios,
            'rescue_amount': rescue_amounts,
            'agent 0 actions': agent_0_actions,
            'agent 1 actions': agent_1_actions,
            'agent 0 assets': agent_0_assets,
            'agent 1 assets': agent_1_assets,
            'distressed bank assets': distressed_bank_assets,
            'debt owed agent 0': debt_owed_agent_0,
            'debt owed agent 1': debt_owed_agent_1,
            'run_identifiers' : run_identifiers,
            }        
        
        df = pd.DataFrame(data=data)
        df.to_csv(
            f'{root_dir}/experimental_data.csv', 
            index=False,
        )  

    ray.shutdown()
