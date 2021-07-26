import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models import ModelCatalog

from custom_model import Custom_discrete_model_with_masking, basic_model_with_masking, full_information_model_with_masking
from env import Volunteers_Dilemma
from utils import custom_eval_function, MyCallbacks, get_args

import numpy as np


def policy_mapping_fn(agent_id):
    return np.random.choice(['policy_0','policy_1','policy_2','policy_3'])

def setup(args):

    env_config = {
            "n_agents":             args.n_agents,
            "n_entities":           args.n_agents + 1,
            "haircut_multiplier":   args.haircut_multiplier,
            'discrete':             args.discrete,
            'max_system_value':     args.max_system_value, 
            'debug':                args.debug,
            'number_of_negotiation_rounds':     args.number_of_negotiation_rounds,
            'alpha':                args.alpha,
            'beta':                 args.beta,
            'scenario':             args.scenario,
            'invert_actions':       args.invert_actions,
            'full_information':     args.full_information,
            'pooled_training':      args.pooled_training,
        }

    env = Volunteers_Dilemma(env_config)
    obs_space = env.observation_space
    action_space = env.action_space
    
    ModelCatalog.register_custom_model("custom_discrete_action_model_with_masking", Custom_discrete_model_with_masking)
    ModelCatalog.register_custom_model("full_information_model_with_masking", full_information_model_with_masking)
    ModelCatalog.register_custom_model("basic_model", basic_model_with_masking)

    config = {
        "env": Volunteers_Dilemma,  
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "policy_0": (None, obs_space, action_space, {"framework": "torch", "beta":0.0}),
                "policy_1": (None, obs_space, action_space, {"framework": "torch", "beta":0.2}),
                "policy_2": (None, obs_space, action_space, {"framework": "torch", "beta":0.4}),
                "policy_3": (None, obs_space, action_space, {"framework": "torch", "beta":0.6}),
                "policy_4": (None, obs_space, action_space, {"framework": "torch", "beta":0.8}),
                "policy_5": (None, obs_space, action_space, {"framework": "torch", "beta":1.0}),
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train":['policy_0','policy_1','policy_2','policy_3']
        },
        "num_workers": args.n_workers,  
        "framework": "torch",
        "num_gpus": args.n_gpus,
        "lr": 1e-3,
        "callbacks": MyCallbacks,  

        # # Evaluation
        # "evaluation_num_workers": 1,

        # # Optional custom eval function.
        # "custom_eval_function": custom_eval_function,

        # # Enable evaluation, once per training iteration.
        # "evaluation_interval": 1,

        # # Run 10 episodes each time evaluation runs.
        # "evaluation_num_episodes": 100,

        # # Override the env config for evaluation.
        # "evaluation_config": {
        #     "env_config": {            
        #         "episode_length":   args.number_of_negotiation_rounds,
        #         },
        #     "explore": False
        # },
    }

    # Discrete action space
    if args.discrete:

        config['exploration_config']= {
            "type": "EpsilonGreedy",
            "initial_epsilon": 0.90, 
            "final_epsilon": 0.10,
            "epsilon_timesteps": args.stop_iters, 
        }

        if args.basic_model: 
            config['model'] = {  
                "custom_model": "basic_model",
                "custom_model_config": {
                }
            }
        elif args.full_information:
            config['model'] = {  
                "custom_model": "full_information_model_with_masking",
                "custom_model_config": {
                    'embedding_size' : 32,
                    'num_embeddings': args.max_system_value,
                },
            }
        else:
            config['model'] = {  
                "custom_model": "custom_discrete_action_model_with_masking",
                "custom_model_config": {
                    'embedding_size' : 32,
                    'num_embeddings': args.max_system_value,
                },
                # "custom_action_dist": "custom_action_distribution",
                # "custom_action_dist": "torch_categorical",    # DQN defaults to categorical

            }

        if args.n_samples == 1:
            config['seed'] = args.seed

    stop = {
        "training_iteration"    : args.stop_iters,
    }

    if args.run == "DQN":
        config['hiddens'] = []
        config['dueling'] = False

    return config, env_config, stop

if __name__ == "__main__":
    args=get_args()
    
    ray.init(local_mode = args.local_mode)

    config, env_config, stop = setup(args)

    results = tune.run( args.run, 
                        config=config, 
                        stop=stop, 
                        local_dir=args.log_dir, 
                        checkpoint_freq = args.checkpoint_frequency,
                        num_samples = args.n_samples,
                    )

    if args.as_test:
        check_learning_achieved(results, stop['episode_reward_mean'])

    ray.shutdown()
