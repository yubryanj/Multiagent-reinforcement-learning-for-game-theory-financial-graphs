import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models import ModelCatalog

from custom_model import basic_model_with_masking, Generalized_model_with_masking
from env import Volunteers_Dilemma
from utils import custom_eval_function, MyCallbacks, get_args

import numpy as np

POLICIES = ['policy_0','policy_1','policy_2','policy_3','policy_4','policy_5']


def policy_mapping_fn(agent_id):
    return np.random.choice(POLICIES)

def setup(args):

    # TODO: Update ENV so that it reads from args, and does not require an env_config
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
            'pool_size':            args.pool_size,

        }

    if hasattr(args, 'reveal_other_agents_identity'):
        env_config['reveal_other_agents_identity'] = args.reveal_other_agents_identity

    if hasattr(args, 'reveal_other_agents_beta'):
        env_config['reveal_other_agents_beta'] = args.reveal_other_agents_beta

    env = Volunteers_Dilemma(env_config)
    obs_space = env.observation_space
    action_space = env.action_space
    
    ModelCatalog.register_custom_model("basic_model", basic_model_with_masking)
    ModelCatalog.register_custom_model("generalized_model_with_masking", Generalized_model_with_masking)

    config = {
        "env": Volunteers_Dilemma,  
        "env_config": env_config,
        "num_workers": args.n_workers,  
        "framework": "torch",
        "num_gpus": args.n_gpus,
        "lr": 1e-3,
        "callbacks": MyCallbacks,  
    }

    policies = {}
    for policy in args.policies:
        policies[policy] = (None, obs_space, action_space, {"framework":"torch", "beta":args.policies[policy]})

    policies_to_train = [policy for policy in args.policies]
    
    config["multiagent"] =  {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": policies_to_train
    }


    # Discrete action space
    if args.discrete:

        config['exploration_config']= {
            "type": "EpsilonGreedy",
            "initial_epsilon": args.initial_epsilon, 
            "final_epsilon": args.final_epsilon,
            "epsilon_timesteps": args.stop_iters, 
        }

        if args.basic_model: 
            config['model'] = {  
                "custom_model": "basic_model",
                "custom_model_config": {
                }
            }
        else:
            config['model'] = {  
                "custom_model": "generalized_model_with_masking",
                "custom_model_config": {
                    'args':                     args,
                    'num_embeddings':           args.max_system_value,
                },
            }
            if hasattr(args, 'reveal_other_agents_identity'):
                config['model']['custom_model_config']['full_information'] = args.full_information

            if hasattr(args, 'reveal_other_agents_identity'):
                config['model']['custom_model_config']['reveal_other_agents_identity'] = args.reveal_other_agents_identity

            if hasattr(args, 'reveal_other_agents_beta'):
                config['model']['custom_model_config']['reveal_other_agents_beta'] = args.reveal_other_agents_beta





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
