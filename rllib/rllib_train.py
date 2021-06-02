import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models import ModelCatalog

from custom_model import Custom_discrete_model_with_masking, basic_model_with_masking
from env import Volunteers_Dilemma
from utils import custom_eval_function, MyCallbacks, get_args






def setup(args):
    env_config = {
            "n_agents":             args.n_agents,
            "haircut_multiplier":   args.haircut_multiplier,
            "episode_length":       args.episode_length,
            'discrete':             args.discrete,
            'max_system_value':     args.max_system_value, 
            'debug':                args.debug,
            'number_of_negotiation_rounds':     args.number_of_negotiation_rounds,
            'alpha':                args.alpha,
            'beta':                 args.beta,
        }

    env = Volunteers_Dilemma(env_config)
    obs_space = env.observation_space
    action_space = env.action_space
    
    ModelCatalog.register_custom_model("custom_discrete_action_model_with_masking", Custom_discrete_model_with_masking)
    ModelCatalog.register_custom_model("basic_model", basic_model_with_masking)
    # ModelCatalog.register_custom_action_dist("custom_action_distribution", Custom_Distribution)

    config = {
        "env": Volunteers_Dilemma,  
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "policy_0": (None, obs_space, action_space, {"framework": "torch"}),
                "policy_1": (None, obs_space, action_space, {"framework": "torch"}),
            },
            "policy_mapping_fn": (lambda agent_id: f"policy_{agent_id}"),
        },
        "num_workers": args.n_workers,  
        "framework": "torch",
        "num_gpus": args.n_gpus,
        "lr": 1e-3,
        "callbacks": MyCallbacks,  

        # Evaluation
        "evaluation_num_workers": 1,

        # Optional custom eval function.
        "custom_eval_function": custom_eval_function,

        # Enable evaluation, once per training iteration.
        "evaluation_interval": 1,

        # Run 10 episodes each time evaluation runs.
        "evaluation_num_episodes": 100,

        # Override the env config for evaluation.
        "evaluation_config": {
            "env_config": {            
                "episode_length":   args.number_of_negotiation_rounds,
                },
            "explore": False
        },
    }

    # Discrete action space
    if args.discrete:

        config['exploration_config']= {
            "type": "EpsilonGreedy",
            "initial_epsilon": 0.90, 
            "final_epsilon": 0.10,
            "epsilon_timesteps": args.stop_iters, 
        }

        config['model'] = {  

            "custom_model": "custom_discrete_action_model_with_masking",
            "custom_model_config": {
                'embedding_size' : 32,
                'num_embeddings': args.max_system_value,
            },
            # "custom_action_dist": "custom_action_distribution",
            # "custom_action_dist": "torch_categorical",    # DQN defaults to categorical

        }

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
                        # checkpoint_at_end = True,
                        num_samples = args.n_samples,
                        restore = args.restore,
                    )

    if args.as_test:
        check_learning_achieved(results, stop['episode_reward_mean'])

    ray.shutdown()
