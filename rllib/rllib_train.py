import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer


import argparse
from custom_model import Custom_Model, Discrete_action_model_with_masking, Custom_discrete_model_with_masking
from custom_callback import MyCallbacks
from custom_distribution import Custom_Distribution
from env import Volunteers_Dilemma
from utils import generate_graph, custom_eval_function



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-test",    action="store_true")
    parser.add_argument("--local-mode", action="store_true")
    parser.add_argument("--discrete",   action="store_true")
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--run",        type=str, default="PG")
    parser.add_argument("--n-agents",   type=int, default=1)
    parser.add_argument("--n-workers",  type=int, default=1)
    parser.add_argument("--n-samples",  type=int, default=3)
    parser.add_argument("--n-gpus",     type=int, default=0)
    parser.add_argument("--stop-iters", type=int)
    parser.add_argument("--checkpoint-frequency", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=1)
    parser.add_argument("--stop-reward", type=float, default=6.0)
    parser.add_argument("--haircut-multiplier", type=float, default=0.50)
    parser.add_argument("--max-system-value", type=int, default=100)
    parser.add_argument("--restore",    type=str)
    parser.add_argument("--note",       type=str)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--experiment-number", type=int, default=000)
    args = parser.parse_args()
    args.log_dir = f"/itet-stor/bryayu/net_scratch/results/{args.experiment_number}"
    return args


def setup(args):
    env_config = {
            "n_agents":             args.n_agents,
            "haircut_multiplier":   args.haircut_multiplier,
            "episode_length":       args.episode_length,
            'discrete':             args.discrete,
            'max_system_value':     args.max_system_value, 
            'debug':                args.debug,
        }

    env = Volunteers_Dilemma(env_config)
    obs_space = env.observation_space
    action_space = env.action_space
    
    ModelCatalog.register_custom_model("custom_discrete_action_model_with_masking", Custom_discrete_model_with_masking)
    ModelCatalog.register_custom_model("custom_distribution", Custom_Distribution)

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
                "episode_length":   1,
                },
            "explore": False
        },
    }

    # Discrete action space
    if args.discrete:
        config['exploration_config']= {
            "type": "EpsilonGreedy",
            "initial_epsilon": 0.90, # Need to update the epsilon greedy component for action masking
            "final_epsilon": 0.10,
            "epsilon_timesteps": 1e3, 
        }
            
        config['model'] = {  
            "custom_model": "custom_discrete_action_model_with_masking",
            # "custom_action_dist": "torch_categorical",    # DQN defaults to categorical

        }
        config['seed'] = args.seed
    else:
    # Continuous action space
        config["model"] = { 
            "custom_model": "my_torch_model",
            "custom_model_config": {},
            # "custom_action_dist": "TorchDiagGaussian",
        }

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
