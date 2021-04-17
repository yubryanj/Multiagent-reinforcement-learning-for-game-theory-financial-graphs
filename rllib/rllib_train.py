import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian, TorchSquashedGaussian, TorchDirichlet, TorchDeterministic
from ray.rllib.agents.ppo import PPOTrainer


import argparse

from custom_model import Custom_Model
from custom_callback import MyCallbacks
from env import Volunteers_Dilemma
from utils import generate_graph


parser = argparse.ArgumentParser()
parser.add_argument("--as-test",    action="store_true")
parser.add_argument("--local-mode", action="store_true")
parser.add_argument("--discrete", action="store_true")
parser.add_argument("--debug",      action="store_true")
parser.add_argument("--run",        type=str, default="PG")
parser.add_argument("--n-agents",   type=int, default=1)
parser.add_argument("--n-workers",  type=int, default=1)
parser.add_argument("--n-samples",  type=int, default=3)
parser.add_argument("--n-gpus",     type=int, default=0)
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--checkpoint-frequency", type=int, default=1)
parser.add_argument("--episode-length", type=int, default=1)
parser.add_argument("--stop-reward", type=float, default=6.0)
parser.add_argument("--haircut-multiplier", type=float, default=0.50)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        args.log_dir = f"/itet-stor/bryayu/net_scratch/results/DEBUG/{args.n_agents}_agents/{args.run}/episode_length_{args.episode_length}"
    else:
        args.log_dir = f"/itet-stor/bryayu/net_scratch/results/discrete_{args.discrete}/{args.n_agents}_agents/{args.run}/episode_length_{args.episode_length}"
    ray.init(local_mode = args.local_mode)

    adjacency_matrix, position = generate_graph(args.debug, args.n_agents)

    env_config = {
            "n_agents":         args.n_agents,
            "adjacency_matrix": adjacency_matrix,
            "position" :        position,
            "haircut_multiplier": args.haircut_multiplier,
            "episode_length":   args.episode_length,
            'discrete':         args.discrete,
            'max_system_cash':  100,
            
        }

    env = Volunteers_Dilemma(env_config)
    obs_space = env.observation_space
    action_space = env.action_space

    ModelCatalog.register_custom_action_dist("TorchDiagGaussian", TorchDiagGaussian)
    ModelCatalog.register_custom_model("my_torch_model", Custom_Model)

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
        
    }

    # Discrete action space
    if args.discrete:
        config['exploration_config']= {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,
                "epsilon_timesteps": 100000, 
            }
    else:
    # Continuous action space
        config["model"] = { "custom_model": "my_torch_model",
                            "custom_model_config": {},
                            "custom_action_dist": "TorchDiagGaussian",
                            }
        # config["extra_grad_process_fn"] = True
        # "lr": tune.grid_search([1e-5]),
        config["entropy_coeff"] = tune.grid_search([0.0,1e-2,1e-3])

        # Set this to enable gradient clipping, as the max gradient
        config["grad_clip"] = 1.0

    stop = {
        # "training_iteration"    : args.stop_iters,
        # "episode_reward_mean"   : args.stop_reward * args.n_agents,
        "episode_reward_mean"   : 13.75,
    }

    results = tune.run( args.run, 
                        config=config, 
                        stop=stop, 
                        local_dir=args.log_dir, 
                        checkpoint_freq = args.checkpoint_frequency,
                        checkpoint_at_end = True,
                        num_samples = args.n_samples,
                        # restore='results/discrete_True/2_agents/DQN/episode_length_1/DQN/DQN_Volunteers_Dilemma_d7cce_00000_0_2021-04-14_09-34-27/checkpoint_939/checkpoint-939',
                    )

    if args.as_test:
        check_learning_achieved(results, stop['episode_reward_mean'])

    ray.shutdown()
