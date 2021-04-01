import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models import ModelCatalog


import argparse

from custom_model import Custom_Model
from custom_callback import MyCallbacks
from env import Volunteers_Dilemma


parser = argparse.ArgumentParser()
parser.add_argument("--as-test",    action="store_true")
parser.add_argument("--local-mode", action="store_true")
parser.add_argument("--debug",      action="store_true")
parser.add_argument("--run",        type=str, default="PG")
parser.add_argument("--n-agents",   type=int, default=1)
parser.add_argument("--n-workers",  type=int, default=1)
parser.add_argument("--n-gpus",     type=int, default=0)
parser.add_argument("--stop-iters", type=int, default=300)
parser.add_argument("--checkpoint-frequency", type=int, default=1)
parser.add_argument("--episode-length", type=int, default=1)
parser.add_argument("--stop-reward", type=float, default=6.0)
parser.add_argument("--haircut-multiplier", type=float, default=0.50)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        args.log_dir = f"./results/DEBUG/{args.n_agents}_agents/{args.run}/episode_length_{args.episode_length}"
    else:
        args.log_dir = f"./results/{args.n_agents}_agents/{args.run}/episode_length_{args.episode_length}"
    ray.init(local_mode = args.local_mode)

    env_config = {
            "n_agents": args.n_agents,
            "adjacency_matrix": [[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [15.0, 15.0, 0.0]],
            "position" :        [20.0, 20.0, 29.0],
            "haircut_multiplier" : args.haircut_multiplier,
            "episode_length" : args.episode_length,
        }

    env = Volunteers_Dilemma(env_config)
    obs_space = env.observation_space
    action_space = env.action_space

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
        "model":{"custom_model": "my_torch_model",
                "custom_model_config": {},
        },
        "num_workers": args.n_workers,  
        "framework": "torch",
        "num_gpus": args.n_gpus,
        "lr": tune.grid_search([1e-5,5e-6]),
        "callbacks": MyCallbacks,
        
    }

    stop = {
        "training_iteration"    : args.stop_iters,
        "episode_reward_mean"   : args.stop_reward * args.n_workers,
    }

    results = tune.run( args.run, 
                        config=config, 
                        stop=stop, 
                        local_dir=args.log_dir, 
                        log_to_file="results.log",
                        checkpoint_freq = args.checkpoint_frequency,
                        checkpoint_at_end = True,
                    )

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
