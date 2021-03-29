import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.test_utils import check_learning_achieved

import argparse

from ray.rllib.models import ModelCatalog
from custom_model import Custom_Model
from env import Volunteers_Dilemma



parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PG")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1000)
parser.add_argument("--stop-reward", type=float, default=5)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(local_mode=True)
    # ray.init()

    env_config = {
            "n_agents": 1,
            "adjacency_matrix": [[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [15.0, 15.0, 0.0]],
            "position" :        [20.0, 20.0, 29.0],
            "haircut_multiplier" : 0.50,
            "episode_length" : 1,
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
                # "policy_1": (None, obs_space, action_space, {"framework": "torch",}),
            },
            "policy_mapping_fn": (lambda agent_id: f"policy_{agent_id}"),
        },
        "model":{"custom_model": "my_torch_model",
                "custom_model_config": {},
        },
        "num_workers": 1,  # parallelism
        "framework": "torch",
        "num_gpus": 0,
        "lr": tune.grid_search([1e-4,1e-5,1e-6,1e-7]),
    }

    stop = {
        "training_iteration"    : args.stop_iters,
        "episode_reward_mean"   : 6.0 * 2,
    }

    results = tune.run( args.run, 
                        config=config, 
                        stop=stop, 
                        local_dir="./results", 
                        log_to_file="results.log",

                    )

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
