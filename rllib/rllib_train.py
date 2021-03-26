"""Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
from copy import deepcopy

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.examples.env.mock_env import MockEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PG")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)


class BasicMultiAgent(MultiAgentEnv):
    """Env of N independent agents, each of which exits after 25 steps."""

    def __init__(self, config):
        self.n_agents = config['n_agents']
        self.haircut_multiplier = config['haircut_multiplier']
        self.position  = np.asarray(config['position'])
        self.adjacency_matrix = np.asarray(config['adjacency_matrix'])
        self.episode_length = config['episode_length']
        self.config = config
        self.distressed_node = 2

        self.observation_space = gym.spaces.Box(low   = -100,
                                                high  = 100,
                                                shape = (self.get_observation_size(),),
                                                dtype = np.float32
                                                )
        self.action_space = gym.spaces.Box( low   = 0,\
                                            high  = 1,\
                                            shape = (1,), 
                                            dtype = np.float32
                                            )

        self.dones = set()
        self.timestep = 0
        self.resetted = False


    def reset(self):
        
        self.resetted = True
        self.dones = set()
        self.timestep =0 

        # Reset the environment
        self.adjacency_matrix = np.asarray(self.adjacency_matrix)
        self.position = np.asarray(self.position)
        
        # Retrieve the observations of the resetted environment        
        observations = {}
        for agent_identifier in range(self.n_agents):
            observations[agent_identifier] = self.get_observation(agent_identifier, reset=True)

        return observations


    def step(self, action_dict):
        # Increment the timestep counter
        self.timestep += 1
                        
        # Retrieve the observations of the resetted environment
        rewards   = self.compute_reward(action_dict)
        
        observations    = {}
        for agent_identifier in range(self.n_agents):
            observations[agent_identifier] = self.get_observation(agent_identifier, reset=False, previous_actions=action_dict)

        done = {"__all__" : self.timestep == self.episode_length }
        info = {}

        return observations, rewards, done, info


    def take_action(self, actions):
        """
        TODO: WRITE ME
        """    

        for agent_identifier in range(self.n_agents):
            transferred_amount = self.position[agent_identifier] * actions[agent_identifier]
            self.position[self.distressed_node] += transferred_amount
            self.position[agent_identifier] -= transferred_amount


    def compute_reward(self, actions):
        """
        Return the requested agent's reward
        """
        position_old = deepcopy(self.position)

        # Allocate the cash as the agents requested
        previous_positions = self.clear()
        self.take_action(actions)
        new_positions = self.clear()

        change_in_position = new_positions - previous_positions
        reward =  change_in_position.reshape(-1,1)[:self.n_agents]

        rewards = {}
        for i in range(self.n_agents):
            rewards[i] = reward.flatten()[i]
        
        
        self.position = deepcopy(position_old)

        return rewards


    def clear(self):
        """
        Clear the system to see where everything stabilizes
        :params None
        :output TODO:WRITE ME
        """
        adjacency_matrix = deepcopy(self.adjacency_matrix)
        position = deepcopy(self.position)
        
        for agent in range(adjacency_matrix.shape[0]):
            
            net_position = self.get_net_position(agent)

            if net_position < 0:
                # Compute the net position
                position[agent] -= np.sum(adjacency_matrix[agent, :])
                adjacency_matrix[agent, : ] *= self.haircut_multiplier

                # Distribute the funds
                position += np.sum(adjacency_matrix, axis=0)
                adjacency_matrix[agent,:] = 0
                adjacency_matrix[:, agent] = 0

        position += np.sum(adjacency_matrix, axis=0)
        position -= np.sum(adjacency_matrix, axis=1)

        return position


    def get_observation(self, agent_identifier=None, reset=False, previous_actions=None):
        """
        Generates the observation matrix displayed to the agent
        :param    None
        :output   np.array  [self.number_of_banks + 1, self.number_of_banks] 
                            matrix stacking the debt and cash position of each agent
        """
        observation = self.clear().flatten().tolist()
        
        if reset:
            for _ in range(self.n_agents):
                observation  = observation + [0]
        if not reset and previous_actions is not None:
            for action in previous_actions:
                observation = observation + [action]

        return observation


    def get_observation_size(self):
        obs = self.get_observation(reset=True)
        return len(obs)


    def get_net_position(self, agent):
        net_position = self.position[agent] - np.sum(self.adjacency_matrix[agent,:]) + np.sum(self.adjacency_matrix[:,agent])
        return net_position


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(local_mode=True)

    env_config = {
            "n_agents": 2,
            "adjacency_matrix": [[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [15.0, 15.0, 0.0]],
            "position" :        [20.0, 20.0, 29.0],
            "haircut_multiplier" : 0.50,
            "episode_length" : 5,
        }

    env = BasicMultiAgent(env_config)
    obs_space = env.observation_space
    action_space = env.action_space

    config = {
        "env": BasicMultiAgent,  
        "env_config": env_config,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "multiagent": {
            "policies": {
                "pg_policy": (None, obs_space, action_space, {"framework": "torch",}),            
            },
            "policy_mapping_fn": (
                lambda agent_id: "pg_policy"),
        },
        "num_workers": 1,  # parallelism
        "framework": "torch",
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()