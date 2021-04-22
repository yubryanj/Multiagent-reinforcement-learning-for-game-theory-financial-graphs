import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from copy import deepcopy

from utils import generate_case_1, generate_graph



class Volunteers_Dilemma(MultiAgentEnv):
    """Env of N independent agents."""

    def __init__(self, config):
        # Generalize the graph for any bank being in distress

        self.n_agents = config['n_agents']
        self.haircut_multiplier = config['haircut_multiplier']
        self.episode_length = config['episode_length']
        self.config = config
        self.distressed_node = 2


        # self.position  = np.asarray(config['position'])
        # self.adjacency_matrix = np.asarray(config['adjacency_matrix'])
        self.adjacency_matrix, self.position = generate_graph(debug = config.get('debug'))

        if config['discrete']:
            self.action_space = gym.spaces.Discrete(config['max_system_value'])

            self.observation_space = gym.spaces.Dict({
                                        "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n, )),
                                        "avail_actions": gym.spaces.Box(-1, 1, shape=(self.action_space.n, )),
                                        "real_obs": gym.spaces.Box(-config['max_system_value'],
                                                                    config['max_system_value'],
                                                                    shape=(self.get_observation_size(),)
                                                                    )
            })
        else:
            self.action_space = gym.spaces.Box( low   = 0,\
                                                high  = 1,\
                                                shape = (1,), 
                                                dtype = np.float32
                                                )
                                            
            self.observation_space = gym.spaces.Box(low   = -100,
                                                    high  = 100,
                                                    shape = (self.get_observation_size(),),
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
        self.adjacency_matrix, self.position = generate_graph(debug = self.config.get('debug'))
        # self.adjacency_matrix = np.asarray(self.adjacency_matrix)
        # self.position = np.asarray(self.position)

        system_value = self.clear()

        # Retrieve the observations of the resetted environment        
        observations = {}
        info ={}
        for agent_identifier in range(self.n_agents):
            observations[agent_identifier] = self.get_observation(agent_identifier, reset=True)
            info[agent_identifier] = system_value.sum()

        return observations


    def step(self, action_dict):
        # Increment the timestep counter
        self.timestep += 1

        # Compute the optimal action
        optimal_allocation = np.sum(self.adjacency_matrix[-1]) - self.position[-1]
        actual_allocation = action_dict[0]
        percentage_of_optimal_allocation = np.clip(float(actual_allocation/optimal_allocation),0.0,1.0)

        starting_system_value = self.clear().sum()
                        
        # Retrieve the observations of the resetted environment
        rewards, ending_system_value, new_positions = self.compute_reward(action_dict)
        
        observations    = {}
        info = {}
        for agent_identifier in range(self.n_agents):
            observations[agent_identifier] = self.get_observation(agent_identifier, reset=False, previous_actions=action_dict)
            info[agent_identifier] = {  
                'starting_position' : self.position,
                'adjacency_matrix' : self.adjacency_matrix,
                'starting_system_value': starting_system_value,
                'ending_system_value': ending_system_value,
                'optimal_allocation': optimal_allocation,
                'actual_allocation': actual_allocation,
                'percentage_of_optimal_allocation': percentage_of_optimal_allocation,
                }

        done = {"__all__" : self.timestep == self.episode_length}

        return observations, rewards, done, info


    def take_action(self, actions):
        """
        TODO: WRITE ME
        """    

        for agent_identifier in actions:

            if self.config['discrete']:
                transferred_amount = actions[agent_identifier]
            else:
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

        system_value = new_positions.sum()
        
        self.position = deepcopy(position_old)

        return rewards, system_value, new_positions


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
                # Compute the amount to redistribute to other nodes
                discounted_position = position[agent] * self.haircut_multiplier
                normalized_adjacencies = adjacency_matrix[agent]/(adjacency_matrix[agent].sum())

                amounts_to_redistribute = discounted_position * normalized_adjacencies

                position += amounts_to_redistribute

                adjacency_matrix[agent,:] = 0 
                adjacency_matrix[:,agent] = 0
                position[agent] = 0
                
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

        observation_dict = {}

        observation = self.clear().flatten().tolist()
        
        if reset:
            for _ in range(self.n_agents):
                observation  = observation + [0]
        if not reset and previous_actions is not None:
            for action in previous_actions:
                action = np.clip(action,0,1)
                observation = observation + [action]

        observation_dict['real_obs'] = observation
        observation_dict['action_mask'] = np.array([0.] * self.action_space.n)
        observation_dict['avail_actions'] = np.array([0.] * self.action_space.n)

        # Mask all actions outside of current position
        observation_dict.get('action_mask')[:int(self.position[agent_identifier])] = 1

        return observation_dict


    def get_observation_size(self):
        obs = self.get_observation(agent_identifier=0, reset=True)
        return len(obs['real_obs'])


    def get_net_position(self, agent):
        net_position = self.position[agent] - np.sum(self.adjacency_matrix[agent,:]) + np.sum(self.adjacency_matrix[:,agent])
        return net_position
