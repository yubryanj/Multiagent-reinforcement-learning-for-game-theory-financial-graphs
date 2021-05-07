import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from copy import deepcopy

from utils import generate_graph



class Volunteers_Dilemma(MultiAgentEnv):
    """Env of N independent agents."""

    def __init__(self, config):
        # Generalize the graph for any bank being in distress

        self.n_agents = config['n_agents']
        self.haircut_multiplier = config['haircut_multiplier']
        self.episode_length = config['episode_length']
        self.config = config
        self.distressed_node = 2
        self.iteration = 0

        # Placeholder to get observation size
        self.adjacency_matrix, self.position = generate_graph(
            debug = config.get('debug'), 
            max_value = config.get('max_system_value')
        )

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
            self.action_space = gym.spaces.Box( 
                low   = 0,\
                high  = 1,\
                shape = (1,), 
                dtype = np.float32
            )
                                            
            self.observation_space = gym.spaces.Box(
                low   = -100,
                high  = 100,
                shape = (self.get_observation_size(),),
                dtype = np.float32
            )



    def reset(self):
        self.timestep =0 

        # Reset the environment
        # TODO: Remove the rescue amounts -- they're being used for testing if the network can learn
        # Across different actions
        self.adjacency_matrix, self.position = generate_graph(
            debug = self.config.get('debug'), 
            max_value = self.config.get('max_system_value'), 
            rescue_amount = (self.iteration % 7) + 1    # TODO: remove this hard-coding
            )

        # Retrieve the observations of the resetted environment        
        observations = {}
        for agent_identifier in range(self.n_agents):
            observations[agent_identifier] = self.get_observation(agent_identifier, reset=True)

        self.iteration += 1

        return observations


    def step(self, action_dict):
        # Increment the timestep counter
        self.timestep += 1

        # Compute the optimal action
        optimal_allocation = np.sum(self.adjacency_matrix[-1]) - self.position[-1]
        starting_system_value = self.clear().sum()
                        
        # Retrieve the observations of the resetted environment
        rewards, ending_system_value = self.compute_reward(action_dict)
        
        observations    = {}
        info = {}
        for agent_identifier in range(self.n_agents):
            observations[agent_identifier] = self.get_observation(agent_identifier, reset=False, previous_actions=action_dict)
            info[agent_identifier] = {  
                'starting_system_value': starting_system_value,
                'ending_system_value': ending_system_value,
                'optimal_allocation': optimal_allocation,
                'actual_allocation': action_dict[agent_identifier],
                'percentage_of_optimal_allocation': action_dict[agent_identifier]/optimal_allocation,
                'agent_0_position': self.position[0],
            }

        # TODO: Include percentage of total reward

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
        # Consider the discounted value
        if self.get_net_position(2) < 0:
            inflows = np.sum(self.adjacency_matrix,axis=0) * self.haircut_multiplier
        else:
            inflows = np.sum(self.adjacency_matrix,axis=0)
        
        bank_value = self.position + inflows

        self.take_action(actions)
        new_bank_value = self.clear()

        change_in_position = new_bank_value - bank_value
        reward =  change_in_position.reshape(-1,1)[:self.n_agents]

        rewards = {}
        for i in range(self.n_agents):
            rewards[i] = reward.flatten()[i]

        system_value = new_bank_value.sum()
        
        self.position = deepcopy(position_old)

        return rewards, system_value


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
                normalized_adjacencies = adjacency_matrix[agent,:]/(adjacency_matrix[agent,:].sum())

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

        def get_obs_discrete(agent_identifier=None, reset=False, previous_actions=None):
            observation_dict = {}

            observation = self.position - np.sum(self.adjacency_matrix,axis=1) + np.sum(self.adjacency_matrix,axis=0)

            # Alternative #1
            observation = np.hstack((observation, self.position, self.adjacency_matrix.flatten()))
            
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

        def get_obs_continuous(agent_identifier=None, reset=False, previous_actions=None):
            observation = self.position - np.sum(self.adjacency_matrix,axis=1) + np.sum(self.adjacency_matrix,axis=0)
            observation = np.hstack((observation, self.position, self.adjacency_matrix.flatten()))

            return observation


        
        if self.config.get('discrete'):
            return get_obs_discrete(agent_identifier, reset, previous_actions)
        else:
            return get_obs_continuous(agent_identifier, reset, previous_actions)



    def get_observation_size(self):
        obs = self.get_observation(agent_identifier=0, reset=True)

        if self.config.get('discrete'):
            return len(obs['real_obs'])
        else:
            return len(obs)


    def get_net_position(self, agent):
        net_position = self.position[agent] - np.sum(self.adjacency_matrix[agent,:]) + np.sum(self.adjacency_matrix[:,agent])
        return net_position
