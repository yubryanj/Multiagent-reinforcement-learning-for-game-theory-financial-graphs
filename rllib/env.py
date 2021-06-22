import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from copy import deepcopy
from utils import generate_graph
from gym.spaces import Discrete, Dict, Box
from generator import Generator



class Volunteers_Dilemma(MultiAgentEnv):
    """Env of N independent agents."""

    def __init__(self, config):
        # Generalize the graph for any bank being in distress

        self.config = config
        self.distressed_node = 2
        self.iteration = 0
        self.generator = Generator()

        # Placeholder to get observation size
        self.config['rescue_amount'] = (self.iteration % 4) + 3
        self.position, self.adjacency_matrix = self.generator.generate_scenario(self.config)


        if config['discrete']:
            self.action_space = Discrete(config['max_system_value'])

            self.observation_space = Dict({
                "action_mask": Box(
                    0,
                    1, 
                    shape=(self.action_space.n, )
                ),
                "real_obs": Box(
                    -config['max_system_value'],
                    config['max_system_value'],
                    shape=(self.get_observation_size(),)
                ),
                "last_offer": Box(
                    0,
                    config['max_system_value'],
                    shape=(1,)
                ),
                "final_round": Box(
                    0,
                    1,
                    shape=(1,)
                ),
                "net_position": Box(
                    0, 
                    config['max_system_value'], 
                    shape=(1, )
                ),
                "rescue_amount": Box(
                    0, 
                    config['max_system_value'], 
                    shape=(1, )
                ),
                "liabilities": Box(
                    0, 
                    config['max_system_value'], 
                    shape=(1, )
                ),
                "assets": Box(
                    0, 
                    config['max_system_value'], 
                    shape=(1, )
                ),
            })


    def reset(self):
        self.timestep =0 

        # Reset the environment
        # TODO: Remove the rescue amounts -- they're being used for testing if the network can learn
        # Across different actions
        self.config['rescue_amount'] = (self.iteration % 4) + 3

        self.position, self.adjacency_matrix = self.generator.generate_scenario(self.config)

        # Retrieve the observations of the resetted environment        
        observations = {}
        for agent_identifier in range(self.config['n_agents']):
            observations[agent_identifier] = self.get_observation(agent_identifier, reset=True)

        self.iteration += 1

        return observations


    def step(self, actions):
        # Increment the timestep counter
        self.timestep += 1

        # Compute the optimal action
        optimal_allocation = np.sum(self.adjacency_matrix[-1]) - self.position[-1]
        starting_system_value = self.clear().sum()
                        
        # Retrieve the observations of the resetted environment
        rewards, ending_system_value = self.compute_reward(actions, round = self.timestep)
        
        observations    = {}
        info = {}
        for agent_identifier in range(self.config['n_agents']):
            observations[agent_identifier] = self.get_observation(agent_identifier, reset=False, actions=actions)
            info[agent_identifier] = {  
                'starting_system_value': starting_system_value,
                'ending_system_value': ending_system_value,
                'optimal_allocation': optimal_allocation,
                'actual_allocation': actions[agent_identifier],
                'percentage_of_optimal_allocation': sum(list(actions.values()))/optimal_allocation,
                'agent_0_position': self.position[0],
            }

        done = {"__all__" : self.timestep == self.config['number_of_negotiation_rounds']}

        return observations, rewards, done, info


    def compute_reward(self, actions, round):
        """
        Returns a reward signal at the end of negotiations
        For all other rounds, returns 0
        """

        # No reward signal if it is not the final negotiation round
        # All offers before is cheaptalk
        if not round == self.config['number_of_negotiation_rounds']:
            rewards = {}
            system_value = self.clear().sum()
            for i in range(self.config['n_agents']):
                rewards[i] = 0
        else:

            position_old = deepcopy(self.position)

            # Allocate the cash as the agents requested
            # Consider the discounted value
            if self.get_net_position(2) < 0:
                inflows = np.sum(self.adjacency_matrix,axis=0) * self.config['haircut_multiplier']
            else:
                inflows = np.sum(self.adjacency_matrix,axis=0)
            
            bank_value = self.position + inflows

            # Modifies the environment according to the agent's requested actions
            for agent_identifier in actions:

                if self.config['discrete']:
                    transferred_amount = actions[agent_identifier]
                else:
                    transferred_amount = self.position[agent_identifier] * actions[agent_identifier] 
                self.position[self.distressed_node] += transferred_amount
                self.position[agent_identifier] -= transferred_amount

            new_bank_value = self.clear()

            change_in_position = new_bank_value - bank_value
            reward =  change_in_position.reshape(-1,1)[:self.config['n_agents']]

            rewards = {}
            for i in range(self.config['n_agents']):
                rewards[i] = self.config.get('alpha') * reward.flatten()[i] + \
                             self.config.get('beta') * reward.flatten()[(i+1)%self.config['n_agents']]

            system_value = new_bank_value.sum()
            
            self.position = deepcopy(position_old)

        return rewards, system_value


    def clear(self):
        """
        Clear the system to see where everything stabilizes
        """
        adjacency_matrix = deepcopy(self.adjacency_matrix)
        position = deepcopy(self.position)
        
        for agent in range(adjacency_matrix.shape[0]):
            
            net_position = self.get_net_position(agent)

            if net_position < 0:
                # Compute the amount to redistribute to other nodes
                discounted_position = position[agent] * self.config['haircut_multiplier']
                normalized_adjacencies = adjacency_matrix[agent,:]/(adjacency_matrix[agent,:].sum())

                amounts_to_redistribute = discounted_position * normalized_adjacencies

                position += amounts_to_redistribute

                adjacency_matrix[agent,:] = 0 
                adjacency_matrix[:,agent] = 0
                position[agent] = 0
                
        position += np.sum(adjacency_matrix, axis=0)
        position -= np.sum(adjacency_matrix, axis=1)

        return position


    def get_observation(self, agent_identifier=None, reset=False, actions=None):
        """
        Generates the observation matrix displayed to the agent
        """

        def get_obs_discrete(agent_identifier=None, reset=False, actions=None):
            observation_dict = {}

            observation = self.position - np.sum(self.adjacency_matrix,axis=1) + np.sum(self.adjacency_matrix,axis=0)

            # Alternative #1
            observation = np.hstack((observation, self.position, self.adjacency_matrix.flatten()))

            observation_dict['real_obs']    = observation
            observation_dict['action_mask'] = np.array([0.] * self.action_space.n)
            observation_dict['assets']      = np.array([self.position[agent_identifier]])
            observation_dict['liabilities'] = np.array([np.sum(self.adjacency_matrix,axis=1)[agent_identifier]])
            observation_dict['net_position']= np.array([observation[agent_identifier]])
            observation_dict['rescue_amount'] = np.array([abs(observation[self.distressed_node])])

            if self.config.get('n_agents') == 1:
                observation_dict['last_offer'] = np.zeros(1)
            else:
                observation_dict['last_offer']  = np.array([actions[(agent_identifier + 1) % 2]]) if actions is not None else np.zeros(1)



            if 'timestep' in self.__dict__.keys() and self.timestep == self.config['number_of_negotiation_rounds']:
                observation_dict['final_round'] = np.ones(1)
            else:
                observation_dict['final_round'] = np.zeros(1)

            # Mask all actions outside of current position
            observation_dict.get('action_mask')[:int(self.position[agent_identifier])+1] = 1
            
            return observation_dict

        def get_obs_continuous(agent_identifier=None, reset=False, actions=None):
            observation = self.position - np.sum(self.adjacency_matrix,axis=1) + np.sum(self.adjacency_matrix,axis=0)
            observation = np.hstack((observation, self.position, self.adjacency_matrix.flatten()))

            return observation


        
        if self.config.get('discrete'):
            return get_obs_discrete(agent_identifier, reset, actions)
        else:
            return get_obs_continuous(agent_identifier, reset, actions)



    def get_observation_size(self):
        obs = self.get_observation(agent_identifier=0, reset=True)

        if self.config.get('discrete'):
            return len(obs['real_obs'])
        else:
            return len(obs)


    def get_net_position(self, agent):
        net_position = self.position[agent] - np.sum(self.adjacency_matrix[agent,:]) + np.sum(self.adjacency_matrix[:,agent])
        return net_position