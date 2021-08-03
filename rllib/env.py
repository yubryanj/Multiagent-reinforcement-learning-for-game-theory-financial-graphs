import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from copy import deepcopy
from gym.spaces import Discrete, Box
from generator import Generator



class Volunteers_Dilemma(MultiAgentEnv):
    """Env of N independent agents."""

    def __init__(
        self, 
        config
        ):

        # Generalize the graph for any bank being in distress
        self.config = config
        self.distressed_node = 2
        self.iteration = 0
        self.generator = Generator()

        # Placeholder to get observation size
        self.rescue_range = self.config['maximum_rescue_amount'] - self.config['minimum_rescue_amount']
        self.config['rescue_amount'] = (self.iteration % self.rescue_range) + self.config['minimum_rescue_amount']
        self.position, self.adjacency_matrix = self.generator.generate_scenario(self.config)

        # Placeholder for individualized betas which is set in the callback
        self.config['agent_0_beta'] = 1.0
        self.config['agent_1_beta'] = 1.0
        self.config['agent_0_policy'] = 'policy_0'
        self.config['agent_1_policy'] = 'policy_0'


        if self.config['discrete']:
            self.action_space = Discrete(self.config['max_system_value'])

            features = {
                "action_mask": Box(
                    0,
                    1, 
                    shape=(self.action_space.n, )
                ),
                "real_obs": Box(
                    -self.config['max_system_value'],
                    self.config['max_system_value'],
                    shape=(self.get_observation_size(),)
                ),
                "last_offer": Box(
                    0,
                    self.config['max_system_value'],
                    shape=(1,)
                ),
                "final_round": Box(
                    0,
                    1,
                    shape=(1,)
                ),
                "net_position": Box(
                    0, 
                    self.config['max_system_value'], 
                    shape=(1, )
                ),
                "rescue_amount": Box(
                    0, 
                    self.config['max_system_value'], 
                    shape=(1, )
                ),
                "liabilities": Box(
                    0, 
                    self.config['max_system_value'], 
                    shape=(1, )
                ),
                "assets": Box(
                    0, 
                    self.config['max_system_value'], 
                    shape=(1, )
                ),
            }

            if self.config.get('full_information'):
                features['other_agents_assets'] = Box(
                    0, 
                    self.config['max_system_value'], 
                    shape=(1, )
                )
                features['other_agents_liabilities'] = Box(
                    0, 
                    self.config['max_system_value'], 
                    shape=(1, )
                )

            if self.config.get('reveal_other_agents_identity'):
                features['other_agents_identity'] = Box(
                    0, 
                    self.config['pool_size'], 
                    shape=(1, )
                )
            
            if self.config.get('reveal_other_agents_beta'):
                # NOTE: Pro-social betas are discretized in steps of 0.01
                features['other_agents_beta'] = Box(
                    0, 
                    100, 
                    shape=(1, )
                )

            self.observation_space = gym.spaces.Dict(features)



    def reset(
        self
        ):
        """
        Resets the environment
        """
        # Reset the timestep counter for multiple-round scenarios
        self.timestep =0 

        # NOTE: Uniform rescue amounts are generated to improve interpretability
        # as rescue amounts are not evenly distributed when randomly generated
        self.config['rescue_amount'] = (self.iteration % self.rescue_range) + self.config['minimum_rescue_amount']

        # Generate the position and adjacency matrix
        self.position, self.adjacency_matrix = self.generator.generate_scenario(self.config)

        # Retrieve the observations of the resetted environment        
        observations = {}
        for agent_identifier in range(self.config['n_agents']):
            observations[agent_identifier] = self.get_observation(agent_identifier, reset=True)

        self.iteration += 1

        return observations


    def step(
        self, 
        actions
        ):
        """
        Takes one transition step in the environment
        :args actions           dictionary containing the actions decided by each agent
        :output observations    dictionary containing the next observations for each agent
        :output rewards         dictionary containing the rewards for each agent
        :output done            dictionary containing __all__ reflecting if the episode is finished
        :output info            dictionary containing any additional episode information
        """

        # Increment the timestep counter
        self.timestep += 1

        # Compute the value of the system before agents make a decision
        starting_system_value = self.clear().sum()

        # If we decide to invert the actions, then the
        # decision of the agent is how much to retain
        if self.config.get('invert_actions'):

            # Storage for the inverted actions
            inverted_actions = {}

            # Invert each agent's actions
            for agent in actions.keys():
                inverted_actions[agent] = self.position[agent] - actions[agent]

            # Update the actions to the inverted actions
            actions = inverted_actions

                        
        # Retrieve the observations of the resetted environment
        rewards, ending_system_value = self.compute_reward(actions, round = self.timestep)
        
        # Allocate memory for the observation and info dictionaries
        observations    = {}
        info            = {}

        # iterate through each and generate their observations and info package
        for agent_identifier in range(self.config['n_agents']):

            # Generate the observation for each agent
            observations[agent_identifier] = self.get_observation(agent_identifier, reset=False, actions=actions)

            # Retrieve the information package for each agent
            info[agent_identifier] = {  
                'starting_system_value': starting_system_value,
                'ending_system_value': ending_system_value,
                'optimal_allocation': self.config.get('rescue_amount'),
                'actual_allocation': actions[agent_identifier],
                # 'percentage_of_optimal_allocation': sum(list(actions.values()))/self.config.get('rescue_amount'),
                'agent_0_position': self.position[0],
            }

        # determine if the episode is completed
        done = {"__all__" : self.timestep == self.config['number_of_negotiation_rounds']}

        return observations, rewards, done, info


    def compute_reward(
        self, 
        actions, 
        round
        ):
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
            if not self.config['pooled_training']:
                for i in range(self.config['n_agents']):
                    rewards[i] = self.config.get('alpha') * reward.flatten()[i] + \
                                self.config.get('beta') * reward.flatten()[(i+1)%self.config['n_agents']]
            else:
                rewards[0] = self.config.get('alpha') * reward.flatten()[0] + \
                                self.config.get('agent_0_beta') * reward.flatten()[1]
                rewards[1] = self.config.get('alpha') * reward.flatten()[1] + \
                                self.config.get('agent_1_beta') * reward.flatten()[0]



            system_value = new_bank_value.sum()
            
            self.position = deepcopy(position_old)

        return rewards, system_value


    def clear(
        self
        ):
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


    def get_observation(
        self, 
        agent_identifier=None, 
        reset=False, 
        actions=None
        ):
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

            # If agents are given full information, reveal the other rescuing banks' assets and liabilities
            if self.config.get('full_information'):
                observation_dict['other_agents_assets']=\
                    np.array([self.position[(agent_identifier + 1) % self.config.get('n_agents')]])
                observation_dict['other_agents_liabilities']=\
                    np.array([self.adjacency_matrix[2,(agent_identifier + 1) % self.config.get('n_agents')]])

            # If agents are given the other_agents's identity, reveal this in the observation vector
            if self.config.get('reveal_other_agents_identity'):
                other_agents_identity = self.config.get(f'agent_{(agent_identifier + 1) % 2}_policy').strip('policy_')
                observation_dict['other_agents_identity']=\
                    np.array([float(other_agents_identity)])

            # If agents are given the other_agents's beta, reveal this in the observation vector 
            if self.config.get('reveal_other_agents_beta'):
                other_agents_beta =  self.config.get(f'agent_{(agent_identifier + 1) % 2}_beta') * 100
                observation_dict['other_agents_beta']=\
                    np.array([float(other_agents_beta)])


            return observation_dict

        def get_obs_continuous(agent_identifier=None, reset=False, actions=None):
            observation = self.position - np.sum(self.adjacency_matrix,axis=1) + np.sum(self.adjacency_matrix,axis=0)
            observation = np.hstack((observation, self.position, self.adjacency_matrix.flatten()))

            return observation


        
        if self.config.get('discrete'):
            return get_obs_discrete(agent_identifier, reset, actions)
        else:
            return get_obs_continuous(agent_identifier, reset, actions)



    def get_observation_size(
        self
        ):
        """
        Returns the size of the observation dictionary/vector
        """
        obs = self.get_observation(agent_identifier=0, reset=True)

        if self.config.get('discrete'):
            return len(obs['real_obs'])
        else:
            return len(obs)


    def get_net_position(
        self, 
        agent
        ):
        """
        Computes the net position of each agent
        """
        net_position = self.position[agent] - np.sum(self.adjacency_matrix[agent,:]) + np.sum(self.adjacency_matrix[:,agent])
        return net_position