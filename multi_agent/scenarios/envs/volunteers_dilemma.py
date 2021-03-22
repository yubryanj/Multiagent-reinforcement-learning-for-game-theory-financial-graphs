import gym
import numpy as np
from gym import spaces
import numpy as np
from copy import deepcopy


class Volunteers_dilemma(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,args):
    self.args                 = args
    self.timestep             = 0

    self.adjacency_matrix = np.asarray(self.args.adjacency_matrix)
    self.position = np.asarray(self.args.position)

    self.observation_space    = []
    self.action_space         = []
    
    self.distressed_node      = 2
    
    for _ in range(args.n_agents):
      self.observation_space.append(spaces.Box( low   = 0,\
                                                high  = 9, \
                                                shape = (1, self.get_observation_size()), \
                                                dtype = np.float32
                                      ))

      # Defines all possible actions
      # Action space will be checked for valid actions in step
      self.action_space.append(spaces.Box(  low   = 0,\
                                            high  = 1,\
                                            shape = (1, 1), 
                                            dtype = np.float32
                                            )) 


  def step(self, actions):
    """
    Takes one step in the environment
    :param  actions     action submitted by the agent
    :output observation Vector representing updated environment state
    :output rewards     Reward vector for agent
    :output done        True/False depending if the episode is finished
    :output info        additional information
    """

    # Increment the timestep counter
    self.timestep += 1
    done = self._determine_if_episode_is_done()
                    
    # Retrieve the observations of the resetted environment
    rewards       = self.compute_reward(actions)
    observations  = []
    
    for agent_identifier in range(self.args.n_agents):
      observations.append(self.get_observation(agent_identifier))

    info = self.get_info()

    return observations, rewards, done, info


  def take_action(self, actions):
    """
    TODO: WRITE ME
    """    

    for agent_identifier, action in enumerate(actions):
      
      if agent_identifier == self.distressed_node:
            continue
      else:
        transferred_amount = self.position[agent_identifier] * action
        self.position[self.distressed_node] += transferred_amount
        self.position[agent_identifier] -= transferred_amount

  
  def compute_reward(self, actions):
    """
    Return the requested agent's reward
    """
    position_old = deepcopy(self.position)

    previous_positions = self.clear()

    # Allocate the cash as the agents requested
    self.take_action(actions)

    new_positions = self.clear()

    self.position = deepcopy(position_old)

    change_in_position = new_positions - previous_positions

    reward =  change_in_position.reshape(-1,1)

    return reward
            

  def reset(self):
    """
    Resets the environment to the initial state
    :param  None
    :output observations  np.matrix representing initial environment state
    """

    # Reset the timestep counter
    self.timestep = 0

    # Reset the environment
    self.adjacency_matrix = np.asarray(self.args.adjacency_matrix)
    self.position = np.asarray(self.args.position)
    
    # Retrieve the observations of the resetted environment
    observations = []
    
    for agent_identifier in range(self.args.n_agents):
      observations.append(self.get_observation(agent_identifier))

    info = self.get_info()

    return observations, info


  def get_info(self):
        info = {
          'cleared_positions': self.clear(),
        }

        return info


  def render(self, mode='human'):
    """
    Outputs a representation of the environment
    :param  mode  defines the representation to output
    :output None
    """

    if mode == 'human':
      pass
    else:
      pass


  def close(self):
    """
    Executed on closure of the program
    :param  None
    :output None
    """
    pass


  def _determine_if_episode_is_done(self):
    """
    Returns a boolean determining whether this episode is completed
    :param    None
    :output   Boolean   True/False depending on whether the episode is finished
    """
    if self.timestep >= self.args.max_episode_len: 
        return True
    
    return False


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
          adjacency_matrix[agent, : ] *= self.args.haircut_multiplier

          # Distribute the funds
          position += np.sum(adjacency_matrix, axis=0)
          adjacency_matrix[agent,:] = 0
          adjacency_matrix[:, agent] = 0

    position += np.sum(adjacency_matrix, axis=0)
    position -= np.sum(adjacency_matrix, axis=1)

    return position


  def get_observation(self, agent_identifier=None):
    """
    Generates the observation matrix displayed to the agent
    :param    None
    :output   np.array  [self.number_of_banks + 1, self.number_of_banks] 
                        matrix stacking the debt and cash position of each agent
    """
    # Full information
    # return np.hstack((self.adjacency_matrix.reshape(1,-1), self.position.reshape(1,-1)))
    observation = self.clear().reshape(1,-1)
    # observation = self.position.reshape(1,-1)
    return observation


  def get_observation_size(self):
    obs = self.get_observation()
    return obs.shape[1]


  def get_net_position(self, agent):
      net_position = self.position[agent] - np.sum(self.adjacency_matrix[agent,:]) + np.sum(self.adjacency_matrix[:,agent])
      return net_position