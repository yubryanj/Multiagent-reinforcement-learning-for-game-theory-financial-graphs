import gym
import numpy as np
from gym import spaces
from .graph import Graph
import numpy as np


class Volunteers_dilemma(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,args):
    self.args                 = args
    self.timestep             = 0

    self.observation_space    = []
    self.action_space         = []
    
    self.graph                = Graph(args)

    self.distressed_node      = 2
    
    for _ in range(args.n_agents):
      self.observation_space.append(spaces.Box( low   = 0,\
                                                high  = 9, \
                                                shape = (1, self.graph.get_observation_size()), \
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
      observations.append(self.graph.get_observation(agent_identifier))

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
        transferred_amount = self.graph.position[agent_identifier] * action
        self.graph.position[self.distressed_node] += transferred_amount
        self.graph.position[agent_identifier] -= transferred_amount

  
  def compute_reward(self, actions):
    """
    Return the requested agent's reward
    """
    previous_positions = self.graph.clear()

    # Allocate the cash as the agents requested
    self.take_action(actions)

    new_positions = self.graph.clear()

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
    self.graph.reset()
    
    # Retrieve the observations of the resetted environment
    observations = []
    
    for agent_identifier in range(self.args.n_agents):
      observations.append(self.graph.get_observation(agent_identifier))

    info = self.get_info()

    return observations, info


  def get_info(self):
        info = {
          'cleared_positions': self.graph.clear(),
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
