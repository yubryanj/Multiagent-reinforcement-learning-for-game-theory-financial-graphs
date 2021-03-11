import gym
import numpy as np
from gym import spaces
from .graph import Graph


class Network(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,args):


    self.args                 = args
    self.timestep             = 0


    self.observation_space = []
    self.action_space = []
    self.adjacency_matrix = args.adjacency_matrix
    self.position = args.position
    
    for _ in range(args.n_agents):
      self.observation_space.append(spaces.Box( low   = 0,\
                                                high  = self.args.maximum_position, \
                                                shape = (1, self.args.n_agents * self.args.n_agents), \
                                                dtype = np.float32
                                      ))

      # Defines all possible actions
      # NOTE: Assumes all-to-all connection between banks
      self.action_space.append(spaces.Box(  low   = 0,\
                                            high  = 1,\
                                            shape = (1, self.args.n_agents), 
                                            dtype = np.float32
                                            )) 

    # TODO: Write me!
    self.graph = Graph()



  def step(self, action):
    """
    Takes one step in the environment
    :param  action      action submitted by the agent
    :output observation Vector representing updated environment state
    :output rewards     Reward vector for agent
    :output done        True/False depending if the episode is finished
    :output info        additional information
    """

    # Convert action into an np array
    action = np.asarray(action)

    # Increment the timestep counter
    self.timestep += 1

    # # Allocate the cash as the agents requested
    # TODO: WRITE ME - COLLECT THE REWARDS
    rewards       = None
    done          = self._determine_if_episode_is_done()

    # TODO: WRITEME Collect information
    info = self.get_info()
                    
    # Retrieve the observations of the resetted environment
    observations = [[],[],[]]
    
    for agent_identifier in range(self.args.n_agents):
      #TODO: WRITE ME - get observations
      observations.append([])

    return observations, rewards, done, info


  def reset(self):
    """
    Resets the environment to the initial state
    :param  None
    :output observations  np.matrix representing initial environment state
    """

    # Reset the timestep counter
    self.timestep = 0

    # Reset the environment
    self.adjacency_matrix = self.args.adjacency_matrix
    self.position = self.args.position

    # Retrieve the observations of the resetted environment
    observations = [[0],[],[]]
    
    for agent_identifier in range(self.args.n_agents):
      # TODO: WRITE ME! - Get observations
      observations.append([])

    # TODO: WRITEME Collect information
    info = self.get_info()

    return observations, info


  def get_info(self):
        pass


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
