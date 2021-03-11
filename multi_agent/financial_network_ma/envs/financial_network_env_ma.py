import gym
import numpy as np
from gym import spaces
from .financial_graph import Financial_Graph
from agent import Agent


DEBUG = False

class Financial_Network_Env_Multi_Agent(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,args):

    print("Initializing environment") if DEBUG else None

    self.args                 = args
    self.timestep             = 0


    self.observation_space = []
    self.action_space = []
    
    for _ in range(args.n_banks):
      self.observation_space.append(spaces.Box( low   = 0,\
                                                high  = self.args.cash_in_circulation, \
                                                shape = (1, self.args.number_of_banks * self.args.number_of_banks + 1), \
                                                dtype = np.float32
                                      ))

      # Defines all possible actions
      # NOTE: Assumes all-to-all connection between banks
      self.action_space.append(spaces.Box(  low   = 0,\
                                            high  = 1,\
                                            shape = (1, self.args.number_of_banks), 
                                            dtype = np.float32
                                            )) 

    # Initialize the debt and cash positions of the banks
    self.financial_graph = Financial_Graph( number_of_banks     = self.args.number_of_banks,\
                                            cash_in_circulation = self.args.cash_in_circulation,\
                                            haircut_multiplier  = self.args.haircut_multiplier,\
                                          )

    print("Finished initializing environment") if DEBUG else None


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
    rewards       = self.financial_graph.take_action(action, reward_mode=self.args.reward_type)
    done          = self._determine_if_episode_is_done()
    info          = { 'net_position':self.financial_graph.get_system_net_position(),\
                      'individual_net_position':self.financial_graph.get_individual_net_position(),\
                      'cash_position': self.financial_graph.cash_position,\
                      'debts': self.financial_graph.debts
                    }
                    
    # Retrieve the observations of the resetted environment
    observations = []
    
    for agent_identifier in range(self.args.n_banks):
      observations.append(self.financial_graph.get_observation(agent_identifier))


    if self.args.reward_type == 'System':
      # Allocate the same reward to all the agents
      rewards = np.ones(self.args.n_banks) * rewards

    return observations, rewards, done, info


  def reset(self, evaluate=False):
    """
    Resets the environment to the initial state
    :param  None
    :output observations  np.matrix representing initial environment state
    """
    print("Resetting the environment") if DEBUG else None

    # Reset the timestep counter
    self.timestep = 0

    # Reset the environment
    self.financial_graph.reset(evaluate=evaluate)

    # Retrieve the observations of the resetted environment
    observations = []
    
    for agent_identifier in range(self.args.n_banks):
      observations.append(self.financial_graph.get_observation(agent_identifier))

    info = { 'net_position':self.financial_graph.get_system_net_position(),\
              'individual_net_position':self.financial_graph.get_individual_net_position(),\
              'cash_position': self.financial_graph.cash_position,\
              'debts': self.financial_graph.debts
          }

    return observations, info


  def render(self, mode='human'):
    """
    Outputs a representation of the environment
    :param  mode  defines the representation to output
    :output None
    """

    if mode == 'human':
      print("Rendering the environment") if DEBUG else None
    else:
      pass


  def close(self):
    """
    Executed on closure of the program
    :param  None
    :output None
    """
    print("Closing the environment") if DEBUG else None
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
