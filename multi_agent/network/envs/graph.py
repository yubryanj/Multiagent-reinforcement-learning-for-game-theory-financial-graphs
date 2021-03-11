import numpy as np

EPSILON = 1e-10

class Graph():

    def __init__(   self,
                    args
                ):

        """
        : WRITE ME
        """
        self.args = args
        self.reset()

    def get_observation(self, agent_identifier):
        """
        Generates the observation matrix displayed to the agent
        :param    None
        :output   np.array  [self.number_of_banks + 1, self.number_of_banks] 
                            matrix stacking the debt and cash position of each agent
        """

        # Full information
        return np.hstack((self.adjacency_matrix.reshape(1,-1), self.position.reshape(1,-1)))

        
    def _initialize_banks(self, adjacency_matrix=None, position=None):
        """
        Initializes self.adjacency_matrix and self.position 
        This function is called whenever the environment is reset.
        """
        # TODO: WRITE ME

        self.adjacency_matrix = adjacency_matrix
        self.position = position


    def compute_system_value(self):
        """
        Computes the value of the system
        :param  mode            which method 
        :output system_value    valuation of the current state of the system
        """
        
        system_value = self.get_system_net_position()

        return system_value


    def take_action(self, action, reward_mode="Individual", disable_default_actions= True):
        """
        Distributes cash as per the action requested by the agents
        :param  action  np.matrix where each cell is the percentage of the cash position to allocate
        :output reward  
        """

        assert(reward_mode in ['Individual','System'])

        if reward_mode == "Individual":
            calculate_net_position = self.get_individual_net_position
        elif reward_mode == 'System':
            calculate_net_position = self.get_system_net_position
        
        old_net_position = calculate_net_position()

        action  = action.reshape(self.number_of_banks, self.number_of_banks)

        # Normalize the cash distribution to 100%
        action = self._normalize_cash_distribution(action)

        # Get the set of bankrupt banks
        # Disable the action of defaulted banks
        if disable_default_actions:
            defaulted_banks = self.get_list_of_defaulting_and_solvent_banks()[0]
            action[defaulted_banks] = 0

        n_rows, n_cols = action.shape

        # Allocate cash as requested by the banks    
        for from_bank in range(n_rows):
            for to_bank in range(n_cols):
                percentage_to_allocate          = action[from_bank, to_bank]
                amount                          = self.position[from_bank] * percentage_to_allocate
                self.position[from_bank]   -= amount
                self.position[to_bank]     += amount

        # Clear the system again after distributing the funds
        self.clear()

        new_net_position = calculate_net_position()

        reward = new_net_position - old_net_position

        return reward


    def get_individual_net_position(self):
        """
        Returns the net position of the individual (i.e. cash -  debt)
        Consider -- that the system should be cleared and debt should be equal to 0
        :params     None
        :outputs    position     position of each agent
        """

        pass        


    def _normalize_cash_distribution(self, action):
        """
        In the case the agent attempts to distribute more than 100%
        of thier cash position, then the system will normalize the amount
        to be distribute 100%
        :param  action  action matrix to be normalized
        :output action  normalized action matrix
        """
        row_sums  = action.sum(axis=1, keepdims=True)
        action    = action / (row_sums + EPSILON)

        return action


    def reset(self, evaluate=False):
        """
        Resets the environment
        :param  None
        :output None
        """
        self.adjacency_matrix = np.asarray(self.args.adjacency_matrix)
        self.position = np.asarray(self.args.position)
        pass

    
    def clear(self):
        """
        Clear the financial system by distributring debt and credit
        Clearing is completed when the system stabilizes (i.e. solvent banks no longer change)
        :params None
        :output None
        """
        pass

