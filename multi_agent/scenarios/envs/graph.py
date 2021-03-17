from copy import deepcopy
import numpy as np

EPSILON = 1e-10

class Graph():

    def __init__(   self,
                    args
                ):

        """
        :args       arguments
        :output     none
        """
        self.args = args
        self.reset()


    def get_observation_size(self):
        # return self.adjacency_matrix.size + self.position.size
        return self.position.size



    def get_observation(self, agent_identifier=None):
        """
        Generates the observation matrix displayed to the agent
        :param    None
        :output   np.array  [self.number_of_banks + 1, self.number_of_banks] 
                            matrix stacking the debt and cash position of each agent
        """
        # Full information
        # return np.hstack((self.adjacency_matrix.reshape(1,-1), self.position.reshape(1,-1)))
        return self.position.reshape(1,-1)


    def reset(self):
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
        Clear the system to see where everything stabilizes
        :params None
        :output TODO:WRITE ME
        """

        adjacency_matrix = deepcopy(self.adjacency_matrix)
        position = deepcopy(self.position)
        for agent in range(adjacency_matrix.shape[0]):
            if position[agent] < 0:
                adjacency_matrix[agent, : ] = 0

        position += np.sum(adjacency_matrix, axis=0)

        return position