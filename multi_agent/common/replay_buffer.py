import numpy as np
import threading

class Buffer:

    def __init__(self, args) -> None:
        self.max_size = args.buffer_size
        self.args = args

        # Current size of the memory
        self.current_size = 0
        
        self.buffer = dict()

        for agent_identifier in range(self.args.n_banks):
            self.buffer[f'observations_{agent_identifier}'] = np.empty([self.max_size, self.args.obs_shape[agent_identifier]])
            self.buffer[f'actions_{agent_identifier}'] = np.empty([self.max_size, self.args.action_shape[agent_identifier]])
            self.buffer[f'rewards_{agent_identifier}'] = np.empty([self.max_size])
            self.buffer[f'next_observations_{agent_identifier}'] = np.empty([self.max_size, self.args.obs_shape[agent_identifier]])

        self.lock = threading.Lock()

    
    def store_episode(self, observation, action, reward, next_observation):
        """
        Store the observation, action, reward, and next observation quadruple from the current episode
        :param  None
        :output None
        """
        # Get the indices of locations to store the episode
        indices = self._get_storage_indices(increment=1)

        # Store the episode into the buffer
        for agent_identifier in range(self.args.n_banks):
            # Retrieve the lock so that the data is updated simultaneously
            with self.lock:
                self.buffer[f'observations_{agent_identifier}'][indices] = observation[agent_identifier]
                self.buffer[f'actions_{agent_identifier}'][indices] = action[agent_identifier]
                self.buffer[f'rewards_{agent_identifier}'][indices] = reward[agent_identifier]
                self.buffer[f'next_observations_{agent_identifier}'][indices] = next_observation[agent_identifier]



    def sample(self, batch_size):
        """
        Sample from the buffer
        :param  batch_size  the amount of samples to sample from the buffer
        """
        # Allocate temporary storage
        buffer = {}
        # Retrieve the indices of the samples to return
        indices = np.random.randint(0,self.current_size, batch_size)
        for key in self.buffer.keys():
            buffer[key] = self.buffer[key][indices]

        return buffer


    def _get_storage_indices(self, increment=1):
        """
        Retrieve the indices of where to store the samples
        :param  increment   the number of samples to be introduced to the buffer
        :output indices     the indices where to store the samples
        """
        if self.current_size + increment < self.max_size:
            # If there is enough space left in the buffer, then just allocate to the end of the buffer
            indices = np.arange(self.current_size, self.current_size + increment)

        elif self.current_size < self.max_size:
            # If there is not enough size left, allocate from the current position to the maximum buffer 
            # then distribute the remainder throughout the buffer
            overflow = increment - (self.max_size - self.current_size)

            indices_remainder = np.arange(self.current_size, self.max_size)
            indicies_overflow = np.random.randint(0,self.current_size, overflow)

            indices = np.concatenate([indices_remainder, indicies_overflow])

        else:
            # If the buffer is already maxed, distribute samples randomly throughout the buffer
            indices = np.random.randint(0, self.max_size, increment)

        # Update the current size
        self.current_size = min(self.max_size, self.current_size + increment)

        return indices

