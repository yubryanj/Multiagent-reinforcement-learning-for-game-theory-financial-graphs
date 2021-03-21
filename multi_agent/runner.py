from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import copy


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def _init_agents(self):
        agents = []
        # Share the same agent and critic; Move below if a different agent/critic per
        
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents


    def run(self):
        returns = []
        average_net_position = float('-inf')

        # Run this loop repeatedly
        for time_step in tqdm(range(self.args.time_steps)):
            
            # reset the environment and get the first sample
            obs, _ = self.env.reset()

            # observations would be modified after taking a "step" if a deep copy is not made
            initial_obs = copy.deepcopy(obs)
            actions = []

            with torch.no_grad():
                # For each agent
                for agent_id, agent in enumerate(self.agents):
                    # select an action
                    action = agent.select_action(obs[agent_id], self.noise, self.epsilon, evaluate=False)
                    # store the action 
                    actions.append(action)

            # Take the next action; retrieve next obs, reward, done, and additional information
            next_obs, reward, done, info = self.env.step(actions)

            # Store the episode in the replay buffer
            self.buffer.store_episode(initial_obs[:self.args.n_agents], actions, reward[:self.args.n_agents], next_obs[:self.args.n_agents])

            # Update the obs
            obs = next_obs

            # If there are enough samples in the buffer
            if self.buffer.current_size >= self.args.batch_size:

                # Get a sample from the buffer of (s,a,r,s')
                transitions = self.buffer.sample(self.args.batch_size)

                # Train each agent
                for agent in self.agents:

                    # Get a list of the other agents
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)

                    # Train the current agent on the world transitions
                    agent.learn(transitions, other_agents)

            # Evaluate the learning
            if time_step >0 and time_step % self.args.evaluate_rate == 0:
                print(f'Timestep {time_step}: Conducting an evaluation:')
                average_net_position = self.evaluate(self.args)
                returns.append(average_net_position)

            # Generate Noise
            self.noise = 0
            self.epsilon = max(0.20, self.epsilon - 0.00005)

            # Save the returns
            # np.save(f'{self.args.save_dir}/returns.pkl', returns)




    def evaluate(self, args=None):
        # Allocate lists to store results from info
        positions = []
        system_configurations = []

        for episode in range(self.args.evaluate_episodes):

            # reset the environment
            s, info = self.env.reset()

            system_configurations.append({'initial_configurations':copy.deepcopy(info)})

            # Obtain the results for a series of trainings
            for time_step in range(self.args.evaluate_episode_len):

                actions = []
                
                # Zero out the gradients
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        # Select the action for the given agent
                        action = agent.select_action(s[agent_id], 0, 0, evaluate=True)
                        actions.append(action)
                
                # Take the next action
                s_next, rewards, done, info = self.env.step(actions)

                # Update the obs
                s = s_next

                # Store the action taken
                system_configurations[-1]['action'] = copy.deepcopy(actions)
                positions.append(info['cleared_positions'])
                
            # Store the cumulative rewards
            system_configurations[-1]['final_configurations'] = copy.deepcopy(info)

        print(f'Average positions: {np.mean(positions,axis=0)}, total: {np.mean(np.sum(positions, axis=1))}')

        np.save(f"{self.args.save_dir}/evaluation-data", system_configurations)

        return np.mean(positions)