import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, args, agent_identifier) -> None:
        """
        Initializes the Actor model
        :param  args                dictionary of args
        :param  agent_identifier    identifier for current agent
        """
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.mean1 = nn.Linear(args.obs_shape[agent_identifier], 64)
        self.mean2 = nn.Linear(64,64)
        self.mean3 = nn.Linear(64,64)
        self.mean = nn.Linear(64, 1)


        self.var1 = nn.Linear(args.obs_shape[agent_identifier], 64)
        self.var2 = nn.Linear(64,64)
        self.var3 = nn.Linear(64,64)
        self.variance = nn.Linear(64, 1)


    def forward(self, x):
        """
        Conduct a forward pass of the model
        :param  x   input of the model
        """
        mean = F.relu(self.mean1(x))
        mean = F.relu(self.mean2(mean))
        mean = F.relu(self.mean3(mean))
        mu = self.mean(mean)

        var = F.relu(self.var1(x))
        var = F.relu(self.var2(var))
        var = F.relu(self.var3(var))
        std = self.variance(var)
        std = torch.exp(std / 2)

        q = torch.distributions.normal.Normal(mu, std)
        action = q.rsample()
        action = torch.sigmoid(action)

        return action


class Critic(nn.Module):
    def __init__(self, args) -> None:
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)

        # Outputs a a scalar - Q-value
        self.q_out = nn.Linear(64,1)

    def forward(self, state, action):
        """
        Conducts a forward pass in the critic network
        :param  state   current state vector
        :param  action  action vector 
        :output q_value valuation of the taken state-action pair
        """
        state = torch.cat(state, dim=1)
        
        # for i in range(len(action)):
        #     action[i] /= self.max_action

        # Concatenate the action vector 
        action = torch.cat(action, dim=1).reshape(-1,1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

