import ray
from rllib_train import get_args, setup
from ray.rllib.agents.dqn import DQNTrainer
from env import Volunteers_Dilemma

from data.heatmap import plot



if __name__ == "__main__":
    args = get_args()
    ray.init(local_mode = args.local_mode)
    config, env_config, stop = setup(args)

    config['explore'] = False

    checkpoints = [1400,1600,1800,2000]


    for checkpoint in checkpoints:
        agent = DQNTrainer(config=config, env=Volunteers_Dilemma)
        agent.restore(f"/itet-stor/bryayu/net_scratch/results/6/DQN/DQN_Volunteers_Dilemma_f75fc_00000_0_2021-05-08_10-39-56/checkpoint_{checkpoint}/checkpoint-{checkpoint}")

        # instantiate env class
        env = Volunteers_Dilemma(env_config)

        agent_0_actions = []
        agent_1_actions = []
        optimal_allocation = []

        for i in range(1000):
            obs = env.reset()
            action_0 = agent.compute_action(obs[0], policy_id='policy_0')
            action_1 = agent.compute_action(obs[1], policy_id='policy_1')

            agent_0_actions.append(action_0)
            agent_1_actions.append(action_1)
            optimal_allocation.append(-obs[0]['real_obs'][2])
        
        pass

        plot(agent_0_actions, optimal_allocation, title=f"Checkpoint {checkpoint}, Agent 0")
        plot(agent_1_actions, optimal_allocation, title=f"Checkpoint {checkpoint}, Agent 1")

    ray.shutdown()
