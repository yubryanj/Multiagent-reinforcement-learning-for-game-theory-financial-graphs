import ray
from rllib_train import get_args, setup
from ray.rllib.agents.dqn import DQNTrainer
from env import Volunteers_Dilemma



if __name__ == "__main__":
    args = get_args()
    ray.init(local_mode = args.local_mode)
    config, env_config, stop = setup(args)

    config['explore'] = False


    agent = DQNTrainer(config=config, env=Volunteers_Dilemma)
    agent.restore("/itet-stor/bryayu/net_scratch/results/0/DQN/DQN_Volunteers_Dilemma_f3a8f_00000_0_2021-05-07_15-05-53/checkpoint_1/checkpoint-1")

    # instantiate env class
    env = Volunteers_Dilemma(env_config)

    agent_0_actions = []
    agent_1_actions = []

    for i in range(10):
        obs = env.reset()
        action_0 = agent.compute_action(obs[0], policy_id='policy_0')
        action_1 = agent.compute_action(obs[1], policy_id='policy_1')

        agent_0_actions.append(action_0)
        agent_1_actions.append(action_1)
    
    pass

    ray.shutdown()
