from financial_network_ma.envs.financial_network_env_ma import Financial_Network_Env_Multi_Agent

def make_env(args):
    env = Financial_Network_Env_Multi_Agent(args)
    args.obs_shape = [env.observation_space[i].shape[1] for i in range(args.n_banks)]
    args.action_shape = [env.action_space[i].shape[1] for i in range(args.n_banks)]
    args.high_action = 1
    args.low_action = 0
    
    return env, args