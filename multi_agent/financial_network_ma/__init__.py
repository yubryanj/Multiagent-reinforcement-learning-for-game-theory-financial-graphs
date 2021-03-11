from gym.envs.registration import register

register(
    id='Financial-network-ma-v0',
    entry_point='financial_network_ma.envs:Financial_Network_Env_Multi_Agent',
    )

