from gym.envs.registration import register

register(
    id='network-ma-v0',
    entry_point='network.envs:Network',
    )

