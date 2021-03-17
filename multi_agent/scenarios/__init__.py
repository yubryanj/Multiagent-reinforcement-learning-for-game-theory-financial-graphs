from gym.envs.registration import register

register(
    id='volunteers_dilemma-ma-v0',
    entry_point='network.envs:Volunteers_Dilemma',
    )

