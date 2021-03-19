from scenarios.envs.volunteers_dilemma import Volunteers_dilemma
import numpy as np

def make_env(args):

    args = load_scenario(args)

    env = Volunteers_dilemma(args)
    args.obs_shape = [env.observation_space[i].shape[1] for i in range(args.n_agents)]
    args.action_shape = [env.action_space[i].shape[1] for i in range(args.n_agents)]
    args.high_action = 1
    args.low_action = 0
    
    return env, args


def load_scenario(args):
    assert args.experiment_name in ['volunteers_dilemma'], "Not a valid experiment setting"

    if args.experiment_name == 'volunteers_dilemma':
        adjacency_matrix, position = volunteers_dilemma()

    args.adjacency_matrix = adjacency_matrix
    args.position = position

    return args


def volunteers_dilemma():
    adjacency_matrix = [[0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [15.0, 15.0, 0.0]]

    position = [20.0, 20.0, 29.0]

    return adjacency_matrix, position