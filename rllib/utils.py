import numpy as np

MAX_POSITION = 100

def generate_graph(debug = True, n_agents = 2, case_1=True):
    # if debug:
    adjacency_matrix = [[0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [15.0, 15.0, 0.0]]

    position = [30.0, 30.0, 29.0]
    # else:
    #     adjacency_matrix, position = generate_case_1(n_entities=3)


    return adjacency_matrix, position


def generate_case_1(n_entities, overdraw = 1.0):
    position = np.random.rand(n_entities)
    position = position/ np.sum(position) * MAX_POSITION

    adjacency_matrix = np.zeros((n_entities, n_entities))
    adjacency_matrix[-1,:-1] = (position[-1] + overdraw )/2.0

    adjacency_matrix = adjacency_matrix.tolist()
    position = position.tolist()

    return adjacency_matrix, position


