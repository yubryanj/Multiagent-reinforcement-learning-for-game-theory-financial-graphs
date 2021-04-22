import numpy as np

MAX_POSITION = 100

def generate_graph(debug = True):
    if debug:
        adjacency_matrix = [[0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [16.0, 16.0, 0.0]]

        position = [35.0, 35.0, 30.0]
    else:
        adjacency_matrix, position = generate_case_1(n_entities=3)


    return adjacency_matrix, position


def generate_case_1(n_entities):

    # Indicator if a valid graph was generated
    generated = False

    while not generated:
        # Sample a distribution for each bank
        position = np.random.multinomial(MAX_POSITION, np.ones(n_entities)/n_entities, size=1)[0]

        # Initialize a zero'd out adjacency matrix
        adjacency_matrix = np.zeros((n_entities, n_entities))

        # Generate debts
        total_debt = position[-1] + np.random.randint(position[:-1]).sum()
        adjacency_matrix[-1,:-1] = np.random.multinomial(total_debt, np.ones(n_entities-1)/(n_entities-1),size=1)[0]

        # adjacency_matrix = adjacency_matrix.tolist()
        position = position.astype(float)

        # Check that one bank is is distressed
        # And that both can rescue
        net_positions = position - np.sum(adjacency_matrix, axis=1)
        savior_banks = [bank_id for bank_id, net_position in enumerate(net_positions[:-1]) if net_position > np.abs(net_positions[-1])]

        # Conditions for successful generation
        distressed_bank_generated = True if net_positions[-1] < 0 else False
        sufficient_savior_banks = True if len(savior_banks) >= (n_entities -1) else False
    
        if distressed_bank_generated and sufficient_savior_banks:
            generated = True

    return adjacency_matrix, position


