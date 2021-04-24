import numpy as np
from numpy.matrixlib.defmatrix import matrix
import ray



def generate_graph(debug = True, max_value = 100):
    if debug:
        adjacency_matrix = [[0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [16.0, 16.0, 0.0]]

        position = [35.0, 35.0, 30.0]
        adjacency_matrix = np.asarray(adjacency_matrix)
        position = np.asarray(position)
    else:
        adjacency_matrix, position = generate_volunteers_dilemma(n_entities=3, max_value = max_value)


    return adjacency_matrix, position


def generate_volunteers_dilemma(n_entities, max_value = 100, haircut_multiplier = 0.50 ):

    n_agents = n_entities - 1 

    # Indicator if a valid graph was generated
    generated = False

    while not generated:
        # Sample a distribution for each bank
        position_generated = False
        while not position_generated:
            position = np.random.multinomial(max_value, np.ones(n_entities)/n_entities, size=1)[0]
            position_generated = (position > 1).all()

        # Initialize a zero'd out adjacency matrix
        adjacency_matrix = np.zeros((n_entities, n_entities))

        # Generate debts
        total_debt = np.random.randint(position[2] + 1 , position[2] + position[:2].min())
        adjacency_matrix[2,:2] = np.random.multinomial(total_debt, np.ones(n_agents )/(n_agents),size=1)[0]

        # adjacency_matrix = adjacency_matrix.tolist()
        position = position.astype(float)


        """Conditions for successful generation"""

        net_positions = position - np.sum(adjacency_matrix, axis=1)
        savior_banks = [bank_id for bank_id, net_position in enumerate(net_positions[:2]) if net_position > np.abs(net_positions[2])]

        # We have at least one distressed bank
        distressed_bank_generated = True if net_positions[2] < 0 else False

        # Each bank can save the distressed bank individually
        sufficient_savior_banks = True if len(savior_banks) == (n_agents) else False

        # Position if rescued
        cost_of_rescue = -net_positions[2]
        inflows = adjacency_matrix[2,:2]
        rescued_positions = position[:n_agents] + inflows - cost_of_rescue

        # Positions if not rescued
        inflows = ((adjacency_matrix[2,:2] / total_debt) * haircut_multiplier * position[2])
        not_rescued_positions = position[:n_agents] + inflows

        # Individual_incentives
        incentives = rescued_positions - not_rescued_positions - cost_of_rescue

        each_savior_has_incentive = (incentives > 0 ).all()

        print(  f'adjacency_matrix:\n{adjacency_matrix}',
                f'\nposition:{position}', 
                f'\nrescued_positions:{rescued_positions}', 
                f'\nnot_rescued_positions:{not_rescued_positions}', 
                f'\nincentives:{incentives}', 
                f'\ncost_of_rescue:{cost_of_rescue}'
            )
    
        if  distressed_bank_generated \
            and sufficient_savior_banks \
            and each_savior_has_incentive:
            generated = True

    return adjacency_matrix, position



from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

def custom_eval_function(trainer, eval_workers):
    """Example of a custom evaluation function.
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    # We configured 2 eval workers in the training config.
    worker_1, worker_2 = eval_workers.remote_workers()

    # Set different env settings for each worker. Here we use a fixed config,
    # which also could have been computed in each worker by looking at
    # env_config.worker_index (printed in SimpleCorridor class above).
    worker_1.foreach_env.remote(lambda env: env.reset())
    worker_2.foreach_env.remote(lambda env: env.reset())

    for i in range(1):
        print("Custom evaluation round", i)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    metrics = {}
    # You can also put custom values in the metrics dict.
    metrics['starting_system_value'] = episodes[0].custom_metrics.get('starting_system_value')
    metrics['ending_system_value'] = episodes[0].custom_metrics.get('ending_system_value')
    metrics['optimal_allocation'] = episodes[0].custom_metrics.get('optimal_allocation')
    metrics['actual_allocation'] = episodes[0].custom_metrics.get('actual_allocation')
    metrics['percentage_of_optimal_allocation'] = episodes[0].custom_metrics.get('percentage_of_optimal_allocation')
    metrics['reward'] = episodes[0].agent_rewards.get((0, 'policy_0'))



    return metrics


if __name__ == "__main__":
    adjacency_matrices = []
    positions = []
    while True:
        adjacency_matrix, position = generate_volunteers_dilemma(n_entities=3, max_value=100)


