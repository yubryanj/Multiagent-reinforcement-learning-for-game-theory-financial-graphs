import numpy as np
import ray


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



from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.examples.env.simple_corridor import SimpleCorridor 

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
    metrics['optimal_allocation'] = episodes[0].custom_metrics.get('optimal_allocation')
    metrics['actual_allocation'] = episodes[0].custom_metrics.get('actual_allocation')
    metrics['percentage_of_optimal_allocation'] = episodes[0].custom_metrics.get('percentage_of_optimal_allocation')
    metrics['reward'] = episodes[0].agent_rewards.get((0, 'policy_0'))



    return metrics
