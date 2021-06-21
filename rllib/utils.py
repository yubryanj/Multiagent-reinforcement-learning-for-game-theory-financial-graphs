from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from typing import Dict
import numpy as np
import ray
import argparse
import json
import os


class MyCallbacks(DefaultCallbacks):

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        # if worker.env_context['discrete']:
        #     episode.custom_metrics[f'current_epsilon'] = policies['policy_0'].exploration.get_info()['cur_epsilon']
        
        
        # episode.custom_metrics[f'starting_system_value'] = episode.last_info_for(0)['starting_system_value']
        # episode.custom_metrics[f'ending_system_value'] = episode.last_info_for(0)['ending_system_value']
        # episode.custom_metrics[f'percentage_of_optimal_allocation'] = episode.last_info_for(0)['percentage_of_optimal_allocation']
        # episode.custom_metrics[f'optimal_allocation'] = episode.last_info_for(0)['optimal_allocation']

        # for i in range(base_env.envs[0].config['n_agents']):
        #     episode.custom_metrics[f'{i}_actual_allocation'] = episode.last_info_for(i)['actual_allocation']


        # log_dir = worker._original_kwargs.get('log_dir')
        pass
        


def generate_graph(
    config,
    rescue_amount = 3
    ):
    if config.get('debug'):
        adjacency_matrix = [[0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [16.0, 16.0, 0.0]]

        position = [35.0, 35.0, 30.0]
        adjacency_matrix = np.asarray(adjacency_matrix)
        position = np.asarray(position)
    elif config.get('scenario') == 'volunteers dilemma':
        adjacency_matrix, position = generate_volunteers_dilemma(
            config,
            rescue_amount=rescue_amount
        )
    elif config.get('scenario') == 'only one can bailout':
        adjacency_matrix, position = generate_only_one_can_bailout(
            config,
            rescue_amount=rescue_amount
        )
    elif config.get('scenario') == 'no one can bailout':
        adjacency_matrix, position = generate_no_one_can_bailout(
            config,
            rescue_amount=rescue_amount
        )
    elif config.get('scenario') == 'no point to bailout':
        adjacency_matrix, position = generate_no_point_to_bailout(
            config,
            rescue_amount=rescue_amount
        )
    elif config.get('scenario') == 'coordination game':
        adjacency_matrix, position = generate_coordination_game(
            config,
            rescue_amount = rescue_amount
        )
    elif config.get('scenario') == 'both are rich enough to bailout':
        adjacency_matrix, position = generate_both_are_rich_enough_to_bailout(
            config,
            rescue_amount = rescue_amount
        )

    else:
        assert False, "Invalid scenario requested.  Select amongst \
            'volunteers dilemma', \
            'only one can bailout', \
            ' no one can bailout', \
            'no point to bailout', \
            'coordination game', \
            'both are rich enough to bailout', \
            "


    return adjacency_matrix, position


def generate_both_are_rich_enough_to_bailout(
    config,
    rescue_amount
    ):
    pass


def generate_coordination_game(
    config,
    rescue_amount
    ):
    pass


def generate_random_adjacency_matrix(
    config,
    position,
    rescue_amount
    ):
    adjacency_matrix = np.zeros((config.get('n_entities'), config.get('n_entities')))

    total_debt = position[2] + rescue_amount
    adjacency_matrix[2,:2] = np.random.multinomial(
        total_debt, 
        np.ones(config.get('n_agents'))
        /(config.get('n_agents')),size=1)[0]

    return adjacency_matrix
    

def volunteers_graph_generator(
    config,
    number_of_saviors,
    rescue_amount,
    ):

    position_generated = False
    while not position_generated:
        position = np.random.multinomial(
            config.get('max_system_value'), 
            np.ones(config.get('n_entities'))
            /config.get('n_entities'), size=1)[0]
        position_generated = (position > 1).all()

    # Generate debts
    adjacency_matrix = generate_random_adjacency_matrix(
        config,
        position,
        rescue_amount
    )


    position = position.astype(float)

    return position, adjacency_matrix


def only_n_can_rescue(
    config,
    number_of_saviors,
    rescue_amount,
    ):
    
    # TODO: DEBUG, there's a bug here for experiment 42

    position = np.zeros((config.get('n_entities')))

    banks = np.arange(0,config.get('n_agents'))
    np.random.shuffle(banks)

    saviors = banks[:number_of_saviors]
    non_saviors = banks[number_of_saviors:]
    remaining_amount_to_allocate = config.get('max_system_value')
    
    position_generated = False
    while not position_generated:
        for bank_id in non_saviors:
            allocation = np.random.randint(
                1,
                rescue_amount-1
            )
            position[bank_id] = allocation
            remaining_amount_to_allocate -= allocation

                
        # Allocate the remainder including distressed bank
        n_to_allocate_to = len(saviors) + 1
        allocations = np.random.multinomial(
            remaining_amount_to_allocate, 
            np.ones(n_to_allocate_to)/n_to_allocate_to
            , size=1)[0]
        for bank_id, allocation in zip(saviors,allocations[:-1]):
            position[bank_id] = allocation

        # Allocate to the distressed bank
        position[2] = allocations[-1]

        if position.sum() <= config.get('max_system_value') and (position>0).all():
            position_generated = True

    # Generate debts
    adjacency_matrix = generate_random_adjacency_matrix(
        config,
        position,
        rescue_amount
    )

    position = position.astype(float)

    return position, adjacency_matrix


def no_point_to_bailout(
    config,
    number_of_saviors,
    rescue_amount,
    ):
        
    position_generated = False
    while not position_generated:
        position = np.random.multinomial(
            config.get('max_system_value'), 
            np.ones(config.get('n_entities'))
            /config.get('n_entities'), size=1)[0]
        position_generated = (position > 1).all()


    adjacency_matrix = np.zeros((config.get('n_entities'), config.get('n_entities')))
    total_debt = position[2] + rescue_amount
    adjacency_matrix[2,:2] = np.random.multinomial(
        total_debt, 
        np.ones(config.get('n_agents'))/(config.get('n_agents')),size=1)[0]

    position = position.astype(float)

    return position, adjacency_matrix


def compute_number_of_defaulted_banks(
    net_positions
    ):

    return (net_positions < 0).sum()


def compute_number_of_savior_banks(
    position,
    adjacency_matrix
    ):
    net_positions = compute_net_position(position, adjacency_matrix)
    savior_banks = [bank_id for bank_id, net_position in enumerate(net_positions[:2]) if net_position > np.abs(net_positions[2])]

    return len(savior_banks), savior_banks


def compute_incentives(
    position,
    adjacency_matrix,
    config,
    cost_of_rescue
    ):
    inflows = adjacency_matrix[2,:2]
    rescued_positions = position[:config.get('n_agents')] + inflows - cost_of_rescue

    # Positions if not rescued
    total_debt = np.sum(adjacency_matrix[2,:2])
    inflows = ((adjacency_matrix[2,:2] / total_debt) * config.get('haircut_multiplier') * position[2])
    not_rescued_positions = position[:config.get('n_agents')] + inflows

    # Individual_incentives
    incentives = rescued_positions - not_rescued_positions - cost_of_rescue

    return incentives


def compute_net_position(
    position, 
    adjacency_matrix
    ):
    return position - np.sum(adjacency_matrix, axis=1)


def each_savior_has_incentive(
    incentives, 
    savior_banks
    ):
    conditions_met = True
    for savior_bank_id in savior_banks:
        if incentives[savior_bank_id] < 0:
            conditions_met = False

    return conditions_met


def no_savior_has_incentive(
    incentives, 
    savior_banks
    ):
    conditions_met = True
    for savior_bank_id in savior_banks:
        if incentives[savior_bank_id] > 0:
            conditions_met = False

    return conditions_met


def create_environment(
    config,
    required_number_of_savior_banks,
    incentive_function,
    generate,
    rescue_amount = 1,
    ):

    # Indicator if a valid graph was generated
    generated = False

    while not generated:
        position, adjacency_matrix = generate(
            config, 
            required_number_of_savior_banks,
            rescue_amount
        )

        net_positions   = compute_net_position(position, adjacency_matrix)
        cost_of_rescue  = -net_positions[2]
        incentives      = compute_incentives(position, adjacency_matrix, config, cost_of_rescue)
        n_savior_banks, savior_banks  = compute_number_of_savior_banks(position, adjacency_matrix)

        """Conditions for successful generation"""
        # We have at least one distressed bank
        distressed_bank_generated = True if net_positions[2] < 0 else False

        # Each bank can save the distressed bank individually
        sufficient_savior_banks = True if n_savior_banks == required_number_of_savior_banks else False

        incentives_met = incentive_function(incentives, savior_banks)

        #  position falls within 0, 100 in total
        valid_net_positions     = (net_positions[:-1]>0).all() and (net_positions<config.get('max_system_value')).all()        

        if  distressed_bank_generated \
            and sufficient_savior_banks \
            and incentives_met \
            and rescue_amount == cost_of_rescue\
            and valid_net_positions:
            generated = True

    return adjacency_matrix, position


def generate_volunteers_dilemma(
    config,
    rescue_amount = 1
    ):

    adjacency_matrix, position = create_environment(
        config, 
        required_number_of_savior_banks= config.get('n_agents'), 
        incentive_function = each_savior_has_incentive,
        generate=volunteers_graph_generator,
        rescue_amount = rescue_amount,
    )

    return adjacency_matrix, position


def generate_only_one_can_bailout(
    config,
    rescue_amount = 2
    ):

    adjacency_matrix, position = create_environment(
        config,
        required_number_of_savior_banks = 1,
        incentive_function = each_savior_has_incentive,
        generate = only_n_can_rescue,
        rescue_amount = rescue_amount
    )

    return adjacency_matrix, position


def generate_no_one_can_bailout(
    config,
    required_n_savior_banks = 0,
    rescue_amount = 3
    ):
    
    adjacency_matrix, position = create_environment(
        config,
        required_number_of_savior_banks = 0,
        incentive_function = each_savior_has_incentive,
        generate = only_n_can_rescue,
        rescue_amount = rescue_amount
    )
    return adjacency_matrix, position


def generate_no_point_to_bailout(
    config,
    required_n_savior_banks = 2,
    rescue_amount = 3
    ):

    # TODO: Check if this is correct!  The generated graph and the incentive structure may not align.
    # Forced by using haircut multiplier == 1.0
    adjacency_matrix, position = create_environment(
        config,
        required_number_of_savior_banks = required_n_savior_banks,
        incentive_function = no_savior_has_incentive,
        generate = volunteers_graph_generator,
        rescue_amount = rescue_amount
    )       

    return adjacency_matrix, position


from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

def custom_eval_function(
    trainer, 
    eval_workers
    ):
    """Example of a custom evaluation function.
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    # We configured 2 eval workers in the training config.
    worker_1 = eval_workers.remote_workers()[0]

    # Set different env settings for each worker. Here we use a fixed config,
    # which also could have been computed in each worker by looking at
    # env_config.worker_index (printed in SimpleCorridor class above).
    worker_1.foreach_env.remote(lambda env: env.reset())

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

    agent_0_allocation = episodes[0].custom_metrics.get('0_actual_allocation') if episodes[0].custom_metrics.get('0_actual_allocation') is not None else 0
    agent_1_allocation = episodes[0].custom_metrics.get('1_actual_allocation') if episodes[0].custom_metrics.get('1_actual_allocation') is not None else 0

    metrics = {}
    # You can also put custom values in the metrics dict.
    metrics['starting_system_value'] = episodes[0].custom_metrics.get('starting_system_value')
    metrics['ending_system_value'] = episodes[0].custom_metrics.get('ending_system_value')
    metrics['optimal_allocation'] = episodes[0].custom_metrics.get('optimal_allocation')
    metrics['actual_allocation'] = episodes[0].custom_metrics.get('actual_allocation')
    metrics['current_epsilon'] = episodes[0].custom_metrics.get('current_epsilon')
    metrics['0_actual_allocation'] = agent_0_allocation
    metrics['1_actual_allocation'] = agent_1_allocation
    metrics['percentage_of_optimal_allocation'] = (agent_0_allocation + agent_1_allocation)/metrics['optimal_allocation']

    return metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-test",    action="store_true")
    parser.add_argument("--local-mode", action="store_true")
    parser.add_argument("--discrete",   action="store_true")
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--basic-model",   action="store_true")
    parser.add_argument("--run",        type=str, default="DQN")
    parser.add_argument("--n-agents",   type=int, default=2)
    parser.add_argument("--n-workers",  type=int, default=1)
    parser.add_argument("--n-samples",  type=int, default=1)
    parser.add_argument("--n-gpus",     type=int, default=0)
    parser.add_argument("--stop-iters", type=int, default=1)
    parser.add_argument("--checkpoint-frequency", type=int, default=50)
    parser.add_argument("--haircut-multiplier", type=float, default=0.50)
    parser.add_argument("--max-system-value", type=int, default=100)
    parser.add_argument("--restore",    type=str)
    parser.add_argument("--seed",       type=int, default=123)
    parser.add_argument("--experiment-number", type=int, default=000)
    parser.add_argument("--number-of-negotiation-rounds", type=int, default=1)
    parser.add_argument("--alpha",      type=int, default=1)    # Prosocial parameter
    parser.add_argument("--beta",       type=int, default=0)    # Prosocial parameter
    parser.add_argument("--scenario",       type=str, default="volunteers dilemma")    # Prosocial parameter
    args = parser.parse_args()
    args.log_dir = f"/itet-stor/bryayu/net_scratch/results/{args.experiment_number}"


    with open('configs.json') as f:
        data = f.read()
    data = json.loads(data)

    if f'{args.experiment_number}' in data.keys():
        configs = data[f'{args.experiment_number}']
        for key in configs:
            setattr(args,key,configs.get(key))


    return args