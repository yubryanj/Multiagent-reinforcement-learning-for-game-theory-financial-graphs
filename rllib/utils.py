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
import pandas as pd
import prettytable



from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import AgentID, PolicyID
from typing import Dict, Optional, TYPE_CHECKING


from generator import Generator


class MyCallbacks(DefaultCallbacks):

    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode,
                         env_index: Optional[int] = None,
                         **kwargs):
        """
        Callback run on the rollout worker before each episode starts.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index (EnvID): Obsoleted: The ID of the environment, which the
                episode belongs to.
            kwargs: Forward compatibility placeholder.
        """

        pass


    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode,
                        env_index: Optional[int] = None,
                        **kwargs):
        """Runs on each episode step.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index (EnvID): Obsoleted: The ID of the environment, which the
                episode belongs to.
            kwargs: Forward compatibility placeholder.
        """

        agent_0_policy  = episode._agent_to_policy[0]
        agent_1_policy  = episode._agent_to_policy[1]

        agent_0_beta    = episode._policies[agent_0_policy].config.get('beta')
        agent_1_beta    = episode._policies[agent_1_policy].config.get('beta')

        # Set agent 0 beta
        base_env.envs[0].config['agent_0_beta']     = agent_0_beta
        base_env.envs[0].config['agent_0_policy']   = agent_0_policy
        worker.env.config['agent_0_beta']           = agent_0_beta
        worker.env.config['agent_0_policy']         = agent_0_policy
        
        # Set agent 1 beta
        base_env.envs[0].config['agent_1_beta']     = agent_1_beta
        base_env.envs[0].config['agent_1_policy']   = agent_1_policy
        worker.env.config['agent_1_beta']           = agent_1_beta
        worker.env.config['agent_1_policy']         = agent_1_policy

        pass



    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        # if worker.env_context['discrete']:
        #     episode.custom_metrics[f'current_epsilon'] = policies['policy_0'].exploration.get_info()['cur_epsilon']
        
        # episode.custom_metrics[f'starting_system_value'] = episode.last_info_for(0)['starting_system_value']
        # episode.custom_metrics[f'ending_system_value'] = episode.last_info_for(0)['ending_system_value']
        # # episode.custom_metrics[f'percentage_of_optimal_allocation'] = episode.last_info_for(0)['percentage_of_optimal_allocation']
        # episode.custom_metrics[f'optimal_allocation'] = episode.last_info_for(0)['optimal_allocation']

        # for i in range(base_env.envs[0].config['n_agents']):
        #     episode.custom_metrics[f'{i}_actual_allocation'] = episode.last_info_for(i)['actual_allocation']


        # # log_dir = worker._original_kwargs.get('log_dir')
        pass


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
    from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes


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
    parser.add_argument("--as-test",                        action="store_true")
    parser.add_argument("--local-mode",                     action="store_true")
    parser.add_argument("--discrete",                       action="store_true")
    parser.add_argument("--debug",                          action="store_true")
    parser.add_argument("--basic-model",                    action="store_true")
    parser.add_argument("--invert-actions",                 action="store_true")
    parser.add_argument("--evaluate-during-training",       action="store_true")
    parser.add_argument("--pooled-training",                action="store_true")
    parser.add_argument("--restore",            type=str)
    parser.add_argument("--run",                type=str,   default="DQN")
    parser.add_argument("--n-agents",           type=int,   default=2)
    parser.add_argument("--embedding-size",     type=int,   default=32)
    parser.add_argument("--n-workers",          type=int,   default=1)
    parser.add_argument("--n-samples",          type=int,   default=1)
    parser.add_argument("--n-gpus",             type=int,   default=0)
    parser.add_argument("--stop-iters",         type=int,   default=1)
    parser.add_argument("--checkpoint-frequency", type=int, default=50)
    parser.add_argument("--haircut-multiplier", type=float, default=0.50)
    parser.add_argument("--initial-epsilon",    type=float, default=0.90)
    parser.add_argument("--final-epsilon",      type=float, default=0.10)
    parser.add_argument("--max-system-value",   type=int,   default=100)
    parser.add_argument("--seed",               type=int,   default=123)
    parser.add_argument("--experiment-number",  type=int,   default=000)
    parser.add_argument("--alpha",              type=int,   default=1)
    parser.add_argument("--beta",               type=int,   default=0)
    parser.add_argument("--scenario",           type=str,   default="volunteers dilemma")
    parser.add_argument("--minimum_rescue_amount",          type=int,   default=3)
    parser.add_argument("--maximum_rescue_amount",          type=int,   default=7)
    parser.add_argument("--number-of-negotiation-rounds",   type=int,   default=1)
    args = parser.parse_args()
    args.log_dir = f"/itet-stor/bryayu/net_scratch/results/{args.experiment_number}"


    with open('configs.json') as f:
        data = f.read()
    data = json.loads(data)

    if f'{args.experiment_number}' in data.keys():
        configs = data[f'{args.experiment_number}']
        for key in configs:
            setattr(args,key,configs.get(key))

    setattr(args,'n_entities',args.n_agents + 1)

    if hasattr(args,'policies'):
        setattr(args,'pool_size',len(args.policies))

    return args


def print_graph(
    position,
    adjacency_matrix, 
    config
    ):
    print('\\begin{figure}[!htb]')
    print('    \centering')
    print('    \\begin{tikzpicture}[node distance={25mm}, thick, main/.style = {draw, circle}]')
    print('        \\node[main] (1) {$A_{' + str(position[2]) + '}$}; ')
    print('        \\node[main] (2) [below left of=1] {$B_{' + str(position[0]) + '}$};')
    print('        \\node[main] (3) [below right of=1] {$C_{' + str(position[1]) + '}$};')
    print('        \draw[->] (1) -- node[midway, above left] {' + str(adjacency_matrix[2][0]) + '} (2);')
    print('        \draw[->] (1) -- node[midway, above right] {' + str(adjacency_matrix[2][1]) + '} (3);')
    print('    \\end{tikzpicture}')
    print('    \\caption{\\textbf{' + f'{config["scenario"]}' +' Financial Graph} Bank $\mathit{B}$ with capitalization of ' + str(position[0]) + ' assets and bank $\mathit{C}$ with capitalized with ' + str(position[1]) + ' of assets face the dilemma of whether or not to rescue bank $\mathit{A}$.  Bank $\mathit{A}$ has capitalization of ' + str(position[2]) + ' assets but total outgoing liabilities of ' + str(np.sum(adjacency_matrix[2])) + ' assets with '  + str(adjacency_matrix[2][0]) + ' due bank $\mathit{B}$ and ' + str(adjacency_matrix[2][1]) + ' due bank $\mathit{C}$.  A rescue would only be successful if '+ f'{config["rescue_amount"]}' + ' units of assets or greater is contributed to bank $\mathit{A}$.}')
    print('    \\label{fig:' + f'{config["scenario"]}' +': financial graph}')
    print('\\end{figure}')


def print_table(
    data,
    config,
    x = 3,
    y = 2):

    x_value = str(eval(data[x][y].strip('\\textbf').strip('{').strip('}'))[0])
    y_value = str(eval(data[x][y].strip('\\textbf').strip('{').strip('}'))[1])

    print('\\begin{figure}[!htb]')
    print('    \\centering')
    print('    \\begin{tabular}{p{5pt}c|cccccc}')
    print('        \multicolumn{1}{}{} & \multicolumn{7}{c}{Bank C} \\\\')
    print('            ~ & ~ &  0 & 1 & 2 & 3 & 4 & 5   \\\\')
    print('        \cline{2-8}')
    print('        \multirow{6}{*}{\\rotatebox[origin=c]{90}{Bank B}}')

    for row_id, row in enumerate(data):
        print(f'            & {row_id} & {" & ".join(row)}\\\\')

    print('    \end{tabular}')
    print('    \caption{\\textbf{' +  f'{config["scenario"]}' +' Payoff Matrix} The rows of the table represent bank $\mathit{B}$\'s contributions and similarly the columns represent bank $\mathit{C}$\'s contributions.  The payoff to invalid actions are reported as a dash (-).  Each cell in the table reports the payoff of a joint-contribution as a tuple with the first number reporting bank $\mathit{B}$\'s payoff and the second number reporting bank $\mathit{C}$\'s payoff.  For example, if bank $\mathit{B}$ contributes ' + f'{x}' + ' assets and bank $\mathit{C}$ contributes ' + f'{y}' + ' assets, the payoff tuple is ' + data[x][y]  + ' reflecting a payoff of '+ x_value +' assets to bank $\mathit{B}$ and ' + y_value +' assets to bank $\mathit{C}$. \\newline \hspace*{10mm} As per figure \\ref{fig:' + f'{config["scenario"]}' + ': financial graph}, a successful rescue occurs if ' + f'{config["rescue_amount"]}' + ' units of assets or greater is contributed to bank $\mathit{A}$.  Made bold within the table are the nash equilibriums, policies where neither bank $\mathit{B}$ nor bank $\mathit{C}$ can unilaterally improve their outcomes. Specific to this financial graph, the nash equilibriums occur when the joint-allocation sum to the rescue amount.}')
    print('    \label{fig:' + f'{config["scenario"]}' + ': Payoff Matrix}')
    print('\end{figure}')




def compute_payoff_matrices():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario",           type=str,   default="Volunteers Dilemma")
    args = parser.parse_args()

    config = {}

    scenarios = {
        "Volunteers Dilemma": {
            "position": np.array(
                [5,5,12]
            ),
            "adjacency_matrix":np.array([
                [0,0,0],
                [0,0,0],
                [8,8,0]
            ])
        },
        "Coordination Game": {
            "position": np.array(
                [3,3,12]
            ),
            "adjacency_matrix":np.array([
                [0,0,0],
                [0,0,0],
                [8,8,0]
            ])
        },
        "No one can rescue": {
            "position": np.array(
                [1,1,12]
            ),
            "adjacency_matrix":np.array([
                [0,0,0],
                [0,0,0],
                [8,8,0]
            ])
        },
        "Only Agent 0 can rescue": {
            "position": np.array(
                [6,2,12]
            ),
            "adjacency_matrix":np.array([
                [0,0,0],
                [0,0,0],
                [8,8,0]
            ])
        },
        "Only Agent 1 can rescue": {
            "position": np.array(
                [2,6,12]
            ),
            "adjacency_matrix":np.array([
                [0,0,0],
                [0,0,0],
                [8,8,0]
            ])
        },
        "Not in Default": {
            "position": np.array(
                [6,6,20]
            ),
            "adjacency_matrix":np.array([
                [0,0,0],
                [0,0,0],
                [8,8,0]
            ])
        },


        
    }

    config['scenario'] = args.scenario
    position = scenarios[config['scenario']]['position']
    adjacency_matrix = scenarios[config['scenario']]['adjacency_matrix']

    config['rescue_amount'] = max(np.sum(adjacency_matrix[2]) - position[2], 0)

    maximum_payoff = 6
    # payoff_matrix = prettytable.PrettyTable([str(i) for i in range(maximum_payoff)])
    data = []

    for i in range(maximum_payoff):
        payoffs = []
        for j in range(maximum_payoff):
            
            if i <= position[0] and j<= position[1]:
            
                agent_0_allocation = i
                agent_1_allocation = j
                total_allocation = agent_0_allocation + agent_1_allocation
                

                if total_allocation >= config['rescue_amount']:
                    b_reward = adjacency_matrix[2][0] - agent_0_allocation
                    c_reward = adjacency_matrix[2][1] - agent_1_allocation
                else:
                    b_reward = int((0.5 * adjacency_matrix[2][0]) - agent_0_allocation)
                    c_reward = int((0.5 * adjacency_matrix[2][1]) - agent_1_allocation)

                if total_allocation == config['rescue_amount']:
                    payoffs.append("\\textbf{" + f'{b_reward}, {c_reward}' + "}")
                else:
                    payoffs.append(f'{b_reward}, {c_reward}')
            else:
                payoffs.append('-')

        data.append(payoffs)



    print("\n\n")
    print_graph(position, adjacency_matrix, config)

    print("\n\n")

    if args.scenario not in ['No one can rescue', 'Only Agent 1 can rescue', 'Only Agent 0 can rescue']:
        print_table(
            data, 
            config
        )
    else:

        print_table(
            data, 
            config, 
            x = position[0] - 1, 
            y = position[1] - 1,
        )




if __name__ == "__main__":
    compute_payoff_matrices()
