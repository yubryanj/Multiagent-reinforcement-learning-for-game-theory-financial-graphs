from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from typing import Dict
import numpy as np



class MyCallbacks(DefaultCallbacks):

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        episode.custom_metrics[f'starting_system_value'] = episode.last_info_for(0)['starting_system_value']
        episode.custom_metrics[f'ending_system_value'] = episode.last_info_for(0)['ending_system_value']
        episode.custom_metrics[f'optimal_allocation'] = episode.last_info_for(0)['optimal_allocation']
        episode.custom_metrics[f'actual_allocation'] = episode.last_info_for(0)['actual_allocation']
        episode.custom_metrics[f'percentage_of_optimal_allocation'] = episode.last_info_for(0)['percentage_of_optimal_allocation']

        pass

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):

        # Store the action distribution of each agent
        
        if not worker.env_context['discrete']:
            action_low = policies[f'policy_{agent_id}'].action_space.low[0]
            action_high = policies[f'policy_{agent_id}'].action_space.high[0]

            for round in range(episode.length):
                episode.custom_metrics[f'action_distribution_agent{agent_id}_round{round}_mu'] = postprocessed_batch['action_dist_inputs'][round][0]
                episode.custom_metrics[f'action_distribution_agent{agent_id}_round{round}_sigma'] = np.exp(postprocessed_batch['action_dist_inputs'][round][1])
                episode.custom_metrics[f'action_agent{agent_id}'] = np.clip(postprocessed_batch['actions'][round], action_low, action_high)
                episode.custom_metrics[f'rewards_agent{agent_id}'] = postprocessed_batch['rewards'][round]
                episode.custom_metrics[f'value_targets_agent{agent_id}'] = postprocessed_batch['value_targets'][round]
                episode.custom_metrics[f'advantages_agent{agent_id}'] = postprocessed_batch['advantages'][round]

            