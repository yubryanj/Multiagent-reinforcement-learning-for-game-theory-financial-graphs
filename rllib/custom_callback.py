from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from typing import Dict



class MyCallbacks(DefaultCallbacks):

    def on_postprocess_trajectory(
            self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):

        # Store the action distribution of each agent
        for round in range(episode.length):
            episode.custom_metrics[f'action_distribution_agent{agent_id}_round{round}_mean'] = postprocessed_batch['action_dist_inputs'][round][0]
            episode.custom_metrics[f'action_distribution_agent{agent_id}_round{round}_variance'] = postprocessed_batch['action_dist_inputs'][round][1]
            episode.custom_metrics[f'action_agent{agent_id}'] = postprocessed_batch['actions'][round]
            episode.custom_metrics[f'rewards_agent{agent_id}'] = postprocessed_batch['rewards'][round]
            episode.custom_metrics[f'value_targets_agent{agent_id}'] = postprocessed_batch['value_targets'][round]
            episode.custom_metrics[f'advantages_agent{agent_id}'] = postprocessed_batch['advantages'][round]

        