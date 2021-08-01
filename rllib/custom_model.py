from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX


torch, nn = try_import_torch()


class embedding_network(nn.Module):
    def __init__(self, number_of_embeddings, embedding_size=100):
        super().__init__()
        self.embedding = nn.Embedding(
            number_of_embeddings, 
            embedding_size
        )

    def forward(self, x):
        return self.embedding(x.long())


class offer_network(nn.Module):
    def __init__(self, obs_size, num_outputs):
        super().__init__()        
        self.layer1 = torch.nn.Linear(obs_size, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, 128)
        self.action = torch.nn.Linear(128, num_outputs)


    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.layer3(x)
        x = nn.functional.relu(x)
        logits = self.action(x)

        return logits


class Custom_discrete_model_with_masking(TorchModelV2, nn.Module):
    """Torch version of FastModel (tf)."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        number_of_embeddings        = model_config.get('custom_model_config').get('num_embeddings')
        embedding_size              = model_config.get('custom_model_config').get('embedding_size')

        self.embed_assets           = embedding_network(number_of_embeddings, embedding_size)
        self.embed_liabilities      = embedding_network(number_of_embeddings, embedding_size)
        self.embed_net_position     = embedding_network(number_of_embeddings, embedding_size)
        self.embed_last_offer       = embedding_network(number_of_embeddings, embedding_size)
        self.embed_rescue_amount    = embedding_network(number_of_embeddings, embedding_size)
        self.embed_final_round      = embedding_network(number_of_embeddings, embedding_size)

        self.combining_network      = nn.Linear(embedding_size * 6, embedding_size)
        self.proposal_network       = offer_network(embedding_size, num_outputs)
        self.value                  = torch.nn.Linear(embedding_size, 1)

        self._value_input           = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        action_mask     = input_dict.get('obs').get('action_mask')

        assets          = self.embed_assets(      input_dict.get('obs').get('assets'))
        liabilities     = self.embed_liabilities( input_dict.get('obs').get('liabilities'))
        net_position    = self.embed_net_position(input_dict.get('obs').get('net_position'))
        rescue_amount   = self.embed_rescue_amount(input_dict.get('obs').get('rescue_amount'))
        # Inflows
        last_offer      = self.embed_last_offer(  input_dict.get('obs').get('last_offer'))
        final_round     = self.embed_final_round( input_dict.get('obs').get('final_round'))

        # Consider attention later
        hidden_vector = torch.cat([
            assets,
            liabilities,
            net_position,
            rescue_amount,
            last_offer,
            final_round,
        ],-1)

        hidden_vector = self.combining_network(hidden_vector)
        hidden_vector = nn.functional.relu(hidden_vector)

        self._value_input = hidden_vector

        logits = self.proposal_network(hidden_vector).squeeze()
        inf_mask = torch.clamp(torch.log(action_mask),FLOAT_MIN, FLOAT_MAX)

        # Apply the masks
        logits = logits + inf_mask

        return logits, []


    @override(ModelV2)
    def value_function(self):
        assert self.hidden_vector is not None, "must call forward first!"
        return torch.reshape(self.value(self._value_input), [-1])


class Generalized_model_with_masking(TorchModelV2, nn.Module):
    """Torch version of FastModel (tf)."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.args               = model_config.get('custom_model_config').get('args')
        number_of_embeddings    = model_config.get('custom_model_config').get('num_embeddings')
        embedding_size          = self.args.embedding_size

        # TODO: Need to know the number of layers, so need to know the configs
        self.layers = {
            'assets':           embedding_network(number_of_embeddings, embedding_size),
            'liabilities':      embedding_network(number_of_embeddings, embedding_size),
            'net_position':     embedding_network(number_of_embeddings, embedding_size),
            'rescue_amount':    embedding_network(number_of_embeddings, embedding_size),
        }

        if self.args.full_information:
            self.layers['other_agents_assets']      = embedding_network(number_of_embeddings, embedding_size)
            self.layers['other_agents_liabilities'] = embedding_network(number_of_embeddings, embedding_size)

        if self.args.number_of_negotiation_rounds > 1:
            self.layers['final_round']  = embedding_network(number_of_embeddings, embedding_size)
            self.layers['last_offer']   = embedding_network(number_of_embeddings, embedding_size)

        if hasattr(self.args,'reveal_other_agents_identity'):
            if self.args.reveal_other_agents_identity:
                self.layers['other_agents_identity']  = embedding_network(self.args.pool_size, embedding_size)

        if hasattr(self.args, 'reveal_other_agents_beta'):
            if self.args.reveal_other_agents_beta:
                # Note: Betas are discretized in steps of 0.01
                # NOTE: 101 because not inclusive. (0 - 100)
                self.layers['other_agents_beta']  = embedding_network(101, embedding_size)

        number_of_layers = len(self.layers.keys())

        self.combining_network      = nn.Linear(embedding_size * number_of_layers, embedding_size)
        self.proposal_network       = offer_network(embedding_size, num_outputs)
        self.value                  = torch.nn.Linear(embedding_size, 1)

        self._value_input           = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        action_mask     = input_dict.get('obs').pop('action_mask')

        embedded_observations = {}
        # NOTE: This is parsing from the layers, not from the observations.  
        # Observations may contain more than used by the model.
        for observation_key in self.layers.keys():
            embedded_observations[observation_key] = self.layers[observation_key](input_dict.get('obs').get(observation_key))
        
        # Concatenate observations into a hidden vector
        # TODO: Consider attention later for continuous actions
        hidden_vector = torch.cat(list(embedded_observations.values()),-1)

        hidden_vector = self.combining_network(hidden_vector)
        hidden_vector = nn.functional.relu(hidden_vector)

        self._value_input = hidden_vector

        logits = self.proposal_network(hidden_vector).squeeze()
        inf_mask = torch.clamp(torch.log(action_mask),FLOAT_MIN, FLOAT_MAX)

        # Apply the masks
        logits = logits + inf_mask

        return logits, []


    @override(ModelV2)
    def value_function(self):
        assert self.hidden_vector is not None, "must call forward first!"
        return torch.reshape(self.value(self._value_input), [-1])


class basic_model_with_masking(TorchModelV2, nn.Module):
    """Torch version of FastModel (tf)."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        import numpy as np
        nn.Module.__init__(self)

        self.obs_size = int(np.product(obs_space.original_space.spaces.get('real_obs').shape))
        self.layer1 = torch.nn.Linear(self.obs_size, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, 128)
        self.action = torch.nn.Linear(128, num_outputs)
        
        self.value = torch.nn.Linear(self.obs_size, 1)

        self._output = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._output = input_dict.get('obs').get('real_obs').float()
        x = self.layer1(self._output)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.layer3(x)
        x = nn.functional.relu(x)
        logits = self.action(x)

        action_mask = input_dict.get('obs').get('action_mask')
        inf_mask = torch.clamp(torch.log(action_mask),FLOAT_MIN, FLOAT_MAX)

        # Apply the masks
        logits = logits + inf_mask

        return logits, []

    @override(ModelV2)
    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return torch.reshape(self.value(self._output), [-1])