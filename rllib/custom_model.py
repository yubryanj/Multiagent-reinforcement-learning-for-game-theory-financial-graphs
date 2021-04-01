from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor

from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()



class Custom_Model(TorchModelV2, nn.Module):
    """Torch version of FastModel (tf)."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.layer1 = torch.nn.Linear(self.obs_size, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, 128)
        self.action = torch.nn.Linear(128, num_outputs)
        
        self.value = torch.nn.Linear(self.obs_size, 1)

        self._output = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._output = input_dict['obs']
        x = self.layer1(self._output)
        x = self.layer2(x)
        x = self.layer3(x)
        action = self.action(x)
        action = torch.sigmoid(action)

        return action, []

    @override(ModelV2)
    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return torch.reshape(self.value(self._output), [-1])