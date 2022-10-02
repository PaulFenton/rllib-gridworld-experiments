from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork



class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        print(f"Model initialization: obs_space={obs_space}, action_space={action_space}")
        super(CustomModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        print(f"Initializing FC Network: obs_space={obs_space},  action_space={action_space}, num_outputs={num_outputs}, model_config={model_config}, name={name}")
        self.model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()