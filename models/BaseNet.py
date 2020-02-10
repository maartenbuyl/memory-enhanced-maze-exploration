import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self, config):
        super(BaseNet, self).__init__()

        self.device = config.device

        self.num_channels = config.observation_dim[0][0]
        self.screen_length = config.observation_dim[0][1]
        self.screen_width = config.observation_dim[0][2]
        self.num_actions = config.action_dim

        self.input_cat_length = self.num_channels * self.screen_length * self.screen_width + config.observation_dim[1]

    def forward(self, observation):
        raise NotImplementedError
