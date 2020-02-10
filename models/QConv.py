import torch.nn.functional as F

from models.BaseNet import *


class DQN(BaseNet):
    def __init__(self, args):
        super(BaseNet, self).__init__()

        self.cnn_depth = args['cnn_depth']
        self.h_depth = args['h_depth']
        self.conv_output_size = (self.screen_length - 1) * (self.screen_width - 1) * self.cnn_depth

        self.read_layer = nn.Conv2d(in_channels=self.num_channels, out_channels=self.cnn_depth, stride=1, kernel_size=(2, 2))
        self.h = nn.Linear(self.conv_output_size, self.h_depth)
        self.out = nn.Linear(self.h_depth, self.num_actions)

    def forward(self, observation):
        read = F.relu(self.read_layer(observation))
        read = read.view(-1, self.conv_output_size)
        h = F.relu(self.h(read))
        out = self.out(h)

        return out
