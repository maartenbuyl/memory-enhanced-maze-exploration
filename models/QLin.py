import torch.nn.functional as F

from models.BaseNet import *


class QN(BaseNet):
    def __init__(self, args):
        super(QN, self).__init__()

        self.h_depth = args['h_depth']

        self.input_size = self.num_channels*self.screen_length*self.screen_width

        self.h = nn.Linear(self.input_size, self.h_depth)
        self.out = nn.Linear(self.h_depth, self.num_actions)

    def forward(self, observation):
        batch_size = observation.size()[0]
        observation_reformatted = observation.view(batch_size, -1)
        h = F.relu(self.h(observation_reformatted))
        out = self.out(h)

        return out
