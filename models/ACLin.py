from models.ACBaseNet import *


class ACLin(ACBaseNet):
    def __init__(self, args):
        super(ACLin, self).__init__(args)

        self.linear1 = nn.Linear(self.input_cat_length, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)

        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, observation):
        x = F.relu(self.linear1(observation))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return x
