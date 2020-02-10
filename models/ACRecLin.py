import torch

from models.ACBaseNet import *


class ACRecLin(ACBaseNet):
    def __init__(self, config):
        super(ACRecLin, self).__init__(config)

        self.hidden_size = 128
        self.batch_size = config.batch_size

        self.linear1 = nn.Linear(self.input_cat_length, 128)
        self.linear2 = nn.Linear(128, 128)
        # self.gru = nn.GRUCell(128, self.hidden_size)
        self.lstm = nn.LSTMCell(128, self.hidden_size)

        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.h = None
        self.c = None
        self.memory_writes_allowed = None

    def forward(self, observation):
        x = F.relu(self.linear1(observation))
        x = F.relu(self.linear2(x))
        #self.h = self.gru(x, self.h)
        h, c = self.lstm(x, (self.h, self.c))
        if self.memory_writes_allowed:
            self.h = h
            self.c = c
        return h

    def new_episode(self, starting_infos):
        # Wipe hidden state.
        self.h = torch.zeros((self.batch_size, self.hidden_size), dtype=torch.float, device=self.device)
        self.c = torch.zeros((self.batch_size, self.hidden_size), dtype=torch.float, device=self.device)

        # Reset memory writes variable for each task.
        self.memory_writes_allowed = True

    def update_task_info(self, non_terminated_indices, infos):
        if len(non_terminated_indices) < self.h.shape[0]:
            self.h = self.h[non_terminated_indices]
            self.c = self.c[non_terminated_indices]

        # Turn off memory writes.
        if 'no_mem_writing' in infos[0] and infos[0]['no_mem_writing']:
            self.memory_writes_allowed = False
