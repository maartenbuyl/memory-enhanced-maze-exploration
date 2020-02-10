import torch
import numpy as np

from models.ACBaseNet import *


class ACMem(ACBaseNet):
    # Constants.
    state_rep_size = 128
    memory_feature_size = 16

    # Layer input dimensions.
    query_input_size = state_rep_size
    write_input_size = state_rep_size + memory_feature_size + memory_feature_size  # state + context + memory at (x,y)
    output_size = state_rep_size + memory_feature_size + memory_feature_size  # state + context + memory at (x,y)

    def __init__(self, config):
        super(ACMem, self).__init__(config)

        self.batch_size = config.batch_size
        self.memory_dim = config.maze_dim

        self.state_f = nn.Sequential(
            nn.Linear(self.input_cat_length, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_rep_size),
            nn.ReLU()
        )

        self.query_f = nn.Sequential(
            nn.Linear(self.query_input_size, self.memory_feature_size),
            nn.ReLU()
        )

        # Write takes as input:
        self.write_f = nn.Linear(self.write_input_size, self.memory_feature_size)

        # Take context as input
        self.actor = nn.Sequential(
            nn.Linear(self.output_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.memory_values = None
        self.locations = None
        self.last_weights = None
        self.memory_writes_allowed = None

    def forward(self, observation):
        # Current number of tasks still running.
        cur_batch_size = self.memory_values.shape[0]

        # Represent state: B x H. B = current batch size, H = amount of hidden layer units.
        state = self.state_f(observation)

        # Compute query vector: B x M.
        query = self.query_f(state)

        # Memory to B x HW x M. M = number of features in memory (= values per memory cell).
        straightened_memory = self.memory_values.view(cur_batch_size, -1, self.memory_feature_size)

        # Query memory using query vector to get scores of each memory cell: B x HW.
        scores = torch.bmm(straightened_memory, query.unsqueeze(2)).squeeze(2)

        # Compute weights based on the scores: B x H x W.
        weights = F.softmax(scores, dim=1).view(-1, *self.memory_dim)

        # Save weights to interpret them later. Detach from autograd to be sure.
        self.last_weights = weights.detach()

        # Represent context by taking the weighted average over the whole maze. Context = B x M.
        # The einsum does element-wise multiplication of each weight with all corresponding memory features.
        weighed_memory = torch.einsum('bxy,bxyf->bxyf', weights, self.memory_values)
        context = weighed_memory.sum(dim=1).sum(dim=1)

        # Locations information is passed on by the agent. Each task has one (x,y) location.
        locations = self.locations

        # Read local memory for each batch element, only in the current location: B x M.
        local_memory = torch.empty((cur_batch_size, self.memory_feature_size), dtype=torch.float, device=self.device)
        for batch_el in range(cur_batch_size):
            local_memory[batch_el] = self.memory_values[batch_el, locations[batch_el, 0], locations[batch_el, 1], :]

        # Calculate new memory.
        write_input = torch.cat((state, context, local_memory), dim=1)
        write_update = self.write_f(write_input)

        # Update memory.
        if self.memory_writes_allowed:
            new_memory = self.memory_values.clone()
            for batch_el in range(cur_batch_size):
                new_memory[batch_el, locations[batch_el, 0], locations[batch_el, 1], :] = write_update[batch_el, :]
            self.memory_values = new_memory

        output = torch.cat((context, state, local_memory), dim=1)
        return output

    def new_episode(self, starting_infos):
        # Wipe memory.
        self.memory_values = torch.zeros((self.batch_size, *self.memory_dim, self.memory_feature_size), dtype=torch.float, device=self.device)

        # Reset location info.
        self.locations = np.array([info['player_pos'] for info in starting_infos])

        # Reset memory writes variable.
        self.memory_writes_allowed = True

    def update_task_info(self, non_terminated_indices, infos):
        # Turn off memory writes.
        if 'no_mem_writing' in infos[0] and infos[0]['no_mem_writing']:
            self.memory_writes_allowed = False

        # Update location info of the tasks that were not yet terminated.
        self.locations = np.array([infos[non_term_index]['player_pos'] for non_term_index in non_terminated_indices])

        # Only keep part of memory that is not yet terminated. Note, otherwise the memory size is just the same.
        self.memory_values = self.memory_values[non_terminated_indices]

    def print_weight_info(self):
        if self.memory_dim[1] == 1:
            line = ""
            for x in range(self.memory_dim[0]):
                line += str(int(np.floor(self.last_weights[0][x].item()*10)))
            print(line, flush=True)
            return

        for x in range(self.memory_dim[0]):
            line = ""
            for y in range(self.memory_dim[1]):
                line += str(int(np.floor(self.last_weights[0][x][y].item()*10)))
            print(line, flush=True)
