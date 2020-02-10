from models.ACMem import *


class ACMemKV(ACMem):
    # Overwrites=
    write_input_size = ACMem.write_input_size + ACMem.memory_feature_size  # Add local key to write input.
    output_size = ACMem.output_size + ACMem.memory_feature_size  # Add local key to output.

    def __init__(self, config):
        self.memory_keys = None

        super(ACMemKV, self).__init__(config)

        # Replace write layer with 'key' write and 'value' write.
        self.write_f = None
        self.write_key_f = nn.Linear(self.write_input_size, self.memory_feature_size)
        self.write_value_f = nn.Linear(self.write_input_size, self.memory_feature_size)

    def forward(self, observation):
        # Current number of tasks still running.
        cur_batch_size = self.memory_keys.shape[0]

        # Represent state: B x H. B = current batch size, H = amount of hidden layer units.
        state = self.state_f(observation)

        # Compute query vector: B x M.
        query = self.query_f(state)

        # Memory to B x HW x M. M = number of features in memory (= values per memory cell).
        straightened_memory_keys = self.memory_keys.view(cur_batch_size, -1, self.memory_feature_size)

        # Query memory using query vector to get scores of each memory cell: B x HW.
        scores = torch.bmm(straightened_memory_keys, query.unsqueeze(2)).squeeze(2)

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
        local_memory_key = torch.empty((cur_batch_size, self.memory_feature_size), dtype=torch.float, device=self.device)
        local_memory_value = torch.empty((cur_batch_size, self.memory_feature_size), dtype=torch.float, device=self.device)
        for batch_el in range(cur_batch_size):
            local_memory_key[batch_el] = self.memory_keys[batch_el, locations[batch_el, 0], locations[batch_el, 1], :]
            local_memory_value[batch_el] = self.memory_values[batch_el, locations[batch_el, 0], locations[batch_el, 1], :]

        # Calculate new memory.
        write_input = torch.cat((state, context, local_memory_key, local_memory_value), dim=1)
        write_update_key = self.write_key_f(write_input)
        write_update_value = self.write_value_f(write_input)

        # Update memory.
        if self.memory_writes_allowed:
            new_memory_keys = self.memory_keys.clone()
            new_memory_values = self.memory_values.clone()
            for batch_el in range(cur_batch_size):
                new_memory_keys[batch_el, locations[batch_el, 0], locations[batch_el, 1], :] = write_update_key[batch_el, :]
                new_memory_values[batch_el, locations[batch_el, 0], locations[batch_el, 1], :] = write_update_value[batch_el, :]
            self.memory_keys = new_memory_keys
            self.memory_values = new_memory_values

        output = torch.cat((context, state, local_memory_key, local_memory_value), dim=1)
        return output

    def new_episode(self, starting_infos):
        super(ACMemKV, self).new_episode(starting_infos)

        self.memory_keys = torch.zeros((self.batch_size, *self.memory_dim, self.memory_feature_size), dtype=torch.float, device=self.device)

    def update_task_info(self, non_terminated_indices, infos):
        super(ACMemKV, self).update_task_info(non_terminated_indices, infos)

        self.memory_keys = self.memory_keys[non_terminated_indices]
