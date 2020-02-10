from models.ACMem import *


class ACMemSimple(ACMem):
    output_size = ACMem.output_size - ACMem.state_rep_size  # context + write at (x,y)

    def __init__(self, config):
        super(ACMemSimple, self).__init__(config)

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

        output = torch.cat((context, write_update), dim=1)
        return output
