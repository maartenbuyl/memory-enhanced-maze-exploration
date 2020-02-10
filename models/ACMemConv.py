from models.ACMem import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ACMemConv(ACMem):
    # Class-specific layer input dimensions
    global_read_size = 16

    # Overwrites
    query_input_size = ACMem.query_input_size + global_read_size
    write_input_size = ACMem.write_input_size + global_read_size
    output_size = ACMem.output_size + global_read_size

    def __init__(self, config):
        super(ACMemConv, self).__init__(config)

        # Add 'global read' layer.
        conv_size = self.memory_dim[0] - 3 + 1
        conv_size = conv_size - 3 + 1
        conv_size = ((conv_size - 3 + 1) ** 2) * 8
        self.global_read_f = nn.Sequential(
            nn.Conv2d(in_channels=self.memory_feature_size, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.ReLU(),
            Flatten(),
            nn.Linear(conv_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.global_read_size),
            nn.ReLU()
        )

    def forward(self, observation):
        # Current number of tasks still running.
        cur_batch_size = self.memory_values.shape[0]

        # Represent state: B x H. B = current batch size, H = amount of hidden layer units.
        state = self.state_f(observation)

        # Global read: B x H x W.
        conv_prepped = self.memory_values.permute(0, 3, 1, 2)
        global_read = self.global_read_f(conv_prepped)

        # Compute query vector: B x M.
        query_input = torch.cat((state, global_read), dim=1)
        query = self.query_f(query_input)

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
        write_input = torch.cat((state, context, local_memory, global_read), dim=1)
        write_update = self.write_f(write_input)

        # Update memory.
        if self.memory_writes_allowed:
            new_memory = self.memory_values.clone()
            for batch_el in range(cur_batch_size):
                new_memory[batch_el, locations[batch_el, 0], locations[batch_el, 1], :] = write_update[batch_el, :]
            self.memory_values = new_memory

        output = torch.cat((context, state, local_memory, global_read), dim=1)
        return output
