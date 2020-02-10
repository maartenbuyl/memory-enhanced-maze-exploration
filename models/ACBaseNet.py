import torch.nn.functional as F

from models.BaseNet import *


class ACBaseNet(BaseNet):
    def __init__(self, config):
        super(ACBaseNet, self).__init__(config)

    def forward(self, observation):
        raise NotImplementedError

    # Only the Actor head
    def get_action_probs(self, observation):
        x = self(observation)
        action_probs = F.softmax(self.actor(x), dim=1)

        return action_probs

    # Only the Critic head
    # def get_state_value(self, observation):
    #     raise NotImplementedError

    # Both heads
    def evaluate_actions(self, observation):
        x = self(observation)

        action_probs = F.softmax(self.actor(x), dim=1)

        state_values = self.critic(x)
        return action_probs, state_values

    def new_episode(self, starting_infos):
        pass

    def update_task_info(self, non_terminated_indices, infos):
        pass

    def print_weight_info(self):
        pass
