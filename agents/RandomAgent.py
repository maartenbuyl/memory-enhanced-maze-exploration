import random

from agents.BaseAgent import *


class RandomAgent(BaseAgent):
    def __init__(self, model, config):
        super(RandomAgent, self).__init__(config)

    def act(self, observation, under_evaluation=False):
        actions = []
        for i in range(self.batch_size):
            actions.append(random.randint(0, self.num_actions - 1))
        return actions

    def experience(self, observations, action, rewards, terminals, infos, next_observations):
        pass

    def experience_in_evaluation(self, terminals, infos):
        pass

    def save_agent(self, epoch, path):
        pass

    def load_agent(self, path):
        pass

    def new_episode(self, starting_infos):
        pass
