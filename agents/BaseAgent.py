

class BaseAgent(object):
    def __init__(self, config):
        self.device = config.device
        self.num_actions = config.action_dim
        self.observation_dim = config.observation_dim
        self.discount_factor = config.discount_factor
        self.grad_clip_val = config.grad_clip_val
        self.batch_size = config.batch_size

        self.iteration_counter = 0

    def act(self, observation, under_evaluation=False):
        raise NotImplementedError

    def experience(self, observations, action, rewards, terminals, infos, next_observations):
        raise NotImplementedError

    def experience_in_evaluation(self, terminals, infos):
        raise NotImplementedError

    def save_agent(self, epoch, path):
        raise NotImplementedError

    def load_agent(self, path):
        raise NotImplementedError
