

class BatchMaze:
    def __init__(self, config, mazes=None):
        self.maze_class = config.maze_class
        self.batch_size = config.batch_size

        if mazes:
            self.mazes = mazes
        else:
            self.mazes = []
            for i in range(self.batch_size):
                self.mazes.append(self.maze_class(config))

    def step(self, actions):
        observations = []
        terminals = []
        rewards = []
        infos = []
        for i in range(self.batch_size):
            o, r, t, f = self.mazes[i].step(actions[i])
            observations.append(o)
            rewards.append(r)
            terminals.append(t)
            infos.append(f)
        return observations, rewards, terminals, infos

    def reset(self, mutable_maze_info):
        observations = []
        infos = []
        for i in range(self.batch_size):
            o, f = self.mazes[i].reset(mutable_maze_info)
            observations.append(o)
            infos.append(f)
        return observations

    def render(self):
        self.mazes[0].render()

    def get_maze_dimension(self):
        return self.mazes[0].get_maze_dimension()

    def get_action_dimension(self):
        return self.mazes[0].get_action_dimension()

    def get_state_dimension(self):
        return self.mazes[0].get_state_dimension()

    def get_start_location(self):
        return self.mazes[0].get_start_location()
