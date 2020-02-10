from __future__ import print_function

from maze_building.IndicatorMaze import *


class FullyVisibleIndicatorMaze(IndicatorMaze):
    def __init__(self, args):
        super(FullyVisibleIndicatorMaze, self).__init__(args)

    def _getAgentObservation(self):
        # Get the full maze
        screen = self._constructMaze()

        if self.min_maze_size != self.max_maze_size:
            raise NotImplementedError("Please choose a constant maze size for now!")
        return screen

    def calculate_optimal_policy(self):
        goal = (0, self.green_goal_pos[0], self.green_goal_pos[1])

        modmaze = np.copy(self.maze)

        opt_value = dijkstra(modmaze, self.agent_mechanism, goal)
        #opt_policy = extract_policy(modmaze, self.agent_mechanism, opt_value)

        return opt_value

    def get_state_dimension(self):
        return 1 + self.num_orient + 4, self.max_maze_size, self.max_maze_size

    def get_action_dimension(self):
        return self.num_actions
