from maze_building.MazeWorld import MazeWorld
from maze_building.utils import *


class LandmarkMaze(MazeWorld):
    def __init__(self, config):
        super(LandmarkMaze, self).__init__(config)

        # Constants
        self.start_pos_phase_1 = (1, 1)
        self.start_pos_phase_2 = (1, 1)
        self.num_actions = 4  # 3 move actions and 1 selection action.
        self.nb_random_points = 3

        # Parsed arguments
        self.positive_reward = config.positive_reward
        self.negative_reward = config.negative_reward
        self.max_nb_landmarks = config.max_nb_landmarks
        self.current_nb_landmarks = self.max_nb_landmarks
        self.view_length = config.view_length_lm_maze

        # Create some specific variables.
        self.phase = None  # int
        self.lm_locs = None  # array of 2-D positions
        self.goal_lm = None  # int
        self.phase_1_points = None
        self.phase_1_points_counter = None

    # Only called in resets.
    def _resetAgentPos(self):
        self.player_pos = self.start_pos_phase_1

    def _postReset(self):
        self.phase = 1

        # Reset goal landmark.
        self.goal_lm = np.random.randint(self.current_nb_landmarks)

        # Randomly assign locations for landmarks.
        self.lm_locs = []
        for lm in range(0, self.current_nb_landmarks):
            # Search for locations that are not in a wall, and not already an LM location.
            while True:
                random_xy = (np.random.randint(1, self.maze_size-1), np.random.randint(1, self.maze_size-1))
                if self.maze[random_xy] == 1 and random_xy not in self.lm_locs:
                    self.lm_locs.append(random_xy)
                    break

        # Find 3 random points to visit during phase 1. Avoid points in the first row or first column.
        self.phase_1_points = []
        for i in range(self.nb_random_points):
            while True:
                random_xy = (np.random.randint(2, self.maze_size-1), np.random.randint(2, self.maze_size-1))
                if self.maze[random_xy] == 1 and random_xy not in self.phase_1_points:
                    self.phase_1_points.append(random_xy)
                    break

    def _generateMaze(self):
        assert self.maze_size % 2 == 1

        # Start with grid without walls.
        maze = np.ones((self.maze_size, self.maze_size), dtype=np.int)

        # Set walls on the sides.
        maze[0, :] = 0
        maze[-1, :] = 0
        maze[:, 0] = 0
        maze[:, -1] = 0

        # Set interior walls in Bomberman-like grid.
        # row_and_column_indices = np.arange(1, (self.maze_size / 2) - 1, dtype=np.int) * 2
        # for row_index in row_and_column_indices:
        #     for column_index in row_and_column_indices:
        #         maze[row_index][column_index] = 0

        self.maze = maze

    def _constructMaze(self):
        # 1 screen for the maze, L for landmarks
        total_maze = np.zeros((1 + self.max_nb_landmarks, self.maze_size, self.maze_size))
        idx_start = 0

        total_maze[idx_start] = super(LandmarkMaze, self)._constructMaze()
        idx_start += 1

        for lm_loc in self.lm_locs:
            total_maze[idx_start, lm_loc[0], lm_loc[1]] = 1
            idx_start += 1

        return total_maze

    def _getNonSpatialObservation(self):
        # phase + orientation + previous action + goal lm.
        non_spatial_observation = np.zeros(2 + self.num_orient + self.num_actions + self.current_nb_landmarks).astype(np.float32)
        idx_start = 0

        # Phase
        non_spatial_observation[idx_start + (self.phase - 1)] = 1
        idx_start += 2

        # Orientation.
        non_spatial_observation[idx_start + self.player_orient] = 1
        idx_start += self.num_orient

        # Previous action.
        if self.previous_action is not None:
            non_spatial_observation[idx_start + self.previous_action] = 1
        idx_start += self.num_actions

        # Goal landmark.
        if self.phase > 1:
            non_spatial_observation[idx_start + self.goal_lm] = 1
        idx_start += self.max_nb_landmarks

        return non_spatial_observation

    def _getInfo(self):
        info = super(LandmarkMaze, self)._getInfo()

        info['phase'] = self.phase

        # If in phase 1, or if in the first step of phase 2, tell the agent to disregard all experiences.
        #if self.phase == 1:
        #    info['no_grad_update'] = True
        #
        #if self.phase == 2:
        #    info['no_mem_writing'] = True

        return info

    def _agent_step(self, act):
        # If phase 1, change action into action that follows phase 1 policy.
        if self.phase == 1:
            act = self._execute_phase_1_policy()

            # If act is None, then we have progressed a phase and should just return.
            if act is None:
                return

            self.previous_action = act

        # Selection action.
        if act == 3:
            # Did we end up in a goal location?
            for lm in range(self.current_nb_landmarks):
                lm_loc = self.lm_locs[lm]

                if self.player_pos == lm_loc:
                    if self.goal_lm == lm:
                        self.reward = self.positive_reward
                    else:
                        self.reward = self.negative_reward
                    self.terminal = True

            self.agent_step_counter += 1
            if self.agent_step_counter > self.kill_early:
                self.reward += self.sloth_reward
                self.terminal = True

            if not self.terminal and self.living_reward is not None:
                self.reward += self.living_reward
            return

        # If it was not the selection action, it was a move:
        MazeWorld._agent_step(self, act)

    def _execute_phase_1_policy(self):
        # Shorthand.
        pos = self.player_pos
        orient = self.player_orient

        # # "Trap" policy
        # if self.player_orient == 1:
        #     if self.player_pos[1] == self.player_pos[0] + 2:
        #         return 1
        #     else:
        #         return 0
        # if self.player_orient == 3:
        #     if self.player_pos[0] == self.player_pos[1]:
        #         return 2
        #     else:
        #         return 0

        # # Circle
        # if self.player_orient == 1 and self.player_pos[1] == self.maze_size - 2:
        #     return 1
        # if self.player_orient == 3 and self.player_pos[0] == self.maze_size - 2:
        #     return 1
        # if self.player_orient == 2 and self.player_pos[1] == 1:
        #     return 1
        # if self.player_orient == 0 and self.player_pos[0] == 1:
        #     return 1
        # return 0

        # Use random point navigation.
        if pos == self.phase_1_points[self.phase_1_points_counter]:
            self.phase_1_points_counter += 1

            if self.phase_1_points_counter >= len(self.phase_1_points):
                self._phase_switch()
                return

        cur_point = self.phase_1_points[self.phase_1_points_counter]

        # Find orientation(s) that would lead to the goal.
        possible_orients = []

        # Vertical:
        if pos[0] > cur_point[0]:
            possible_orients.append(0)  # N
        elif pos[0] < cur_point[0]:
            possible_orients.append(3)  # S

        # Horizontal:
        if pos[1] < cur_point[1]:
            possible_orients.append(1)  # E
        elif pos[1] > cur_point[1]:
            possible_orients.append(2)  # W

        # If orientation is already ok, go forward.
        if orient in possible_orients:
            return 0

        # If orientation is not ok, try neighbouring orients.
        clockwise_neighbours = [1, 3, 0, 2]
        if clockwise_neighbours[orient] in possible_orients:
            return 1
        else:
            return 2

    def _phase_switch(self):
        if self.phase == 1:
            self.phase = 2
            self.agent_step_counter = 0
            self.player_pos = self.start_pos_phase_2
            self.player_orient = 1

    def get_state_dimension(self):
        return (1 + self.max_nb_landmarks, self.view_length, 3), 2 + self.num_orient + self.num_actions + self.current_nb_landmarks

    def get_action_dimension(self):
        return self.num_actions

    def render(self):
        if self.phase == 2 and self.agent_step_counter == 0:
            print("Phase 1 is over, moving to phase 2!", flush=True)

        view = []
        for y in range(self.maze_size):
            line = ""
            for x in range(self.maze_size):
                if self.maze[y][x] == 0.:
                    line += u"\u258B"
                elif (self.player_pos[0] == y) and (self.player_pos[1] == x):
                    if self.player_orient == 0:  # N
                        line += "^"
                    elif self.player_orient == 1:  # E
                        line += '>'
                    elif self.player_orient == 2:  # W
                        line += '<'
                    else:  # S
                        line += 'v'
                elif (y, x) in self.lm_locs:
                    letter = "A"
                    line += chr(ord(letter) + self.lm_locs.index((y, x)))
                else:
                    line += ' '
            view.append(line)

        observations = super(LandmarkMaze, self)._getAgentObservation()
        view_length = observations[0].shape[1]
        for view_observation in observations[0]:
            view_i = 0
            for y in range(max(0, view_length - len(view)), view_length):
                view[view_i] += "      "
                for x in range(view_observation.shape[1]):
                    view[view_i] += str(int(view_observation[y][x])) + " "
                view_i += 1

        # Print non-spatial observations as well.
        view[0] += "      "
        for x in range(observations[1].shape[0]):
            if x == 2 or x == 2 + self.num_orient or x == 2 + self.num_orient + self.num_actions or x == 2 + self.num_orient + self.num_actions + self.max_nb_landmarks:
                view[0] += "| "

            view[0] += str(int(observations[1][x])) + " "

        view[0] += "   Find goal " + chr(self.goal_lm + ord('A')) + "!"

        for line in view:
            print(line, end='\n', flush=True)

    def generate_sample_maze(self, generate_training_mazes):
        # For the IndicatorMaze, all mazes are the same. So ignore argument.
        mutable_maze_info = {'long_life': None}
        self.reset(mutable_maze_info=mutable_maze_info)

        maze = self._constructMaze()

        extra_info = np.zeros_like(maze[0])

        # Mark position (0, 0) with the goal command. This will later be overwritten.
        extra_info[0, 0] = self.goal_lm

        # Mark random points that are to be visited with the rank in which they must be visited
        rank = 1
        for point in self.phase_1_points:
            extra_info[point[0], point[1]] = rank
            rank += 1

        extra_info = np.expand_dims(extra_info, axis=0)
        maze = np.concatenate((maze, extra_info), axis=0)

        # Construct maze to save it to a file.
        return maze

    @staticmethod
    def reset_from_existing_array(config, existing_maze, mutable_maze_info):
        maze_object = LandmarkMaze(config)

        if mutable_maze_info['long_life']:
            maze_object.kill_early = maze_object.kill_early_eval
        else:
            maze_object.kill_early = maze_object.kill_early_training

        # Fill in the information that is normally randomly generated or parsed.
        maze_object.maze = np.copy(existing_maze[0])
        maze_object.maze_size = existing_maze.shape[1]
        maze_object.max_nb_landmarks = existing_maze.shape[0] - 2

        # Parse goal command from the last dimension.
        maze_object.goal_lm = int(existing_maze[-1, 0, 0])

        maze_object.lm_locs = []

        maze_object.phase_1_points = [(-1, -1) for i in range(maze_object.nb_random_points)]

        # Iterate over dimensions and maze to look for lm locs and phase 1 navigation points.
        # Ignore the walls on the first row and column.
        for dim in range(maze_object.max_nb_landmarks+1):
            for y in range(1, maze_object.maze_size):
                for x in range(1, maze_object.maze_size):
                    if existing_maze[1+dim][y][x] != 0.:
                        if dim < maze_object.max_nb_landmarks:
                            maze_object.lm_locs.append((y, x))

                        # If final phase, then the number is the rank of the phase 1 random point.
                        else:
                            rank = int(existing_maze[1+maze_object.max_nb_landmarks][y][x])
                            maze_object.phase_1_points[rank-1] = (y, x)

        # Reset phase.
        maze_object.phase = 1
        maze_object.phase_1_points_counter = 0

        # Set agent positions.
        maze_object._resetAgentPos()
        maze_object.last_player_pos = (maze_object.player_pos[0], maze_object.player_pos[1])

        # Reset orientations to 1.
        maze_object.player_orient = 1
        maze_object.last_player_orient = maze_object.player_orient

        # Reset player rewards&terminals&step counters
        maze_object.reward = 0
        maze_object.terminal = False
        maze_object.agent_step_counter = 0

        state, _, _t, info = maze_object._getState()
        return maze_object, state, info

    @staticmethod
    def serialize(maze):
        result_lines = []
        nb_lms = maze.shape[0]-2

        for y in range(maze.shape[1]):
            line = ""
            for x in range(maze.shape[2]):
                char_printed = False

                if y == 0 and x == 0:
                    line += chr(int(maze[-1][y][x]) + ord('A'))
                    char_printed = True

                if not char_printed and maze[0][y][x] == 0.:
                    line += "0"
                    char_printed = True

                for lm_dim in range(nb_lms):
                    if not char_printed and maze[1 + lm_dim, y, x] == 1.:
                        line += chr(lm_dim + ord('A'))
                        char_printed = True

                if not char_printed:
                    line += ' '

            result_lines.append(line)

        # Write random points that must be visited + their order.
        rank = 1
        while True:
            loc = np.where(maze[maze.shape[0]-1] == rank)
            if loc[0].shape[0] == 0:
                # If this rank is not found...
                break
            result_lines[0] += " | " + str(int(loc[0][-1])) + "," + str(int(loc[1][-1]))
            rank += 1

        result = "\n"
        for line in result_lines:
            result += line + "\n"
        return result + "\n"

    @staticmethod
    def deserialize(maze_string_lines):
        maze_size = len(maze_string_lines)

        maze = [np.zeros((maze_size, maze_size))]
        extra_info = np.zeros_like(maze[0])

        nb_landmarks_found = 0
        for y in range(maze_size):
            for x in range(maze_size):
                char = maze_string_lines[y][x]

                # Top left just shows goal lm.
                if y == 0 and x == 0:
                    extra_info[0][0] = (ord(char) - ord('A'))

                # If not top left and not a wall...
                elif char != "0":
                    maze[0][y][x] = 1.

                    # If there is a different spot in the maze, then it is a landmark location.
                    if char != " ":
                        lm_rank = (ord(char) - ord('A')) + 1

                        # If we did not expect this many landmarks yet, then add more dimensions to the maze.
                        while lm_rank > nb_landmarks_found:
                            maze.append(np.zeros_like(maze[0]))
                            nb_landmarks_found += 1
                        maze[lm_rank][y][x] = 1.

        # Parse random points.
        points = maze_string_lines[0].split(" | ")
        for i in range(1, len(points)):
            point = points[i].split(",")
            extra_info[int(point[0]), int(point[1])] = i

        maze.append(extra_info)

        maze = np.array(maze)
        return maze

    def generate_maze_composition_info(self):
        goal_location = self.lm_locs[self.goal_lm]

        distances_to_goal = new_dijkstra(self.maze, goal_location)

        # For each distance, add 1 extra step for the selection action.
        for i in range(distances_to_goal.shape[0]):
            distances_to_goal[i] += 1

        return distances_to_goal
