from __future__ import print_function
import numpy as np

from maze_building.MazeWorld import MazeWorld


class LandmarkRail(MazeWorld):
    def __init__(self, config):
        super(LandmarkRail, self).__init__(config)

        # Constants
        self.view_length = 1  # How far an agent can see beyond his own spot.
        self.maze_length = 21
        self.phase_1_duration = int((self.maze_length - 1) / 2) - 1
        self.phase_2_duration = self.maze_length - 3  # Enough time to make small mistakes, not enough to visit both ends.
        self.start_pos_phase_2 = (int((self.maze_length - 1) / 2), 0)

        # Relevant fields for superclass.
        self.num_orient = 0
        self.num_actions = 3

        # Parsed arguments
        self.positive_reward = config.positive_reward
        self.negative_reward = config.negative_reward
        self.max_nb_landmarks = config.max_nb_landmarks

        # Size of the observation that the agent actually sees.
        self.view_size = self.view_length * 2 + 1

        # Create some specific variables.
        self.phase = None  # int
        self.lm_locs = None  # array of 2-D positions
        self.goal_lm = None  # int
        self.current_nb_landmarks = None  # boolean
        self.start_pos_phase_1 = None  # (int, int)

        # Create landmark locations.
        self.possible_lm_locs = []
        for i in range(0, int((self.maze_length - 3) / 2)):
            self.possible_lm_locs.append((i, 0))
            self.possible_lm_locs.append((self.maze_length - 1 - i, 0))

    def reset(self, mutable_maze_info):
        # Generate start position 1 on either the left side or the right side.
        if np.random.randint(2) == 0:
            self.start_pos_phase_1 = (0, 0)
        else:
            self.start_pos_phase_1 = (self.maze_length - 1, 0)

        # Initialize
        self.player_pos = self.start_pos_phase_1
        self.agent_step_counter = 0
        self.reward = 0
        self.terminal = False

        self.phase = 1

        # Create 'maze' of size maze_length x 1.
        self.maze = np.ones((self.maze_length, 1), dtype=int)

        # If doing curriculum learning, sample difficulty in terms of number of landmarks.
        if 'curriculum_learning' in mutable_maze_info and mutable_maze_info['curriculum_learning']:
            # Note that the 'high' value for randint is exclusive
            self.current_nb_landmarks = np.random.randint(low=1, high=self.max_nb_landmarks+1)
        else:
            self.current_nb_landmarks = self.max_nb_landmarks

        # Reset goal landmark.
        self.goal_lm = np.random.randint(self.current_nb_landmarks)

        # Randomly assign locations for landmarks.
        self.lm_locs = []
        random_indices = np.random.permutation(len(self.possible_lm_locs))
        for lm in range(self.current_nb_landmarks):
            random_index = random_indices[lm]
            self.lm_locs.append(self.possible_lm_locs[random_index])

        observation, _, _t, info = self._getState()
        return observation, info

    def _constructMaze(self):
        # 1 screen for the maze, L for landmarks
        total_maze = np.zeros((1 + self.max_nb_landmarks, self.maze_length, 1))
        idx_start = 0

        total_maze[idx_start] = self.maze
        idx_start += 1

        for lm_loc in self.lm_locs:
            total_maze[idx_start, lm_loc[0]] = 1
            idx_start += 1

        return total_maze

    def _getAgentObservation(self):
        total_maze = self._constructMaze()

        view = np.zeros((total_maze.shape[0], self.view_length*2+1, 1))
        left_view_limit = self.player_pos[0] - self.view_length
        right_view_limit = self.player_pos[0] + self.view_length

        view_i = 0
        # Fill in all the observation that are within the agent's viewing range.
        for i in range(left_view_limit, right_view_limit + 1):
            if 0 <= i < self.maze_length:
                view[:, view_i, :] = total_maze[:, i, :]
            view_i += 1

        # 0-1: phase, 2-2+(L-1): landmark that is the goal, 2+L-L+(num_actions-1): action observation.
        non_spatial_observation = np.zeros(2 + self.max_nb_landmarks + self.num_actions)
        idx_start = 0

        non_spatial_observation[idx_start + (self.phase - 1)] = 1
        idx_start += 2

        if self.phase > 1:
            non_spatial_observation[idx_start + self.goal_lm] = 1
        idx_start += self.max_nb_landmarks

        if self.previous_action is not None:
            non_spatial_observation[idx_start + self.previous_action] = 1
        idx_start += self.num_actions

        return view, non_spatial_observation

    def _getInfo(self):
        info = super(LandmarkRail, self)._getInfo()

        # If in phase 1, or if in the first step of phase 2, tell the agent to disregard all experiences.
        if self.phase == 1 or (self.phase == 2 and self.agent_step_counter == 0):
            info['no_grad_update'] = True

        if self.phase == 2:
            info['no_mem_writing'] = True

        return info

    def _agent_step(self, act):
        # Note that we simplify the movement process, so Mazeworld._agent_step is never called.

        # If phase 1, change action into action that follows phase 1 policy.
        if self.phase == 1:
            act = self._execute_phase_1_policy()
            self.previous_action = act

        # Check if a new phase should be started, before we actually process the action.
        if self._check_phase_end():
            return

        # Move left.
        if act == 0:
            if self.player_pos[0] > 0:
                self.player_pos = (self.player_pos[0]-1, 0)

        # Choose this position. Note that this is not possible in phase 1, where we just ignore the action.
        elif act == 1 and self.phase > 1:
            self.terminal = True

            # Correct choice.
            if self.player_pos == self.lm_locs[self.goal_lm]:
                self.reward += self.positive_reward

            # Wrong goal chosen.
            elif self.player_pos in self.lm_locs:
                self.reward += self.negative_reward

            # Non-goal spot chosen.
            else:
                self.reward += self.sloth_reward

        # Move right.
        elif act == 2:
            if self.player_pos[0] < self.maze_length - 1:
                self.player_pos = (self.player_pos[0]+1, 0)

        self.agent_step_counter += 1
        return

    def _check_phase_end(self):
        # If phase has to end, carry out phase switch.
        if self.phase == 1 and self.agent_step_counter >= self.phase_1_duration:
            self.phase = 2
            self.agent_step_counter = 0
            self.player_pos = self.start_pos_phase_2
            return True

        if self.phase == 2 and self.agent_step_counter >= self.phase_2_duration:
            self.terminal = True
            self.reward += self.sloth_reward
            return True
        return False

    def _execute_phase_1_policy(self):
        # If started left, go to the right.
        if self.start_pos_phase_1[0] == 0:
            action = 2
        # If started right, go to the left.
        elif self.start_pos_phase_1[0] == self.maze_length - 1:
            action = 0
        else:
            action = None

        return action

    def get_maze_dimension(self):
        return self.maze_length, 1

    def get_state_dimension(self):
        return (1 + self.max_nb_landmarks, self.view_size, 1), 2 + self.max_nb_landmarks + self.num_actions

    def get_action_dimension(self):
        return self.num_actions

    def render(self):
        if self.phase == 2 and self.agent_step_counter == 0:
            print("Phase 1 is over, moving to phase 2!", flush=True)

        lm_locs_x = [lm_loc[0] for lm_loc in self.lm_locs]

        assert (self.maze is not None)
        line = ""
        for x in range(self.maze_length):
            if self.maze[x] == 0.:
                line += u"\u258B"
            elif self.player_pos[0] == x:
                line += "@"
            elif x in lm_locs_x:
                letter = "A"
                line += chr(ord(letter) + lm_locs_x.index(x))
            else:
                line += ' '

        observations = self._getAgentObservation()
        view_length = observations[0].shape[1]
        for view_observation in observations[0]:
            line += "      "
            for x in range(0, view_length):
                line += str(int(view_observation[x])) + " "

        line += "      "
        for non_spatial_val in observations[1]:
            line += str(int(non_spatial_val)) + " | "

        line += "      "
        line += "Find goal " + chr(self.goal_lm + ord('A'))

        print(line, end='\n', flush=True)

    def generate_sample_maze(self, generate_training_mazes):
        # For the LandmarkRail, all mazes are the same. So ignore argument.
        mutable_maze_info = {
            'long_life': None,
            'curriculum_learning': None
        }

        self.reset(mutable_maze_info=mutable_maze_info)

        # Construct maze to save it to a file.
        maze = self._constructMaze()

        # Mark position (0, 0) with the goal command, coded as "2 + self.goal_lm". 2 is a magic number.
        maze[0, 0, 0] = 2 + self.goal_lm

        # Mark position (1, 0) with the start_pos_phase_1 position, coded as "2 + pos". 2 is a magic number.
        maze[0, 1, 0] = 2 + self.start_pos_phase_1[0]
        return maze

    @staticmethod
    def reset_from_existing_array(args, existing_maze, mutable_maze_info):
        maze_object = LandmarkRail(args)

        # Fill in the information that is normally randomly generated.
        maze_object.maze = np.copy(existing_maze[0])
        maze_object.maze[:] = 1

        maze_object.maze_length = existing_maze.shape[1]

        # Read goal command.
        maze_object.goal_lm = int(existing_maze[0, 0, 0] - 2)

        # Read start pos 1.
        maze_object.start_pos_phase_1 = (int(existing_maze[0, 1, 0]) - 2, 0)

        maze_object.max_nb_landmarks = existing_maze.shape[0] - 1
        lm_locs = []
        # Iterate over dimensions.
        for lm_dim in range(0, maze_object.max_nb_landmarks):
            # Iterate over array.
            for x in range(maze_object.maze_length):
                if existing_maze[1+lm_dim, x, 0] == 1.:
                    lm_locs.append((x, 0))
        maze_object.lm_locs = lm_locs

        # Set agent positions
        maze_object.player_pos = maze_object.start_pos_phase_1
        maze_object.phase = 1

        # Reset player rewards&terminals&step counters
        maze_object.reward = 0
        maze_object.terminal = False
        maze_object.agent_step_counter = 0

        state, _, _t, info = maze_object._getState()
        return maze_object, state, info

    @staticmethod
    def serialize(maze):
        result = "\n"
        nb_lms = maze.shape[0]-1

        for x in range(maze.shape[1]):
            char_printed = False

            for lm_dim in range(nb_lms):
                if not char_printed and maze[1+lm_dim, x, 0] == 1.:
                    result += chr(lm_dim + ord('A'))
                    char_printed = True

            if not char_printed:
                result += " "

        # Leave 4 empty spaces.
        result += "    "

        # Print goal command.
        result += chr(int(maze[0, 0, 0] - 2) + ord('A'))

        # Print start pos.
        result += str(int(maze[0, 1, 0] - 2 > 0))

        result += "\n"
        return result + "\n"

    @staticmethod
    def deserialize(maze_string_lines):
        # maze_size = total line length - 4 (for empty spaces) - 1 (for goal command) - 1 (for start_pos) - 1 (for \n).
        maze_size = len(maze_string_lines[0]) - 4 - 1 - 1 - 1

        line = maze_string_lines[0]

        maze = [np.ones((maze_size, 1))]
        nb_landmarks_found = 0
        for x in range(maze_size):
            if line[x] != " ":
                lm_rank = (ord(line[x]) - ord('A')) + 1
                while lm_rank > nb_landmarks_found:
                    maze.append(np.zeros((maze_size, 1)))
                    nb_landmarks_found += 1

                maze[lm_rank][x, 0] = 1.

        # Other arguments are saved after the maze size + 4 extra spaces.
        index = maze_size + 4

        # Parse goal command.
        maze[0][0, 0] = ord(line[index]) - ord('A') + 2
        index += 1

        # Parse start pos.
        if line[index] == 0:
            start_pos = maze_size - 1
        else:
            start_pos = 0
        maze[0][1, 0] = start_pos + 2

        maze = np.array(maze)
        return maze
