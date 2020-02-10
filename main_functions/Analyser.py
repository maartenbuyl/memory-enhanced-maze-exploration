import numpy as np


class Analyser:
    def __init__(self, experiment_setup, config):
        self.config = config
        self.total_nb_mazes = config.total_nb_mazes

        # Initialize analysis objects.
        self.maze_analysis_containers = []
        for i in range(self.total_nb_mazes):
            if config.maze_class.__name__ == "IndicatorMaze":
                self.maze_analysis_containers.append(IndicatorMazeBatchAnalysis(i, experiment_setup, config))
            elif config.maze_class.__name__ == "LandmarkRail":
                self.maze_analysis_containers.append(LandmarkRailBatchAnalysis(i, experiment_setup, config))
            elif config.maze_class.__name__ == "LandmarkMaze":
                self.maze_analysis_containers.append(LandmarkMazeBatchAnalysis(i, experiment_setup, config))
            else:
                self.maze_analysis_containers.append(MazeBatchAnalysis(i, experiment_setup, config))
        self.current_index = -1

        # Success rates.
        self.success_rates = np.zeros(self.total_nb_mazes, dtype=np.float)

        # Fail rates.
        self.fail_rates = np.zeros(self.total_nb_mazes, dtype=np.float)

        # Sloth rates.
        self.sloth_rates = np.zeros(self.total_nb_mazes, dtype=np.float)

        # Recall 50: how many tasks with >50% success rate?
        self.recall_50_check = np.zeros(self.total_nb_mazes, dtype=np.int)

        # Recall 90 how many tasks with >50% success rate?
        self.recall_90_check = np.zeros(self.total_nb_mazes, dtype=np.int)

        # Wall collisions
        self.wall_collisions = np.zeros(self.total_nb_mazes, dtype=np.float)

        # Inefficient turns: Turn in opposite ways, Turn > 2 times in a row
        self.inefficient_turns = np.zeros(self.total_nb_mazes, dtype=np.float)

        # Inefficient revisits: return to spot where agent has been, in inefficient manner.
        # Inefficient = new position further from goal.
        self.inefficient_revisits = np.zeros(self.total_nb_mazes, dtype=np.float)

        # Distance-inefficiency: ratio of taken path / optimal path
        self.distance_inefficiencies = np.zeros(self.total_nb_mazes, dtype=np.float)

        # IndicatorMaze: Closure-inefficiency: ratio of time to goal / fastest time to goal.
        self.closure_inefficiencies = np.zeros(self.total_nb_mazes, dtype=np.float)

        # LandmarkMaze: Exploitation-inefficiency: ratio of taken path to goal / shortest path to goal during exploring.
        self.exploitation_inefficiencies = np.zeros(self.total_nb_mazes, dtype=np.float)

    def process_stats(self, observations, rewards, terminals, infos, actions):
        self.maze_analysis_containers[self.current_index].process_step(observations, rewards, terminals, infos, actions)

    def initialize_maze_analysis(self, maze_index, starting_infos):
        self.current_index = maze_index
        self.maze_analysis_containers[self.current_index].initialize_maze_analysis(starting_infos)

    def write_results(self, folder_name, name_extra):
        self._aggregate_results()

        results_file = open(folder_name + "/post_results" + name_extra + ".txt", "w")
        results_file.write(self._stringify_results())
        results_file.close()

    def _aggregate_results(self):
        if self.config.only_eval_render_maze:
            range_of_mazes = [self.config.render_maze]
        else:
            range_of_mazes = range(self.total_nb_mazes)

        for i in range_of_mazes:
            maze_analyzer = self.maze_analysis_containers[i]

            self.success_rates[i] = maze_analyzer.success_rate

            if self.config.maze_class.__name__ != "LandmarkRail":
                self.fail_rates[i] = maze_analyzer.fail_rate
                self.sloth_rates[i] = maze_analyzer.sloth_rate
                self.recall_50_check[i] = int(maze_analyzer.success_rate > 0.5)
                self.recall_90_check[i] = int(maze_analyzer.success_rate > 0.9)
                self.wall_collisions[i] = maze_analyzer.wall_collisions
                self.inefficient_turns[i] = maze_analyzer.inefficient_turns
                self.inefficient_revisits[i] = maze_analyzer.inefficient_revisits

                if self.config.maze_class.__name__ == "IndicatorMaze":
                    self.distance_inefficiencies[i] = maze_analyzer.distance_inefficiency
                    self.closure_inefficiencies[i] = maze_analyzer.closure_inefficiency
                elif self.config.maze_class.__name__ == "LandmarkMaze":
                    self.exploitation_inefficiencies[i] = maze_analyzer.exploitation_inefficiency

        # Aggregate mazes:
        t_size = self.config.training_set_size
        self.mean_success_rate = [self.success_rates[:t_size].mean(), self.success_rates[t_size:].mean()]
        self.mean_fail_rate = [self.fail_rates[:t_size].mean(), self.fail_rates[t_size:].mean()]
        self.mean_sloth_rate = [self.sloth_rates[:t_size].mean(), self.sloth_rates[t_size:].mean()]
        self.recall_50 = [self.recall_50_check[:t_size].mean(), self.recall_50_check[t_size:].mean()]
        self.recall_90 = [self.recall_90_check[:t_size].mean(), self.recall_90_check[t_size:].mean()]
        self.mean_wall_collisions = [np.mean(self.wall_collisions[:t_size]), np.mean(self.wall_collisions[t_size:])]
        self.mean_inefficient_turns = [np.mean(self.inefficient_turns[:t_size]), np.mean(self.inefficient_turns[t_size:])]
        self.mean_inefficient_revisits = [np.mean(self.inefficient_revisits[:t_size]), np.mean(self.inefficient_revisits[t_size:])]
        self.mean_distance_inefficiency = [np.nanmean(self.distance_inefficiencies[:t_size]), np.nanmean(self.distance_inefficiencies[t_size:])]
        self.mean_closure_inefficiency = [np.nanmean(self.closure_inefficiencies[:t_size]), np.nanmean(self.closure_inefficiencies[t_size:])]
        self.mean_exploitation_inefficiency = [np.nanmean(self.exploitation_inefficiencies[:t_size]), np.nanmean(self.exploitation_inefficiencies[t_size:])]

    def _stringify_results(self):
        total_string = "---------------------------------------------------------------\n"

        if self.config.only_eval_render_maze:
            range_of_mazes = [self.config.render_maze]

        else:
            for i in range(2):
                if i == 0:
                    summary = "Summary of training set: \n"
                else:
                    summary = "Summary of validation set: \n"

                summary += "mean_success_rate: " + str(self.mean_success_rate[i]) + "\n" + \
                           "mean_fail_rate: " + str(self.mean_fail_rate[i]) + "\n" + \
                           "mean_sloth_rate: " + str(self.mean_sloth_rate[i]) + "\n" + \
                           "recall_50: " + str(self.recall_50[i]) + "\n" + \
                           "recall_90: " + str(self.recall_90[i]) + "\n" + \
                           "mean_wall_collisions: " + str(self.mean_wall_collisions[i]) + "\n" + \
                           "mean_inefficient_turns: " + str(self.mean_inefficient_turns[i]) + "\n" + \
                           "mean_inefficient_revisits: " + str(self.mean_inefficient_revisits[i]) + "\n"

                if self.config.maze_class.__name__ == "IndicatorMaze":
                    summary += "mean_distance_inefficiency: " + str(self.mean_distance_inefficiency[i]) + "\n" + \
                        "mean_closure_inefficiency: " + str(self.mean_closure_inefficiency[i]) + "\n"

                if self.config.maze_class.__name__ == "LandmarkMaze":
                    summary += "mean_exploitation_inefficiency: " + str(self.mean_exploitation_inefficiency[i]) + "\n"
                total_string += summary + "---------------------------------------------------------------\n"

            range_of_mazes = range(self.total_nb_mazes)

        for i in range_of_mazes:
            total_string += self.maze_analysis_containers[i].stringify_maze_analysis()

        return total_string


class MazeBatchAnalysis:
    # Analysis of a single maze.

    def __init__(self, maze_index, experiment_setup, config):
        self.maze_index = maze_index

        # Keep track of proxy object.
        self.batch_proxy = experiment_setup.maze

        # Keep track of config.
        self.config = config
        self.batch_size = config.batch_size

        # Setup temporary datastructures.
        self.already_terminated = np.zeros(self.batch_size, dtype=np.bool)
        self.previous_locations = np.ones((self.batch_size, 2), dtype=np.int) * -1
        self.turn_action_checker = np.zeros(self.batch_size, dtype=np.int)
        self.visited = np.zeros((self.batch_size, *self.config.maze_dim), dtype=np.bool)
        self.nb_steps_so_far = np.zeros(self.batch_size, dtype=np.int)
        self.distances_to_goal = None
        self.shortest_distances = None

        # Setup some metric counts.
        self.successful_maze = np.zeros(self.batch_size, dtype=np.bool)
        self.fail_maze = np.zeros(self.batch_size, dtype=np.bool)
        self.distance_inefficiencies = np.zeros(self.batch_size, dtype=np.float)
        self.wall_collisions = 0
        self.inefficient_turns = 0
        self.inefficient_revisits = 0

    def process_step(self, observations, rewards, terminals, infos, actions):
        non_terminated = np.logical_not(self.already_terminated)
        non_terminated_indices = np.arange(0, self.batch_size)[non_terminated]
        for i in non_terminated_indices:
            self._process_single_maze(i, observations[i][0], rewards[i], terminals[i], infos[i], actions[i])

        if np.all(terminals):
            self._compute_statistics()

    def _process_single_maze(self, i, agent_screen, reward, terminal, info, action):
        self.nb_steps_so_far[i] += 1

        if terminal:
            assert not self.already_terminated[i]
            self.already_terminated[i] = True

            if np.isclose(reward, self.config.positive_reward):
                self.successful_maze[i] = True
            elif np.isclose(reward, self.config.negative_reward):
                self.fail_maze[i] = True

            if self.nb_steps_so_far[i] == 0:
                self.distance_inefficiencies[i] = 1
            else:
                self.distance_inefficiencies[i] = self.nb_steps_so_far[i] / self.shortest_distances[i]
            return  # We assume that no other metrics should be recorded for a terminal maze.

        location = info['player_pos']
        prev_location = self.previous_locations[i]

        # Action-specific processing.
        # Forward.
        if action == 0:
            # Set turn-action-checker to "0": no turns so far.
            self.turn_action_checker[i] = 0

            # Check for wall collision.
            wall_collision_occurred = np.all(location == prev_location)
            if wall_collision_occurred:
                self.wall_collisions += 1

            # Check for revisit.
            if not wall_collision_occurred and self.visited[i, location[0], location[1]]:
                # Check if it is an inefficient revisit, by comparing distances to the goal.
                orient = info['player_orient']
                previous_distance = self.distances_to_goal[
                    prev_location[0], prev_location[1], orient]
                new_distance = self.distances_to_goal[location[0], location[1], orient]
                if new_distance > previous_distance:
                    self.inefficient_revisits += 1

            # Mark the spot as visited.
            self.visited[i, location[0], location[1]] = True

        # Turn right:
        if action == 1:
            if self.turn_action_checker[i] == 0:
                # Set to 1, meaning: turned once to the right.
                self.turn_action_checker[i] = 1

            elif self.turn_action_checker[i] == 1:
                # Set to 3, meaning: turned twice in a row.
                # Since it was in the same direction, it is not inefficient.
                self.turn_action_checker[i] = 3

            else:
                # If previous turn was left, or if there were already >= 2 turns, then the turn is inefficient.
                self.turn_action_checker[i] = 3
                self.inefficient_turns += 1

        # Turn left:
        if action == 2:
            if self.turn_action_checker[i] == 0:
                # Set to 2, meaning: turned once to the left.
                self.turn_action_checker[i] = 2

            elif self.turn_action_checker[i] == 2:
                # Set to 3, meaning: turned twice in a row.
                # Since it was in the same direction, it is not inefficient.
                self.turn_action_checker[i] = 3

            else:
                # If previous turn was left, or if there were already >= 2 turns, then the turn is inefficient.
                self.turn_action_checker[i] = 3
                self.inefficient_turns += 1

        self.previous_locations[i] = location

    # Initialize maze-specific information.
    def initialize_maze_analysis(self, starting_infos):
        # Get a ground truth maze.
        maze = self.batch_proxy.get_first_maze_from_batch()

        # For each location and orientation, compute the distance to the goal.
        self.distances_to_goal = maze.generate_maze_composition_info()

        self.shortest_distances = np.zeros(self.batch_size, dtype=np.int)
        for i in range(len(starting_infos)):
            starting_loc = starting_infos[i]['player_pos']
            self.previous_locations[i] = starting_loc

            starting_orient = starting_infos[i]['player_orient']

            self.shortest_distances[i] = self.distances_to_goal[starting_loc[0], starting_loc[1], starting_orient]

            self.visited[i, starting_loc[0], starting_loc[1]] = True

    def _compute_statistics(self):
        # Success rate.
        self.success_rate = np.where(self.successful_maze)[0].shape[0] / self.batch_size

        # Fail rate.
        self.fail_rate = np.where(self.fail_maze)[0].shape[0] / self.batch_size

        # Sloth rate.
        self.sloth_rate = 1 - self.success_rate - self.fail_rate

        # Wall collisions.
        self.wall_collisions /= self.batch_size

        # Inefficient turns.
        self.inefficient_turns /= self.batch_size

        # Inefficient revisits: return to spot where agent has been, in inefficient manner.
        # Inefficient = new position further from goal.
        self.inefficient_revisits /= self.batch_size

        # Distance-inefficiency: ratio of taken path / optimal path, for successful runs!
        self.distance_inefficiency = np.mean(self.distance_inefficiencies[np.where(self.successful_maze)[0]])

        if self.maze_index == self.config.render_maze:
            print(self.stringify_maze_analysis(), flush=True)

    def stringify_maze_analysis(self):
        line = "Maze " + str(self.maze_index) + \
               "| Su: " + str(self.success_rate) + \
               "| Fa: " + str(self.fail_rate) + \
               "| Sl: " + str(self.sloth_rate) + \
               "| WC: " + str(self.wall_collisions) + \
               "| IT: " + str(self.inefficient_turns) + \
               "| IR: " + str(self.inefficient_revisits) + \
               "| DI: " + str(self.distance_inefficiency) + \
               "\n"
        return line


class IndicatorMazeBatchAnalysis(MazeBatchAnalysis):
    def __init__(self, maze_index, experiment_setup, config):
        super(IndicatorMazeBatchAnalysis, self).__init__(maze_index, experiment_setup, config)
        self.indicator_color = None

        # Temporary data structures.
        self.fastest_time_to_goal_after_seen = np.ones(self.batch_size, dtype=np.int) * -1
        self.goal_seen_time = np.ones(self.batch_size, dtype=np.int) * -1

        # Additional metric.
        self.closure_inefficiencies = np.zeros(self.batch_size, dtype=np.float)

    def _process_single_maze(self, i, agent_screen, reward, terminal, info, action):
        super(IndicatorMazeBatchAnalysis, self)._process_single_maze(i, agent_screen, reward, terminal, info, action)

        if terminal:
            actual_time_to_goal_after_seen = self.nb_steps_so_far[i] - self.goal_seen_time[i]
            self.closure_inefficiencies[i] = actual_time_to_goal_after_seen / self.fastest_time_to_goal_after_seen[i]

        # If goal was not seen yet:
        if not terminal and self.goal_seen_time[i] == -1:
            indicator_goal_index = 3 + self.indicator_color

            if np.any(agent_screen[indicator_goal_index] == 1):
                self.goal_seen_time[i] = self.nb_steps_so_far[i]

                loc = info['player_pos']
                self.fastest_time_to_goal_after_seen[i] = self.distances_to_goal[loc[0], loc[1], info['player_orient']]

    def initialize_maze_analysis(self, starting_infos):
        super(IndicatorMazeBatchAnalysis, self).initialize_maze_analysis(starting_infos)

        # Get a ground truth maze.
        maze = self.batch_proxy.get_first_maze_from_batch()

        self.indicator_color = maze.indicator_color

    def _compute_statistics(self):
        # IndicatorMaze: Closure-inefficiency: ratio of time to goal / fastest time to goal.
        self.closure_inefficiency = np.mean(self.closure_inefficiencies[np.where(self.successful_maze)[0]])

        super(IndicatorMazeBatchAnalysis, self)._compute_statistics()

    def stringify_maze_analysis(self):
        line = super(IndicatorMazeBatchAnalysis, self).stringify_maze_analysis()

        # Cut off the "\n"
        line = line[:len(line)-1]

        line += "| CI: " + str(self.closure_inefficiency) + \
                "\n"
        return line


class LandmarkRailBatchAnalysis:
    def __init__(self, maze_index, experiment_setup, config):
        self.maze_index = maze_index

        # Keep track of proxy object.
        self.batch_proxy = experiment_setup.maze

        # Keep track of config.
        self.config = config
        self.batch_size = config.batch_size

        # Setup temporary datastructures.
        self.already_terminated = np.zeros(self.batch_size, dtype=np.bool)

        # Setup some metric counts.
        self.successful_maze = np.zeros(self.batch_size, dtype=np.bool)

    def process_step(self, observations, rewards, terminals, infos, actions):
        non_terminated = np.logical_not(self.already_terminated)
        non_terminated_indices = np.arange(0, self.batch_size)[non_terminated]
        for i in non_terminated_indices:
            self._process_single_maze(i, observations[i][0], rewards[i], terminals[i], infos[i], actions[i])

        if np.all(terminals):
            self._compute_statistics()

    def _process_single_maze(self, i, agent_screen, reward, terminal, info, action):
        if terminal:
            assert not self.already_terminated[i]
            self.already_terminated[i] = True

            if np.isclose(reward, self.config.positive_reward):
                self.successful_maze[i] = True
            return  # We assume that no other metrics should be recorded for a terminal maze.

    # Initialize maze-specific information.
    def initialize_maze_analysis(self, starting_infos):
        pass

    def _compute_statistics(self):
        # Success rate.
        self.success_rate = np.where(self.successful_maze)[0].shape[0] / self.batch_size

        if self.maze_index == self.config.render_maze:
            print(self.stringify_maze_analysis(), flush=True)

    def stringify_maze_analysis(self):
        line = "Maze " + str(self.maze_index) + \
               "| Su: " + str(self.success_rate) + \
               "\n"
        return line


class LandmarkMazeBatchAnalysis(MazeBatchAnalysis):
    def __init__(self, maze_index, experiment_setup, config):
        super(LandmarkMazeBatchAnalysis, self).__init__(maze_index, experiment_setup, config)
        self.goal_lm = None
        self.previous_phases = np.zeros(self.batch_size, dtype=np.int)

        # Temporary data structures.
        self.fastest_time_to_goal = np.ones(self.batch_size, dtype=np.int) * -1
        self.phase_1_steps = np.zeros(self.batch_size, dtype=np.int)

        # Additional metric.
        self.exploitation_inefficiencies = np.zeros(self.batch_size, dtype=np.float)

    def _process_single_maze(self, i, agent_screen, reward, terminal, info, action):
        phase = info['phase']

        # During a phase switch, do not record any information.
        if self.previous_phases[i] != phase:
            self.previous_phases[i] = phase
            return
        self.previous_phases[i] = phase

        if phase == 1:
            self.phase_1_steps[i] += 1

            # If goal was not seen yet:
            if self.fastest_time_to_goal[i] == -1:
                goal_lm_index = 1 + self.goal_lm

                if np.any(agent_screen[goal_lm_index] == 1):
                    loc = info['player_pos']
                    time_from_here_to_goal = self.distances_to_goal[loc[0], loc[1], info['player_orient']]

                    self.fastest_time_to_goal[i] = self.phase_1_steps[i] + time_from_here_to_goal
            return  # In phase 1, we do not record other information.

        super(LandmarkMazeBatchAnalysis, self)._process_single_maze(i, agent_screen, reward, terminal, info, action)

        if terminal:
            if self.fastest_time_to_goal[i] < 0:
                self.exploitation_inefficiencies[i] = np.nan
            else:
                self.exploitation_inefficiencies[i] = self.nb_steps_so_far[i] / self.fastest_time_to_goal[i]

    # Initialize maze-specific information.
    def initialize_maze_analysis(self, starting_infos):
        super(LandmarkMazeBatchAnalysis, self).initialize_maze_analysis(starting_infos)

        # Get a ground truth maze.
        maze = self.batch_proxy.get_first_maze_from_batch()

        self.goal_lm = maze.goal_lm

        self.previous_phases[:] = starting_infos[0]['phase']

        # If the goal is in the start location, then set the fastest goal time to 1 action.
        if maze.lm_locs[maze.goal_lm] == (1, 1):
            self.fastest_time_to_goal[:] = 1

    def _compute_statistics(self):
        # LandmarkMaze: Exploitation-inefficiency: ratio of taken path to goal / shortest path to goal during exploring.
        self.exploitation_inefficiency = np.mean(self.exploitation_inefficiencies[np.where(self.successful_maze)[0]])

        super(LandmarkMazeBatchAnalysis, self)._compute_statistics()

    def stringify_maze_analysis(self):
        line = super(LandmarkMazeBatchAnalysis, self).stringify_maze_analysis()

        # Cut off the "\n"
        line = line[:len(line)-1]

        line += "| EI: " + str(self.exploitation_inefficiency) + \
                "\n"
        return line
