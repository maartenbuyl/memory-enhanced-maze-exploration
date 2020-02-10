import numpy as np
import os.path

from maze_building import BatchMaze


class PreComputedMazeSet:
    def __init__(self, training_set_size, validation_set_size, config):
        self.training_set_size = training_set_size
        self.validation_set_size = validation_set_size
        self.total_set_size = training_set_size + validation_set_size
        self.maze_class = config.maze_class
        self.maze_size = config.maze_size
        self.decimation = config.decimation

        self.training_mazes = []
        self.validation_mazes = []

        # Name to recognize data file.
        if self.maze_class.__name__ == "IndicatorMaze":
            file_name = str(self.training_set_size) + "-" + str(self.validation_set_size) \
                + "_" + self.maze_class.__name__ + "s" + "_siz" + str(self.maze_size) + "_dec" + str(self.decimation)
        elif self.maze_class.__name__ == "LandmarkRail":
            file_name = str(self.training_set_size) + "-" + str(self.validation_set_size) \
                        + "_" + self.maze_class.__name__ + "s" + "_nblm" + str(config.max_nb_landmarks)
        elif self.maze_class.__name__ == "LandmarkMaze":
            file_name = str(self.training_set_size) + "-" + str(self.validation_set_size) \
                        + "_" + self.maze_class.__name__ + "s" + "_siz" + str(self.maze_size) + \
                        "_nblm" + str(config.max_nb_landmarks)
        else:
            file_name = None
        self.file_path = os.path.abspath(os.path.dirname(__file__)) + "/.." + "/datasets/" + file_name + ".txt"

        if os.path.isfile(self.file_path):
            self._load_dataset()
        else:
            self._create_dataset(config)

    def _create_dataset(self, config):
        validation_set_fraction = self.training_set_size / self.validation_set_size

        both_maze_sets = []
        i = 0
        while_loop_iterations = 0
        # Loop until we have enough training mazes AND validation mazes.
        while i < self.training_set_size+self.validation_set_size:
            generate_training_mazes = len(self.training_mazes) <= np.floor(len(self.validation_mazes)*validation_set_fraction)

            # Generate a new maze.
            new_maze = self.maze_class(config).generate_sample_maze(generate_training_mazes)

            # If it is unique, add it to the correct sets.
            if self._check_maze_unique(new_maze, both_maze_sets):
                both_maze_sets.append(new_maze)
                if generate_training_mazes:
                    self.training_mazes.append(new_maze)
                else:
                    self.validation_mazes.append(new_maze)
                i += 1

            while_loop_iterations += 1
            if while_loop_iterations > 10000:
                print("Already tried 10000 mazes, number of distinct mazes is probably impossible!!!")

        self._write_file()

    def _load_dataset(self):
        self._load_file()
        
        # Use this to check that the validation set still distinct from training.
        # assert(self._check_training_set_again())

        print("Loaded " + str(len(self.training_mazes)) + " training mazes and " +
              str(len(self.validation_mazes)) + " validation mazes succesfully.", flush=True)

    def _write_file(self):
        file = open(self.file_path, "w")
        file.write("Training set\n")
        for i in range(self.training_set_size):
            file.write(str(i))
            file.write(self.maze_class.serialize(self.training_mazes[i]))
        file.write("\nValidation set\n")
        for i in range(self.validation_set_size):
            file.write(str(i + self.training_set_size))
            file.write(self.maze_class.serialize(self.validation_mazes[i]))

        file.close()
        print("New dataset created at path " + self.file_path, flush=True)

    def _load_file(self):
        file = open(self.file_path, "r")
        lines = file.readlines()
        assert (lines[0] == "Training set\n")

        i = 2
        while lines[i] != "Validation set\n":
            maze_lines = []
            while lines[i] != "\n":
                maze_lines.append(lines[i])
                i += 1
            self.training_mazes.append(self.maze_class.deserialize(maze_lines))
            i += 2

        i += 2  # Skip the "Validation set" line.
        while i < len(lines):
            maze_lines = []
            while lines[i] != "\n":
                maze_lines.append(lines[i])
                i += 1
            self.validation_mazes.append(self.maze_class.deserialize(maze_lines))
            i += 2

        file.close()

    @staticmethod
    def _check_maze_unique(new_maze, other_mazes):
        for other_maze in other_mazes:
            if np.array_equal(new_maze, other_maze):
                return False

        return True

    def _check_training_set_again(self):
        for validation_maze in self.validation_mazes:
            if not self._check_maze_unique(validation_maze, self.training_mazes):
                return False
        return True


class BatchMazeProxy:
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size

        training_set_size = config.training_set_size
        validation_set_size = config.validation_set_size

        self.maze_set = PreComputedMazeSet(training_set_size, validation_set_size, config)

        self.current_batch = BatchMaze(config, mazes=[])

    def step(self, actions):
        return self.current_batch.step(actions)

    def reset(self, mutable_maze_info):
        if 'force_maze_index' in mutable_maze_info:
            indices = np.empty(self.batch_size, dtype=np.int)
            indices[:] = mutable_maze_info['force_maze_index']
        else:
            # Generate indices of mazes.
            if mutable_maze_info['sample_from_training_set']:
                min_index = 0
                max_index = self.maze_set.training_set_size
            else:
                min_index = self.maze_set.training_set_size
                max_index = min_index + self.maze_set.validation_set_size
            indices = np.random.randint(low=min_index, high=max_index, size=self.batch_size)

        # Convert each maze array into a maze object.
        actual_mazes = []
        first_observations = []
        first_infos = []
        for index in indices:
            if index < self.maze_set.training_set_size:
                maze_array = self.maze_set.training_mazes[index]
            else:
                maze_array = self.maze_set.validation_mazes[index - self.maze_set.training_set_size]
            maze_object, first_observation, first_info = self.maze_set.maze_class.reset_from_existing_array(self.config, maze_array, mutable_maze_info)
            actual_mazes.append(maze_object)
            first_observations.append(first_observation)
            first_infos.append(first_info)

        # Build batch maze object.
        self.current_batch = BatchMaze(self.config, actual_mazes)
        return first_observations, first_infos

    def render(self):
        self.current_batch.render()

    def get_maze_dimension(self):
        return self.current_batch.get_maze_dimension()

    def get_action_dimension(self):
        return self.current_batch.get_action_dimension()

    def get_state_dimension(self):
        return self.current_batch.get_state_dimension()

    def get_start_location(self):
        return self.current_batch.get_start_location()

    def get_first_maze_from_batch(self):
        return self.current_batch.mazes[0]
