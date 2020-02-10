import torch

from maze_building import *
from agents import *
from models import *


class Config:
    def __init__(self, log_folder_name):
        ########################
        # Maze config.

        # Meta maze config.
        self.maze_class = LandmarkMaze
        self.batch_size = 32  # Number of environments trained in one batch. More envs -> slower but smoother learning.
        self.training_set_size = 5000
        self.validation_set_size = 1000

        # Config for "MazeWorld" mazes.
        self.maze_size = 11
        self.decimation = 0.1
        self.view_length_ind_maze = 7
        self.view_length_lm_maze = 3
        self.drive = True  # Meaning: actions are "forward, turn left, turn right" if true.

        # Reward structure.
        self.living_reward = -0.1
        self.sloth_reward = -0.1
        self.positive_reward = 2
        self.negative_reward = -1
        self.exit_reward = 1

        # Difficulty settings.
        self.kill_early_training = 99
        self.kill_early_eval = 499
        self.max_nb_landmarks = 4
        self.curriculum_learning = False

        ########################
        # Model config.
        self.cpu_or_cuda = "cpu"
        self.model_class = ACRecLin

        ########################
        # Learning config.
        self.agent_class = A2CRecAgent
        self.learning_rate = 1e-3
        self.discount_factor = 0.95
        self.entropy_scaler = 0.1
        self.grad_clip_val = 0.1

        ########################
        # Logging config.
        self.comment = ""

        #########################
        # Experiment config.
        self.nb_training_epochs = 500
        self.total_nb_mazes_per_epoch = 2500
        self.total_nb_validation_mazes_per_epoch = 200

        ########################
        # Ask user for config.
        self._ask_for_classes()
        self._ask_for_other_config()

        #########################
        # Set deduced args.
        self.device = torch.device(self.cpu_or_cuda)

        # Generate example maze to get info about dimensions.
        example_maze = self.maze_class(self)
        self.maze_dim = example_maze.get_maze_dimension()
        self.action_dim = example_maze.get_action_dimension()
        self.observation_dim = example_maze.get_state_dimension()

        self.total_nb_mazes = self.training_set_size + self.validation_set_size

        self.nb_mazes_per_epoch = int(self.total_nb_mazes_per_epoch / self.batch_size)
        self.nb_validation_mazes_per_epoch = int(self.total_nb_validation_mazes_per_epoch / self.batch_size)

        self.folder_name = log_folder_name + \
            self.maze_class.__name__ + "/" + \
            self.model_class.__name__ + "/" + \
            "_lr=" + str(self.learning_rate) + \
            "_gamma=" + str(self.discount_factor)

        if self.maze_class.__name__ != "LandmarkRail":
            self.folder_name += "_maz_siz=" + str(self.maze_size) + \
                "_deci=" + str(self.decimation)

        if self.maze_class.__name__ == "LandmarkMaze" or self.maze_class.__name__ == "LandmarkRail":
            self.folder_name += "_nblm=" + str(self.max_nb_landmarks)

        self.folder_name += "_" + self.comment

        #########################
        # Post-analysis config.
        self.nb_analysis_batch_cycles = 1
        self.render_maze = 28
        self.only_eval_render_maze = True  # If true, we will only compute results on the specific maze.

        print("Config loaded! The folder name is: " + self.folder_name)

    def _ask_for_classes(self):
        print("If you want to use non-default classes, then please just enter them in any order, separated by spaces.")
        print("For example: \"IndicatorMaze ACMem\", otherwise, input an empty string.")
        print("Currently, the default classes are: " + self.maze_class.__name__ + ", " + self.model_class.__name__ +
              ", " + self.agent_class.__name__)

        inp = input()
        if len(inp) == 0:
            return
        else:
            input_array = inp.split(" ")
            for input_el in input_array:
                # Maze classes.
                if input_el == "IndicatorMaze":
                    self.maze_class = IndicatorMaze
                elif input_el == "LandmarkRail":
                    self.maze_class = LandmarkRail
                elif input_el == "LandmarkMaze":
                    self.maze_class = LandmarkMaze

                # Model classes.
                elif input_el == "ACMem":
                    self.model_class = ACMem
                elif input_el == "ACMemConv":
                    self.model_class = ACMemConv
                elif input_el == "ACMemGRU":
                    self.model_class = ACMemGRU
                elif input_el == "ACMemGRUc":
                    self.model_class = ACMemGRUc
                elif input_el == "ACMemKV":
                    self.model_class = ACMemKV
                elif input_el == "ACLin":
                    self.model_class = ACLin
                elif input_el == "ACRecLin":
                    self.model_class = ACRecLin
                elif input_el == "ACMemSimple":
                    self.model_class = ACMemSimple

                # Agent classes.
                elif input_el == "RandomAgent":
                    self.agent_class = RandomAgent
                elif input_el == "A2CRecAgent":
                    self.agent_class = A2CRecAgent

    def _ask_for_other_config(self):
        print("If you want to use non-default config options, then please just enter them in any order, separated by spaces.")
        print("For example: \"maze_size=20 decimation=0.3\", otherwise, input an empty string.")
        print("Currently, the default values are:")
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

        inp = input()
        if len(inp) == 0:
            return

        input_array = inp.split(" ")
        for input_el in input_array:
            equation = input_el.split("=")
            typeee = type(getattr(self, equation[0]))
            setattr(self, equation[0], typeee(equation[1]))

