import os

from main_functions import *

if __name__ == '__main__':
    log_folder_name = os.path.abspath(os.path.dirname(__file__)) + "/../results/"

    config = Config(log_folder_name)

    experiment_setup = Experiment(config)

    experiment_setup.run_training_experiment()
