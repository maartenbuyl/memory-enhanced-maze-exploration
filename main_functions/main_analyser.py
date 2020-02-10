import os

from main_functions import *

if __name__ == '__main__':
    log_folder_name = os.path.abspath(os.path.dirname(__file__)) + "/../results/"

    config = Config(log_folder_name)

    experiment_setup = Experiment(config)

    post_analyser = Analyser(experiment_setup, config)

    # Code to do time profiling.
    # import cProfile
    # cProfile.run('experiment_setup.post_analysis_experiment(post_analyser)')

    experiment_setup.post_analysis_experiment(post_analyser)

    name_extra = ""
    if config.only_eval_render_maze:
        name_extra += "_maze_" + str(config.render_maze)
    post_analyser.write_results(config.folder_name, name_extra)
