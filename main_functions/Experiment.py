import sys
import io
import os
import numpy as np
from tensorboard_logger import configure, log_value

from maze_precomputing import *


class Experiment:
    def __init__(self, config):
        self.config = config

        # Fix console for Atom, see https://github.com/rgbkrk/atom-script/issues/1166#issuecomment-264268895.
        sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

        # Build proxy of a batch maze object: the proxy will load a dataset or compute it.
        self.maze = BatchMazeProxy(config)

        # Build model and set it to the correct device.
        model = config.model_class(config)
        model.to(config.device)
        self.model = model

        # Build agent and pass model so that it can update some model-specific info.
        self.agent = config.agent_class(model, config)

        # Check if we can resume from checkpoint.
        self.agent_object_filename = config.folder_name + "/saved_agent"
        if os.path.isfile(self.agent_object_filename):
            # Load model and optimizer values.
            self.starting_epoch = self.agent.load_agent(self.agent_object_filename)
            print("Resuming from a saved agent with starting epoch " + str(self.starting_epoch), flush=True)
        else:
            self.starting_epoch = 0

    def run_training_experiment(self):
        # Setup tensorboard
        configure(self.config.folder_name)

        # Shorthand variables.
        config = self.config
        agent = self.agent

        # Run experiment loops.
        for epoch in range(self.starting_epoch, config.nb_training_epochs):
            results = ""

            # Training loop.
            self._run_experiment_loop(only_evaluation=False, sample_training_set=True)

            # Evaluation on training set.
            correct_fraction, wrong_fraction = self._run_experiment_loop(only_evaluation=True, sample_training_set=True)
            results += "T:" + str(correct_fraction) + ","
            results += "|" + str(wrong_fraction) + ","
            log_value('train/correct', correct_fraction, epoch)
            log_value('train/wrong', wrong_fraction, epoch)
            print("Training score: " + str(correct_fraction) + "|" + str(wrong_fraction), flush=True)

            # Evaluation on validation set.
            correct_fraction, wrong_fraction = self._run_experiment_loop(only_evaluation=True, sample_training_set=False)
            results += "V:" + str(correct_fraction) + ","
            results += "|" + str(wrong_fraction) + ","
            log_value('validation/correct', correct_fraction, epoch)
            log_value('validation/wrong', wrong_fraction, epoch)
            print("Validation score: " + str(correct_fraction) + "|" + str(wrong_fraction), flush=True)

            print("This was epoch " + str(epoch) + " out of " + str(config.nb_training_epochs), flush=True)

            agent.save_agent(epoch + 1, self.agent_object_filename)
            results += "\n"
            results_file = open(config.folder_name + "/results.txt", "a")
            results_file.write(results)
            results_file.close()

    # Function to bundle experiment code.
    def _run_experiment_loop(self, only_evaluation=True, sample_training_set=False):
        # Shorthand variables.
        config = self.config
        maze = self.maze
        agent = self.agent
        model = self.model

        # Use curriculum learning if it is mentioned in the config and if this is not a validation set run.
        use_curriculum_learning = config.curriculum_learning and sample_training_set

        # Fill in mutable maze info: a way to pass loop-specific information to the maze.
        mutable_maze_info = {
            'curriculum_learning': use_curriculum_learning,
            'sample_from_training_set': sample_training_set,
            'long_life': only_evaluation
        }

        # Keep track of how many times we chose a correct goal or landmark, and how many times we chose a wrong one.
        nb_correct_goal_chosen = 0
        nb_wrong_goal_chosen = 0

        epoch_length = config.nb_mazes_per_epoch if not only_evaluation else config.nb_validation_mazes_per_epoch
        for maze_i in range(epoch_length):
            observations, starting_infos = maze.reset(mutable_maze_info)
            agent.new_episode(starting_infos)

            # Render one maze during this round if we are evaluating on the training set.
            rendering = only_evaluation and sample_training_set and maze_i == 0
            if rendering:
                new_round_string = "\n\n\n\n\n" \
                                   "############################################################################\n" \
                                   "## New validation round starting, here's an example run on the training set:"
                print(new_round_string, flush=True)
                maze.render()

            # Run until all mazes have terminated.
            all_terminal = False
            while not all_terminal:
                # Sample an action.
                actions = agent.act(observations, under_evaluation=only_evaluation)

                # Print weight info if available.
                if rendering:
                    model.print_weight_info()

                # Process action in environment.
                next_observations, rewards, terminals, infos = maze.step(actions)

                # Experience results of the action.
                if not only_evaluation:
                    agent.experience(observations, actions, rewards, terminals, infos, next_observations)
                else:
                    agent.experience_in_evaluation(terminals, infos)

                # Maintain loop variables.
                observations = next_observations
                all_terminal = np.all(terminals)

                # Render again.
                if rendering:
                    print("############################################################################", flush=True)
                    maze.render()
                    if terminals[0]:
                        print("Terminated! Got reward of " + str(rewards[0]), flush=True)
                        rendering = False

                nb_correct_goal_chosen += np.where(np.isclose(rewards, config.positive_reward))[0].shape[0]
                nb_wrong_goal_chosen += np.where(np.isclose(rewards, config.negative_reward))[0].shape[0]

        _correct_fraction = nb_correct_goal_chosen / (epoch_length * config.batch_size)
        _wrong_fraction = nb_wrong_goal_chosen / (epoch_length * config.batch_size)
        return _correct_fraction, _wrong_fraction

    # Could probably be refactored to use _run_experiment_loop code.
    def post_analysis_experiment(self, analyser):
        # Shorthand variables.
        config = self.config
        maze = self.maze
        agent = self.agent
        model = self.model

        # Option to only evaluate the "render_maze". Results are saved to different file.
        if config.only_eval_render_maze:
            mazes_to_analyse = [config.render_maze]
        else:
            mazes_to_analyse = range(config.total_nb_mazes)

        for cycle in range(config.nb_analysis_batch_cycles):
            for maze_i in mazes_to_analyse:
                # Fill in mutable maze info: a way to pass loop-specific information to the maze.
                mutable_maze_info = {
                    'curriculum_learning': False,
                    'long_life': True,
                    'force_maze_index': maze_i
                }
                observations, starting_infos = maze.reset(mutable_maze_info)
                agent.new_episode(starting_infos)
                analyser.initialize_maze_analysis(maze_i, starting_infos)

                # Render one maze during this round if we are evaluating on the training set.
                rendering = maze_i == config.render_maze
                if rendering:
                    new_round_string = "\n\n\n\n\n" \
                                       "############################################################################\n" \
                                       "## New post-analysis round starting, we render maze " + str(config.render_maze) + ":"
                    print(new_round_string, flush=True)
                    maze.render()

                all_terminal = False
                while not all_terminal:
                    # Sample an action.
                    actions = agent.act(observations, under_evaluation=True)

                    # Print weight info if available.
                    if rendering:
                        model.print_weight_info()

                    # Process action in environment.
                    observations, rewards, terminals, infos = maze.step(actions)

                    # Process reaction in agent.
                    agent.experience_in_evaluation(terminals, infos)

                    all_terminal = np.all(terminals)

                    # Render again.
                    if rendering:
                        print("############################################################################",
                              flush=True)
                        maze.render()
                        if terminals[0]:
                            print("Terminated! Got reward of " + str(rewards[0]), flush=True)
                            rendering = False

                    # Pass along statistics to analyser.
                    analyser.process_stats(observations, rewards, terminals, infos, actions)
