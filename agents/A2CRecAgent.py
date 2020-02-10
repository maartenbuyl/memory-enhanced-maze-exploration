import torch.nn as nn
import torch.optim as optim
import torch.distributions

from tensorboard_logger import log_value
from agents.BaseAgent import *
from agents.utils import *


class A2CRecAgent(BaseAgent):
    def __init__(self, model, config):
        super(A2CRecAgent, self).__init__(config)

        self.a2c_net = model
        self.optimizer = optim.Adam(self.a2c_net.parameters(), lr=config.learning_rate)
        self.rollouts = RolloutStorage()

        self.entropy_scaler = config.entropy_scaler

        self.batch_counter = 0
        self.loss_counts = 0

        # Last calculated actor and critic values.
        self.last_action_probs = None
        self.last_value_preds = None

        # Data structures to keep track of termination of episodes.
        self.terminated_mazes = np.zeros(self.batch_size, dtype=np.bool)
        self.episode_lengths = np.ones(self.batch_size, dtype=np.int)*-1

    def act(self, observations, under_evaluation=False):
        ##########
        # Prepare observations.

        # Indices where episodes are not yet terminated.
        relevant_indices = np.where(np.logical_not(self.terminated_mazes))[0]

        # From maze shape to flat observation.
        flat_relevant_observations = []
        for rel_idx in relevant_indices:
            flat_screen = observations[rel_idx][0].flatten()
            flattened_observation = np.concatenate((flat_screen, observations[rel_idx][1]))
            flat_relevant_observations.append(flattened_observation)
        flat_relevant_observations = torch.tensor(flat_relevant_observations, device=self.device, dtype=torch.float)

        def _relevant_to_sparse_values(values):
            # Check old shape.
            new_shape = values.size()
            larger_tensor = torch.zeros((self.batch_size, new_shape[1]), device=values.device, dtype=values.dtype)
            larger_tensor[relevant_indices] = values
            return larger_tensor

        ##########
        # Take actions.
        if under_evaluation:
            # Sample action from the policy, without remembering info for back-propagation.
            action_probs = self.a2c_net.get_action_probs(flat_relevant_observations)
            actual_actions = action_probs.multinomial(1)
            actions = _relevant_to_sparse_values(actual_actions)
        else:
            # Sample action.
            action_probs, value_preds = self.a2c_net.evaluate_actions(flat_relevant_observations)
            actual_actions = action_probs.multinomial(1)
            actions = _relevant_to_sparse_values(actual_actions)

            # Store actor and critic value, we will place them together with the whole time step later.
            self.last_action_probs = _relevant_to_sparse_values(action_probs)
            self.last_value_preds = _relevant_to_sparse_values(value_preds)

        return actions

    def experience(self, observations, actions, rewards, terminals, infos, next_observations):
        self._update_network_info(terminals, infos)

        # If the task tells us not to update gradient, don't do so. All information from this experience will be lost.
        #if 'no_grad_update' in infos[0] and infos[0]['no_grad_update']:
        #    return

        # Termination info for the agent.
        for i in range(len(terminals)):
            if self.terminated_mazes[i] != terminals[i]:
                self.episode_lengths[i] = self.batch_counter + 1
                self.terminated_mazes[i] = terminals[i]

        # Ignore "observation", "next_observation" since we don't use replay memory, but consecutive rollouts.
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        masks = torch.tensor([not terminal for terminal in terminals], device=self.device, dtype=torch.float)
        self.rollouts.insert(rewards, self.last_value_preds, actions, self.last_action_probs, masks)

        self.iteration_counter += 1
        self.batch_counter += 1

        # If all episodes are terminated, reflect.
        if np.all(self.terminated_mazes):
            # Find returns of all values to calculate the losses.
            self.rollouts.compute_returns(self.discount_factor)
            self._loss()

            # Clear rollout values.
            self.rollouts.continue_after_update()

    def experience_in_evaluation(self, terminals, infos):
        self._update_network_info(terminals, infos)

        if np.all(terminals):
            self.terminated_mazes = np.zeros(self.batch_size, dtype=np.bool)
        else:
            self.terminated_mazes = terminals

    def new_episode(self, starting_infos):
        self.batch_counter = 0
        self.terminated_mazes = np.zeros(self.batch_size, dtype=np.bool)
        self.episode_lengths = np.ones(self.batch_size, dtype=np.int) * -1

        # Send starting infos to the model.
        self.a2c_net.new_episode(starting_infos)

    def save_agent(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'iteration_counter': self.iteration_counter,
            'model_state_dict': self.a2c_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_agent(self, path):
        saved_agent = torch.load(path)

        epoch = saved_agent['epoch']
        self.iteration_counter = saved_agent['iteration_counter']
        self.a2c_net.load_state_dict(saved_agent['model_state_dict'])
        self.optimizer.load_state_dict(saved_agent['optimizer_state_dict'])

        return epoch

    def _loss(self):
        # Wipe gradient.
        self.optimizer.zero_grad()

        # Compute loss.
        entropy = torch.empty(self.batch_size, device=self.device)
        action_loss = torch.empty(self.batch_size, device=self.device)
        value_loss = torch.empty(self.batch_size, device=self.device)

        def _reshape(list_of_tensors):
            tensor = torch.stack(list_of_tensors, dim=0)
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(2)
            if len(tensor.shape) == 3:  # == Also the case if the previous condition was fullfilled.
                tensor = tensor.permute(1, 0, 2)
            else:
                raise NotImplementedError
            return tensor
        action_probs = _reshape(self.rollouts.action_probs)
        chosen_actions = _reshape(self.rollouts.actions)
        returns = _reshape(self.rollouts.returns)
        value_preds = _reshape(self.rollouts.value_preds)

        # Iterate over batch.
        for b_c in range(self.batch_size):
            ep_len = self.episode_lengths[b_c]

            these_action_probs = action_probs[b_c, :ep_len]
            action_log_probs = (these_action_probs + torch.ones_like(these_action_probs) * 1e-5).log()
            entropy_sum = -(these_action_probs * action_log_probs).sum(1)
            this_entropy = entropy_sum.mean(0)

            these_chosen_actions = chosen_actions[b_c, :ep_len]
            chosen_action_log_probs = action_log_probs.gather(1, these_chosen_actions).squeeze(1)

            advantages = (returns[b_c, :ep_len] - value_preds[b_c, :ep_len]).squeeze(1)

            # TD error.
            this_action_loss = -(chosen_action_log_probs * advantages.detach()).mean(0)
            this_value_loss = advantages.pow(2).mean(0)

            entropy[b_c] = this_entropy
            action_loss[b_c] = this_action_loss
            value_loss[b_c] = this_value_loss

        batch_value_loss = value_loss.mean(0)
        batch_action_loss = action_loss.mean(0)
        batch_entropy = entropy.mean(0)
        loss = batch_value_loss*0.5 + (batch_action_loss - self.entropy_scaler*batch_entropy)

        if loss != loss:
            print(loss)
            print(value_preds)
            print(value_loss)
            print(action_loss)
            print(entropy)

        # Propagate gradient.
        loss.backward()

        # Clip gradient.
        nn.utils.clip_grad_norm_(self.a2c_net.parameters(), self.grad_clip_val)

        # Update weights.
        self.optimizer.step()

        # Log loss from time to time
        self.loss_counts += 1
        if self.loss_counts % 10 == 0:
            log_value('train/action_loss', batch_action_loss, self.iteration_counter)
            log_value('train/value_loss', batch_value_loss, self.iteration_counter)
            self.loss_counts = 0

    def _update_network_info(self, terminals, infos):
        # Termination info for hidden state units in recurrent network.
        nb_non_terminated = 0   # Number of mazes that weren't terminated before this function call.
        non_terminated_indices = []  # Indices of those mazes.
        for i in range(len(terminals)):
            if not terminals[i]:
                non_terminated_indices.append(nb_non_terminated)
            if not self.terminated_mazes[i]:
                nb_non_terminated += 1

        self.a2c_net.update_task_info(non_terminated_indices, infos)
