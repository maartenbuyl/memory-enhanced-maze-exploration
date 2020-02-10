from collections import deque

import torch
import random
import numpy as np


class ReplayMemoryWithHidden(object):
    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, observation, h_state, action, reward, terminal, next_observation, next_h_state):
        self.buffer.append((observation, h_state.squeeze(0), action, reward, terminal, next_observation, next_h_state.squeeze(0)))

    def sample(self, sample_size):
        # Put all states, actions, ... next to eachother in one array
        observation, h_state, action, reward, terminal, next_observation, next_h_state = zip(*random.sample(self.buffer, sample_size))

        # Copy to GPU
        observation = torch.tensor(observation, device=self.device, dtype=torch.float)
        next_observation = torch.tensor(next_observation, device=self.device, dtype=torch.float)
        h_state = torch.tensor(torch.stack(h_state), device=self.device, dtype=torch.float)
        next_h_state = torch.tensor(torch.stack(next_h_state), device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        terminal = torch.tensor(terminal, device=self.device, dtype=torch.float)

        return observation, h_state, action, reward, terminal, next_observation, next_h_state

    def __len__(self):
        return len(self.buffer)


class ReplayMemory(object):
    def __init__(self, capacity, device):
        self.device = device
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, observation, action, reward, terminal, next_observation):
        self.buffer.append((observation, action, reward, terminal, next_observation))

    def sample(self, sample_size):
        # Put all states, actions, ... next to eachother in one array
        observation, action, reward, terminal, next_observation = zip(*random.sample(self.buffer, sample_size))

        # Copy to GPU
        observation = torch.tensor(observation, device=self.device, dtype=torch.float)
        next_observation = torch.tensor(next_observation, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        terminal = torch.tensor(terminal, device=self.device, dtype=torch.float)

        return observation, action, reward, terminal, next_observation

    def __len__(self):
        return len(self.buffer)


class OneRolloutStorage(object):
    def __init__(self):
        self.rewards = []
        self.value_preds = []
        self.actions = []
        self.action_probs = []
        self.returns = []

    def insert(self, reward, value_pred, action, action_probs):
        self.rewards.append(reward)
        self.value_preds.append(value_pred)
        self.actions.append(action)
        self.action_probs.append(action_probs)

    def wipe_rollout(self):
        self.rewards = []
        self.value_preds = []
        self.actions = []
        self.action_probs = []

    # Assuming that last return was from a terminal state.
    def compute_returns(self, possible_reward, gamma):
        if possible_reward:
            self.returns = [torch.tensor(possible_reward(), dtype=torch.float).unsqueeze(0)]
        else:
            self.returns = [torch.tensor(self.rewards[-1], dtype=torch.float).unsqueeze(0)]
        for step in reversed(range(len(self.rewards)-1)):
            previous_return = self.returns[-1] * gamma + self.rewards[step]
            self.returns.append(previous_return)
        self.returns.reverse()


class RolloutStorage(object):
    def __init__(self):
        self.rewards = []
        self.value_preds = []
        self.actions = []
        self.action_probs = []
        self.masks = []
        self.returns = []

    def insert(self, reward, value_pred, action, action_probs, mask):
        self.rewards.append(reward)
        self.value_preds.append(value_pred)
        self.actions.append(action)
        self.action_probs.append(action_probs)
        self.masks.append(mask)

    def continue_after_update(self):
        self.rewards = []
        self.value_preds = []
        self.actions = []
        self.action_probs = []
        self.masks = []

    def compute_returns(self, gamma):
        self.returns = [self.rewards[-1]]
        for step in reversed(range(len(self.actions)-1)):
            next_return = self.returns[-1] * gamma * self.masks[step] + self.rewards[step]
            self.returns.append(next_return)
        self.returns.reverse()

# class RolloutStorage(object):
#     def __init__(self, num_steps, action_size, device, num_processes=1):
#         self.num_steps = num_steps
#         self.device = device
#
#         self.rewards = torch.zeros((num_steps+1, num_processes, 1), device=device, dtype=torch.float)
#         self.value_preds = torch.zeros((num_steps+1, num_processes, 1), device=device, dtype=torch.float)
#         self.actions = torch.zeros((num_steps+1, num_processes, 1), device=device, dtype=torch.long)
#         self.action_probs = torch.zeros((num_steps+1, num_processes, action_size), device=device, dtype=torch.float)
#         self.masks = torch.ones((num_steps+1, num_processes, 1), device=device, dtype=torch.float)
#         self.returns = torch.zeros((num_steps+1, num_processes, 1), device=device, dtype=torch.float)
#
#     def insert(self, step, reward, value_pred, action, action_probs, mask):
#         self.rewards[step] = reward
#         self.value_preds[step].copy_(value_pred)
#         self.actions[step] = action
#         self.action_probs[step].copy_(action_probs)
#         self.masks[step] = float(mask)
#
#     def after_update(self):
#         self.rewards[0].copy_(self.rewards[-1])
#         self.value_preds[0].copy_(self.value_preds[-1])
#         self.actions[0].copy_(self.actions[-1])
#         self.action_probs[0].copy_(self.action_probs[-1])
#         self.masks[0].copy_(self.masks[-1])
#
#     def compute_returns(self, gamma):
#         self.returns[-1] = self.value_preds[-1]
#         for step in reversed(range(self.num_steps)):
#             self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]
