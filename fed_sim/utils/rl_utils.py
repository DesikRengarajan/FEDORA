import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6),gpu_index = 0):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')



	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def convert_D4RL(self, dataset,lower_lim=None,upper_lim=None):
		if ((lower_lim is None) and (upper_lim is None)):
			self.state = dataset['observations']
			print(len(self.state))
			self.action = dataset['actions']
			self.next_state = dataset['next_observations']
			self.reward = dataset['rewards'].reshape(-1,1)
			self.not_done = 1. - dataset['terminals'].reshape(-1,1)
			self.size = self.state.shape[0]
		
		else:
			self.state = dataset['observations'][lower_lim:upper_lim]
			self.action = dataset['actions'][lower_lim:upper_lim]
			self.next_state = dataset['next_observations'][lower_lim:upper_lim]
			self.reward = dataset['rewards'][lower_lim:upper_lim].reshape(-1,1)
			self.not_done = 1. - dataset['terminals'][lower_lim:upper_lim].reshape(-1,1)
			self.size = self.state.shape[0]
		print("Length of data set:",self.size)

	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std