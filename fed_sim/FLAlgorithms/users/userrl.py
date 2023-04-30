D4RL_SUPPRESS_IMPORT_ERROR=1 
import torch 
import os 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import gym
import copy
from utils.rl_utils import *
import d4rl

 ### Defining the actor and critic models ###
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class UserFedRL:
	def __init__(
		self,
		userid=1,
		gpu_index=0,
		eval_env="hopper-expert-v0",
		start_index=0, 
		stop_index=2000,
		alpha_0=0,
		alpha_1=0,
		alpha_2=0,
		batch_size=256,
		seed=0,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
		decay_rate=0.995
	): 
		self.userid = userid
		self.eval_env = eval_env
		self.alpha_0 = alpha_0
		self.alpha_1 = alpha_1
		self.alpha_2 = alpha_2
		self.batch_size = batch_size
		self.seed = seed
		env = gym.make(self.eval_env)
		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0] 
		max_action = float(env.action_space.high[0])        
		self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
		
		self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.prev_actor = copy.deepcopy(self.actor) ###

		self.critic = Critic(state_dim, action_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha
		self.total_it = 0
		self.pol_val = 0
		self.server_val = 0
		self.decay = 1
		self.decay_rate = decay_rate

		self.replay_buffer = ReplayBuffer(state_dim, action_dim,gpu_index=gpu_index)
		dataset = d4rl.qlearning_dataset(env)
		self.replay_buffer.convert_D4RL(dataset,lower_lim=start_index,upper_lim=stop_index)

	def get_parameters_actor(self):
		for param in self.actor.parameters():
			param.detach()
		return self.actor.parameters()

	def get_parameters_critic(self):
		for param in self.critic.parameters():
			param.detach()
		return self.critic.parameters()


	def set_parameters_actor(self,server_actor):
		for old_param, new_param in zip(self.actor.parameters(), server_actor.parameters()):
			old_param.data = new_param.data.clone().to(self.device)
			
	def set_parameters_critic(self,server_critic):
		for old_param, new_param in zip(self.critic.parameters(), server_critic.parameters()):
			old_param.data = new_param.data.clone().to(self.device)


	def train(self,local_epochs,server_actor,server_critic):
		server_actor = server_actor.to(self.device)
		server_critic = server_critic.to(self.device)
		total_epochs = (self.replay_buffer.size // self.batch_size)  * local_epochs
		self.server_val = self.eval_pol(server_actor,server_critic)
		for eph in range(total_epochs):
			self.train_TD3(server_actor,server_critic)
		self.pol_val = self.eval_pol(self.actor,self.critic) #value of the current policy according to the current dataset
		self.prev_actor = copy.deepcopy(self.actor) ###
		if self.server_val > self.pol_val:
			self.decay = self.decay * self.decay_rate


	def eval_pol(self,actor,critic):
		state, action, next_state, reward, not_done = self.replay_buffer.sample(self.replay_buffer.size)
		with torch.no_grad():
			pi = actor(state)
			pol_val = critic.Q1(state, pi).mean().cpu().numpy()
		return pol_val
		


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train_TD3(self,server_actor,server_critic):
		self.total_it += 1
		# Sample replay buffer 
		state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)
			#Computing the minimum of the server Qs, to take min for the target update
			fed_Q1, fed_Q2 = server_critic(next_state,next_action)
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			fed_min = torch.min(fed_Q1,fed_Q2)
			target_Q = torch.max(target_Q,fed_min) ##
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		with torch.no_grad():
			fed_Q1, fed_Q2 = server_critic(state,action)

		
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + self.alpha_1 * F.mse_loss(current_Q1, fed_Q1) + self.alpha_1 * F.mse_loss(current_Q2, fed_Q2)
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			server_pi = server_actor(state).detach()
			prev_pi = self.prev_actor(state).detach() ###
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha/Q.abs().mean().detach()
			actor_loss = -lmbda * self.decay * Q.mean() + self.decay * F.mse_loss(pi, action) + self.alpha_0 * F.mse_loss(pi,server_pi) + self.alpha_2 * F.mse_loss(pi,prev_pi) ###
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
	


	## Policy Evaluation ##
	## Given a policy, run its evaluation ##
	def eval_policy(self, mean=0, std=1, seed_offset=0, eval_episodes=2):
		eval_env = gym.make(self.eval_env)
		eval_env.seed(self.seed)
		avg_reward = 0.
		for _ in range(eval_episodes):
			state, done = eval_env.reset(), False
			while not done:
				state = (np.array(state).reshape(1,-1) - mean)/std
				action = self.select_action(state)
				state, reward, done, _ = eval_env.step(action)
				avg_reward += reward

		avg_reward /= eval_episodes
		d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
		print(f"Evaluation of Client {self.userid} avg_reward: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f} dataset: {self.eval_env} decay: {self.decay:.3f}")
		return avg_reward






