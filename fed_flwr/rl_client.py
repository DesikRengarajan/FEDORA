import flwr as fl
import torch 
import numpy as np 
import torch.nn.functional as F
from collections import OrderedDict
import gym
import d4rl
import argparse
import yaml
import copy
from utils.rl_utils import ReplayBuffer
from utils.nets import Actor, Critic



class ClientFedRL:

	def __init__(self, gpu_index, eval_env, start_index, stop_index, \
		c_config):
		self.eval_env = eval_env
		self.seed = c_config["seed"]
		torch.manual_seed(self.seed)
		np.random.seed(self.seed)
		self.device = torch.device('cuda', index=gpu_index) \
			if (torch.cuda.is_available() and gpu_index > -1) \
			else torch.device('cpu')	
		env = gym.make(self.eval_env)
		state_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0] 
		max_action = float(env.action_space.high[0])        
		self.max_action = max_action
		self.alpha_0 = c_config["alpha_0"] 
		self.alpha_1 = c_config["alpha_1"] 
		self.alpha_2 = c_config["alpha_2"] 
		self.batch_size = c_config["batch_size"] 
		self.discount = c_config["discount"] 
		self.tau = c_config["tau"]  
		self.policy_noise = c_config["policy_noise_f"] * max_action
		self.noise_clip = c_config["noise_clip_f"] * max_action
		self.policy_freq = c_config["policy_freq"] 
		self.alpha = c_config["alpha"]
		self.decay_rate =c_config["decay_rate"]
		self.l_r = c_config["l_r"]  
		self.local_epochs = c_config["local_epochs"]
 
		self.total_it = 0
		self.pol_val = 0
		self.server_val = 0
		self.decay = 1

		self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.l_r)
		self.prev_actor = copy.deepcopy(self.actor)
		self.critic = Critic(state_dim, action_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.l_r)
		self.len_param_actor = len(self.actor.state_dict().keys())
		self.len_param_critic = len(self.critic.state_dict().keys())

		self.replay_buffer = ReplayBuffer(state_dim, action_dim, gpu_index=gpu_index)
		dataset = d4rl.qlearning_dataset(env)
		self.replay_buffer.convert_D4RL(dataset, lower_lim=start_index, upper_lim=stop_index)


	def get_parameters_actor(self):
		return [val.cpu().numpy() for _, val in self.actor.state_dict().items()]


	def get_parameters_critic(self):
		return [val.cpu().numpy() for _, val in self.critic.state_dict().items()]


	def get_parameters_combined(self):
		param_actor = self.get_parameters_actor()
		param_critic = self.get_parameters_critic()
		param_combined = param_actor + param_critic
		return param_combined


	def set_parameters_actor(self, params):
		params_dict = zip(self.actor.state_dict().keys(), params)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.actor.load_state_dict(state_dict, strict=True)


	def set_parameters_critic(self, params):
		params_dict = zip(self.critic.state_dict().keys(), params)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.critic.load_state_dict(state_dict, strict=True)


	def set_parameters_combined(self, params):
		if len(params) != self.len_param_actor + self.len_param_critic:
			raise SystemExit("Error: Actor and Critic parameter length mismatch.")
		param_actor = params[:self.len_param_actor]
		param_critic = params[self.len_param_actor:]
		self.set_parameters_actor(param_actor)
		self.set_parameters_critic(param_critic)


	def eval_pol(self, actor, critic):
		state, action, next_state, reward, not_done = \
			self.replay_buffer.sample(self.replay_buffer.size)
		with torch.no_grad():
			pi = actor(state)
			pol_val = critic.Q1(state, pi).mean().cpu().numpy()
		return pol_val
		

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self):
		server_actor = copy.deepcopy(self.actor)
		server_critic = copy.deepcopy(self.critic)
		total_epochs = (self.replay_buffer.size // self.batch_size) * self.local_epochs
		self.server_val = self.eval_pol(server_actor, server_critic)
		for epoch in range(total_epochs):
			self.train_TD3(server_actor, server_critic)
		# Value of the current policy according to the current dataset
		self.pol_val = self.eval_pol(self.actor, self.critic).item() 
		self.prev_actor = copy.deepcopy(self.actor)
		if self.server_val > self.pol_val:
			self.decay = self.decay * self.decay_rate


	def train_TD3(self, server_actor, server_critic):
		self.total_it += 1
		# Sample replay buffer 
		state, action, next_state, reward, not_done = \
			self.replay_buffer.sample(self.batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)
			# Computing the minimum of the server Qs, to take min for the target update
			fed_Q1, fed_Q2 = server_critic(next_state,next_action)
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			fed_min = torch.min(fed_Q1,fed_Q2)
			target_Q = torch.max(target_Q,fed_min)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
		with torch.no_grad():
			fed_Q1, fed_Q2 = server_critic(state,action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) \
			+ F.mse_loss(current_Q2, target_Q) + self.alpha_1 \
			* F.mse_loss(current_Q1, fed_Q1) + self.alpha_1 \
			* F.mse_loss(current_Q2, fed_Q2)
		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			server_pi = server_actor(state).detach()
			prev_pi = self.prev_actor(state).detach()
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha/Q.abs().mean().detach()
			actor_loss = -lmbda * self.decay * Q.mean() \
				+ self.decay * F.mse_loss(pi, action) \
				+ self.alpha_0 * F.mse_loss(pi,server_pi) \
				+ self.alpha_2 * F.mse_loss(pi,prev_pi)
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in \
			zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data \
					+ (1 - self.tau) * target_param.data)

			for param, target_param in \
			zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data \
					+ (1 - self.tau) * target_param.data)
	

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
		# print(f"Evaluation of client ; c_rwd: {avg_reward:.3f}, \
		# 	D4RL score: {d4rl_score:.3f} dataset: {self.eval_env} c_decay: {self.decay:.3f}")
		return avg_reward, d4rl_score, self.decay



class NumPyClientRL(fl.client.NumPyClient):
	def get_parameters(self, config=None):
		return client.get_parameters_combined()


	def set_parameters(self, params):
		client.set_parameters_combined(params)


	def fit(self, params, config=None):
		self.set_parameters(params)
		client.train()
		c_pol_val = client.pol_val
		return self.get_parameters(), 1, {"c_pol_val":c_pol_val}


	def evaluate(self, params, config=None):
		#self.set_parameters(params)
		c_rwd, c_d4rl_score, c_decay = client.eval_policy()
		return 0.0, 1, {"c_rwd":c_rwd, "c_d4rl_score":c_d4rl_score, \
			"c_decay":c_decay}



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu-index", type=int, default=-1, \
		help="GPU index for training, default:CPU")
	parser.add_argument("--eval-env", type=str, default="hopper-expert-v0", \
		help="client gym environment")
	parser.add_argument("--start-index", type=int, default=0, \
		help="start index of d4rl sample")
	parser.add_argument("--stop-index", type=int, default=2000, \
		help="stop index of d4rl sample")
	args = parser.parse_args()

	with open("config/c_config.yml", "r") as config_file:
		c_config = yaml.safe_load(config_file)
	
	client = ClientFedRL(args.gpu_index, args.eval_env, \
		args.start_index, args.stop_index, c_config)

	fl.client.start_numpy_client(server_address=c_config["server_ip"], \
		client=NumPyClientRL())