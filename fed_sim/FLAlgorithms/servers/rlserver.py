import torch 
import os 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import gym
import copy
from FLAlgorithms.users.userrl import UserFedRL
from torch.utils.tensorboard import SummaryWriter
import datetime

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


## Defining the RL federation server ###

class RLFedServer:
	def __init__(self,datasets,num_users,num_users_per_round,batch_size,alpha_0,alpha_1,alpha_2,local_epochs,global_iters,dataset_size,seed,temp_a,temp_c,decay_rate,gpu_index_1,gpu_index_2):
		self.env = datasets[0]
		self.num_users = num_users
		self.num_users_per_round = num_users_per_round
		server_env = gym.make(datasets[0])
		self.server_device = torch.device('cpu')
		state_dim = server_env.observation_space.shape[0]
		action_dim = server_env.action_space.shape[0] 
		max_action = float(server_env.action_space.high[0])
		## Creating server versions of Actor and Critic
		self.server_actor = Actor(state_dim, action_dim, max_action).to(self.server_device)
		self.server_critic = Critic(state_dim, action_dim).to(self.server_device)
		self.users = []
		self.total_train_samples = 0
		self.dataset_size = dataset_size
		self.local_epochs = local_epochs
		self.global_iter = global_iters
		total_env = datasets[0] + "_" + datasets[1]
		run_id = "FEDORA_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
		log_path = "Results/" + total_env + "/" + run_id + "/"
		self.writer = SummaryWriter(log_path)
		self.temp_a = temp_a
		self.temp_c = temp_c
		self.writer.add_text("Num Clients",str(self.num_users))
		self.writer.add_text("Num Clients per Round",str(self.num_users_per_round))
		self.writer.add_text("Each User Samples",str(self.dataset_size))
		self.writer.add_text("Local Epochs",str(self.local_epochs))
		self.writer.add_text("Alpha_0",str(alpha_0))
		self.writer.add_text("Alpha_1",str(alpha_1))
		self.writer.add_text("Alpha_2",str(alpha_2))
		self.writer.add_text("Temp A",str(temp_a))
		self.writer.add_text("Temp C",str(temp_c))
		self.writer.add_text("Seed",str(seed))
		self.writer.add_text("decay_rate",str(decay_rate))

		kwargs = {
			"discount": 0.99,
			"tau": 0.005,
			"policy_noise": 0.2 * max_action,
			"noise_clip": 0.5 * max_action,
			"policy_freq": 2,
			"alpha": 2.5,
			"seed":seed,
			"alpha_0":alpha_0,
			"alpha_1":alpha_1,
			"alpha_2":alpha_2,
			"batch_size":batch_size,
			"decay_rate":decay_rate
			}



		### Creating clients with different datasets ###
		for i in range(self.num_users // 2):
			print("Creating client ",i)
			print("Env: " + datasets[0])
			self.writer.add_text("Env/"+str(i),str(datasets[0]))
			print("Start Index", i*dataset_size)
			print("Stop Index", (i+1)*dataset_size)                    
			user = UserFedRL(userid=i,gpu_index=gpu_index_1,eval_env=datasets[0], start_index=i*dataset_size, stop_index=(i+1)*dataset_size,**kwargs)
			self.users.append(user)
			print("Replay Buffer Size",user.replay_buffer.size)
			print("="* 20)
			self.total_train_samples += user.replay_buffer.size

		for i in range(self.num_users // 2,self.num_users):
			print("Creating client ",i)
			print("Env " + datasets[1])
			self.writer.add_text("Env/"+str(i),str(datasets[1]))
			print("Start Index", i*dataset_size)
			print("Stop Index", (i+1)*dataset_size)                    
			user = UserFedRL(userid=i,gpu_index=gpu_index_2,eval_env=datasets[1], start_index=i*dataset_size, stop_index=(i+1)*dataset_size,**kwargs)
			self.users.append(user)
			print("Replay Buffer Size",user.replay_buffer.size)
			print("="* 20)
			self.total_train_samples += user.replay_buffer.size

	def send_parameters_actor(self):
		assert (self.users is not None and len(self.users) > 0)
		for user in self.users:
			user.set_parameters_actor(self.server_actor)

	def send_parameters_critic(self):
		assert (self.users is not None and len(self.users) > 0)
		for user in self.users:
			user.set_parameters_critic(self.server_critic)
	

	def add_parameters_actor(self, user, ratio):
		for server_actor_param, user_actor_param in zip(self.server_actor.parameters(), user.get_parameters_actor()):
			server_actor_param.data = server_actor_param.data + user_actor_param.data.cpu().clone() * ratio
	

	def add_parameters_critic(self, user, ratio):
		for server_critic_param, user_critic_param in zip(self.server_critic.parameters(), user.get_parameters_critic()):
			server_critic_param.data = server_critic_param.data + user_critic_param.data.cpu().clone() * ratio


	def select_users(self, round, num_users):
		if(num_users == len(self.users)):
			print("All users are selected")
			return self.users

		num_users = min(num_users, len(self.users))
		return np.random.choice(self.users, num_users, replace=False)  
	
	
	def aggregate_actor_parameters(self,glob_iter):
		assert (self.users is not None and len(self.users) > 0)
		for param in self.server_actor.parameters():
			param.data = torch.zeros_like(param.data)
		total_train = 0
		
		for user in self.selected_users:
			total_train += np.exp(self.temp_a * user.pol_val)
		for user in self.selected_users:
			ratio = np.exp(self.temp_a * user.pol_val) / total_train
			print(ratio)
			self.writer.add_scalar("c_ratio/"+str(user.userid),ratio,glob_iter+1)
			self.add_parameters_actor(user,ratio)

	def aggregate_critic_parameters(self): ####
		assert (self.users is not None and len(self.users) > 0) #####
		for param in self.server_critic.parameters(): ####
			param.data = torch.zeros_like(param.data) #####
		total_train = 0
		for user in self.selected_users:
			total_train += np.exp(self.temp_c * user.pol_val)
		for user in self.selected_users:
			ratio = np.exp(self.temp_c * user.pol_val) / total_train
			self.add_parameters_critic(user, ratio)

	def train(self):

		for glob_iter in range(self.global_iter):
			avg_rwd = []
			print("-------------Round number: ",glob_iter, " -------------")
			self.send_parameters_actor()
			self.send_parameters_critic()
			global_reward = self.eval_server_policy()
			self.writer.add_scalar("s_rwd",global_reward,glob_iter+1)
			self.selected_users = self.select_users(glob_iter,self.num_users_per_round)
			for user in self.selected_users:
				server_actor_clone = copy.deepcopy(self.server_actor)
				server_critic_clone = copy.deepcopy(self.server_critic)
				user.train(self.local_epochs,server_actor_clone,server_critic_clone)
				user_reward = user.eval_policy()
				avg_rwd.append(user_reward)
				self.writer.add_scalar("c_rwd/"+str(user.userid),user_reward,glob_iter+1)
				self.writer.add_scalar("c_decay/"+str(user.userid),user.decay,glob_iter+1)
			print('Average reward over client:',np.mean(avg_rwd))
			self.writer.add_scalar("avg_c_rwd",np.mean(avg_rwd),glob_iter+1)
			self.aggregate_actor_parameters(glob_iter=glob_iter)
			self.aggregate_critic_parameters()

	## Policy Evaluation ##
	## Given a policy, run its evaluation ##
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.server_device)
		return self.server_actor(state).cpu().data.numpy().flatten()

	def eval_server_policy(self, mean=0, std=1, seed_offset=0, eval_episodes=2):
		eval_env = gym.make(self.env)
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
		print(f"Evaluation of Server avg_reward: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
		return avg_reward