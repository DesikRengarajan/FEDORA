import torch 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
import gym
import d4rl
import yaml
import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.nets import Actor, Critic
from utils.flwr_utils import aggregate_rl
import flwr as fl
from typing import Dict, List, Optional, Tuple, Union, OrderedDict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.common.logger import log
from flwr.common import (
	EvaluateRes,
	FitIns,
	FitRes,
	Parameters,
	Scalar,
	parameters_to_ndarrays,
	ndarrays_to_parameters,
)


class ServerFedRL:
	def __init__(self, c_config, s_config) -> None:
		self.seed = c_config["seed"]
		torch.manual_seed(self.seed)
		np.random.seed(self.seed)
		self.env_name = s_config["env_1"]
		server_env = gym.make(self.env_name)
		self.server_device = torch.device('cpu')
		state_dim = server_env.observation_space.shape[0]
		action_dim = server_env.action_space.shape[0] 
		max_action = float(server_env.action_space.high[0])
		self.server_actor = Actor(state_dim, action_dim, max_action).to(self.server_device)
		self.server_critic = Critic(state_dim, action_dim).to(self.server_device)
		self.temp_a = s_config["temp_a"]
		self.temp_c = s_config["temp_c"]
		self.len_param_actor = len(self.server_actor.state_dict().keys())
		self.len_param_critic = len(self.server_critic.state_dict().keys())


	def set_parameters_actor(self, params):
		params_dict = zip(self.server_actor.state_dict().keys(), params)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.server_actor.load_state_dict(state_dict, strict=True)


	def set_parameters_critic(self, params):
		params_dict = zip(self.server_critic.state_dict().keys(), params)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.server_critic.load_state_dict(state_dict, strict=True)


	def set_parameters(self, params):
		if len(params) != self.len_param_actor + self.len_param_critic:
			raise SystemExit("Error: Actor and Critic parameter length mismatch.")
		param_actor = params[:self.len_param_actor]
		param_critic = params[self.len_param_actor:]
		self.set_parameters_actor(param_actor)
		self.set_parameters_critic(param_critic)


	## Policy Evaluation ##
	## Given a policy, run its evaluation ##
	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.server_device)
		return self.server_actor(state).cpu().data.numpy().flatten()


	def eval_server_policy(self, mean=0, std=1, seed_offset=0, eval_episodes=2):
		eval_env = gym.make(self.env_name)
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
		print(f"Evaluation of Server avg_reward: {avg_reward:.3f},\
			 D4RL score: {d4rl_score:.3f}")
		return avg_reward, d4rl_score



class CustomMetricStrategy(fl.server.strategy.FedAvg):

	def set_server(self, server: ServerFedRL):
		self.server = server


	def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
		ret_sup = super().configure_fit(server_round, parameters, client_manager)
		s_rwd, s_d4rl_score = self.server.eval_server_policy()
		writer.add_scalar("s_rwd", s_rwd, server_round)
		# print("round {}, \ts_rwd {:.3f}"\
		# 	.format(server_round, s_rwd))
		return ret_sup


	def aggregate_fit(
		self,
		server_round: int,
		results: List[Tuple[ClientProxy, FitRes]],
		failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
	) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
		
		pol_val = []
		for client in range(len(results)):
			res = results[client]
			pol_val.append(res[1].metrics["c_pol_val"])
			c_id = res[0].cid.split("ipv6:")[-1]

		if not results:
			return None, {}
		# Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures:
			return None, {}

		# Convert results
		weights_results = [
			(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
			for _, fit_res in results
		]

		# Weigh and compute weights
		weights_updated = aggregate_rl(weights_results, pol_val, server.len_param_actor, \
			server.temp_a, server.temp_c)
		parameters_aggregated = ndarrays_to_parameters(weights_updated)

		# Update server model weights
		self.server.set_parameters(weights_updated)

		metrics_aggregated = {}
		aggregated_weights = (parameters_aggregated, metrics_aggregated)

		return aggregated_weights


	def aggregate_evaluate(
		self,
		server_round: int,
		results: List[Tuple[ClientProxy, EvaluateRes]],
		failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
	) -> Tuple[Optional[float], Dict[str, Scalar]]:

		avg_c_rwd = 0.0
		for client in range(len(results)):
			res = results[client]
			c_id = res[0].cid.split("ipv6:")[-1]
			c_rwd = res[1].metrics["c_rwd"]
			c_d4rl_score = res[1].metrics["c_d4rl_score"]
			c_decay = res[1].metrics["c_decay"]
			avg_c_rwd += c_rwd

			writer.add_scalar("c_rwd/" + c_id, \
				c_rwd, server_round)
			# writer.add_scalar("c_d4rl_score/" + c_id, \
			# 	c_d4rl_score, server_round)
			writer.add_scalar("c_decay/" + c_id, \
				c_decay, server_round)

		avg_c_rwd /= len(results)
		writer.add_scalar("avg_c_rwd", avg_c_rwd, server_round)

		# s_rwd, s_d4rl_score = self.server.eval_server_policy()
		# writer.add_scalar("s_rwd", s_rwd, server_round)

		# print("round {}, \ts_rwd {:.3f}, \ts_d4rl_score {:.3f} \tavg_c_rwd {:.3f}"\
		# 	.format(server_round, s_rwd, s_d4rl_score, avg_c_rwd))

		print("round {}, \tavg_c_rwd {:.3f}"\
			.format(server_round, avg_c_rwd))

		return super().aggregate_evaluate(server_round, results, failures)



if __name__ == "__main__":
	with open("config/c_config.yml", "r") as config_file:
		c_config = yaml.safe_load(config_file)
	with open("config/s_config.yml", "r") as config_file:
		s_config = yaml.safe_load(config_file)

	total_env = s_config["env_1"] + "_" + s_config["env_2"]
	run_id = "FEDORA_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
	log_path = "Results/" + total_env + "/" + run_id + "/"
	writer = SummaryWriter(log_path)
	writer.add_text("Num Rounds", str(s_config["n_rounds"]))
	writer.add_text("Num Clients", str(s_config["n_clients"]))
	writer.add_text("Num Clients per Round", str(s_config["ncpr"]))
	writer.add_text("Local Epochs", str(c_config["local_epochs"]))
	writer.add_text("Alpha_0", str(c_config["alpha_0"]))
	writer.add_text("Alpha_1", str(c_config["alpha_1"]))
	writer.add_text("Alpha_2", str(c_config["alpha_2"]))
	writer.add_text("Temp A", str(s_config["temp_a"]))
	writer.add_text("Temp C", str(s_config["temp_c"]))
	writer.add_text("Seed", str(c_config["seed"]))
	writer.add_text("decay_rate", str(c_config["decay_rate"]))

	fraction_c = s_config["ncpr"] / s_config["n_clients"]
	min_c = s_config["ncpr"]
	num_c = s_config["n_clients"]

	server = ServerFedRL(c_config, s_config)

	strategy = CustomMetricStrategy(
		fraction_fit = fraction_c,
		fraction_evaluate = fraction_c,
		min_fit_clients = min_c,
		min_evaluate_clients = min_c,
		min_available_clients = num_c,
	)

	strategy.set_server(server)

	hist = fl.server.start_server(
		config = fl.server.ServerConfig(num_rounds=s_config["n_rounds"]),
		strategy = strategy,
		server_address = c_config["server_ip"]
	)