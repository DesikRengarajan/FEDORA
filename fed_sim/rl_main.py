D4RL_SUPPRESS_IMPORT_ERROR=1 
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from FLAlgorithms.servers.rlserver import RLFedServer
import torch
import d4rl
import gym

"""
Server:
	Critic: Average Critics of all selected Users
	Actor: Average Actors of all selected Users 
User:
	Critic: Init policy to federated ciritc, keep a copy of federated critic, use this to take max amongst the current Q and the federated Q, use alpha_1 to prevent deviatio
	Actor: Init policy to federated actor, keep a copy of federated actor, use it to prevent deviation, with alpha_0, keep a copy of previous actor, 
	use it to prevent deviation with alpha_2
Federation Weights of Actor and Critic:
	According to the eval of the current policy on the local dataset
Decay:
	Decay the weight of TD3-BC according to policy_val and server_val
"""

def main(datasets,num_users,num_users_per_round,batch_size,alpha_0,alpha_1,alpha_2,local_epochs,global_iters,dataset_size,seed,temp_a,temp_c,decay_rate,gpu_index_1,gpu_index_2):
	torch.manual_seed(seed)
	np.random.seed(seed)
	server = RLFedServer(datasets,num_users,num_users_per_round,batch_size,alpha_0,alpha_1,alpha_2,local_epochs,global_iters,dataset_size,seed,temp_a,temp_c,decay_rate,gpu_index_1,gpu_index_2)
	server.train()
	torch.cuda.empty_cache()
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env-1", type=str, default="hopper-expert-v2")
	parser.add_argument("--env-2", type=str, default="hopper-medium-v2")
	parser.add_argument("--n-clients", type=int, default=10)
	parser.add_argument("--ncpr", type=int, default=10,help="Number of clients per round")
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--alpha-0", type=float, default=1.0) #Prox to fed policy
	parser.add_argument("--alpha-1", type=float, default=0.0) #Prox of critic
	parser.add_argument("--alpha-2", type=float, default=1.0) #Prox to prev policy
	parser.add_argument("--local-epochs", type=int, default=20)
	parser.add_argument("--n-rounds", type=int, default=1000)
	parser.add_argument("--dataset-size", type=int, default=5000)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--temp-a", type=float, default=0.1)
	parser.add_argument("--temp-c", type=float, default=0.1)
	parser.add_argument("--decay-rate", type=float, default=0.995)
	parser.add_argument("--gpu-index-1", type=int, default=0)
	parser.add_argument("--gpu-index-2", type=int, default=1)
	args = parser.parse_args()

	datasets = [args.env_1,args.env_2]
	main(
		datasets=datasets,
		num_users = args.n_clients,
		num_users_per_round=args.ncpr,
		batch_size=args.batch_size,
		alpha_0=args.alpha_0,
		alpha_1=args.alpha_1,
		alpha_2=args.alpha_2,
		local_epochs=args.local_epochs,
		global_iters=args.n_rounds,
		dataset_size= args.dataset_size,
		seed = args.seed,
		temp_a = args.temp_a,
		temp_c = args.temp_c,
		decay_rate = args.decay_rate,
		gpu_index_1 = args.gpu_index_1,
		gpu_index_2 = args.gpu_index_2
		)
