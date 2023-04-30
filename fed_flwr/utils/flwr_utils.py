from typing import Callable, Dict, List, Optional, Tuple, Union, OrderedDict
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.common.logger import log
from flwr.common import (
	EvaluateIns,
	EvaluateRes,
	FitIns,
	FitRes,
	MetricsAggregationFn,
	Parameters,
	Scalar,
	parameters_to_ndarrays,
	ndarrays_to_parameters,
	NDArrays
)
from functools import reduce
from logging import WARNING
import numpy as np

def aggregate_rl(results: List[Tuple[NDArrays, int]], pol_val: List[float], \
len_param_actor: int, temp_a: float=1.0, temp_c: float=1.0) -> NDArrays:
	"""Compute exponentiated weighted average."""

	results_val = list(zip([weights for weights, _ in results], pol_val))

	results_a = [(weights[:len_param_actor], pol_val) \
				for weights, pol_val in results_val]
	results_c = [(weights[len_param_actor:], pol_val) \
				for weights, pol_val in results_val]

	# Calculate the total exponentiated value from training
	exp_val_a_total = sum([np.exp(temp_a * pol_val) for _, pol_val in results_a])
	exp_val_c_total = sum([np.exp(temp_c * pol_val) for _, pol_val in results_c])
	
	# Create a list of weights, each multiplied by the related policy values
	weighted_weights_a = [
		[layer * np.exp(temp_a * pol_val) for layer in weights] \
			for weights, pol_val in results_a
	]
	weighted_weights_c = [
		[layer * np.exp(temp_c * pol_val) for layer in weights] \
			for weights, pol_val in results_c
	]

	# Compute average weights of each layer
	weights_prime_a: NDArrays = [
		reduce(np.add, layer_updates) / exp_val_a_total
		for layer_updates in zip(*weighted_weights_a)
	]
	weights_prime_c: NDArrays = [
		reduce(np.add, layer_updates) / exp_val_c_total
		for layer_updates in zip(*weighted_weights_c)
	]
	weights_prime = weights_prime_a + weights_prime_c
	return weights_prime