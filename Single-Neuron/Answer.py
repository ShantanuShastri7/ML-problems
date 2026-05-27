import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	X = np.array(features)
	W = np.array(weights)
	b = np.array((bias))
	Z = X@W.T + b
	Y = np.array(labels)

	probabilities = 1/(1+np.exp(-Z))

	SE = np.power(probabilities-Y , 2)@np.ones((Y.shape[0]))
	mse = SE/Y.shape[0]
	mse = np.round(mse, 4)
	probabilities = np.round(probabilities, 4)
	return probabilities.tolist(), mse.tolist()