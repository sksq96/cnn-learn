# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-26 05:50:51
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-26 16:12:06

import numpy as np

class TwoLayerNet:
	"""
	A two-layer fully connected Neural Network.
	Layers: N (input) -- H (hidden) -- C (output)
	Loss: Softmax + L2 regularization
	Structure: input - fully connected layer - ReLU - fully connected layer - softmax
	"""

	def __init__(self, input_size, hidden_size, output_size, std=1e-4):
		self.params = {}
		self.params['W1'] = std * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)
	
	def loss(self, X, y=None, reg=0.0):
		"""Compute scores, loss and gradient"""
		
		# Foward Pass: Scores and Loss
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']

		num_examples, dim = X.shape

		# first fully connected
		hidden = np.dot(X, W1) + b1
		# RELU
		hidden[hidden < 0] = 0
		# second fully connected
		scores = np.dot(hidden, W2) + b2

		# If the targets are not given then jump out, we're done
		if y is None:
			return scores

		# compute loss
		data_loss = - scores[range(num_examples), y] + np.log(np.sum(np.exp(scores), axis=1)).reshape(-1, 1)
		data_loss = np.mean(data_loss)
		reg_loss = 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))
		loss = data_loss + reg_loss

		# Backward Pass: Gradient
		

		return loss, {}

	def train():
		pass

	def predict():
		pass


