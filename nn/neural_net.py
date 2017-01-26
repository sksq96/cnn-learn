# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-26 05:50:51
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-26 06:09:58

import numpy as np

class TwoLayerNet:
	"""
	A two-layer fully connected Neural Network.
	Layers: N (input) -- H (hidden) -- C (output)
	Loss: Softmax + L2 regularization
	Structure: input - fully connected layer - ReLU - fully connected layer - softmax
	"""

	def __init__(self, input_size, hidden_size, output_size, , std=1e-4):
		self.params = {}
		self.params['W1'] = np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.random.randn(hidden_size)
		self.params['W2'] = np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.random.randn(output_size)
	
	def train():
		pass

	def predict():
		pass

	def loss():
		pass

