# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-26 05:50:51
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-28 16:28:12

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

		# class probabiliies and negative log 
		exp_scores = np.exp(scores)
		prob = exp_scores / np.sum(exp_scores, axis=1).reshape(-1, 1)
		negative_log_prob = - np.log(prob[range(num_examples), y])

		# loss
		data_loss = np.mean(negative_log_prob)
		reg_loss = 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))
		loss = data_loss + reg_loss

		# Backward Pass: Gradient
		grad = {}

		# backprob the gradient
		dscores = prob
		dscores[range(num_examples), y] -= 1
		dscores /= num_examples

		grad['W2'] = np.dot(hidden.T, dscores)
		grad['b2'] = np.sum(dscores, axis=0)

		dhidden = np.dot(dscores, W2.T)
		dhidden[hidden <= 0] = 0

		grad['W1'] = np.dot(X.T, dhidden)
		grad['b1'] = np.sum(dhidden, axis=0)

		grad['W2'] += reg * W2
		grad['W1'] += reg * W1

		return loss, grad

	def train(self, X, y, Xval, yval, learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
		
		num_examples, dim = X.shape
		num_classes = np.max(y) + 1
		iterations_per_epoch = max(num_examples / batch_size, 1)

		loss_history, train_acc_history, val_acc_history = [], [], []
		
		for it in range(num_iters):
			# sample a mini batch
			idx = np.random.choice(num_examples, batch_size)
			Xbatch = X[idx]
			ybatch = y[idx]

			# find direction to update weights
			loss, grads = self.loss(Xbatch, y=ybatch, reg=reg)
			loss_history.append(loss)

			for param in self.params:
				self.params[param] -= grads[param] * learning_rate


			if verbose and it % 100 == 0:
				# Check accuracy
				train_acc = (self.predict(Xbatch) == ybatch).mean()
				val_acc = (self.predict(Xval) == yval).mean()
				train_acc_history.append(train_acc)
				val_acc_history.append(val_acc)
				print('iteration {} / {}: loss {}, train accuracy {}, val accuracy {}'.format(it, num_iters, loss, train_acc, val_acc))

			# Every epoch, check train and val accuracy and decay learning rate.
			if it % iterations_per_epoch == 0:
				# Decay learning rate
				learning_rate *= learning_rate_decay

		return loss_history, train_acc_history, val_acc_history


	def predict(self, X):
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		
		h1 = np.dot(X, W1) + b1
		h1[h1 < 0] = 0
		scores = np.dot(h1, W2) + b2

		return np.argmax(scores, axis=1)

