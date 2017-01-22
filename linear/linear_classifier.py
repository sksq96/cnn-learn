# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-22 19:28:48
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-22 20:55:33

import numpy as np
from classifiers.linear_svm import svm_loss_vectorized

class LinearClassifier(object):
	def __init__(self):
		self.W = None

	# Train using stochastic gradient descent
	def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
		num_examples, dim = X.shape
		num_classes = np.max(y) + 1

		if self.W is None:
			self.W = 0.001 * np.random.randn(dim, num_classes)

		# Run stochastic gradient descent to optimize W
		loss_history = []
		for it in range(num_iters):
			# Sample a batch
			idxs = np.random.choice(num_examples, batch_size)
			Xbatch = X[idxs]
			ybatch = y[idxs]
			
			# Compute gradient of loss fn
			loss, dW = self.loss(Xbatch, ybatch, reg)
			loss_history.append(loss)

			# Move in opposite direction of gradient
			self.W -= dW * learning_rate
			
			if verbose and it % 100 == 0:
				print("iteration {} / {}: loss {}".format(it, num_iters, loss))
		
		return loss_history

	def predict(self, X):
		scores = np.dot(X, self.W)
		return np.argmax(scores, axis=1)

	def loss(self, X, y, reg):
		pass


class LinearSVM(LinearClassifier):
	def loss(self, X, y, reg):
		return svm_loss_vectorized(self.W, X, y, reg)
