# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-20 17:56:03
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-20 21:28:20

import numpy as np
from collections import Counter

class KNearestNeighbour:
	
	def train(self, X, y):
		self.Xtrain = X
		self.ytrain = y

	def predict(self, dists, K=1):
		# predict correct class 
		# given distance matrix

		n_test = dists.shape[0]
		ypred = np.zeros(n_test)
		
		for row in range(n_test):
			# select top k labels and
			# take vote among them
			top_k = self.ytrain[np.argsort(dists[row])[:K]]
			pred = Counter(top_k).most_common()[0][0]
			ypred[row] = pred

		return ypred

	def distance(self, x, y):
		# L2 norm
		return np.sum((x-y)**2)

	def distance_two_loops(self, Xtest):
		# computes distance between each test
		# exampe and each train example

		n_test, n_train = Xtest.shape[0], self.Xtrain.shape[0]
		dist = np.zeros((n_test, n_train))

		for i in range(n_test):
			for j in range(n_train):
				dist[i][j] = self.distance(Xtest[i], self.Xtrain[j])
		
		return dist

	# could be slower for large data
	# http://stackoverflow.com/q/39502630/4335937
	def distance_one_loop(self, Xtest):
		n_test, n_train = Xtest.shape[0], self.Xtrain.shape[0]
		dist = np.zeros((n_test, n_train))

		for i in range(n_test):
			dist[i] = np.sum((self.Xtrain - Xtest[i])**2, axis=1)
		
		return dist


	def distance_no_loop(self, Xtest):
		a_square = np.sum(self.Xtrain**2, axis=1).reshape(1, -1)
		b_square = np.sum(Xtest**2, axis=1).reshape(-1, 1)
		a_into_b = np.dot(Xtest, self.Xtrain.T)
		
		return a_square + b_square - 2 * a_into_b

