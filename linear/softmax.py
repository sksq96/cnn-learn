# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-21 15:41:21
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-23 01:18:34

import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
	
	data_loss = 0.0
	dW = np.zeros_like(W)
	
	num_classes = W.shape[1]
	num_examples = X.shape[0]
	
	# for each example in training
	for i in range(num_examples):
		scores = np.dot(X[i], W)
		data_loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))

	data_loss /= num_examples
	reg_loss = 0.5 * reg * np.sum(W*W)
	loss = data_loss + reg_loss

	return loss, dW


def softmax_loss_vectorized(W, X, y, reg=0.0):
	
	num_examples = X.shape[0]
	scores = np.dot(X, W)
	
	data_loss = -scores[range(num_examples), y] + np.log(np.sum(np.exp(scores), axis=1))
	data_loss = np.mean(data_loss)
	
	reg_loss = 0.5 * reg * np.sum(W*W)
	loss = data_loss + reg_loss

	dscores = np.exp(scores)
	dscores /= np.sum(dscores, axis=1).reshape(-1, 1)
	dscores[range(num_examples), y] -= 1

	dW = np.dot(X.T, dscores)/num_examples
	dW += reg * W
	
	return loss, dW

