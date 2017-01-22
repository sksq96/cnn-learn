# -*- coding: utf-8 -*-
# @Author: shubham
# @Date:   2017-01-21 15:41:21
# @Last Modified by:   shubham
# @Last Modified time: 2017-01-22 20:16:19

import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
	
	data_loss = 0.0
	dW = np.zeros_like(W)
	
	num_classes = W.shape[1]
	num_examples = X.shape[0]
	
	S = []

	# for each example in training
	for i in range(num_examples):
		scores = np.dot(X[i], W)
		S.append(scores)
		for j in range(num_classes):
			data_loss += max(0, scores[j] - scores[y[i]] + 1)
		data_loss -= 1

	S = np.vstack(S)
	
	data_loss /= num_examples
	reg_loss = 0.5 * reg * np.sum(W*W)
	loss = data_loss + reg_loss

	return loss, dW


def svm_loss_vectorized(W, X, y, reg=0.0):
	
	num_examples = X.shape[0]
	delta = 1

	scores = np.dot(X, W)
	margin = (scores - scores[range(num_examples), y].reshape(-1, 1) + delta).clip(0)
	
	# for each example, extra delta was added in margin
	# so subtract: num_examples*delta
	data_loss = margin.sum() - num_examples * delta
	data_loss /= num_examples
	reg_loss = 0.5 * reg * np.sum(W*W)
	loss = data_loss + reg_loss

	dmargin = np.copy(margin)
	dmargin[dmargin > 0] = 1
	dmargin[range(num_examples), y] = -(np.sum(dmargin, axis=1) - 1)

	dW = np.dot(X.T, dmargin)/num_examples
	dW += reg * W
	
	return loss, dW

