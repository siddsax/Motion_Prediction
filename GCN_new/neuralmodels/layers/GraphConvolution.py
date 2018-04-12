from __future__ import print_function
from __future__ import division
import theano
import numpy as np
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.sparse.basic as sp
from theano.tensor.elemwise import CAReduce
# from activations import *
# from inits import *
# from utils import *
# from Dropout import Dropout
from headers import *


class GraphConvolution_hetro(object):

	def __init__(self, size, adjacency, num_features_nonzero=False, drop_value=None, rng=None, init='glorot', bias=False, sparse_inputs=False, dropout=True, activation_str='rectify', weights=False, featureless=False):

		self.settings = locals()
		del self.settings['self']
		self.sparse_inputs = sparse_inputs
		self.size = size
		self.rng = rng
		# temp = inits()
		self.init = getattr(inits, init)
		# temp = activations()
		self.activation = getattr(activations, activation_str)
		self.featureless = featureless
		self.weights = weights
		self.bias = bias
		self.adjacency = adjacency
		if dropout:
			self.drop_value = drop_value
		else:
			self.drop_value = 0
		self.numparams = 0
		if self.sparse_inputs:
			self.num_features_nonzero = num_features_nonzero

	def connect(self, layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size

		self.W = list()
		if self.bias:
			self.b = list()
		self.nonzeros = {}
		for i in range(np.shape(self.adjacency)[0]):
			count = 0
			self.nonzeros[i] = []
			for j in range(np.shape(self.adjacency)[1]):
				if(self.adjacency[i, j]):
					self.nonzeros[i].append(j)
					count += 1
			self.W.append(self.init((count, self.inputD, self.size), rng=self.rng))
			self.numparams += count*self.inputD*self.size
			if self.bias:
				self.b = zero0s((count, self.size))
				self.numparams += count*self.size

		self.params = []
		self.params += self.W
		if self.bias:
			self.params += self.b

		if (self.weights):
			for param, weight in zip(self.params, self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))

		self.L2_sqr = 0
		for W in self.W:
			self.L2_sqr += (W ** 2).sum()

	def output(self, seq_output=True):
		x = self.layer_below.output(seq_output=seq_output)
		for i in range(np.shape(self.adjacency)[0]):
			for j in range(1, len(self.nonzeros[i])):
				if(j==0):
					out_d = T.tensordot(x[:, :, self.nonzeros[i][j], :],self.W[i][j, :, :], axes=[2, 0])
				else:
					out_d += T.tensordot(x[:, :, self.nonzeros[i][j], :],self.W[i][j, :, :], axes=[2, 0])
				if self.bias:
					out_d += self.b[i][j, :]
			if(i == 0):
				out = out_d.reshape((out_d.shape[0], out_d.shape[1], 1, out_d.shape[2]))
			else:
				out = T.concatenate((out, out_d.reshape((out_d.shape[0], out_d.shape[1], 1, out_d.shape[2]))), axis=2)

		return self.activation(out)


class GraphConvolution_homo(object):

	def __init__(self, size, adjacency, num_features_nonzero=False, drop_value=None, rng=None, init='glorot', bias=False, sparse_inputs=False, dropout=True, activation_str='rectify', weights=False):

		self.settings = locals()
		del self.settings['self']
		self.sparse_inputs = sparse_inputs
		self.size = size
		self.rng = rng
		# temp = inits()
		self.init = getattr(inits, init)
		# temp = activations()
		self.activation = getattr(activations, activation_str)
		self.weights = weights
		self.bias = bias
		self.adjacency = adjacency
		if dropout:
			self.drop_value = drop_value
		else:
			self.drop_value = 0
		self.numparams = 0
		if self.sparse_inputs:
			self.num_features_nonzero = num_features_nonzero

	def connect(self, layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size
		# self.W = list()
		# for i in range(len(self.adjacency)):
		# 	self.W.append(self.init((self.inputD,self.size),rng=self.rng))
		self.W = self.init((self.inputD, self.size), rng=self.rng)

		self.numparams += self.inputD * self.size
		if self.bias:
			self.b = zero0s((self.size))
			self.numparams += self.size
			self.params = [self.W, self.b]
		else:
			self.params = [self.W]

		if (self.weights):
			for param, weight in zip(self.params, self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))

		# for i in range(len(self.W)):
		# 	self.L2_sqr = (self.W[i] ** 2).sum()
		self.L2_sqr = (self.W ** 2).sum()

	def output(self, seq_output=True):
		x = self.layer_below.output(seq_output=seq_output)
		pre_sup = T.tensordot(x, self.W, axes=[3, 0])
		support = T.tensordot(self.adjacency, pre_sup, axes=[0, 2])
		sp = support.shape
		support = support.reshape((sp[1], sp[2], sp[0], sp[3]))
		output = support

		if self.bias:
			output += self.b

		# for i in range(np.shape(self.adjacency)[0]):
		# 	for j in range(1, len(self.nonzeros[i])):
		# 		if(j==0):
		# 			out_d = T.tensordot(x[:, :, self.nonzeros[i][j], :],self.W[i][j, :, :], axes=[2, 0])
		# 		else:
		# 			out_d += T.tensordot(x[:, :, self.nonzeros[i][j], :],self.W[i][j, :, :], axes=[2, 0])
		# 		if self.bias:
		# 			out_d += self.b[i][j, :]
		# 	if(i == 0):
		# 		out = out_d.reshape((out_d.shape[0], out_d.shape[1], 1, out_d.shape[2]))
		# 	else:
		# 		out = T.concatenate((out, out_d.reshape((out_d.shape[0], out_d.shape[1], 1, out_d.shape[2]))), axis=2)

		return self.activation(out)
