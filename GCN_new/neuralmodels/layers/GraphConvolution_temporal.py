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


class GraphConvolution_t(object):

	def __init__(self, size, adjacency, rng=None, init='glorot', bias=False, activation_str='rectify', weights=False, g_low=-10.0, g_high=10.0):

		self.settings = locals()
		del self.settings['self']
		self.size = size
		self.rng = rng
		self.init = getattr(inits, init)
		self.activation = getattr(activations, activation_str)
		self.weights = weights
		self.bias = bias
		self.adjacency = adjacency
		self.numparams = 0
		self.g_low = g_low
		self.g_high = g_high

	def connect(self, layer_below):
		self.layer_below = layer_below
		self.inputD = layer_below.size

		self.W = list()
		self.W_t = list()

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
			self.W_t.append(self.init((self.size, self.size), rng=self.rng))
			self.numparams += self.size*self.size
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
		
		self.h0 = zero0s((1,1,self.size))
	
	def recurrence_efficient(self,x,h_tm1):
	
		h_tm1 = theano.gradient.grad_clip(h_tm1, self.g_low, self.g_high)
		for i in range(np.shape(self.adjacency)[0]):
			t = x[:, i, :]
			c = h_tm1[:, i, :]
			#print(self.size.__repr__())
			#print(self.W_t[i].shape.__repr__())
			#print("0000000000000000000")
			b = T.tensordot(c, self.W_t[i], axes=[1, 0])
			a = t + b
			a = a.reshape((a.shape[0], 1, a.shape[1]))
			if(i==0):
				out = a
			else:
		 		out = T.concatenate((out, a), axis=1)
		 		
		return self.activation(out)

	def output(self, seq_output=True):
		x = self.layer_below.output(seq_output=seq_output)
		for i in range(np.shape(self.adjacency)[0]):
			for j in range(len(self.nonzeros[i])):
				if(j==0):
					out_d = T.tensordot(x[:, :, self.nonzeros[i][j], :],self.W[i][j, :, :], axes=[2, 0])
				else:
					out_d += T.tensordot(x[:, :, self.nonzeros[i][j], :],self.W[i][j, :, :], axes=[2, 0])
				if self.bias:
					out_d += self.b[i][j, :]
			if(i == 0):
				outpt = out_d.reshape((out_d.shape[0], out_d.shape[1], 1, out_d.shape[2]))
			else:
				outpt = T.concatenate((outpt, out_d.reshape((out_d.shape[0], out_d.shape[1], 1, out_d.shape[2]))), axis=2)

	
		# return self.activation(out)

		h_init = T.extra_ops.repeat(self.h0, x.shape[2], axis=1)
		h_init = T.extra_ops.repeat(h_init, x.shape[1], axis=0)

		out, _ = theano.scan(fn=self.recurrence_efficient,
				sequences=[outpt],
				outputs_info=[h_init],
				n_steps=x.shape[0],
			)
		return out
