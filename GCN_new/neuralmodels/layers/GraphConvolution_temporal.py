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

	def __init__(self, size, adjacency, rng=None, init='glorot', bias=False, activation_str='rectify', weights=False):

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
		
		self.h0 = zero0s((self.size))
	
	def recurrence_efficient(self,x,h_tm1):
	
		print("-------------------")
		print(np.shape(self.adjacency)[0])
		print("-------------------")
		
		# out = T.zeros_like(x)
		for i in range(np.shape(self.adjacency)[0]):
			a = x[:, i, :] + T.tensordot(h_tm1[:, i, :], self.W, axes=[1, 0])
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
				out = out_d.reshape((out_d.shape[0], out_d.shape[1], 1, out_d.shape[2]))
			else:
				out = T.concatenate((out, out_d.reshape((out_d.shape[0], out_d.shape[1], 1, out_d.shape[2]))), axis=2)

	
		# return self.activation(out)

		h_init = T.extra_ops.repeat(self.h0, x.shape[2], axis=0)
		h_init = T.extra_ops.repeat(h_init, x.shape[1], axis=0)
		#T.zeros((x.shape[1], x.shape[2], self.size))
		[out, cells], _ = theano.scan(fn=self.recurrence_efficient,
				sequences=[out],
				outputs_info=[h_init],
				n_steps=x.shape[0],
				# truncate_gradient=self.truncate_gradient
			)
		return out
		# if get_cell:
			# return cells

# class GraphConvolution_homo(object):

# 	def __init__(self, size, adjacency, num_features_nonzero=False, drop_value=None, rng=None, init='glorot', bias=False, sparse_inputs=False, dropout=True, activation_str='rectify', weights=False):

# 		self.settings = locals()
# 		del self.settings['self']
# 		self.sparse_inputs = sparse_inputs
# 		self.size = size
# 		self.rng = rng
# 		# temp = inits()
# 		self.init = getattr(inits, init)
# 		# temp = activations()
# 		self.activation = getattr(activations, activation_str)
# 		self.weights = weights
# 		self.bias = bias
# 		self.adjacency = adjacency
# 		if dropout:
# 			self.drop_value = drop_value
# 		else:
# 			self.drop_value = 0
# 		self.numparams = 0
# 		if self.sparse_inputs:
# 			self.num_features_nonzero = num_features_nonzero

# 	def connect(self, layer_below):
# 		self.layer_below = layer_below
# 		self.inputD = layer_below.size
# 		# self.W = list()
# 		# for i in range(len(self.adjacency)):
# 		# 	self.W.append(self.init((self.inputD,self.size),rng=self.rng))
# 		self.W = self.init((self.inputD, self.size), rng=self.rng)

# 		self.numparams += self.inputD * self.size
# 		if self.bias:
# 			self.b = zero0s((self.size))
# 			self.numparams += self.size
# 			self.params = [self.W, self.b]
# 		else:
# 			self.params = [self.W]

# 		if (self.weights):
# 			for param, weight in zip(self.params, self.weights):
# 				param.set_value(np.asarray(weight, dtype=theano.config.floatX))
# 		self.nonzeros = {}
#                 for i in range(np.shape(self.adjacency)[0]):
#                         count = 0
#                         self.nonzeros[i] = []
#                         for j in range(np.shape(self.adjacency)[1]):
#                                 if(self.adjacency[i, j]):
#                                         self.nonzeros[i].append(j)
#                                         count += 1
		
# 		# for i in range(len(self.W)):
# 		# 	self.L2_sqr = (self.W[i] ** 2).sum()
# 		self.L2_sqr = (self.W ** 2).sum()

# 	def recurrence_efficient(self,x_i,x_f,x_c,x_o,h_tm1_unclip,c_tm1_unclip):


# 		for i in range(np.shape(self.adjacency)[0]):
# 		 	for j in range(len(self.nonzeros[i])):
# 		 		if(j==0):
# 		 			out_d = T.tensordot(x[:, :, self.nonzeros[i][j], :],self.W, axes=[2, 0])
# 		 		else:
# 		 			out_d += T.tensordot(x[:, :, self.nonzeros[i][j], :],self.W, axes=[2, 0])
# 		 		if self.bias:
# 		 			out_d += self.b[i][j, :]
# 		 	if(i == 0):
# 		 		out = out_d.reshape((out_d.shape[0], out_d.shape[1], 1, out_d.shape[2]))
# 		 	else:
# 		 		out = T.concatenate((out, out_d.reshape((out_d.shape[0], out_d.shape[1], 1, out_d.shape[2]))), axis=2)
# 		# h_tm1 = theano.gradient.grad_clip(h_tm1_unclip,self.g_low,self.g_high)
# 		# c_tm1 = theano.gradient.grad_clip(c_tm1_unclip,self.g_low,self.g_high)

# 		# i_t = self.activation_gate(x_i + T.dot(h_tm1, self.U_i) + T.dot(c_tm1, T.nlinalg.diag(self.V_i)))
# 		# f_t = self.activation_gate(x_f + T.dot(h_tm1, self.U_f) + T.dot(c_tm1, T.nlinalg.diag(self.V_f)))
# 		# c_tilda_t = self.activation(x_c + T.dot(h_tm1, self.U_c))
# 		# c_t = f_t * c_tm1 + i_t * c_tilda_t
# 		# o_t = self.activation_gate(x_o + T.dot(h_tm1, self.U_o) + T.dot(c_t, T.nlinalg.diag(self.V_o)))
# 		# h_t = o_t * self.activation(c_t)

# 		return h_t,c_t


# 	def output(self, seq_output=True):
# 		x = self.layer_below.output(seq_output=seq_output)
# 		#pre_sup = T.tensordot(x, self.W, axes=[3, 0])
# 		#support = T.tensordot(self.adjacency, pre_sup, axes=[0, 2])
# 		#sp = support.shape
# 		#support = support.reshape((sp[1], sp[2], sp[0], sp[3]))
# 		#output = support

# 		#if self.bias:
# 		#	output += self.b

# 		X_i = T.dot(x, self.W_i) + self.b_i

#         # h_init = T.extra_ops.repeat(self.h0,X.shape[1],axis=0)
# 		# c_init =  T.extra_ops.repeat(self.c0,X.shape[1],axis=0)
# 		[out, cells], _ = theano.scan(fn=self.recurrence_efficient,
# 				sequences=[X_i],#,X_f,X_c,X_o],
# 				#outputs_info=[T.extra_ops.repeat(self.h0,X.shape[1],axis=0), T.extra_ops.repeat(self.c0,X.shape[1],axis=0)],
# 				outputs_info=[h_init,c_init],
# 				n_steps=X_i.shape[0],
# 				truncate_gradient=self.truncate_gradient
# 			)
        

# 		return self.activation(out)
