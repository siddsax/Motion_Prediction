from headers import *
import numpy as np

class unConcatenateVectors(object):
	
	# def __init__(self,idxValues,weights=None):
	# 	self.settings = locals()
	# 	del self.settings['self']
	# 	self.input=T.tensor3(dtype=theano.config.floatX)
	# 	self.idxValues = idxValues
	# 	self.params=[]
	# 	self.weights=weights
	# 	self.L2_sqr = theano.shared(value=np.float32(0.0))
	# def output(self,idx):
	# 	low = self.idxValues[idx][0]
	# 	high = self.idxValues[idx][1]
	# 	return self.input[:,:,low:high]

	def __init__(self,idxValues,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.input=T.dtensor3(name="Data")#, dtype=theano.config.floatX)
		self.input.tag.test_value = np.random.rand(5,150, 294)
		self.idxValues = idxValues
		self.params=[]
		self.weights=weights
		self.L2_sqr = theano.shared(value=np.float32(0.0))
		
		self.base = 0
		self.temp = 0
		self.norm = 0

	def output(self,idx,nm,inputM):

		if(idx=='temporal'):
			self.temp = 1
		if(idx=='normal'):
			self.norm = 1
		low = self.idxValues[nm][idx][0] + self.base
		high = self.idxValues[nm][idx][1] + self.base

		if((self.norm) and (self.temp)):
			self.temp = 0
			self.norm = 0
			self.base = self.idxValues[nm]['temporal'][1] + self.base

		return self.input[:,:,low:high]
