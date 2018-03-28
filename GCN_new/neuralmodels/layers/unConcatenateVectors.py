from headers import *
import numpy as np


class unConcatenateVectors(object):

	def __init__(self, idxValues, weights=None, flag=1):
		self.settings = locals()
		del self.settings['self']
		self.input = T.tensor3(name="Data", dtype=theano.config.floatX)
		self.input.tag.test_value = np.random.rand(7, 150, 162)
		self.idxValues = idxValues
		self.params = []
		self.weights = weights
		self.L2_sqr = theano.shared(value=np.float32(0.0))

		self.base = 0
		self.temp = 0
		self.norm = 0
		self.flag = flag

	def output(self, idx, nm=None):

		if(self.flag):
			if(idx == nm + '_temporal'):
				self.temp = 1
			if(idx == nm + '_normal'):
				self.norm = 1
			low = self.idxValues[nm][idx][0] + self.base
			high = self.idxValues[nm][idx][1] + self.base

			if((self.norm) and (self.temp)):
				self.temp = 0
				self.norm = 0
				self.base = self.idxValues[nm][nm + '_temporal'][1] + self.base

			return self.input[:, :, low:high]
		else:
			low = self.idxValues[idx][0]
			high = self.idxValues[idx][1]
			return self.input[:, :, low:high]
