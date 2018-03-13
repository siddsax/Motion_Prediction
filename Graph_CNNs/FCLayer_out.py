from headers import *

class FCLayer_out(object):
	def __init__(self,activation_str='tanh',init='orthogonal',size=128,weights=None,rng=None,flag=1):
		self.settings = locals()
		del self.settings['self']
		self.activation = getattr(activations,activation_str)
		self.size = size
		self.init = getattr(inits,init)
		self.weights = weights
		self.rng = rng
		self.numparams = 0
		self.flag = flag

	def connect(self,layer_below,indx=None):
		self.layer_below = layer_below
		if(self.flag):
			self.indx = indx
		self.inputD = layer_below.size
		self.W = self.init((self.inputD,self.size),rng=self.rng)
		self.b = zero0s((self.size))
		self.params = [self.W, self.b]
		self.numparams += self.inputD*self.size + self.size 
		print("----FCLayer_out--------------")
		print(self.numparams)
		print("{0}x{1} + {2}".format(self.inputD,self.size,self.size))
		print("----FCLayer_out--------------")
		
		if self.weights is not None:
			for param, weight in zip(self.params,self.weights):
				param.set_value(np.asarray(weight, dtype=theano.config.floatX))
		
		self.L2_sqr = (self.W ** 2).sum() 

	def output(self,seq_output=True):
		if(self.flag):
			X = self.layer_below.output(seq_output=seq_output)[:,:,self.indx,:]
		else:
			X = self.layer_below.output(seq_output=seq_output)			
		return self.activation(T.dot(X, self.W) + self.b)
