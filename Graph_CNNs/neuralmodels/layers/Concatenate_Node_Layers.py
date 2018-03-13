from headers import *

'''
This layer concatenates high-level representations from 
top-layer of multiple deep networks into a single features vector.
'''

class Concatenate_Node_Layers(object):
	def __init__(self,weights=None):
		self.settings = locals()
		del self.settings['self']
		self.params=[]
		self.weights=weights
		self.L2_sqr = theano.shared(value=np.float32(0.0))

	def connect(self,layers_below):
		self.size = layers_below[0].size
		# self.size = 0
		self.layers_below = layers_below
		# for layer in self.layers_below:

	def output(self,seq_output=True):
		
		is_tensor3 = False
		it = 0
		for layer in self.layers_below:
			x = layer.output(seq_output=seq_output)
			xr = x.reshape((x.shape[0], x.shape[1] , 1 , x.shape[2]))
			if(it==0):
				concatenate_output = xr
			else:
				concatenate_output = T.concatenate([concatenate_output,xr],axis=2)
			it = it + 1
		return concatenate_output 
