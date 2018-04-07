from headers import *

class AddNoiseToInput(object):
	def __init__(self,weights=None,rng=None,skip_input=False,jump_up=False,dropout_noise=True):
		self.settings = locals()
		del self.settings['self']
		self.rng = rng
		self.weights = weights
		self.params = []
		self.std=T.scalar(dtype=theano.config.floatX)
		self.std.tag.test_value = .5
		self.skip_input = skip_input
		self.jump_up = jump_up
		self.dropout_noise = dropout_noise
		if rng is None:
			self.rng = np.random
		self.theano_rng = T.shared_randomstreams.RandomStreams(self.rng.randint(2 ** 30))

		self.numparams = 0
	def connect(self,layer_below):
		self.layer_below = layer_below
		self.size = self.layer_below.size
		self.inputD = self.size
	
	def output(self,seq_output=True):
		X = self.layer_below.output(seq_output=seq_output)
		if self.dropout_noise:
			binomial_probab = T.extra_ops.repeat(self.theano_rng.binomial(size=(X.shape[0],X.shape[1],X.shape[2],1),p=0.5,dtype=theano.config.floatX),X.shape[3],axis=3)

			a = T.le(self.std,theano.shared(value=0.0))
			b = (binomial_probab*X)#self.theano_rng.normal(size=X.shape,std=self.std,dtype=theano.config.floatX))
			out = T.switch(a,X,b)
			return out
		else:
			out = T.switch(T.le(self.std,theano.shared(value=0.0)),X,(X + self.theano_rng.normal(size=X.shape,std=self.std,dtype=theano.config.floatX)))
			return out
