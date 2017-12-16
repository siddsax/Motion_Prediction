from headers import *

class GCNN(object):
	def __init__(self,preGraphNets,nodeNames,temporalNodeRNN,nodeRNNs,topLayer,edgeListComplete,cost,nodeLabels,learning_rate,clipnorm=0.0,update_type=RMSprop(),weight_decay=0.0):
		'''
		edgeRNNs and nodeRNNs are dictionary with keys as RNN name and value is a list of layers
		
		preGraphNets is a dictionary with keys as nodeNames and value is another dictionary whose keys are types of pregraphNetworls that is 'temporal' or 'normal' and value is a list of size-2 which indicates the features to choose from the unConcatenateLayer 

		nodeLabels is a dictionary with keys as node names and values as Theano matrix
		'''
		self.settings = locals()
		del self.settings['self']
		
		self.topLayer = topLayer
		self.nodeRNNs = nodeRNNs
		self.temporalNodeRNN = temporalNodeRNN
		self.nodeToEdgeConnections = nodeToEdgeConnections
		self.edgeListComplete = edgeListComplete
		self.nodeLabels = nodeLabels
		self.learning_rate = learning_rate
		self.clipnorm = clipnorm
		self.weight_decay = weight_decay
		
		nodeTypes = nodeRNNs.keys()
		edgeTypes = edgeRNNs.keys()

		self.cost = {}
		self.X = {}
		self.Y_pr = {}
		self.Y_pr_last_timestep = {}
		self.Y = {}
		self.params = {}
		self.updates = {}
		self.train_node = {}
		self.predict_node = {}
		self.predict_node_last_timestep = {}
		self.masterlayer = {}
		self.grads = {}
		self.predict_node_loss = {}
		self.grad_norm = {}
		self.norm = {}
		self.get_cell = {}

		self.update_type = update_type
		self.update_type.lr = self.learning_rate
		self.update_type.clipnorm = self.clipnorm

		self.std = T.scalar(dtype=theano.config.floatX)
		self.preGraphNetsTypes = ['temporal', 'normal' ]
		for nm in nodeNames:
			layers = self.nodeRNNs[nm]
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])
				if layers[i].__class__.__name__ == 'AddNoiseToInput':
					layers[i].std = self.std

			layers = self.temporalNodeRNN[nm]
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])
				if layers[i].__class__.__name__ == 'AddNoiseToInput':
					layers[i].std = self.std

			self.params[nm] = []
			self.masterlayer[nm] = unConcatenateVectors(preGraphNets[nm])
			self.X[nm] = self.masterlayer[nm].input
			
			layers_below = []
			for pgnT in self.preGraphNetsTypes: # ADDED BY ME!!!!!!!!!!!!!!!!!!		
				if (pgnT =='temporal'):
					nodeLayers = self.nodeRNNs[pgnT]
				elif(pgnT == 'normal'):
					nodeLayers = self.temporalNodeRNNs[pgnT]

				layers_below.append(nodeLayers)
				nodeLayers[0].input = self.masterlayer[nm].output(pgnT)
				for l in nodeLayers:
					if hasattr(l,'params'):
						self.params[nt].extend(l.params)

			cv = ConcatenateVectors()
			cv.connect(layers_below)
			nodeTopLayer = self.topLayer[nm]
			nodeTopLayer[0].connect(cv)
			for i in range(1,len(nodeTopLayer)):
				nodeTopLayer[i].connect(nodeTopLayer[i-1])

			for l in nodeTopLayer:
				if hasattr(l,'params'):
					self.params[nt].extend(l.params)

