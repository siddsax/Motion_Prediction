from headers import *
import theano.d3viz as d3v
# from py_server import ssh
import sys
# import theano.sandbox.cuda
# theano.sandbox.cuda.use("gpu0")
class GCNN_new(object):
	def __init__(self,graphLayers,finalLayer,preGraphNets,nodeNames,temporalNodeRNNs,nodeRNNs,topLayer,cost,nodeLabels,learning_rate,new_idx,featureRange,clipnorm=0.0,update_type=RMSprop(),weight_decay=0.0):
		'''
		edgeRNNs and nodeRNNs are dictionary with keys as RNN name and value is a list of layers
		
		nodeToEdgeConnections is a dictionary with keys as nodeRNNs name and value is another dictionary whose keys are edgeRNNs the nodeRNN is connected to and value is a list of size-2 which indicates the features to choose from the unConcatenateLayer 

		nodeLabels is a dictionary with keys as node names and values as Theano matrix
		'''
		self.settings = locals()
		del self.settings['self']
		
		self.topLayer = topLayer
		self.nodeRNNs = nodeRNNs
		self.temporalNodeRNNs = temporalNodeRNNs
		self.nodeLabels = nodeLabels
		self.graphLayers = graphLayers
		self.finalLayer = finalLayer
		self.learning_rate = learning_rate
		self.clipnorm = clipnorm
		self.weight_decay = weight_decay
		

		self.cost = {}
		self.X = {}
		self.Y_pr = {}
		self.Y_pr_all = []
		self.Y_pr_last_timestep = {}
		self.Y = {}
		self.params = {}
		self.updates = {}
		self.masterlayer = {}
		self.grads = {}
		self.grad_norm = {}
		self.norm = {}
		self.get_cell = {}
		self.update_type = update_type
		self.update_type.lr = self.learning_rate
		self.update_type.clipnorm = self.clipnorm
		self.node_features = []
		self.std = T.scalar(dtype=theano.config.floatX)
		self.std.tag.test_value = .5
		self.preGraphNetsTypes = ['temporal', 'normal' ]
		self.num_params = 0
		self.params = {}
		self.train_node = {}
		self.predict_node = {}
		self.predict_node_last_timestep = {}
		self.predict_node_loss = {}
		self.num_params = 0
		
		nmG = ""
		for nm in nodeNames:
			layers = self.nodeRNNs[nm]
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])
				if layers[i].__class__.__name__ == 'AddNoiseToInput':
					layers[i].std = self.std
			print("======---------=======")
			layers = self.temporalNodeRNNs[nm]
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])
				if layers[i].__class__.__name__ == 'AddNoiseToInput':
					layers[i].std = self.std
			print("======---------=======")

			layers_below = []
			self.params[nm] = []
			self.masterlayer[nm] = unConcatenateVectors(preGraphNets[nm],flag=0)
			self.X[nm] = self.masterlayer[nm].input
			for pgnT in self.preGraphNetsTypes: # ADDED BY ME!!!!!!!!!!!!!!!!!!		
				# a = self.preGraphNetsTypes[nm]
				# self.masterlayer[nm] = unConcatenateVectors()
				if (pgnT =='temporal'):
					nodeLayers = self.temporalNodeRNNs[nm]
				elif(pgnT == 'normal'):
					nodeLayers = self.nodeRNNs[nm]
				else:
					print("Error in file GCNN.py, the pgnT is neither temporal or normal")	

				layers_below.append(nodeLayers)
				nodeLayers[0].input = self.masterlayer[nm].output(pgnT)
				for l in nodeLayers:
					if hasattr(l,'params'):
						self.params[nm].extend(l.params)
						self.num_params += l.numparams
			cv = ConcatenateVectors()
			cv.connect(layers_below)
			nodeTopLayer = self.topLayer[nm]
			nodeTopLayer[0].connect(cv)
			for i in range(1,len(nodeTopLayer)):
				nodeTopLayer[i].connect(nodeTopLayer[i-1])
			print("======---------=======")

			for l in nodeTopLayer:
				if hasattr(l,'params'):
					self.params[nm].extend(l.params)
					self.num_params += l.numparams

			flayer = self.finalLayer[nm]
			flayer[0].connect(nodeTopLayer[-1])
			for i in range(1,len(flayer)):
				flayer[i].connect(flayer[i-1])
			print("======---------=======")

			for l in flayer:
				if hasattr(l,'params'):
					self.params[nm].extend(l.params)
					self.num_params += l.numparams


			self.Y_pr[nm] = flayer[-1].output()
			self.Y[nm] = T.dtensor3(name="labels")

			self.cost[nm] = cost(self.Y_pr[nm],self.Y[nm]) + self.weight_decay * flayer[-1].L2_sqr

			[self.updates[nm],self.grads[nm]] = self.update_type.get_updates(self.params[nm],self.cost[nm])

			self.train_node[nm] = theano.function([self.X[nm],self.Y[nm],self.learning_rate,self.std],self.cost[nm],updates=self.updates[nm],on_unused_input='ignore')

			self.predict_node[nm] = theano.function([self.X[nm],self.std],self.Y_pr[nm],on_unused_input='ignore')

			self.predict_node_loss[nm] = theano.function([self.X[nm],self.Y[nm],self.std],self.cost[nm],on_unused_input='ignore')

			self.norm[nm] = T.sqrt(sum([T.sum(g**2) for g in self.grads[nm]]))

			self.grad_norm[nm] = theano.function([self.X[nm],self.Y[nm],self.std],self.norm[nm],on_unused_input='ignore')

			# self.get_cell[nm] = theano.function([self.X[nm],self.std],nodeLayers[0].layers[0].output(get_cell=True),on_unused_input='ignore')
			nmG = nm
			# theano.printing.pydotprint(out[nmG], ou	tfile="out" + nmG + ".png", var_with_name_simple=True)
			d3v.d3viz(self.Y_pr[nmG], "out" + nmG + ".html")
			# theano.printing.pydotprint(self.nodeRNNs[nmG][0].output(), outfile="after_temporal_nodeinput" + nmG + ".png", var_with_name_simple=True)
			d3v.d3viz(self.topLayer[nmG][-1].output(), "after_temporal_nodeinput" + nmG + ".html")
			# theano.printing.pydotprint(indv_node_layers[-1].output(), outfile="after_nodetoplayer" + nmG + ".png", var_with_name_simple=True)
			print("wabba")
			sys.exit()

		# d3v.d3viz(indv_node_layers[-1].output(), "after_nodetoplayer" + nmG + ".html")			
		print 'Number of parameters in DRA: ',self.num_params
		print("=============")
		
	# def fitModel(self,trX,trY,snapshot_rate=1,path=None,pathD=None,epochs=30,batch_size=50,learning_rate=1e-3,
	# 	learning_rate_decay=0.97,std=1e-5,decay_after=-1,trX_validation=None,trY_validation=None,
	# 	trX_forecasting=None,trY_forecasting=None,trX_forecast_nodeFeatures=None,rng=np.random.RandomState(1234567890),iter_start=None,
	# 	decay_type=None,decay_schedule=None,decay_rate_schedule=None,
	# 	use_noise=False,noise_schedule=None,noise_rate_schedule=None,
	# 	new_idx=None,featureRange=None,poseDataset=None,graph=None,maxiter=10000,unNormalizeData=None):
	def fitModel(self,trX,trY,snapshot_rate=10,path=None,pathD=None,epochs=30,batch_size=50,learning_rate=1e-3,
		learning_rate_decay=0.97,std=1e-5,decay_after=-1,trX_validation=None,trY_validation=None,
		trX_forecasting=None,trY_forecasting=None,trX_forecast_nodeFeatures=None,rng=np.random.RandomState(1234567890),iter_start=None,
		decay_type=None,decay_schedule=None,decay_rate_schedule=None,
		use_noise=False,noise_schedule=None,noise_rate_schedule=None,
		new_idx=None,featureRange=None,poseDataset=None,graph=None,maxiter=10000,unNormalizeData=None):
	
		from neuralmodels.loadcheckpoint import saveGCNN
		test_ground_truth = self.convertToSingleVec(trY_forecasting, new_idx, featureRange)
		test_ground_truth_unnorm = np.zeros((np.shape(test_ground_truth)[0],np.shape(test_ground_truth)[1],len(new_idx)))
		for i in range(np.shape(test_ground_truth)[1]):
			test_ground_truth_unnorm[:,i,:] = unNormalizeData(test_ground_truth[:,i,:],poseDataset.data_mean,poseDataset.data_std,poseDataset.dimensions_to_ignore)

		fname = 'test_ground_truth_unnorm'
		self.saveForecastedMotion(test_ground_truth_unnorm,path,fname)

		'''If loading an existing model then some of the parameters needs to be restored'''
		epoch_count = 0
		iterations = 0
		validation_set = []
		skel_loss_after_each_minibatch = []
		loss_after_each_minibatch = []
		complete_logger = ''
		if iter_start > 0:
			# if path:
			# 	lines = open('{0}logfile'.format(path)).readlines()
			# 	for i in range(iter_start):
			# 		line = lines[i]
			# 		values = line.strip().split(',')
			# 		print values
			# 		if len(values) == 1:
			# 			skel_loss_after_each_minibatch.append(float(values[0]))
			# 			validation_set.append(-1)
			# 		elif len(values) == 2:
			# 			skel_loss_after_each_minibatch.append(float(values[0]))
			# 			validation_set.append(float(values[1]))
				#if os.path.exists('{0}complete_log'.format(path)):
				#	complete_logger = open('{0}complete_log'.format(path)).read()
				#	complete_logger = complete_logger[:epoch_count]
			iterations = iter_start + 1

		tr_X = {}
		tr_Y = {}
		Nmax = 0
		outputDim = 0
		unequalSize = False
		numExamples = {}
		seq_length = 0
		skel_dim = 0
		
		nodeNames = trX.keys()
		for nm in nodeNames:
			tr_X[nm] = []
			tr_Y[nm] = []

		for nm in nodeNames:
			N = trX[nm].shape[1]
			seq_length = trX[nm].shape[0]
			skel_dim += trY[nm].shape[2]

			outputDim = trY[nm].ndim
			numExamples[nm] = N
			if Nmax == 0:
				Nmax = N
			if not Nmax == N:
				if N > Nmax:
					Nmax = N
				unequalSize = True
				
		if trY_forecasting is not None and new_idx is not None:
			trY_forecasting = self.convertToSingleVec(trY_forecasting,new_idx,featureRange)
			print 'trY_forecasting shape: {0}'.format(trY_forecasting.shape)
			assert(skel_dim == trY_forecasting.shape[2])

		'''Comverting validation set to a single array when doing drop joint experiments'''
		gth = None
		T1 = -1
		N1 = -1	
		if poseDataset.drop_features and unNormalizeData is not None:
			trY_validation = self.convertToSingleVec(trY_validation,new_idx,featureRange)
			[T1,N1,D1] = trY_validation.shape
			trY_validation_new = np.zeros((T1,N1,poseDataset.data_mean.shape[0]))
			for i in range(N1):
				trY_validation_new[:,i,:] = np.float32(unNormalizeData(trY_validation[:,i,:],poseDataset.data_mean,poseDataset.data_std,poseDataset.dimensions_to_ignore))
			gth = trY_validation_new[poseDataset.drop_start-1:poseDataset.drop_end-1,:,poseDataset.drop_id]

		if unequalSize:
			batch_size = Nmax
		
		batches_in_one_epoch = 1
		for nm in nodeNames:
			N = trX[nm].shape[1]
			batches_in_one_epoch = int(np.ceil(N*1.0 / batch_size))
			break

		print "batches in each epoch ",batches_in_one_epoch
		print nodeNames	
		#iterations = epoch_count * batches_in_one_epoch * 1.0
		numrange = np.arange(Nmax)
		# print(Nmax)
		#for epoch in range(epoch_count,epochs):
		epoch = 0
		while iterations <= maxiter:
			t0 = time.time()

			'''Learning rate decay.'''	
			if decay_type:
				if decay_type == 'continuous' and decay_after > 0 and epoch > decay_after:
					learning_rate *= learning_rate_decay
				elif decay_type == 'schedule' and decay_schedule is not None:
					for i in range(len(decay_schedule)):
						if decay_schedule[i] > 0 and iterations > decay_schedule[i]:
							learning_rate *= decay_rate_schedule[i]
							decay_schedule[i] = -1

			'''Set noise level.'''	
			if use_noise and noise_schedule is not None:
				for i in range(len(noise_schedule)):
					if noise_schedule[i] > 0 and iterations >= noise_schedule[i]:
						std = noise_rate_schedule[i]
						noise_schedule[i] = -1

			'''Loading noisy data'''
			noisy_data = graph.readCRFgraph(poseDataset,noise=std)
			trX = noisy_data[8]
			trY = noisy_data[4]
			trX_validation = noisy_data[9]
			trY_validation = noisy_data[5]



			'''Permuting before mini-batch iteration'''
			if not unequalSize:
				shuffle_list = rng.permutation(numrange)
				for nm in nodeNames:
					trX[nm] = trX[nm][:,shuffle_list,:]
					if outputDim == 2:
						trY[nm] = trY[nm][:,shuffle_list]
					elif outputDim == 3:
						trY[nm] = trY[nm][:,shuffle_list,:]

			for j in range(batches_in_one_epoch):

				examples_taken_from_node = 0	
				for nm in nodeNames:
					# nt = nm.split(':')[1]
					if(len(tr_X[nm])) == 0:
						examples_taken_from_node = min((j+1)*batch_size,numExamples[nm]) - j*batch_size
						tr_X[nm] = copy.deepcopy(trX[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm]),:])
						if outputDim == 2:
							tr_Y[nm] = copy.deepcopy(trY[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm])])
						elif outputDim == 3:
							tr_Y[nm] = copy.deepcopy(trY[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm]),:])
					else:
						tr_X[nm] = np.concatenate((tr_X[nm],trX[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm]),:]),axis=1)
						if outputDim == 2:
							tr_Y[nm] = np.concatenate((tr_Y[nm],trY[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm])]),axis=1)
						elif outputDim == 3:
							tr_Y[nm] = np.concatenate((tr_Y[nm],trY[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm]),:]),axis=1)

				loss = 0.0
				skel_loss = 0.0
				grad_norms = []
				for nm in nodeNames:
					loss_for_current_node = self.train_node[nm](tr_X[nm],tr_Y[nm],learning_rate,std)
					g = self.grad_norm[nm](tr_X[nm],tr_Y[nm],std)
					grad_norms.append(g)
					skel_loss_for_current_node = loss_for_current_node*tr_X[nm].shape[1]*1.0 / examples_taken_from_node
					loss += loss_for_current_node
					skel_loss += skel_loss_for_current_node
				iterations += 1
				loss_after_each_minibatch.append(loss)
				validation_set.append(-1)
				skel_loss_after_each_minibatch.append(skel_loss)
				termout = 'e={1} iter={8} m={2} lr={5} g_l2={4} noise={7} loss={0} normalized={3} skel_err={6}'.format(loss,epoch,j,(skel_loss*1.0/(seq_length*skel_dim)),grad_norms,learning_rate,np.sqrt(skel_loss*1.0/seq_length),std,iterations)
				complete_logger += termout + '\n'
				print termout
			
				del tr_X
				del tr_Y
							
				tr_X = {}
				tr_Y = {}
				for nm in nodeNames:
					tr_X[nm] = []
					tr_Y[nm] = []

				# if int(iterations) % snapshot_rate == 0:
				# 	print 'saving snapshot checkpoint.{0}'.format(int(iterations))
				# 	saveGCNN(self,"{0}checkpoint.{1}".format(path,int(iterations)),"{0}checkpoint.{1}".format(pathD,int(iterations)))
		
				# '''Trajectory forecasting on validation set'''
				# if (trX_forecasting is not None) and (trY_forecasting is not None) and path :
				# 	# print(trX_forecasting.keys())
				# 	forecasted_motion = self.predict_sequence(trX_forecasting,trX_forecast_nodeFeatures,sequence_length=trY_forecasting.shape[0],poseDataset=poseDataset,graph=graph)
				# 	# forecasted_motion = self.convertToSingleVec(forecasted_motion,new_idx,featureRange)
				# 	forecasted_motion = self.convertToSingleVec(forecasted_motion,new_idx,featureRange)

				# 	test_forecasted_motion_unnorm = np.zeros(np.shape(test_ground_truth_unnorm))
				# 	for i in range(np.shape(test_forecasted_motion_unnorm)[1]):
				# 		test_forecasted_motion_unnorm[:,i,:] = unNormalizeData(forecasted_motion[:,i,:],poseDataset.data_mean,poseDataset.data_std,poseDataset.dimensions_to_ignore)	
					# print("-------------------------")

					
					# if (int(iterations) % snapshot_rate == 0):
					# 	fname = 'forecast_iteration_unnorm'#_{0}'.format(int(iterations))
					# 	self.saveForecastedMotion(test_forecasted_motion_unnorm,path,fname)


			t1 = time.time()
			termout = 'Epoch took {0} seconds'.format(t1-t0)
			#complete_logger += termout + '\n'
			print termout
			epoch += 1

	def saveForecastError(self,skel_err,err_per_dof,path,fname):
		f = open('{0}{1}'.format(path,fname),'w')
		for i in range(skel_err.shape[0]):
			f.write('T={0} {1}, {2}\n'.format(i,skel_err[i],err_per_dof[i]))
		f.close()

	def saveForecastedMotion(self,forecast,path,fname):
		T = forecast.shape[0]
		N = forecast.shape[1]
		D = forecast.shape[2]
		for j in range(N):
			motion = forecast[:,j,:]
			file = '{0}{2}_N_{1}'.format(path,j,fname)
			string = ''
			for i in range(T):
				st = ''
				for k in range(D):
					st += str(motion[i,k]) + ','
				st = st[:-1]
				string += st+'\n'
			# print(string)
			# if(j==0):
			ssh( "echo " + "'" + string + "'" + " > " + file)
			# else:
				# ssh( "echo " + '"' + string + '"' + " >> " + file)

	
	def saveCellState(self,cellstate,path,fname):
		nodeNames = cellstate.keys()
		nm = nodeNames[0]
		print nodeNames
		T = cellstate[nm].shape[0]
		N = cellstate[nm].shape[1]
		D = cellstate[nm].shape[2]
		for j in range(N):
			f = open('{0}{2}_N_{1}'.format(path,j,fname),'w')
			for nm in nodeNames:
				motion = cellstate[nm][:,j,:]
				for i in range(T):
					st = ''
					for k in range(D):
						st += str(motion[i,k]) + ','
					st = st[:-1]
					f.write(st+'\n')
			f.close()


	def predict_output(self,teX):
		nodeNames = teX.keys()
		predict = {}
		for nm in nodeNames:
			nt = nm.split(':')[1]
			predict[nm] = self.predict_nodenm(teX[nm],1e-5)
		return predict


	def predict_sequence(self,teX_original,teX_original_nodeFeatures,sequence_length=100,poseDataset=None,graph=None):
		teX = copy.deepcopy(teX_original)
		nodeNames = teX.keys()

		teY = {}
		to_return = {}
		T = 0
		nodeFeatures_t_1 = {}
		for nm in nodeNames:
			[T,N,D] = teX[nm].shape
			to_return[nm] = np.zeros((T+sequence_length,N,D),dtype=theano.config.floatX)
			to_return[nm][:T,:,:] = teX[nm]
			teY[nm] = []
			nodeName = nm.split(':')[0]
			nodeFeatures_t_1[nodeName] = teX_original_nodeFeatures[nm][-1:,:,:]


		for i in range(sequence_length):
			nodeFeatures = {}
			for nm in nodeNames:
				nt = nm.split(':')[1]
				nodeName = nm.split(':')[0]
				prediction = self.predict_nodenm(to_return[nm][:(T+i),:,:],1e-5)
				#nodeFeatures[nodeName] = np.array([prediction])
				nodeFeatures[nodeName] = prediction[-1:,:,:]
				teY[nm].append(nodeFeatures[nodeName][0,:,:])
			for nm in nodeNames:
				nt = nm.split(':')[1]
				nodeName = nm.split(':')[0]
				nodeRNNFeatures = graph.getNodeFeature(nodeName,nodeFeatures,nodeFeatures_t_1,poseDataset)
				to_return[nm][T+i,:,:] = nodeRNNFeatures[0,:,:]
			nodeFeatures_t_1 = copy.deepcopy(nodeFeatures)
		for nm in nodeNames:
			teY[nm] = np.array(teY[nm])
		del teX
		return teY

	def predict_nextstep(self,teX):
		nodeNames = teX.keys()
		prediction = {}
		for nm in nodeNames:
			nt = nm.split(':')[1]
			prediction[nm] = self.predict_nodenm(teX[nm],1e-5)
		return prediction
		
	def predict_cell(self,teX_original,teX_original_nodeFeatures,sequence_length=100,poseDataset=None,graph=None):
		teX = copy.deepcopy(teX_original)
		nodeNames = teX.keys()

		teY = {}
		to_return = {}
		T = 0
		nodeFeatures_t_1 = {}
		for nm in nodeNames:
			[T,N,D] = teX[nm].shape
			to_return[nm] = np.zeros((T+sequence_length,N,D),dtype=theano.config.floatX)
			to_return[nm][:T,:,:] = teX[nm]
			teY[nm] = []
			nodeName = nm.split(':')[0]
			nodeFeatures_t_1[nodeName] = teX_original_nodeFeatures[nm][-1:,:,:]
		for i in range(sequence_length):
			nodeFeatures = {}
			for nm in nodeNames:
				nt = nm.split(':')[1]
				nodeName = nm.split(':')[0]
				prediction = self.predict_nodenm(to_return[nm][:(T+i),:,:],1e-5)
				#nodeFeatures[nodeName] = np.array([prediction])
				nodeFeatures[nodeName] = prediction[-1:,:,:]
				teY[nm].append(nodeFeatures[nodeName][0,:,:])
			for nm in nodeNames:
				nt = nm.split(':')[1]
				nodeName = nm.split(':')[0]
				nodeRNNFeatures = graph.getNodeFeature(nodeName,nodeFeatures,nodeFeatures_t_1,poseDataset)
				to_return[nm][T+i,:,:] = nodeRNNFeatures[0,:,:]
			nodeFeatures_t_1 = copy.deepcopy(nodeFeatures)
		cellstates = {}
		for nm in nodeNames:
			nt = nm.split(':')[1]
			nodeName = nm.split(':')[0]
			cellstates[nm] = self.get_cellnm(to_return[nm],1e-5)
		return cellstates

	def concatenateDimensions(self,dictToconcatenate,axis=2):
		conctArr = []
		for k in dictToconcatenate.keys():
			if len(conctArr) == 0:
				conctArr = copy.deepcopy(dictToconcatenate[k])	
			else:
				conctArr = np.concatenate((conctArr,dictToconcatenate[k]),axis=axis)
		return conctArr
	
	def convertToSingleVec(self,X,new_idx,featureRange):
		keys = X.keys()
		[T,N,D]  = X[keys[0]].shape
		D = len(new_idx) - len(np.where(new_idx < 0)[0])
		single_vec = np.zeros((T,N,D),dtype=np.float32)
		for k in keys:
			nm = k.split(':')[0]
			idx = new_idx[featureRange[nm]]
			insert_at = np.delete(idx,np.where(idx < 0))
			single_vec[:,:,insert_at] = X[k]
		return single_vec
