import sys

try:
	sys.path.remove('/usr/local/lib/python2.7/dist-packages/Theano-0.6.0-py2.7.egg')
except:
	print 'Theano 0.6.0 version not found'

import os
os.environ['PATH'] += ':/usr/local/cuda/bin'
import argparse
import numpy as np
import theano
from theano import tensor as T
from neuralmodels.utils import permute 
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import softmax_loss, euclidean_loss
from neuralmodels.models import * 
from neuralmodels.predictions import OutputMaxProb, OutputSampleFromDiscrete
from neuralmodels.layers import * 
from neuralmodels.updates import Adagrad,RMSprop,Momentum,Adadelta
import cPickle
import pdb
import socket as soc
import copy
import readCRFgraph as graph
from unNormalizeData import unNormalizeData
global rng


theano.config.optimizer='None'
#theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high' 
# theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

rng = np.random.RandomState(1234567890)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--decay_type',type=str,default='schedule')
parser.add_argument('--decay_after',type=int,default=-1)
parser.add_argument('--initial_lr',type=float,default=1e-3)
parser.add_argument('--learning_rate_decay',type=float,default=1.0)
parser.add_argument('--decay_schedule',nargs='*')
parser.add_argument('--decay_rate_schedule',nargs='*')
parser.add_argument('--node_lstm_size',type=int,default=10)
parser.add_argument('--lstm_size',type=int,default=10)
parser.add_argument('--fc_size',type=int,default=500)
parser.add_argument('--lstm_init',type=str,default='uniform')
parser.add_argument('--fc_init',type=str,default='uniform')
parser.add_argument('--snapshot_rate',type=int,default=10)
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--batch_size',type=int,default=3000)
parser.add_argument('--clipnorm',type=float,default=25.0)
parser.add_argument('--use_noise',type=int,default=1)
parser.add_argument('--noise_schedule',nargs='*')
parser.add_argument('--noise_rate_schedule',nargs='*')
parser.add_argument('--momentum',type=float,default=0.99)
parser.add_argument('--g_clip',type=float,default=25.0)
parser.add_argument('--truncate_gradient',type=int,default=50)
parser.add_argument('--use_pretrained',type=int,default=0)
parser.add_argument('--iter_to_load',type=int,default=None)
parser.add_argument('--model_to_train',type=str,default='gcnn')
parser.add_argument('--checkpoint_path',type=str,default='checkpoint')
parser.add_argument('--sequence_length',type=int,default=150)
parser.add_argument('--sequence_overlap',type=int,default=50)
parser.add_argument('--maxiter',type=int,default=15000)
parser.add_argument('--crf',type=str,default='')
parser.add_argument('--copy_state',type=int,default=0)
parser.add_argument('--full_skeleton',type=int,default=0)
parser.add_argument('--weight_decay',type=float,default=0.0)
parser.add_argument('--train_for',type=str,default='validate')
parser.add_argument('--dra_type',type=str,default='simple')
parser.add_argument('--temporal_features',type=int,default=0)
parser.add_argument('--dataset_prefix',type=str,default='')
parser.add_argument('--drop_features',type=int,default=0)
parser.add_argument('--subsample_data',type=int,default=1)
parser.add_argument('--drop_id',type=str,default='')
parser.add_argument('--hidden1',type=int,default=100)
parser.add_argument('--drop_value',type=int,default=.5)
args = parser.parse_args()

convert_list_to_float = ['decay_schedule','decay_rate_schedule','noise_schedule','noise_rate_schedule']
for k in convert_list_to_float:
	if getattr(args,k) is not None:
		temp_list = []
		for v in getattr(args,k):
			temp_list.append(float(v))
		setattr(args,k,temp_list)

print args
if args.use_pretrained:
	print 'Loading pre-trained model with iter={0}'.format(args.iter_to_load)
gradient_method = Momentum(momentum=args.momentum)

drop_ids = args.drop_id.split(',')
drop_id = []
for dids in drop_ids:
	drop_id.append(int(dids))


'''Loads H3.6m dataset'''
sys.path.insert(0,'CRFProblems/H3.6m')
import processdata as poseDataset
poseDataset.T = args.sequence_length
poseDataset.delta_shift = args.sequence_length - args.sequence_overlap
poseDataset.num_forecast_examples = 24
poseDataset.copy_state = args.copy_state
poseDataset.full_skeleton = args.full_skeleton
poseDataset.train_for = args.train_for
poseDataset.temporal_features = args.temporal_features
poseDataset.crf_file = './CRFProblems/H3.6m/crf' + args.crf
poseDataset.dataset_prefix = args.dataset_prefix
poseDataset.drop_features = args.drop_features
poseDataset.drop_id = drop_id
poseDataset.subsample_data = args.subsample_data
poseDataset.runall()
if poseDataset.copy_state:
	args.batch_size = poseDataset.minibatch_size

print '**** H3.6m Loaded ****'

def saveForecastedMotion(forecast,path,prefix='ground_truth_forecast_N_'):
	T = forecast.shape[0]
	N = forecast.shape[1]
	D = forecast.shape[2]
	for j in range(N):
		motion = forecast[:,j,:]
		f = open('{0}{2}{1}'.format(path,j,prefix),'w')
		for i in range(T):
			st = ''
			for k in range(D):
				st += str(motion[i,k]) + ','
			st = st[:-1]
			f.write(st+'\n')
		f.close()





def GCNNmodelRegression(preGraphNets,nodeList,nodeFeatureLength, temporalNodeFeatureLength):

	temporalNodeRNN = {} # --- Node_Feature:(Node_Feature - Node_Feature_-1)
	nodeRNNs = {} 
	topLayer = {}
	nodeLabels = {}
	nodeNames = nodeList.keys()
	for nm in nodeNames:
		num_classes = nodeList[nm]
		LSTMs = [LSTM('tanh','sigmoid',args.lstm_init,truncate_gradient=args.truncate_gradient,size=args.node_lstm_size,rng=rng,g_low=-args.g_clip,g_high=args.g_clip)
			] # ------------------- NEED TO CHECK THE ARGUMENTS ARE IN LINE

		# ---------------------Original---------------------------------------------------------	 
		# nodeRNNs[nm] = [TemporalInputFeatures(nodeFeatureLength[nm])
		# 		multilayerLSTM(LSTMs,skip_input=True,skip_output=True,input_output_fused=True),
		# 		FCLayer('rectify',args.fc_init,size=args.fc_size,rng=rng),
		# 		FCLayer('rectify',args.fc_init,size=100,rng=rng),
		# 		FCLayer('linear',args.fc_init,size=num_classes,rng=rng)
		# 		]
		# -------------------------------------------------------------------------------------
		

		nodeRNNs[nm] = [TemporalInputFeatures(nodeFeatureLength[nm]),
				# FCLayer('rectify',args.fc_init,size=100,rng=rng),
				# FCLayer('linear',args.fc_init,size=50,rng=rng),
				# multilayerLSTM(LSTMs,skip_input=True,skip_output=True,input_output_fused=True),
				]

		temporalNodeRNN[nm] = [TemporalInputFeatures(temporalNodeFeatureLength[nm]),
				#AddNoiseToInput(rng=rng),
				# FCLayer('rectify',args.fc_init,size=args.fc_size,rng=rng),
				# FCLayer('linear',args.fc_init,size=args.fc_size,rng=rng)
				]
		topLayer[nm] = [
				# multilayerLSTM(LSTMs,skip_input=True,skip_output=True,input_output_fused=True),
				# FCLayer('rectify',args.fc_init,size=args.fc_size,rng=rng),
				# FCLayer('rectify',args.fc_init,size=100,rng=rng),
				FCLayer('linear',args.fc_init,size=100,rng=rng)
				]

		# nodeRNNs[nm] = [TemporalInputFeatures(nodeFeatureLength[nm]),
		# 		FCLayer('rectify',args.fc_init,size=100,rng=rng),
		# 		FCLayer('linear',args.fc_init,size=50,rng=rng),
		# 		multilayerLSTM(LSTMs,skip_input=True,skip_output=True,input_output_fused=True),
		# 		]

		# temporalNodeRNN[nm] = [TemporalInputFeatures(temporalNodeFeatureLength[nm]),
		# 		#AddNoiseToInput(rng=rng),
		# 		FCLayer('rectify',args.fc_init,size=args.fc_size,rng=rng),
		# 		FCLayer('linear',args.fc_init,size=args.fc_size,rng=rng)
		# 		]
		# topLayer[nm] = [
		# 		# multilayerLSTM(LSTMs,skip_input=True,skip_output=True,input_output_fused=True),
		# 		FCLayer('rectify',args.fc_init,size=args.fc_size,rng=rng),
		# 		FCLayer('rectify',args.fc_init,size=100,rng=rng),
		# 		FCLayer('linear',args.fc_init,size=num_classes,rng=rng)
		# 		]

		nodeLabels[nm] = T.tensor3(dtype=theano.config.floatX)

	learning_rate = T.scalar(dtype=theano.config.floatX)
	adjacency = T.matrix('adjacency_matrix', dtype=theano.config.floatX)

	# ----------------------- Add graph CNN related variables -------------------------------
	graphLayers = [GraphConvolution_input(args.hidden1,adjacency,drop_value=args.drop_value),
        GraphConvolution(num_classes,adjacency,activation_str='linear',drop_value=args.drop_value)
	] 
	#----------------------------------------------------------------------------------------
	gcnn = GCNN(graphLayers,preGraphNets,nodeNames,temporalNodeRNN,nodeRNNs,topLayer,euclidean_loss,nodeLabels,learning_rate,clipnorm=args.clipnorm,update_type=gradient_method,weight_decay=args.weight_decay)
	return gcnn

def trainGCNN():
	crf_file = './CRFProblems/H3.6m/crf' + args.crf

	path_to_checkpoint = './{0}/'.format(args.checkpoint_path)
	print path_to_checkpoint
	if not os.path.exists(path_to_checkpoint):
		os.mkdir(path_to_checkpoint)
	saveNormalizationStats(path_to_checkpoint)
	
	# --------- Read the graph from here ------------------
	# Things that we need are nodeNames, the feature length of temoral nodes*, 
	[nodeList, # {node name} = output dimension
	temporalNodeFeatureLength, # dictonary {node name} = temp node feat. length
	nodeFeatureLength, # {node name} = node feat. length
	nodeConnections,# {node name} = node names it is connected to
	trY,trY_validation,trY_forecasting, # the output values of the model i.e. the coordinates of the different joints as a dictionary {node name} = value of the coordinate 
	trX_forecast_nodeFeatures, # {node name} -> ? these are position of joints using which the input fetures are made. trX_forecasting[nodeName] is nothing but just a concatenation of the node_features(which are just this in my model) and temporalfeatures (which is this:[this-this_{-1})

	trX,trX_validation,trX_forecasting, # {node name}  = concatenation of [node feature] + [temporal node feature] this is identified using the preGraphnets variable
	preGraphNets, # {node name} = {temporal/normal} = {high,low}  
	adjacency] = graph.readCRFgraph(poseDataset)

	print("###########################################################################")
	print("nodeList")
	print(nodeList)
	print("###########################################################################")
	print("temporalNodeFeatureLength")
	print(temporalNodeFeatureLength)
	print("###########################################################################")
        print("nodeFeatureLength")
        print(nodeFeatureLength)
	print("###########################################################################")
        print("nodeConnections")
        print(nodeConnections)
	print("###########################################################################")
        print("trX_forecast_nodeFeatures right_arm")
        #print(trX_forecast_nodeFeatures)
	print(np.shape(trX_forecast_nodeFeatures['right_arm']))
	print("###########################################################################")
	print("trY right_arm")
	print(np.shape(trY['right_arm']))
	print("###########################################################################")
	print("trX right_arm")
	print(np.shape(trX['right_arm']))

	new_idx = poseDataset.new_idx # all the dimensions which are not used ( the dimensions that have very little varianc ) are -1 and other have their respective index number like 0 1 2 3 -1 4 5 ......
	featureRange = poseDataset.nodeFeaturesRanges # range of torso, right hand, left hand ...
	gcnn = []
	# ------------- Here we load the model using the function ------------------------------
	if args.use_pretrained == 1:
		gcnn = loadDRA(path_to_checkpoint+'checkpoint.'+str(args.iter_to_load))
		print 'DRA model loaded successfully'
	else:
		args.iter_to_load = 0
		gcnn = GCNNmodelRegression(preGraphNets,nodeList,nodeFeatureLength,temporalNodeFeatureLength)
	# ----------------------- To be used finally commented as work in this part is not done --------------------
	saveForecastedMotion(gcnn.convertToSingleLongVec(trY_forecasting,poseDataset, new_idx, featureRange),path_to_checkpoint)
	saveForecastedMotion(gcnn.convertToSingleLongVec(trX_forecast_nodeFeatures,poseDataset,new_idx,featureRange),path_to_checkpoint,'motionprefix_N_')
	#-------------------------------------------------------------------------------------------------------------

	gcnn.fitModel(trX, trY, adjacency,snapshot_rate=args.snapshot_rate, path=path_to_checkpoint, epochs=args.epochs, batch_size=args.batch_size,
		decay_after=args.decay_after, learning_rate=args.initial_lr, learning_rate_decay=args.learning_rate_decay, trX_validation=trX_validation,
		trY_validation=trY_validation, trX_forecasting=trX_forecasting, trY_forecasting=trY_forecasting,trX_forecast_nodeFeatures=trX_forecast_nodeFeatures, iter_start=args.iter_to_load,
		decay_type=args.decay_type, decay_schedule=args.decay_schedule, decay_rate_schedule=args.decay_rate_schedule,
		use_noise=args.use_noise, noise_schedule=args.noise_schedule, noise_rate_schedule=args.noise_rate_schedule,
		new_idx=new_idx,featureRange=featureRange,poseDataset=poseDataset,graph=graph,maxiter=args.maxiter,unNormalizeData=unNormalizeData)



	
def saveNormalizationStats(path):
	activities = {}
	activities['walking'] = 14
	activities['eating'] = 4
	activities['smoking'] = 11
	activities['discussion'] = 3
	activities['walkingdog'] = 15

	cPickle.dump(poseDataset.data_stats,open('{0}h36mstats.pik'.format(path),'wb'))
	forecastidx = poseDataset.data_stats['forecastidx']
	num_forecast_examples = len(forecastidx.keys())
	f = open('{0}forecastidx'.format(path),'w')
	for i in range(num_forecast_examples):
		tupl = forecastidx[i]
		st = '{0},{1},{2},{3}\n'.format(i,activities[tupl[0]],tupl[2],tupl[1])
		f.write(st)
	f.close()
	print "************Done saving the stats*********"


if __name__ == '__main__':


	if args.model_to_train == 'malik':
		trainMaliks()
	elif args.model_to_train == 'dra':
		trainDRA()
	elif args.model_to_train == 'lstm':
		trainLSTM()
	elif args.model_to_train == 'gcnn':
		trainGCNN()	
	else:
		print "Unknown model type ... existing"
