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
from neuralmodels.layers import * 
from neuralmodels.updates import Adagrad, RMSprop, Momentum, Adadelta
import cPickle
import pdb
import socket as soc
import copy
import readCRFgraph as graph
import sys
sys.setrecursionlimit(50000)


global rng

# import theano.sandbox.cuda
# theano.sandbox.cuda.use("gpu0")



#theano.config.scan.allow_gc =True
#theano.config.scan.allow_output_prealloc =False
#theano.optimizer_excluding=scanOp_pushout_seqs_ops
#theano.optimizer_excluding=scan_pushout_dot1
#theano.optimizer_excluding=scanOp_pushout_output
#theano.optimizer_excluding="more_mem"
#theano.config.optimizer.excluding = "scan"
#theano.config.optimizer='fast_run'
#theano.config.optimizer_including=local_remove_all_assert
#theano.config.optimizer='None'
# theano.config.exception_verbosity='high'
# theano.config.compute_test_value = 'warn'
# theano.config.print_test_value = True
#theano.config.floatX = 'float64'

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
parser.add_argument('--epochs',type=int,default=2000)
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
parser.add_argument('--model_to_train',type=str,default='dra')
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
parser.add_argument('--ssh',type=str,default=0)
parser.add_argument('--test',type=str,default=0)
parser.add_argument('--dump_path', type=str, default='checkpoint')
parser.add_argument('--drop_value', type=int, default=.2)
parser.add_argument('--homo', type=int, default=0)
args = parser.parse_args()

if(int(args.homo)):
	from neuralmodels.layers.GraphConvolution import GraphConvolution_homo as GraphConvolution
else:
	from neuralmodels.layers.GraphConvolution import GraphConvolution_hetro as GraphConvolution
from neuralmodels.layers.GraphConvolution_temporal import GraphConvolution_t
if(int(args.test)):
	theano.config.optimizer='None'
	theano.exception_verbosity='high'
	# theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature

	print "---------------------We are in testing phase-------------------------"

convert_list_to_float = ['decay_schedule','decay_rate_schedule','noise_schedule','noise_rate_schedule']
for k in convert_list_to_float:
	if getattr(args,k) is not None:
		temp_list = []
		for v in getattr(args,k):
			temp_list.append(float(v))
		setattr(args,k,temp_list)


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

def DRAmodelRegression(nodeNames, nodeList, edgeList, edgeListComplete, edgeFeatures, nodeFeatureLength, nodeToEdgeConnections, new_idx, featureRange, adjacency):
	edgeRNNs = {}
	nodeRNNs = {}
	finalLayer = {}
	nodeLabels = {}
	edgeListComplete = []
	graphLayers = []
	for nm in nodeNames:
		num_classes = nodeList[nm]
		if(int(args.test)==1):

			nodeRNNs[nm] = [FCLayer('linear', args.fc_init, size=100, rng=rng)]

			et = nm+'_temporal'
			edgeListComplete.append(et)
			edgeRNNs[et] = [TemporalInputFeatures(edgeFeatures[et]),
							FCLayer('rectify', args.fc_init,size=args.fc_size, rng=rng)
			]

			et = nm+'_normal'
			edgeListComplete.append(et)
			edgeRNNs[et] = [TemporalInputFeatures(edgeFeatures[et]),
							FCLayer('rectify', args.fc_init,size=args.fc_size, rng=rng)
			]

		else:
			
			LSTMs = [LSTM('tanh', 'sigmoid', args.lstm_init, truncate_gradient=args.truncate_gradient, size=args.node_lstm_size, rng=rng, g_low=-args.g_clip, g_high=args.g_clip)]
			nodeRNNs[nm] = [multilayerLSTM(LSTMs, skip_input=True,skip_output=True, input_output_fused=True),
							FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
							FCLayer('linear',args.fc_init,size=args.fc_size,rng=rng),
							]
			et = nm+'_temporal'
			edgeListComplete.append(et)
			
			edgeRNNs[et] = [TemporalInputFeatures(edgeFeatures[et]),
							FCLayer('rectify', args.fc_init,size=args.fc_size, rng=rng),
							FCLayer('linear', args.fc_init,size=args.fc_size, rng=rng)
							]

			et = nm+'_normal'
			edgeListComplete.append(et)
			edgeRNNs[et] = [TemporalInputFeatures(edgeFeatures[et]),
							FCLayer('rectify', args.fc_init,size=args.fc_size, rng=rng),
							FCLayer('linear', args.fc_init,size=args.fc_size, rng=rng)
							]

			nodeLabels[nm] = T.tensor3(dtype=theano.config.floatX)
		if(int(args.test)==1):
			graphLayers = [
				GraphConvolution(args.fc_size, adjacency),
				GraphConvolution_t(args.fc_size, adjacency),
				AddNoiseToInput(rng=rng, dropout_noise=True),
				AddNoiseToInput(rng=rng, dropout=True),
				]
		else:
			graphLayers = [
							GraphConvolution(args.fc_size,adjacency),
							#AddNoiseToInput(rng=rng, dropout_noise=True),
							GraphConvolution(args.fc_size, adjacency),
							AddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							AddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency, activation_str='linear'),
							AddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							#AddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							#ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							#ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							#AddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							#ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							#ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency, activation_str='linear'),
							#ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(len(nodeNames)*args.fc_size,adjacency),
							#ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency, activation_str='linear'),
						]
		for nm in nodeNames:
			num_classes = nodeList[nm]
			if(int(args.test)==1):
				finalLayer[nm] = [
								FCLayer_out('rectify', args.fc_init, size=args.fc_size, rng=rng, flag=len(graphLayers)),
								# FCLayer('rectify',args.fc_init,size=100,rng=rng),
								FCLayer('linear',args.fc_init,size=num_classes,rng=rng),
								]
			else:
				finalLayer[nm] = [
								FCLayer_out('rectify', args.fc_init, size=args.fc_size, rng=rng, flag=len(graphLayers)),
								FCLayer('rectify',args.fc_init,size=100,rng=rng),
								FCLayer('linear',args.fc_init,size=num_classes,rng=rng),
								]

	learning_rate = T.scalar(dtype=theano.config.floatX)
	dra = DRA(graphLayers, finalLayer, nodeNames, edgeRNNs, nodeRNNs, nodeToEdgeConnections, edgeListComplete, euclidean_loss, nodeLabels, learning_rate, new_idx, featureRange, clipnorm=args.clipnorm, update_type=gradient_method, weight_decay=args.weight_decay)
	
	return dra

def temporalGCNN(nodeNames, nodeList, edgeList, edgeListComplete, edgeFeatures, nodeFeatureLength, nodeToEdgeConnections, new_idx, featureRange, adjacency):
	edgeRNNs = {}
	nodeRNNs = {}
	finalLayer = {}
	nodeLabels = {}
	edgeListComplete = []
	graphLayers = []
	for nm in nodeNames:
		num_classes = nodeList[nm]
		if(int(args.test)==1):

			nodeRNNs[nm] = [FCLayer('linear', args.fc_init, size=100, rng=rng)]

			et = nm+'_temporal'
			edgeListComplete.append(et)
			edgeRNNs[et] = [
							TemporalInputFeatures(edgeFeatures[et]),
							FCLayer('rectify', args.fc_init,size=args.fc_size, rng=rng)
							]

			et = nm+'_normal'
			edgeListComplete.append(et)
			edgeRNNs[et] = [
							TemporalInputFeatures(edgeFeatures[et]),
							FCLayer('rectify', args.fc_init,size=args.fc_size, rng=rng)
							]

		else:
# ---------------------------------------------------------------------------------------------
			LSTMs = [LSTM('tanh', 'sigmoid', args.lstm_init, truncate_gradient=args.truncate_gradient, size=args.node_lstm_size, rng=rng, g_low=-args.g_clip, g_high=args.g_clip)]
			nodeRNNs[nm] = [
							# multilayerLSTM(LSTMs, skip_input=True,skip_output=True, input_output_fused=True),
							FCLayer('rectify', args.fc_init, size=args.fc_size, rng=rng),
							FCLayer('linear',args.fc_init,size=args.fc_size,rng=rng),
							]

			et = nm+'_temporal'
			edgeListComplete.append(et)
			
			edgeRNNs[et] = [
							TemporalInputFeatures(edgeFeatures[et]),
							FCLayer('rectify', args.fc_init,size=args.fc_size, rng=rng),
							FCLayer('linear', args.fc_init,size=args.fc_size, rng=rng)
							]

			et = nm + '_normal'
			edgeListComplete.append(et)
			edgeRNNs[et] = [
							TemporalInputFeatures(edgeFeatures[et]),
							FCLayer('rectify', args.fc_init,size=args.fc_size, rng=rng),
							FCLayer('linear', args.fc_init,size=args.fc_size, rng=rng)
							]

			nodeLabels[nm] = T.tensor3(dtype=theano.config.floatX)
		if(int(args.test)==1):
			graphLayers = [
							GraphConvolution(args.fc_size, adjacency),
							GraphConvolution_t(args.fc_size, adjacency),
							AddNoiseToInput(rng=rng, dropout_noise=True),
							AddNoiseToInput(rng=rng, dropout=True),
						  ]
		else:
			graphLayers = [
							GraphConvolution_t(args.fc_size,adjacency),
							# #AddNoiseToInput(rng=rng, dropout_noise=True),
							GraphConvolution(args.fc_size, adjacency),
							# #AddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							# #AddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							# #AddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							# #AddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							# #ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							# #ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							# #AddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							# #ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							# #ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency),
							# #ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(len(nodeNames)*args.fc_size,adjacency),
							# #ddNoiseToInput(rng=rng, dropout=True),
							GraphConvolution(args.fc_size, adjacency, activation_str='linear'),
						]
		for nm in nodeNames:
			num_classes = nodeList[nm]
			if(int(args.test)==1):
				finalLayer[nm] = [
								FCLayer_out('rectify', args.fc_init, size=args.fc_size, rng=rng, flag=len(graphLayers)),
								FCLayer('linear',args.fc_init,size=num_classes,rng=rng),
								]
			else:
				finalLayer[nm] = [
								FCLayer_out('rectify', args.fc_init, size=args.fc_size, rng=rng, flag=len(graphLayers)),
								FCLayer('rectify',args.fc_init,size=100,rng=rng),
								FCLayer('linear',args.fc_init,size=num_classes,rng=rng),
								]
# ---------------------------------------------------------------------------------------------

	learning_rate = T.scalar(dtype=theano.config.floatX)
	gcnn = DRA(graphLayers, finalLayer, nodeNames, edgeRNNs, nodeRNNs, nodeToEdgeConnections, edgeListComplete, euclidean_loss, nodeLabels, learning_rate, new_idx, featureRange, clipnorm=args.clipnorm, update_type=gradient_method, weight_decay=args.weight_decay)
	
	return gcnn




def trainDRA():
	
	path_to_checkpoint = '{0}/'.format(args.checkpoint_path)
	path_to_dump = '../dump/'
	print path_to_checkpoint
	if(int(args.ssh) == 1):
		script = "'if [ ! -d \"" + path_to_checkpoint + "\" ]; \n then mkdir " + path_to_checkpoint + "\nfi'"
		ssh( "echo " + script + " > file.sh")
		ssh("bash file.sh")
	else:
		if not os.path.exists(path_to_checkpoint):
			os.mkdir(path_to_checkpoint)


	[nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeListComplete,edgeFeatures,nodeToEdgeConnections,trX,trY,trX_validation,trY_validation,trX_forecasting,trY_forecasting,trX_forecast_nodeFeatures,adjacency] = graph.readCRFgraph(poseDataset)



	nodeNames = nodeNames.keys()
	
	new_idx = poseDataset.new_idx
	featureRange = poseDataset.nodeFeaturesRanges
	dra = []
	# all the dimensions which are not used ( the dimensions that have very little varianc ) are -1 and other have their respective index number like 0 1 2 3 -1 4 5 ......

	new_idx = poseDataset.new_idx
	featureRange = poseDataset.nodeFeaturesRanges

	if args.use_pretrained == 1:
		path_to_checkpoint = '/users/btech/siddsax/'
		dra = loadDRA(path_to_checkpoint+'checkpoint.'+str(args.iter_to_load))
		print 'DRA model loaded successfully'
	else:
		args.iter_to_load = 0
		#dra = DRAmodelRegression(nodeNames, nodeList, edgeList, edgeListComplete, edgeFeatures, nodeFeatureLength, nodeToEdgeConnections, new_idx, featureRange,adjacency)
		gcnn = temporalGCNN(nodeNames, nodeList, edgeList, edgeListComplete, edgeFeatures, nodeFeatureLength, nodeToEdgeConnections, new_idx, featureRange,adjacency)

	thefile = open('logger.txt', 'w')
	thefile.write("Edge RNNs TEMPORAL\n")
	for item in gcnn.edgeRNNs[nodeNames[0] + "_temporal"]:
		thefile.write("%s\n" % item)
	thefile.write("--------------------------- \n")

	thefile.write("Edge RNNs Normal\n")
	for item in gcnn.edgeRNNs[nodeNames[0] + "_normal"]:
		thefile.write("%s\n" % item)
	thefile.write("--------------------------- \n")

	thefile.write("Node RNNs \n")
	for item in gcnn.nodeRNNs[nodeNames[0]]:
		thefile.write("%s\n" % item)
	thefile.write("--------------------------- \n")

	thefile.write("graphLayers \n")
	for item in gcnn.graphLayers:
		thefile.write("%s\n" % item)
	thefile.write("--------------------------- \n")

	thefile.write("finalLayer \n")
	for item in gcnn.finalLayer[nodeNames[0]]:
		thefile.write("%s\n" % item)
	thefile.write("--------------------------- \n")

	thefile.write("actions \n")
	for item in poseDataset.actions:
		thefile.write("%s\n" % item)
	thefile.write("--------------------------- \n")
	thefile.close()
	print "saving log"
	if(int(args.ssh) == 1):
	 	from py_server import ssh
		file = open("logger.txt", "r")
		ssh("echo " + "'" + file.read() + "'" + " > " + path_to_checkpoint + "/logger.txt")
		file.close() 	
		#ssh('logger.txt', dst=path_to_checkpoint + "/logger.txt", copy=1)
	else:
		os.rename("logger.txt", path_to_checkpoint + "/logger.txt")

	gcnn.fitModel(trX, trY, snapshot_rate=args.snapshot_rate, path=path_to_checkpoint, pathD=path_to_dump, epochs=args.epochs, batch_size=args.batch_size,
		decay_after=args.decay_after, learning_rate=args.initial_lr, learning_rate_decay=args.learning_rate_decay, trX_validation=trX_validation,
		trY_validation=trY_validation, trX_forecasting=trX_forecasting, trY_forecasting=trY_forecasting,trX_forecast_nodeFeatures=trX_forecast_nodeFeatures, iter_start=args.iter_to_load,
		decay_type=args.decay_type, decay_schedule=args.decay_schedule, decay_rate_schedule=args.decay_rate_schedule,
		use_noise=args.use_noise, noise_schedule=args.noise_schedule, noise_rate_schedule=args.noise_rate_schedule,
			  new_idx=new_idx, featureRange=featureRange, poseDataset=poseDataset, graph=graph, maxiter=args.maxiter, ssh_f=args.ssh,log=True)

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


	trainDRA()
