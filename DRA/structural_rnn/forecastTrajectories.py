import sys
try:
	sys.path.remove('/usr/local/lib/python2.7/dist-packages/Theano-0.6.0-py2.7.egg')
except:
	print 'Theano 0.6.0 version not found'

import numpy as np
import argparse
from neuralmodels.utils import readCSVasFloat
import theano
import os
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
sys.path.insert(0,'CRFProblems/H3.6m')
import processdata
import copy
import readCRFgraph as graph
import time
from unNormalizeData import unNormalizeData
from convertToSingleVec import convertToSingleVec 
from euler_error import euler_error
# from trainDRA import DRAmodelRegression
global rng
# from neuralmodels.models.DRA import convertToSingleLongVec
# from neuralmodels.models.DRA import convertToSingleVec
rng = np.random.RandomState(1234567890)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--checkpoint',type=str,default='savedModels/srnn_walking')
parser.add_argument('--forecast',type=str,default='dra')
parser.add_argument('--iteration',type=int,default=4500)
parser.add_argument('--motion_prefix',type=int,default=50)
parser.add_argument('--motion_suffix',type=int,default=100)
parser.add_argument('--temporal_features',type=int,default=0)
parser.add_argument('--full_skeleton',type=int,default=1)
parser.add_argument('--dataset_prefix',type=str,default='')
parser.add_argument('--train_for',type=str,default='final')
parser.add_argument('--drop_features',type=int,default=0)
parser.add_argument('--drop_id',type=int,default=9)
args = parser.parse_args()

'''Loads H3.6m dataset'''
print 'Loading H3.6m'
sys.path.insert(0,'CRFProblems/H3.6m')
import processdata as poseDataset
poseDataset.T = 150
poseDataset.delta_shift = 100
poseDataset.num_forecast_examples = 24
poseDataset.motion_prefix = args.motion_prefix
poseDataset.motion_suffix = args.motion_suffix
poseDataset.temporal_features = args.temporal_features
poseDataset.full_skeleton = args.full_skeleton
poseDataset.dataset_prefix = args.dataset_prefix
poseDataset.crf_file = './CRFProblems/H3.6m/crf'
poseDataset.train_for = args.train_for
poseDataset.drop_features = args.drop_features
poseDataset.drop_id = [args.drop_id]
poseDataset.runall()
print '**** H3.6m Loaded ****'

iteration = args.iteration
new_idx = poseDataset.new_idx
featureRange = poseDataset.nodeFeaturesRanges
base_dir = '.'
path = '{0}/{1}/'.format(base_dir,args.checkpoint)
if not os.path.exists(path):
	print 'Checkpoint path does not exist. Exiting!!'
	sys.exit()
	
crf_file = './CRFProblems/H3.6m/crf'


def convertToSingleVec(X,new_idx,featureRange):
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

def convertToSingleLongVec(X,poseDataset,new_idx,featureRange):
	keys = X.keys()
	[T,N,D]  = X[keys[0]].shape
	D = len(new_idx)
	# print(new_idx)
	single_vec = np.zeros((T,N,D),dtype=np.float32)
	k2 = 0
	# print(featureRange)
	for k in keys:
		nm = k.split(':')[0]
		# idx = new_idx[featureRange[nm]]
		k1 = np.size(featureRange[nm])
		p = 0
		for i in range(k1):
			if(new_idx[featureRange[nm][i]]==-1):
				a = poseDataset.data_stats['mean'][featureRange[nm][i]]
				single_vec[:,:,featureRange[nm][i]] = np.tile(a,(T,N))		
			else:
				single_vec[:,:,featureRange[nm][i]] = X[k][:,:,p]
				p+=1	

	return single_vec

def saveForecastedMotion(forecast,path,fname):
	T = forecast.shape[0]
	N = forecast.shape[1]
	D = forecast.shape[2]
	for j in range(N):
		motion = forecast[:,j,:]
		f = open('{0}{2}_N_{1}'.format(path,j,fname),'w')
		for i in range(T):
			st = ''
			for k in range(D):
				st += str(motion[i,k]) + ','
			st = st[:-1]
			f.write(st+'\n')
		f.close()


# path_to_checkpoint = '{0}checkpoint.pik'.format(path)#'{0}checkpoint.{1}'.format(path,iteration)
# if os.path.exists(path_to_checkpoint):
# 	[nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeListComplete,edgeFeatures,nodeToEdgeConnections,trX,trY,trX_validation,trY_validation,X_test,Y_test,beginning_motion] = graph.readCRFgraph(poseDataset,noise=0.7,forecast_on_noisy_features=True)

# 	print beginning_motion.keys()
# 	print 'Loading the model'
# 	model = loadDRA(path_to_checkpoint)
# 	print 'Loaded DRA: ',path_to_checkpoint

# ------------------------------------------------------------------------------------------------------------

for i in range(24):
	# gt = convertToSingleVec(Y_test,new_idx,featureRange)
	# gt_full = convertToSingleLongVec(Y_test,poseDataset,new_idx,featureRange)
	# gt_full[:,:,0:6] = 0
	# beginning_motion_dropped_ = convertToSingleVec(beginning_motion,new_idx,featureRange)
	# beginning_motion_full_ = convertToSingleLongVec(beginning_motion,poseDataset,new_idx,featureRange)
	# predicted_test = model.predict_sequence(X_test,beginning_motion,sequence_length=gt.shape[0],poseDataset=poseDataset,graph=graph)
	# predicted_test_dropped = convertToSingleVec(predicted_test,new_idx,featureRange)
	# predicted_test_full = convertToSingleLongVec(predicted_test,poseDataset,new_idx,featureRange)
	
	gt = readCSVasFloat('savedModels/srnn_walking/ground_truth__N_' + str(i))
	beginning_motion_dropped = readCSVasFloat('savedModels/srnn_walking/beginning_motion_N_' + str(i))
	predicted_test_dropped = readCSVasFloat('savedModels/srnn_walking/predicted_test_4500_N_' + str(i))
	


	gt_full = unNormalizeData(gt, processdata.data_stats['mean'], processdata.data_stats['std'], processdata.data_stats['ignore_dimensions'])
	beginning_motion_full_ = unNormalizeData(beginning_motion_dropped, processdata.data_stats['mean'], processdata.data_stats['std'], processdata.data_stats['ignore_dimensions'])
	predicted_test_full = unNormalizeData(predicted_test_dropped, processdata.data_stats['mean'], processdata.data_stats['std'], processdata.data_stats['ignore_dimensions'])
	
	np.savetxt('ground_truth_unnorm_' + str(i), gt_full)
	np.savetxt('beginning_motion_full_unnorm_' + str(i), beginning_motion_full_)
	np.savetxt('predicted_test_full_unnorm_' + str(i), predicted_test_full)

# 	fname = 'ground_truth_'
# 	saveForecastedMotion(gt,path,fname)
# 	fname = 'ground_truth_full'
# 	saveForecastedMotion(gt_full,path,fname)

# 	fname = 'predicted_test_{0}'.format(iteration)
# 	saveForecastedMotion(predicted_test_dropped,path,fname)
# 	fname = 'predicted_test_full_{0}'.format(iteration)
# 	saveForecastedMotion(predicted_test_full,path,fname)
	

# 	fname = 'beginning_motion'
# 	saveForecastedMotion(beginning_motion_dropped_,path,fname)
# 	fname = 'beginning_motion_full'
# 	saveForecastedMotion(beginning_motion_full_,path,fname)
	

	
# # ----------------------------------------------------------- ERRORS -------------------------------------
	
# 	val_error = euler_error(predicted_test_full, gt)
# 	seq_length_out = len(val_error)
# 	for ms in [1,3,7,9,13,24]:
# 		if seq_length_out >= ms+1:
# 			print(" {0:.3f} |".format( val_error[ms] ))
# 		else:
# 			print("   n/a |")
# 	skel_err = np.mean(np.sqrt(np.sum(np.square((predicted_test_dropped - gt)),axis=2)),axis=1)
# 	err_per_dof = skel_err / gt.shape[2]
