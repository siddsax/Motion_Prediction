import sys
try:
    sys.path.remove('/usr/local/lib/python2.7/dist-packages/Theano-0.6.0-py2.7.egg')
except:
    print 'Theano 0.6.0 version not found'
from py_server import ssh

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

# from trainDRA import DRAmodelRegression
global rng
# from neuralmodels.models.DRA import convertToSingleLongVec
# from neuralmodels.models.DRA import convertToSingleVec
rng = np.random.RandomState(1234567890)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--checkpoint',type=str,default='checkpoints/savedModels/srnn_walking')#savedModels/srnn_walking')
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


rng = np.random.RandomState(1234567890)

args = parser.parse_args()

# convert_list_to_float = ['decay_schedule','decay_rate_schedule','noise_schedule','noise_rate_schedule']
# for k in convert_list_to_float:
#     if getattr(args,k) is not None:
#         temp_list = []
#         for v in getattr(args,k):
#             temp_list.append(float(v))
#         setattr(args,k,temp_list)

# print args
# if args.use_pretrained:
#     print 'Loading pre-trained model with iter={0}'.format(args.iter_to_load)
# gradient_method = Momentum(momentum=args.momentum)

# drop_ids = args.drop_id.split(',')
# drop_id = []
# for dids in drop_ids:
#     drop_id.append(int(dids))


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

new_idx = poseDataset.new_idx
featureRange = poseDataset.nodeFeaturesRanges
base_dir = '.'
# path = '{0}/{1}/'.format(base_dir,args.checkpoint)
path = '..'
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
        file = '{0}{2}_N_{1}'.format(path,j,fname)
        print(file)
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

path_to_checkpoint = '{0}/checkpoint.pik'.format(path)#'{0}checkpoint.{1}'.format(path,iteration)
print(path_to_checkpoint)
if os.path.exists(path_to_checkpoint):
    [nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeListComplete,edgeFeatures,nodeToEdgeConnections,trX,trY,trX_validation,trY_validation,X_test,Y_test,beginning_motion] = graph.readCRFgraph(poseDataset,noise=0.7,forecast_on_noisy_features=True)

    print beginning_motion.keys()
else:
    print("ERROR: Checkpoint not found")

gt = convertToSingleVec(Y_test,new_idx,featureRange)
beginning_motion_dropped_ = convertToSingleVec(beginning_motion,new_idx,featureRange)

gt_full = np.zeros((np.shape(gt)[0],np.shape(gt)[1],len(new_idx)))
beginning_motion_full_ = np.zeros((np.shape(beginning_motion_dropped_)[0],np.shape(beginning_motion_dropped_)[1],len(new_idx)))

print(np.shape(gt)[1])


if os.path.exists(path_to_checkpoint):
    print 'Loading the model'
    print(path_to_checkpoint)
    model = loadDRA(path_to_checkpoint)
    print 'Loaded DRA: ',path_to_checkpoint

# ------------------------------------------------------------------------------------------------------------

predicted_test = model.predict_sequence(X_test,beginning_motion,sequence_length=gt.shape[0],poseDataset=poseDataset,graph=graph)
predicted_test_dropped = convertToSingleVec(predicted_test,new_idx,featureRange)
predicted_test_full = np.zeros((np.shape(predicted_test_dropped)[0],np.shape(predicted_test_dropped)[1],len(new_idx)))


for i in range(8):
    
    gt_full[:,i,:] = unNormalizeData(gt[:,i,:], processdata.data_stats['mean'], processdata.data_stats['std'], processdata.data_stats['ignore_dimensions'])
    beginning_motion_full_[:,i,:] = unNormalizeData(beginning_motion_dropped_[:,i,:], processdata.data_stats['mean'], processdata.data_stats['std'], processdata.data_stats['ignore_dimensions'])
    predicted_test_full[:,i,:] = unNormalizeData(predicted_test_dropped[:,i,:], processdata.data_stats['mean'], processdata.data_stats['std'], processdata.data_stats['ignore_dimensions'])

# # ----------------------------------------------------------- ERRORS -------------------------------------
path = '/new_data/gpu/siddsax/motion_pred_checkpoints/forecast/'
print(path)
fname = 'test_ground_truth_unnorm'
saveForecastedMotion(gt_full,path,fname)

fname = 'forecast_iteration_unnorm'#_{0}'.format(int(iterations))
saveForecastedMotion(predicted_test_full,path,fname)

fname = 'motion_prefix_unnorm'#_{0}'.format(int(iterations))
saveForecastedMotion(beginning_motion_full_,path,fname)

#val_error = euler_error(predicted_test_full, gt_full)
seq_length_out = len(val_error)
#for ms in [1,3,7,9,13,24]:
#    if seq_length_out >= ms+1:
#        print(" {0:.3f} |".format( val_error[ms] ))
#    else:
#        print("   n/a |")
#
error = 0
for nm in nodeNames:
    error+=model.predict_node_loss[nm](predicted_test[nm],Y_test[nm],.5)
print(error)
# skel_err = np.mean(np.sqrt(np.sum(np.square((predicted_test_dropped - gt)),axis=2)),axis=1)
# err_per_dof = skel_err / gt.shape[2]
