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
from numpy import genfromtxt
import cPickle
import pdb
import socket as soc
sys.path.insert(0,'CRFProblems/H3.6m')
import processdata
import copy
import readCRFgraph as graph
from py_server import ssh
import time
sys.path.append('utils')
from utils import *
from euler_error import euler_error
global rng
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

parser.add_argument('--crf', type= str, default = '')
parser.add_argument('--checkpoint_path', required=True, type=str)


if __name__ == '__main__':
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
    poseDataset.crf_file = './CRFProblems/H3.6m/crf' + args.crf
    poseDataset.train_for = args.train_for
    poseDataset.drop_features = args.drop_features
    poseDataset.drop_id = [args.drop_id]
    poseDataset.runall()
    print '**** H3.6m Loaded ****'

    getError(args, poseDataset)


def getError(args, poseDataset, model = None):
    new_idx = poseDataset.new_idx
    featureRange = poseDataset.nodeFeaturesRanges

    if model is None:
        model = loadModel(args.checkpoint_path)

    [nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeListComplete,edgeFeatures,nodeToEdgeConnections,trX,trY,trX_validation,trY_validation,X_test,Y_test,beginning_motion,adjacency] = graph.readCRFgraph(poseDataset,noise=0.7,forecast_on_noisy_features=True)


    #------------------------------------------------------------------------------------------------------------

    if os.path.exists(args.checkpoint_path):
        if(len(self.graphLayers)):
            predicted_test_dropped = model.predict_sequence(X_test,beginning_motion,featureRange,new_idx, sequence_length=gt.shape[0],poseDataset=poseDataset,graph=graph)
        else:
            predicted_test_dropped = model.predict_sequence_indep(X_test,beginning_motion, sequence_length=gt.shape[0],poseDataset=poseDataset,graph=graph)
            predicted_test_dropped = convertToSingleVec(predicted_test_dropped,new_idx,featureRange)
    # else:
    #     predicted_test = np.zeros((np.shape(gt)[0],np.shape(gt)[1],len(new_idx)))
    #     for i in range(8):
    #         predicted_test[:,i,:] = genfromtxt('../savedModels/checkpoints_dra_T_150_bs_100_tg_100_ls_512_fc_256_demoforecast_iteration_unnorm_N_' + str(i), delimiter=',')
    
    gt_dropped = convertToSingleVec(Y_test,new_idx,featureRange)
    beginning_motion_dropped = convertToSingleVec(beginning_motion,new_idx,featureRange)
    
    gt_full = np.zeros((np.shape(gt)[0],np.shape(gt)[1],len(new_idx)))
    beginning_motion_full = np.zeros((np.shape(beginning_motion_dropped)[0],np.shape(beginning_motion_dropped_)[1],len(new_idx)))
    predicted_test_full = np.zeros((np.shape(predicted_test_dropped)[0],np.shape(predicted_test_dropped)[1],len(new_idx)))

    for i in range(np.shape(predicted_test)[1]):
        predicted_test_full[:,i,:] = unNormalizeData(predicted_test_dropped[:,i,:],poseDataset.data_mean,poseDataset.data_std,poseDataset.dimensions_to_ignore)
        
        gt_full[:,i,:] = unNormalizeData(gt_dropped[:,i,:], poseDataset.data_mean,poseDataset.data_std,poseDataset.dimensions_to_ignore)
        
        beginning_motion_full[:, i, :] = unNormalizeData(beginning_motion_dropped[:,i,:], poseDataset.data_mean,poseDataset.data_std,poseDataset.dimensions_to_ignore)

    val_error = euler_error(predicted_test_full, gt_full)
    seq_length_out = len(val_error)
    for ms in [1,3,7,9,13,24]:
        if seq_length_out >= ms+1:
            print(" {0:.3f} |".format( val_error[ms] ))
        else:
            print("   n/a |")

    error = 0
    for nm in nodeNames:
        error+=model.predict_node_loss[nm](predicted_test[nm],Y_test[nm],.5)

    print("Model loss :{}".format(error))


