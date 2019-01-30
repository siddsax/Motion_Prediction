import os
import argparse
import numpy as np
import theano
from theano import tensor as T
from neuralmodels.utils import permute 
from neuralmodels.loadcheckpoint import *
from neuralmodels.costs import temp_euc_loss, euclidean_loss, hinge_euclidean_loss
from neuralmodels.models import * 
from neuralmodels.layers import * 
from neuralmodels.updates import Adagrad, RMSprop, Momentum, Adadelta
import cPickle
import pdb
import socket as soc
import copy
import readCRFgraph as graph
import sys
sys.path.append('utils')
from utils import *
from defineGCN import *
sys.setrecursionlimit(50000)

theano.config.optimizers='fast_run'
def trainModel(params):
    
    drop_ids = params.drop_id.split(',')
    dropIdList = []
    for dids in drop_ids:
        dropIdList.append(int(dids))

    import processdata as poseDataset

    '''Loads H3.6m dataset'''
    sys.path.insert(0,'CRFProblems/H3.6m')
    poseDataset.T = params.sequence_length
    poseDataset.delta_shift = params.sequence_length - params.sequence_overlap
    poseDataset.num_forecast_examples = 24
    poseDataset.copy_state = params.copy_state
    poseDataset.full_skeleton = params.full_skeleton
    poseDataset.train_for = params.train_for
    poseDataset.temporal_features = params.temporal_features
    poseDataset.crf_file = './CRFProblems/H3.6m/crf' + params.crf
    poseDataset.dataset_prefix = params.dataset_prefix
    poseDataset.drop_features = params.drop_features
    poseDataset.drop_id = dropIdList
    poseDataset.subsample_data = params.subsample_data
    poseDataset.runall()
    if poseDataset.copy_state:
        params.batch_size = poseDataset.minibatch_size

    path_to_dump = '../dump/'
    [nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeListComplete,edgeFeatures,nodeToEdgeConnections,trX,trY,trX_validation,trY_validation,trX_forecasting,trY_forecasting,trX_forecast_nodeFeatures,adjacency] = graph.readCRFgraph(poseDataset)
    print '**** H3.6m Loaded ****'

    nodeNames = nodeNames.keys()
    
    new_idx = poseDataset.new_idx
    featureRange = poseDataset.nodeFeaturesRanges

    if params.use_pretrained == 1:
        print 'Loading pre-trained model with iter={0}'.format(params.iter_to_load)
        params.checkpoint_path = '/users/btech/siddsax/'
        model = loadModel(params.checkpoint_path+'checkpoint.'+str(params.iter_to_load))
        print 'DRA model loaded successfully'
    else:
        params.iter_to_load = 0
        model = defineGCN(params, nodeNames, nodeList, edgeList, edgeListComplete, edgeFeatures, nodeFeatureLength, nodeToEdgeConnections, new_idx, featureRange,adjacency)

    thefile = open('logger.txt', 'w')

    logModel(params, thefile, nodeNames, model, poseDataset)


    model.fitModel(trX, trY, snapshot_rate=params.snapshot_rate, path=params.checkpoint_path, pathD=path_to_dump, epochs=params.epochs, batch_size=params.batch_size,
        decay_after=params.decay_after, learning_rate=params.initial_lr, learning_rate_decay=params.learning_rate_decay, trX_validation=trX_validation,
        trY_validation=trY_validation, trX_forecasting=trX_forecasting, trY_forecasting=trY_forecasting,trX_forecast_nodeFeatures=trX_forecast_nodeFeatures, iter_start=params.iter_to_load,
        decay_type=params.decay_type, decay_schedule=params.decay_schedule, decay_rate_schedule=params.decay_rate_schedule,
        use_noise=params.use_noise, noise_schedule=params.noise_schedule, noise_rate_schedule=params.noise_rate_schedule,
               new_idx=new_idx, featureRange=featureRange, poseDataset=poseDataset, graph=graph, maxiter=params.maxiter, ssh_f=len(params.remoteBase), log=True, num_batches= params.curriculum)

