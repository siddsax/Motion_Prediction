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

global rng
rng = np.random.RandomState(1234567890)


def defineGCN(params, nodeNames, nodeList, edgeList, edgeListComplete, edgeFeatures, nodeFeatureLength, nodeToEdgeConnections, new_idx, featureRange, adjacency):

    gradient_method = Momentum(momentum=params.momentum)

    if(params.gcnType == 0):
        print("-------")
        from neuralmodels.layers.GraphConvolution import GraphConvolution
    elif(params.gcnType == 1):
        print("=======")
        from neuralmodels.layers.GraphConvolution_temporal import GraphConvolution_t as GraphConvolution
    elif(params.gcnType == 2):
        print("########")
        from neuralmodels.layers.GraphConvolution_temporal_pairwise import GraphConvolution_tp as GraphConvolution

    edgeRNNs = {}
    nodeRNNs = {}
    finalLayer = {}
    nodeLabels = {}
    edgeListComplete = []

    for nm in nodeNames:
        num_classes = nodeList[nm]
        if(params.test==1):

            nodeRNNs[nm] = [FCLayer('linear', params.fc_init, size=1, rng=rng)]

            et = nm+'_temporal'
            edgeListComplete.append(et)
            edgeRNNs[et] = [
                            TemporalInputFeatures(edgeFeatures[et]),
                            FCLayer('rectify', params.fc_init,size=params.fc_size, rng=rng)
                            ]

            et = nm+'_normal'
            edgeListComplete.append(et)
            edgeRNNs[et] = [
                            TemporalInputFeatures(edgeFeatures[et]),
                            FCLayer('rectify', params.fc_init,size=params.fc_size, rng=rng)
                            ]

            finalLayer[nm] = [
                            FCLayer_out('rectify', params.fc_init, size=params.fc_size, rng=rng, flag=1),
                            FCLayer('linear',params.fc_init,size=num_classes,rng=rng),
                            ]

        else:
            LSTMs = [LSTM('tanh', 'sigmoid', params.lstm_init, truncate_gradient=params.truncate_gradient, size=params.node_lstm_size, rng=rng, g_low=-params.g_clip, g_high=params.g_clip)]
            nodeRNNs[nm] = [
                            #multilayerLSTM(LSTMs, skip_input=True,skip_output=True, input_output_fused=True),
                            FCLayer('rectify', params.fc_init, size=params.fc_size, rng=rng),
                            FCLayer('linear',params.fc_init,size=params.fc_size,rng=rng),
                            ]

            et = nm+'_temporal'
            edgeListComplete.append(et)
            
            edgeRNNs[et] = [
                            TemporalInputFeatures(edgeFeatures[et]),
                            FCLayer('rectify', params.fc_init,size=params.fc_size, rng=rng),
                            FCLayer('linear', params.fc_init,size=params.fc_size, rng=rng)
                            ]

            et = nm + '_normal'
            edgeListComplete.append(et)
            edgeRNNs[et] = [
                            TemporalInputFeatures(edgeFeatures[et]),
                            FCLayer('rectify', params.fc_init,size=params.fc_size, rng=rng),
                            FCLayer('linear', params.fc_init,size=params.fc_size, rng=rng)
                            ]

            nodeLabels[nm] = T.tensor3(dtype=theano.config.floatX)

            finalLayer[nm] = [
                            FCLayer_out('rectify', params.fc_init, size=params.fc_size, rng=rng, flag=1),
                            FCLayer('rectify',params.fc_init,size=100,rng=rng),
                            FCLayer('linear',params.fc_init,size=num_classes,rng=rng),
                            ]

    if(params.test==1):
        graphLayers = [
                        GraphConvolution(params.fc_size, adjacency),
                        AddNoiseToInput(rng=rng, dropout_noise=True),
                        AddNoiseToInput(rng=rng, dropout=True),
                        ]
    else:        
        graphLayers = [
                        GraphConvolution(params.fc_size,adjacency),
                        GraphConvolution(params.fc_size,adjacency),
                        # AddNoiseToInput(rng=rng, dropout_noise=True),
                        GraphConvolution(params.fc_size, adjacency),
                        # AddNoiseToInput(rng=rng, dropout=True),
                        GraphConvolution(params.fc_size, adjacency),
                        # AddNoiseToInput(rng=rng, dropout=True),
                        GraphConvolution(params.fc_size, adjacency),
                        # AddNoiseToInput(rng=rng, dropout=True),
                        GraphConvolution(params.fc_size, adjacency),
                        # AddNoiseToInput(rng=rng, dropout=True),
                        GraphConvolution(params.fc_size, adjacency, activation_str='linear'),
                    ]
# ---------------------------------------------------------------------------------------------

    learning_rate = T.scalar(dtype=theano.config.floatX)
    learning_rate.tag.test_value = 1.0
    gcnn = GCNN(params, graphLayers, finalLayer, nodeNames, edgeRNNs, nodeRNNs, nodeToEdgeConnections, edgeListComplete, euclidean_loss, nodeLabels, learning_rate, new_idx, featureRange, clipnorm=params.clipnorm, update_type=gradient_method, weight_decay=params.weight_decay)
    
    return gcnn
