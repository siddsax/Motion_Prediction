import subprocess as sbp
import sys
import argparse
sys.dont_write_bytecode = True
import os
import copy
import socket as soc
from datetime import datetime
import sys
from py_server import *
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
import socket as soc
import readCRFgraph as graph

def saveModel(model,path,pathD,ssh_f=0):
    sys.setrecursionlimit(10000)

    edgeRNNs = getattr(model,'edgeRNNs')
    edgeRNN_saver = {}
    for k in edgeRNNs.keys():
        layer_configs = []
        for layer in edgeRNNs[k]:
            if hasattr(layer,'nested_layers'):
                if layer.nested_layers:
                    layer = CreateSaveableModel(layer,['layers'])
            layer_config = layer.settings
            layer_name = layer.__class__.__name__
            weights = [p.get_value() for p in layer.params]
            layer_config['weights'] = weights
            layer_configs.append({'layer':layer_name, 'config':layer_config})
        edgeRNN_saver[k] = layer_configs
    model.settings['edgeRNNs'] = edgeRNN_saver

    nodeRNNs = getattr(model,'nodeRNNs')
    nodeRNN_saver = {}
    for k in nodeRNNs.keys():
        layer_configs = []
        for layer in nodeRNNs[k]:
            if hasattr(layer,'nested_layers'):
                if layer.nested_layers:
                    layer = CreateSaveableModel(layer,['layers'])
            layer_config = layer.settings
            layer_name = layer.__class__.__name__
            weights = [p.get_value() for p in layer.params]
            layer_config['weights'] = weights
            layer_configs.append({'layer':layer_name, 'config':layer_config})
            # print(layer_configs)
            # print(-1)
        nodeRNN_saver[k] = layer_configs
    model.settings['nodeRNNs'] = nodeRNN_saver

    nodeRNNs = getattr(model, 'finalLayer')
    nodeRNN_saver = {}
    for k in nodeRNNs.keys():
        layer_configs = []
        for layer in nodeRNNs[k]:
            if hasattr(layer, 'nested_layers'):
                if layer.nested_layers:
                    layer = CreateSaveableModel(layer, ['layers'])
            layer_config = layer.settings
            layer_name = layer.__class__.__name__
            weights = [p.get_value() for p in layer.params]
            layer_config['weights'] = weights
            layer_configs.append({'layer': layer_name, 'config': layer_config})
            # print(layer_configs)
            # print(-1)
        nodeRNN_saver[k] = layer_configs
    model.settings['finalLayer'] = nodeRNN_saver
    serializable_model = {
        'model': model.__class__.__name__, 'config': model.settings}
    cPickle.dump(serializable_model, open(pathD, 'wb'))
    if ssh_f:
        from py_server import ssh
        ssh(pathD, dst=path, copy=1)
        os.remove(pathD)

def loadModel(path):
    model = cPickle.load(open(path))
    model_class = eval(model['model'])  #getattr(models, model['model'])

    edgeRNNs = {}
    for k in model['config']['edgeRNNs'].keys():
        layerlist = model['config']['edgeRNNs'][k]
        edgeRNNs[k] = []
        for layer in layerlist:
            if 'nested_layers' in layer['config'].keys():
                if layer['config']['nested_layers']:
                    layer = loadLayers(layer,['layers'])
            edgeRNNs[k].append(eval(layer['layer'])(**layer['config']))
        #edgeRNNs[k] = [eval(layer['layer'])(**layer['config']) for layer in layerlist]
    model['config']['edgeRNNs'] = edgeRNNs

    nodeRNNs = {}
    # print(model['config']['nodeRNNs'].keys())
    # print(1)
    for k in model['config']['nodeRNNs'].keys():
        layerlist = model['config']['nodeRNNs'][k]
        nodeRNNs[k] = []
        # print(layerlist)
        # print(2)
        for layer in layerlist:
            # print layer['config'].keys()
            # print 3
            if 'nested_layers' in layer['config'].keys():
                if layer['config']['nested_layers']:
                    layer = loadLayers(layer,['layers'])
            nodeRNNs[k].append(eval(layer['layer'])(**layer['config']))
        #nodeRNNs[k] = [eval(layer['layer'])(**layer['config']) for layer in layerlist]
    model['config']['nodeRNNs'] = nodeRNNs

    nodeRNNs = {}
    # print(model['config']['finalLayer'].keys())
    # print 1
    for k in model['config']['finalLayer'].keys():
        layerlist = model['config']['finalLayer'][k]
        nodeRNNs[k] = []
        # print(layerlist)
        # print(2)
        for layer in layerlist:
            # print layer['config'].keys()
            # print 3
            if 'nested_layers' in layer['config'].keys():
                if layer['config']['nested_layers']:
                    layer = loadLayers(layer,['layers'])
            nodeRNNs[k].append(eval(layer['layer'])(**layer['config']))
    model['config']['finalLayer'] = nodeRNNs

    model = model_class(**model['config'])

    return model
