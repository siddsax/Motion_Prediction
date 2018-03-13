import theano
import numpy as np
import cPickle
from theano import tensor as T 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math
import pdb
import sys 
import os
from neuralmodels.layers import *
from neuralmodels.models import *
from GraphConvolution import GraphConvolution
from FCLayer_out import FCLayer_out
# from py_server import ssh
'''
def loadLayers(model,layers_to_load):
	for layer_name in layers_to_load:
		model['config'][layer_name] = [eval(layer['layer'])(**layer['config']) for layer in model['config'][layer_name]]
	return model
'''

'''
def CreateSaveableModel(model,layers_to_save):
	for layerName in layers_to_save:
		layer_configs = []
		for layer in getattr(model,layerName):
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		model.settings[layerName] = layer_configs
	
	return model
'''
def CreateSaveableModel(model,layers_to_save):
	for layerName in layers_to_save:
		layer_configs = []
		for layer in getattr(model,layerName):
			if hasattr(layer,'nested_layers'):
				if layer.nested_layers:
					layer = CreateSaveableModel(layer,['layers'])
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		model.settings[layerName] = layer_configs
	return model
def loadLayers(model,layers_to_load):
	for layer_name in layers_to_load:
		layers_init = []
		for layer in model['config'][layer_name]:

			if 'nested_layers' in layer['config'].keys():
				if layer['config']['nested_layers']:
					layer = loadLayers(layer,['layers'])

			layers_init.append(eval(layer['layer'])(**layer['config']))
		model['config'][layer_name] = layers_init 
	return model


def loadGCNN(path):
	model = cPickle.load(open(path))
	model_class = eval(model['model'])  #getattr(models, model['model'])


	nodeRNNs = {}
	for k in model['config']['nodeRNNs'].keys():
		layerlist = model['config']['nodeRNNs'][k]
		nodeRNNs[k] = []
		for layer in layerlist:
			if 'nested_layers' in layer['config'].keys():
				if layer['config']['nested_layers']:
					layer = loadLayers(layer,['layers'])
			nodeRNNs[k].append(eval(layer['layer'])(**layer['config']))
		#nodeRNNs[k] = [eval(layer['layer'])(**layer['config']) for layer in layerlist]
	model['config']['nodeRNNs'] = nodeRNNs

	temporalNodeRNNs = {}
	for k in model['config']['temporalNodeRNNs'].keys():
		layerlist = model['config']['temporalNodeRNNs'][k]
		temporalNodeRNNs[k] = []
		for layer in layerlist:
			if 'nested_layers' in layer['config'].keys():
				if layer['config']['nested_layers']:
					layer = loadLayers(layer,['layers'])
			temporalNodeRNNs[k].append(eval(layer['layer'])(**layer['config']))
		#edgeRNNs[k] = [eval(layer['layer'])(**layer['config']) for layer in layerlist]
	model['config']['temporalNodeRNNs'] = temporalNodeRNNs

	topLayer = {}
	for k in model['config']['topLayer'].keys():
		layerlist = model['config']['topLayer'][k]
		topLayer[k] = []
		for layer in layerlist:
			if 'nested_layers' in layer['config'].keys():
				if layer['config']['nested_layers']:
					layer = loadLayers(layer,['layers'])
			topLayer[k].append(eval(layer['layer'])(**layer['config']))
		#edgeRNNs[k] = [eval(layer['layer'])(**layer['config']) for layer in layerlist]
	model['config']['topLayer'] = topLayer

	layerlist = model['config']['graphLayers']#[k]
	print(layerlist[3])
	graphLayers = []
	for layer in layerlist:
		# print(layer['config'].keys())
		if 'nested_layers' in layer['config'].keys():
			if layer['config']['nested_layers']:
				layer = loadLayers(layer,['layers'])
		graphLayers.append(eval(layer['layer'])(**layer['config']))
	model['config']['graphLayers'] = graphLayers

	finalLayer = {}
	for k in model['config']['finalLayer'].keys():
		layerlist = model['config']['finalLayer'][k]
		finalLayer[k] = []
		for layer in layerlist:
			if 'nested_layers' in layer['config'].keys():
				if layer['config']['nested_layers']:
					layer = loadLayers(layer,['layers'])
			finalLayer[k].append(eval(layer['layer'])(**layer['config']))
		#edgeRNNs[k] = [eval(layer['layer'])(**layer['config']) for layer in layerlist]
	model['config']['finalLayer'] = finalLayer

	model = model_class(**model['config'])
	return model



def saveGCNN(model,path,pathD):
	sys.setrecursionlimit(10000) #################### ------ ##################

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
		nodeRNN_saver[k] = layer_configs
	model.settings['nodeRNNs'] = nodeRNN_saver

	temporalNodeRNNs = getattr(model,'temporalNodeRNNs')
	temporalNodeRNN_saver = {}
	for k in temporalNodeRNNs.keys():
		layer_configs = []
		for layer in temporalNodeRNNs[k]:
			if hasattr(layer,'nested_layers'):
				if layer.nested_layers:
					layer = CreateSaveableModel(layer,['layers'])
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		temporalNodeRNN_saver[k] = layer_configs
	model.settings['temporalNodeRNNs'] = temporalNodeRNN_saver
	
	topLayer = getattr(model,'topLayer')
	topLayer_saver = {}
	for k in topLayer.keys():
		layer_configs = []
		for layer in topLayer[k]:
			if hasattr(layer,'nested_layers'):
				if layer.nested_layers:
					layer = CreateSaveableModel(layer,['layers'])
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		topLayer_saver[k] = layer_configs
	model.settings['topLayer'] = topLayer_saver

	graphLayers = getattr(model,'graphLayers')
	layer_configs = []
	for layer in graphLayers:
		if hasattr(layer,'nested_layers'):
			if layer.nested_layers:
				layer = CreateSaveableModel(layer,['layers'])
		layer_config = layer.settings
		layer_name = layer.__class__.__name__
		weights = [p.get_value() for p in layer.params]
		layer_config['weights'] = weights
		layer_configs.append({'layer':layer_name, 'config':layer_config})
	print("DADASDASSDSAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
	model.settings['graphLayers'] = layer_configs
	
	finalLayer = getattr(model,'finalLayer')
	finalLayer_saver = {}
	for k in finalLayer.keys():
		layer_configs = []
		for layer in finalLayer[k]:
			if hasattr(layer,'nested_layers'):
				if layer.nested_layers:
					layer = CreateSaveableModel(layer,['layers'])
			layer_config = layer.settings
			layer_name = layer.__class__.__name__
			weights = [p.get_value() for p in layer.params]
			layer_config['weights'] = weights
			layer_configs.append({'layer':layer_name, 'config':layer_config})
		finalLayer_saver[k] = layer_configs
	model.settings['finalLayer'] = finalLayer_saver

	serializable_model = {'model':model.__class__.__name__, 'config':model.settings}
	cPickle.dump(serializable_model, open(pathD, 'wb'))
	ssh(pathD,dst=path,copy=1)
	os.remove(pathD)

def plot_loss(lossfile):
	f = open(lossfile,'r')
	lines = f.readlines()
	f.close()
	loss = [float(l.strip()) for l in lines]
	iterations = range(len(loss))
	plt.plot(iterations,loss)
	plt.show()
