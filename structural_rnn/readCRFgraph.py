import numpy as np
import copy

def readCRFgraph(poseDataset,noise=1e-10,forecast_on_noisy_features=False):
	'''
	Understanding data structures
	preGraphNets: node_type ---> [edge_types]
	nodeConnections: node_name ---> [node_names]
	nodeNames: node_name ---> node_type
	nodeList: node_type ---> output_dimension
	nodeFeatureLength: node_type ---> feature_dim_into_nodeRNN

	edgeList: list of edge types
	edgeFeatures: edge_type ---> feature_dim_into_edgeRNN
	'''
	
	filename = poseDataset.crf_file

	global nodeNames, nodeList, preGraphNets, nodeConnections, nodeFeatureLength, edgeList, edgeFeatures

	lines = open(filename).readlines()
	# nodeOrder = []
	nodeNames = []
	# nodeList = {}
	# preGraphNets = {}
	temporalNodeFeatureLength = {}
	nodeFeatureLength = {}
	for node_name in lines[0].strip().split(','):
		nodeNames.append(node_name)
		# nodeNames[node_name] = node_type
		# nodeList[node_type] = 0
		preGraphNets[node_name] = {}
		preGraphNets[node_name]['temporal'] = [0,0]
		preGraphNets[node_name]['normal'] = [0,0]
		temporalNodeFeatureLength[node_name] = 0
		nodeFeatureLength[node_name] = 0
	
	# edgeList = []
	# edgeFeatures = {}
	nodeConnections = {}
	edgeListComplete = []
	for i in range(2,len(lines)):
		first_nodeName = nodeNames[i-2]
		# first_nodeType = nodeNames[first_nodeName]
		nodeConnections[first_nodeName] = []
		connections = lines[i].strip().split(',')
		for j in range(len(connections)):
			if connections[j] == '1':
				second_nodeName = nodeNames[j]
				# second_nodeType = nodeNames[second_nodeName]
				nodeConnections[first_nodeName].append(second_nodeName)
		
				# edgeType_1 = first_nodeType + '_' + second_nodeType
				# edgeType_2 = second_nodeType + '_' + first_nodeType
				# edgeType = ''
				# if edgeType_1 in edgeList:
				# 	edgeType = edgeType_1
				# 	continue
				# elif edgeType_2 in edgeList:
				# 	edgeType = edgeType_2
				# 	continue
				# else:
				# 	edgeType = edgeType_1
				# edgeList.append(edgeType)
				# edgeListComplete.append(edgeType)

				# if (first_nodeType + '_input') not in edgeListComplete:
				# 	edgeListComplete.append(first_nodeType + '_input')
				# if (second_nodeType + '_input') not in edgeListComplete:
				# 	edgeListComplete.append(second_nodeType + '_input')

				# edgeFeatures[edgeType] = 0
				# nodeToEdgeConnections[first_nodeType][edgeType] = [0,0]
				# nodeToEdgeConnections[second_nodeType][edgeType] = [0,0]

	trX = {}
	trY = {}
	trX_validate = {}
	trY_validate = {}
	trX_forecast = {}
	trY_forecast = {}
	trX_nodeFeatures = {}
	node_features = {}
	temporal_node_features = {}
	forecast_node_features = {}
	temporal_validate_node_features = {}
	forecast_node_features = {}
	temporal_forecast_node_features = {}
	poseDataset.addNoiseToFeatures(noise=noise)
	for nodeName in nodeNames:
		# edge_features = {}
		# validate_edge_features = {}
		# forecast_edge_features = {}

		# nodeType = nodeNames[nodeName]
		# edgeTypesConnectedTo = nodeToEdgeConnections[nodeType].keys()
		low = 0
		high = 0
		# for edgeType in edgeTypesConnectedTo:
		[node_features[nodeName],temporal_node_features[nodeName],
		 validate_node_features[nodeName],temporal_validate_node_features[nodeName],
		 forecast_node_features[nodeName],temporal_forecast_node_features[nodeName]] =
		 poseDataset.getfeatures(nodeName,forecast_on_noisy_features=forecast_on_noisy_features)

		nodeFeatureLength[nodeName] = node_features[nodeName].shape[2]
		temporalNodeFeatureLength[nodeName] = temporal_node_features[nodeName].shape[2]
		# edgeType = nodeType + '_input'
		# D = edge_features[edgeType].shape[2]
		#  --------------------------- FIGURE THIS OUT -------------------------------
		high += nodeFeatureLength[nodeName]
		preGraphNets[nodeName]['normal'][0] = low
		preGraphNets[nodeName]['normal'][1] = high
		low = high

		high += temporalNodeFeatureLength[nodeName]
		preGraphNets[nodeName]['temporal'][0] = low
		preGraphNets[nodeName]['temporal'][1] = high
		low = high

		#  ---------------------------------------------------------------------------
		# nodeRNNFeatures = copy.deepcopy(edge_features[edgeType])
		# validate_nodeRNNFeatures = copy.deepcopy(validate_edge_features[edgeType])
		# forecast_nodeRNNFeatures = copy.deepcopy(forecast_edge_features[edgeType])

		# for edgeType in edgeList:
		# 	if edgeType not in edgeTypesConnectedTo:
		# 		continue
		# 	D = edge_features[edgeType].shape[2]
		# 	edgeFeatures[edgeType] = D
		# 	high += D
		# 	nodeToEdgeConnections[nodeType][edgeType][0] = low
		# 	nodeToEdgeConnections[nodeType][edgeType][1] = high
		# 	low = high
		# 	nodeRNNFeatures = np.concatenate((nodeRNNFeatures,edge_features[edgeType]),axis=2)	
		# 	validate_nodeRNNFeatures = np.concatenate((validate_nodeRNNFeatures,validate_edge_features[edgeType]),axis=2)	
		# 	forecast_nodeRNNFeatures = np.concatenate((forecast_nodeRNNFeatures,forecast_edge_features[edgeType]),axis=2)	

		[Y,Y_validate,Y_forecast,X_forecast] = poseDataset.getlabels(nodeName)
		# nodeList[nodeType] = num_classes
		
		# trX is same as node_features.....

		# idx = nodeName + ':' + nodeType
		trX[nodeName] = np.concatenate((node_features[nodeName],temporal_node_features[nodeName]),axis=2)
		trX_validate[nodeName] = np.concatenate((validate_node_features[nodeName],temporal_validate_node_features[nodeName]),axis=2)
		trX_forecast[nodeName] = np.concatenate((forecast_node_features[nodeName],temporal_forecast_node_features[nodeName]),axis=2)
		
		trY[nodeName] = Y # Output ground truths i.e. the limb position at next time step for each name
		trY_validate[nodeName] = Y_validate # -- Validation set
		trY_forecast[nodeName] = Y_forecast # -- Test set
		
		trX_nodeFeatures[nodeName] = X_forecast # these are position of joints using which the input fetures are made. trX_forecast[nodeName] is nothing but just a concatenation of the node_features(which are just this in my model) and temporalfeatures (which is this:[this-this_{-1}) 
	print nodeToEdgeConnections

	return nodeNames,temporalNodeFeatureLength,nodeFeatureLength,nodeConnections,trY,trY_validate,trY_forecast,trX_nodeFeatures,
	  trX,
	  trX_validate,
		trX_forecast,
		preGraphNets	


def getNodeFeature(nodeName,nodeFeatures,nodeFeatures_t_1,poseDataset):
	edge_features = {}
	nodeType = nodeNames[nodeName]
	edgeTypesConnectedTo = nodeToEdgeConnections[nodeType].keys()
	low = 0
	high = 0

	for edgeType in edgeTypesConnectedTo:
		edge_features[edgeType] = poseDataset.getDRAfeatures(nodeName,edgeType,nodeConnections,nodeNames,nodeFeatures,nodeFeatures_t_1)

	edgeType = nodeType + '_input'
	nodeRNNFeatures = copy.deepcopy(edge_features[edgeType])

	for edgeType in edgeList:
		if edgeType not in edgeTypesConnectedTo:
			continue
		nodeRNNFeatures = np.concatenate((nodeRNNFeatures,edge_features[edgeType]),axis=2)

	return nodeRNNFeatures
