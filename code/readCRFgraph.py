import numpy as np
import copy
import sys

def readCRFgraph(poseDataset,noise=1e-10,forecast_on_noisy_features=False):
    '''
    Understanding data structures
    nodeToEdgeConnections: node_type ---> [edge_types]
    nodeConnections: node_name ---> [node_names]
    nodeNames: node_name ---> node_type
    nodeList: node_type ---> output_dimension
    nodeFeatureLength: node_type ---> feature_dim_into_nodeRNN

    edgeList: list of edge types
    edgeFeatures: edge_type ---> feature_dim_into_edgeRNN
    '''



    filename = poseDataset.crf_file

    global nodeNames, nodeList, nodeToEdgeConnections, nodeConnections, nodeFeatureLength, edgeList, edgeFeatures

    nodeOrder = []
    nodeNames = {}
    nms = poseDataset.nodeNames
    nts = poseDataset.nodeTypes
    nodeList = {}
    nodeToEdgeConnections = {}
    nodeFeatureLength = {}
    edgeList = []
    
    for node_name, node_type in zip(nms,nts):
        nodeOrder.append(node_name)
        nodeNames[node_name] = node_type
        nodeList[node_name] = 0


        nodeToEdgeConnections[node_name] = {}
        nodeToEdgeConnections[node_name][node_name+'_normal'] = [0, 0]
        nodeToEdgeConnections[node_name][node_name+'_temporal'] = [0, 0]
        edgeList.append(node_name+'_normal')
        edgeList.append(node_name+'_temporal')


        nodeFeatureLength[node_name] = 0
    
    edgeFeatures = {}
    nodeConnections = {}#dump useless
    edgeListComplete = []

    trX = {}
    trY = {}
    trX_validate = {}
    trY_validate = {}
    trX_forecast = {}
    trY_forecast = {}
    trX_nodeFeatures = {}
    poseDataset.addNoiseToFeatures(noise=noise)
    

    for nm in nodeNames:
        edge_features = {}
        validate_edge_features = {}
        forecast_edge_features = {}

        edgeTypesConnectedTo = nodeToEdgeConnections[nm].keys()
        low = 0
        high = 0

        for edgeType in edgeTypesConnectedTo:
            [edge_features[edgeType],validate_edge_features[edgeType],forecast_edge_features[edgeType]] = poseDataset.getfeatures(nm,edgeType,nodeNames,forecast_on_noisy_features=forecast_on_noisy_features)
        flag = 0
        for edgeType in edgeList:
            if edgeType not in edgeTypesConnectedTo:
                continue
            D = edge_features[edgeType].shape[2]
            edgeFeatures[edgeType] = D
            high += D
            nodeToEdgeConnections[nm][edgeType][0] = low
            nodeToEdgeConnections[nm][edgeType][1] = high
            low = high
            if(flag):
                nodeRNNFeatures = np.concatenate((nodeRNNFeatures,edge_features[edgeType]),axis=2)	
                validate_nodeRNNFeatures = np.concatenate((validate_nodeRNNFeatures,validate_edge_features[edgeType]),axis=2)	
                forecast_nodeRNNFeatures = np.concatenate((forecast_nodeRNNFeatures,forecast_edge_features[edgeType]),axis=2)	
            else:
                nodeRNNFeatures = edge_features[edgeType]
                validate_nodeRNNFeatures = validate_edge_features[edgeType]
                forecast_nodeRNNFeatures = forecast_edge_features[edgeType]
                flag = 1

        [Y,Y_validate,Y_forecast,X_forecast,num_classes] = poseDataset.getlabels(nm)
        nodeList[nm] = num_classes
        
        trX[nm] = nodeRNNFeatures
        trX_validate[nm] = validate_nodeRNNFeatures
        trX_forecast[nm] = forecast_nodeRNNFeatures
        trY[nm] = Y
        trY_validate[nm] = Y_validate
        trY_forecast[nm] = Y_forecast
        trX_nodeFeatures[nm] = X_forecast
        adjacency = poseDataset.adjacency
    return nodeNames, nodeList, nodeFeatureLength, nodeConnections, edgeList, edgeListComplete, edgeFeatures, nodeToEdgeConnections, trX, trY, trX_validate, trY_validate, trX_forecast, trY_forecast, trX_nodeFeatures, adjacency

def getNodeFeature(nodeName,nodeFeatures,nodeFeatures_t_1,poseDataset):
    edge_features = {}
    edgeTypesConnectedTo = nodeToEdgeConnections[nodeName].keys()
    low = 0
    high = 0
    its = 0
    for edgeType in edgeTypesConnectedTo:
        edge_features[edgeType] = poseDataset.getDRAfeatures(nodeName,edgeType,nodeNames,nodeFeatures,nodeFeatures_t_1)
        if(its):
            nodeRNNFeatures = np.concatenate((nodeRNNFeatures,edge_features[edgeType]),axis=2)
        else:
            nodeRNNFeatures = edge_features[edgeType]
            its=1
            
    

    return nodeRNNFeatures
