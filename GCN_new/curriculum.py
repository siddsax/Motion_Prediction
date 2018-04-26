import numpy as np
import readCRFgraph as graph
# import CRFProblems.processdata as poseDataset
import processdata as poseDataset

poseDataset.T = 150
poseDataset.delta_shift = 100
poseDataset.num_forecast_examples = 24
poseDataset.copy_state = 0
poseDataset.full_skeleton = 1
poseDataset.train_for = 'final'
poseDataset.temporal_features = 1
poseDataset.crf_file = './CRFProblems/H3.6m/crf' + ''
poseDataset.dataset_prefix = ''
poseDataset.drop_features = 0
poseDataset.drop_id = '9'
poseDataset.subsample_data = 1
poseDataset.runall()
if poseDataset.copy_state:
	args.batch_size = poseDataset.minibatch_size

[nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeListComplete,edgeFeatures,nodeToEdgeConnections,trX,trY,trX_validation,trY_validation,trX_forecasting,trY_forecasting,trX_forecast_nodeFeatures,adjacency] = graph.readCRFgraph(poseDataset)

names = trX.keys()
# print()
# print(np.shape(trX[trX.keys()[0]]))

for nm in names:
    v = np.var(trX[nm],axis=1)
    print(np.shape(v))