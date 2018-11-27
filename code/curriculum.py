import numpy as np
import readCRFgraph as graph
# import CRFProblems.processdata as poseDataset
import processdata as poseDataset


def curriculum(num_batches, poseDataset, trX, trY):

    names = trX.keys()
    v = {}
    batchesX = []
    batchesY = []
    k = int(np.shape(trX[names[0]])[1]/num_batches)
    print(k)
    for nm in names:
        v[nm] = np.argsort(np.mean(np.var(trX[nm], axis=0), axis=1))

    batchesX = [dict() for x in range(num_batches+1)]
    batchesY = [dict() for x in range(num_batches+1)]
    for i in range(num_batches+1):
        for nm in names:
            if(i==0):
                print("====")
                batchesX[i][nm] = trX[nm][:, v[nm][i*k:(i+1)*k], : ]
                batchesY[i][nm] = trY[nm][:, v[nm][i*k:(i+1)*k], : ]
            elif(i == num_batches):
                batchesX[i][nm] = np.concatenate((batchesX[i-1][nm], trX[nm][:, v[nm][i*k:], :]), axis = 1)
                batchesY[i][nm] = np.concatenate((batchesY[i-1][nm], trY[nm][:, v[nm][i*k:], :]), axis = 1)
            else:
                batchesX[i][nm] = np.concatenate((batchesX[i-1][nm], trX[nm][:, v[nm][i*k:(i+1)*k], :]), axis = 1)
                batchesY[i][nm] = np.concatenate((batchesY[i-1][nm], trY[nm][:, v[nm][i*k:(i+1)*k], :]), axis = 1)

        print(np.shape(batchesX[i]['torso']))
    
    print("-----------------------")

    return batchesX, batchesY

# poseDataset.T = 150
# poseDataset.delta_shift = 100
# poseDataset.num_forecast_examples = 24
# poseDataset.copy_state = 0
# poseDataset.full_skeleton = 1
# poseDataset.train_for = 'final'
# poseDataset.temporal_features = 1
# poseDataset.crf_file = './CRFProblems/H3.6m/crf' + ''
# poseDataset.dataset_prefix = ''
# poseDataset.drop_features = 0
# poseDataset.drop_id = '9'
# poseDataset.subsample_data = 1
# poseDataset.runall()
# if poseDataset.copy_state:
#     args.batch_size = poseDataset.minibatch_size

# [nodeNames,nodeList,nodeFeatureLength,nodeConnections,edgeList,edgeListComplete,edgeFeatures,nodeToEdgeConnections,trX,trY,trX_validation,trY_validation,trX_forecasting,trY_forecasting,trX_forecast_nodeFeatures,adjacency] = graph.readCRFgraph(poseDataset)
# batchesX, batchesY = curriculum(5, poseDataset, trX, trY)

# for i in range(len(batchesX)):
#     print(np.shape(batchesX[i]['torso']))
