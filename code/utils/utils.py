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


def saveNormalizationStats(path):
    activities = {}
    activities['walking'] = 14
    activities['eating'] = 4
    activities['smoking'] = 11
    activities['discussion'] = 3
    activities['walkingdog'] = 15

    cPickle.dump(poseDataset.data_stats,open('{0}h36mstats.pik'.format(path),'wb'))
    forecastidx = poseDataset.data_stats['forecastidx']
    num_forecast_examples = len(forecastidx.keys())
    f = open('{0}forecastidx'.format(path),'wb')
    for i in range(num_forecast_examples):
        tupl = forecastidx[i]
        st = '{0},{1},{2},{3}\n'.format(i,activities[tupl[0]],tupl[2],tupl[1])
        f.write(st)
    f.close()
    print "************Done saving the stats*********"

def logModel(params, thefile, nodeNames, model, poseDataset):
    numparams = 0 
    print("Edge RNNs --- " + nodeNames[0] + " --- TEMPORAL\n")
    thefile.write("Edge RNNs --- " + nodeNames[0] + " --- TEMPORAL\n")
    for item in model.edgeRNNs[nodeNames[0] + "_temporal"]:
        numparams += item.numparams
        s = item.__class__.__name__ + '        ' + item.paramstr
        thefile.write(s + '\n')
        
    thefile.write("--------------------------- \n")

    thefile.write("Edge RNNs Normal\n")
    for item in model.edgeRNNs[nodeNames[0] + "_normal"]:
        s = item.__class__.__name__ + '        ' + item.paramstr
        thefile.write(s + '\n')
        numparams += item.numparams
    thefile.write("--------------------------- \n")

    thefile.write("Node RNNs \n")
    for item in model.nodeRNNs[nodeNames[0]]:
        s = item.__class__.__name__ + '        ' + item.paramstr
        thefile.write(s + '\n')
        numparams += item.numparams
    thefile.write("--------------------------- \n")

    thefile.write("graphLayers \n")
    for item in model.graphLayers:
        s = item.__class__.__name__ + '        ' + item.paramstr
        thefile.write(s + '\n')
        numparams += item.numparams
    thefile.write("--------------------------- \n")

    thefile.write("finalLayer \n")
    for item in model.finalLayer[nodeNames[0]]:
        s = item.__class__.__name__ + '        ' + item.paramstr
        thefile.write(s + '\n')
        numparams += item.numparams
    thefile.write("--------------------------- \n")

    thefile.write("actions \n")
    for item in poseDataset.actions:
        thefile.write("%s\n" % item)
    thefile.write("--------------------------- \n")

    str = "Total Number of Parameters is = {0}".format(numparams)
    thefile.write(str)
    thefile.close()

    if(len(params.remoteBase)):
        from py_server import ssh
        file = open("logger.txt", "r")
        ssh("echo " + "'" + file.read() + "'" + " > " + params.remoteBase + "/logger.txt")
        file.close() 	
    else:
        print params.checkpoint_path
        import shutil
        shutil.move("logger.txt", params.checkpoint_path + "/logger.txt")

def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore):
        T = normalizedData.shape[0]
        D = data_mean.shape[0]
        origData = np.zeros((T, D), dtype=np.float32)
        dimensions_to_use = []
        for i in range(D):
                if i in dimensions_to_ignore:
                        continue
                dimensions_to_use.append(i)
        dimensions_to_use = np.array(dimensions_to_use)

        if not len(dimensions_to_use) == normalizedData.shape[1]:
                return []

        origData[:, dimensions_to_use] = normalizedData

        stdMat = data_std.reshape((1, D))
        stdMat = np.repeat(stdMat, T, axis=0)
        meanMat = data_mean.reshape((1, D))
        meanMat = np.repeat(meanMat, T, axis=0)
        origData = np.multiply(origData, stdMat) + meanMat
        return origData

def print_num(x):
    x = str(x)[::-1]
    count = 1
    out = ""
    for i in x:
        out += i 
        if count%3 == 0:
            if count == len(x):
                break
            out += ","
        count+=1
    out = out[::-1]
    return out	

def saveForecastedMotion(forecast,path,fname,ssh_flag=0):
    T = forecast.shape[0]
    
    N = forecast.shape[1]
    D = forecast.shape[2]
    for j in range(N):
        motion = forecast[:,j,:]
        file = '{0}/{1}_N_{2}'.format(path, fname, j)
        if(ssh_flag!=1):
            f = open(file,'wb')
        string = ''
        for i in range(T):
            st = ''
            for k in range(D):
                st += str(motion[i,k]) + ','
            st = st[:-1]
            string += st+'\n'
        # if(j==0):
        if(ssh_flag==1):
            ssh( "echo " + "'" + string + "'" + " > " + file)

        else:
            f.write(string)
    if(ssh_flag!=1):
        f.close()

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

def theano_convertToSingleVec(X,new_idx,featureRange):
    keys = X.keys()
    # [T,N,D]  = X[keys[0]].shape
    D = len(new_idx) - len(np.where(new_idx < 0)[0])
    single_vec = X[keys[0]]
    for k in range(1,len(keys)):
        single_vec = T.concatenate((single_vec,X[keys[k]]),axis=2) 
    for k in keys:
        nm = k.split(':')[0]
        idx = new_idx[featureRange[nm]]
        insert_at = np.delete(idx,np.where(idx < 0))
        single_vec = T.set_subtensor(single_vec[:,:,insert_at],X[k])
    return single_vec
