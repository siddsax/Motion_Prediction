import subprocess as sbp
import sys
import argparse
sys.dont_write_bytecode = True
import os
import copy
import socket as soc
from datetime import datetime
import sys
import theano
from trainModel import trainModel
sys.path.append('utils')
sys.setrecursionlimit(5000)


parser = argparse.ArgumentParser()
parser.add_argument('--initial_lr', type=float, default= 1e-3) 
parser.add_argument('--lstm_init', type=str, default='uniform')  # Initialization of lstm weights
parser.add_argument('--fc_init', type=str, default= 'uniform') # Initialization of fc layer weights
parser.add_argument('--clipnorm', type=float, default= 25.0) 
parser.add_argument('--use_noise', type=int, default= 1) 
parser.add_argument('--momentum', type=float, default= 0.99) 
parser.add_argument('--g_clip', type=float, default= 25.0) 
parser.add_argument('--truncate_gradient', type=int, default= 100)
parser.add_argument('--sequence_length', type=int, default= 150)
parser.add_argument('--sequence_overlap', type=int, default= 50)
parser.add_argument('--batch_size', type=int, default= 100)
parser.add_argument('--lstm_size', type=int, default = 512)
parser.add_argument('--node_lstm_size', type=int, default = 512)
parser.add_argument('--fc_size', type=int, default = 256)
parser.add_argument('--snapshot_rate', type=int, default = 10)
parser.add_argument('--test', type=int, default = 0)
parser.add_argument('--remoteBase', type=str , default = '')
parser.add_argument('--learning_rate_decay',type=float,default=1.0)

## Tweak these hyperparameters only if you want to try out new models etc. This is only for 'Advanced' users
parser.add_argument('--train_for', type= str, default = 'final')
parser.add_argument('--use_pretrained', type=int , default = 0)
parser.add_argument('--iter_to_load', type=int , default = 20)
parser.add_argument('--model_to_train', type= str, default = 'dra')
parser.add_argument('--crf', type= str, default = '')
parser.add_argument('--copy_state', type=int , default = 0)
parser.add_argument('--full_skeleton', type=int , default = 1)
parser.add_argument('--weight_decay', type=float , default = 0.0)
parser.add_argument('--temporal_features', type=int , default = 1)
parser.add_argument('--dra_type', type=str , default = 'simple')
parser.add_argument('--dataset_prefix', type=str , default = '')
parser.add_argument('--drop_features', type=int , default = 0)
parser.add_argument('--drop_id', type=str , default = '9')
parser.add_argument('--subsample_data', type=int , default = 1)
parser.add_argument('--decay_type',type=str,default='schedule')
parser.add_argument('--decay_after',type=int,default=-1)
parser.add_argument('--epochs',type=int,default=2000)

parser.add_argument('--maxiter',type=int,default=15000)
parser.add_argument('--dump_path', type=str, default='checkpoint')

parser.add_argument('--gcnType', dest='gcnType', type=int , default = 0)
parser.add_argument('--curriculum', dest='curriculum', type=int , default = 5)
parser.add_argument('--modelName', dest='modelName', type=str , default = '')
parser.add_argument('--homo', type=int, default=0)

params = parser.parse_args()
print(params)

params.decay_schedule = [1.5e3,4.5e3] # Decrease learning rate after these many iterations
params.decay_rate_schedule = [0.1,0.1] # Multiply the current learning rate by this factor
params.noise_schedule = [250,0.5e3,1e3,1.3e3,2e3,2.5e3,3.3e3] # Add noise after these many iterations
params.noise_rate_schedule = [0.01,0.05,0.1,0.2,0.3,0.5,0.7] # Variance of noise to add

if(len(params.modelName)==0):
    params.modelName = 'checkpoints_{0}_T_{2}_bs_{1}_tg_{3}_ls_{4}_fc_{5}_demo'.format(params.model_to_train ,params.batch_size ,params.sequence_length ,params.truncate_gradient ,params.lstm_size ,params.fc_size)

if params.test:
    params.lstm_size = 1
    params.node_lstm_size = 1 
    params.fc_size = 1 
    params.snapshot_rate = 1
    theano.config.optimizer='None'
    theano.exception_verbosity='high'
    theano.config.compute_test_value = 'warn' # Use 'warn' to activate this feature
    theano.config.floatX = 'float64'
    print "---------------------We are in testing phase-------------------------"


if(len(params.remoteBase)):
    params.checkpoint_path = params.remoteBase + '/savedModels'
    script = "'if [ ! -d \"" + params.checkpoint_path + \
        "\" ]; \n then mkdir " + params.checkpoint_path + "\nfi'"
    ssh("echo " + script + " > file.sh")
    ssh("bash file.sh")
    params.checkpoint_path = params.remoteBase + '/savedModels/' + params.modelName
    script = "'if [ ! -d \"" + params.checkpoint_path + \
        "\" ]; \n then mkdir " + params.checkpoint_path + "\nfi'"
    ssh("echo " + script + " > file.sh")
    ssh("bash file.sh")
else:
    if not os.path.exists('../savedModels'):
        os.mkdir('../savedModels')
    
    params.checkpoint_path = '../savedModels/' + params.modelName
    if not os.path.exists(params.checkpoint_path):
        os.mkdir(params.checkpoint_path)

trainModel(params)





