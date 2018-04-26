import subprocess as sbp
import sys
sys.dont_write_bytecode = True
import os
import copy
import socket as soc
from datetime import datetime
import sys
sys.setrecursionlimit(5000)
if(int(sys.argv[1])):
	from py_server import ssh

if(int(sys.argv[1])==2):
	base_dir = '/home/siddhartha/Motion_Prediction/GCN_new'
else:
	base_dir = '/new_data/gpu/siddsax/motion_pred_checkpoints'
gpus = [0,1,3]
## Set gpus = [gpu_id] if you don't have a gpu then set gpus = []

## Hyper parameters for training S-RNN
params = {}

## These hyperparameters are OKAY to tweak. They will affect training, convergence etc.
params['initial_lr'] = 1e-3
params['decay_schedule'] = [1.5e3,4.5e3] # Decrease learning rate after these many iterations
params['decay_rate_schedule'] = [0.1,0.1] # Multiply the current learning rate by this factor
params['lstm_init'] = 'uniform' # Initialization of lstm weights
params['fc_init'] = 'uniform' # Initialization of FC layer weights
params['clipnorm'] = 25.0
params['use_noise'] = 1
params['noise_schedule'] = [250,0.5e3,1e3,1.3e3,2e3,2.5e3,3.3e3] # Add noise after these many iterations
params['noise_rate_schedule'] = [0.01,0.05,0.1,0.2,0.3,0.5,0.7] # Variance of noise to add
params['momentum'] = 0.99
params['g_clip'] = 25.0
params['truncate_gradient'] = 100
params['sequence_length'] = 150 # Length of each sequence fed to RNN
params['sequence_overlap'] = 50 
params['batch_size'] = 100
if (int(sys.argv[2])):
	params['lstm_size'] = 1#512 #
	params['node_lstm_size'] = 1#512 # 
	params['fc_size'] = 1#256# 
	params['snapshot_rate'] = 1#25 #100# Save the model after every 250 iterations
else:
	params['lstm_size'] = 512 #
	params['node_lstm_size'] = 512 # 
	params['fc_size'] = 256# 
	params['snapshot_rate'] = 10 #100# Save the model after every 250 iterations
params['train_for'] = 'final' 
params['test'] =  int(sys.argv[2])

## Tweak these hyperparameters only if you want to try out new models etc. This is only for 'Advanced' users
params['use_pretrained'] = 0
params['iter_to_load'] = 20
params['model_to_train'] = 'dra'
params['crf'] = ''
params['copy_state'] = 0
params['full_skeleton'] = 1
params['weight_decay'] = 0.0
params['temporal_features'] = 1
params['dra_type'] = 'simple'
params['dataset_prefix'] = ''
params['drop_features'] = 0
params['drop_id'] = '9'
params['subsample_data'] = 1
params['ssh'] = int(int(sys.argv[1])==1)
my_env = os.environ
my_env['PATH'] += ':/usr/local/cuda/bin'
use_gpu = 1 


params['gcnType'] = int(sys.argv[3])

# Setting directory to dump trained models and then executing trainDRA.py

#if params['model_to_train'] == 'dra':

if(len(sys.argv)==5):
	name = sys.argv[4]
else:
	name = 'checkpoints_{0}_T_{2}_bs_{1}_tg_{3}_ls_{4}_fc_{5}_demo'.format(params['model_to_train'],params['batch_size'],params['sequence_length'],params['truncate_gradient'],params['lstm_size'],params['fc_size'])
params['checkpoint_path'] = base_dir + '/GCNN/' + name
path_to_checkpoint = '{0}/'.format(params['checkpoint_path'])

if(int(sys.argv[1])==1):
	script = "'if [ ! -d \"" + path_to_checkpoint + \
		"\" ]; \n then mkdir " + path_to_checkpoint + "\nfi'"
	ssh("echo " + script + " > file.sh")
	ssh("bash file.sh")
else:
	if not os.path.exists(path_to_checkpoint):
		os.mkdir(path_to_checkpoint)

print 'Dir: {0}'.format(path_to_checkpoint)
args = ['python','trainDRA.py']
for k in params.keys():
	args.append('--{0}'.format(k))
	if not isinstance(params[k],list):
		args.append(str(params[k]))
	else:
		for x in params[k]:
			args.append(str(x))

# FNULL = open('{0}stdout.txt'.format(path_to_checkpoint),'w')
#p=sbp.Popen(args,env=my_env,shell=False,stdout=FNULL,stderr=sbp.STDOUT)
p=sbp.Popen(args,env=my_env)
pd = p.pid
p.wait()


