import subprocess as sbp
import os
import copy
import socket as soc
from datetime import datetime
import sys


base_dir = '.'
gpus = []

params = {}
params['decay_type'] = 'schedule'
params['decay_after'] =  -1
params['initial_lr'] = 1e-3
params['learning_rate_decay'] = 1.0
params['decay_schedule'] = []
params['decay_rate_schedule'] = [] 
params['lstm_init'] = 'uniform'
params['fc_init'] = 'uniform'
params['epochs'] = 2000
params['clipnorm'] = 0.0
params['use_noise'] = 1
params['noise_schedule'] = []
params['noise_rate_schedule'] = []
params['momentum'] = 0.99
params['g_clip'] = 25.0

params['truncate_gradient'] = 100
params['use_pretrained'] = 0
params['iter_to_load'] = 2500
params['model_to_train'] = 'gcnn'
params['sequence_length'] = 150
params['sequence_overlap'] = 50
params['batch_size'] = 100
params['lstm_size'] = 512
params['node_lstm_size'] = 10#512
params['fc_size'] = 10#256
params['snapshot_rate'] = 25
params['crf'] = ''
params['copy_state'] = 0
params['full_skeleton'] = 1
params['weight_decay'] = 0.0
params['train_for'] = 'eating'
params['temporal_features'] = 0
params['dra_type'] = 'NoEdge'
params['dataset_prefix'] = ''
params['drop_features'] = 0
params['drop_id'] = '9'
params['subsample_data'] = 1

def listToString(ll):
	st = '['
	for v_ in ll:
		st += str(v_) + ','
	st = st[:-1] + ']'
	return st 


my_env = os.environ

# Adding CUDA to path
my_env['PATH'] += ':/usr/local/cuda/bin'

use_gpu = 0

if len(gpus) > 0:
	if use_gpu >= len(gpus):
		use_gpu = 0
	my_env['THEANO_FLAGS']='mode=FAST_RUN,device=gpu{0},floatX=float32'.format(gpus[use_gpu])
	use_gpu += 1
else:
	my_env['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32'.format(use_gpu)
if params['model_to_train'] == 'dra':
	params['checkpoint_path'] = 'checkpoints/checkpoints_{0}_T_{2}_bs_{1}_tg_{3}_ls_{4}_fc_{5}_'.format(params['model_to_train'],params['batch_size'],params['sequence_length'],params['truncate_gradient'],params['lstm_size'],params['fc_size'])
	if not params['node_lstm_size'] == params['lstm_size']:
		params['checkpoint_path'] += 'nls_{0}_'.format(params['node_lstm_size'])

else:
	if(len(sys.argv)==3):
		params['checkpoint_path'] = sys.argv[2]
	else:
		params['checkpoint_path'] = 'checkpoints/checkpoints_{0}_T_{2}_bs_{1}_tg_{3}_'.format(params['model_to_train'],params['batch_size'],params['sequence_length'],params['truncate_gradient'])

if params['weight_decay'] > 1e-6:
	params['checkpoint_path'] += '_wd_{0}'.format(params['weight_decay'])
if params['drop_features']:
	params['checkpoint_path'] += '_df_' + params['drop_id']
path_to_checkpoint = base_dir + '/{0}/'.format(params['checkpoint_path'])
print(path_to_checkpoint)
if not os.path.exists(path_to_checkpoint):
	os.mkdir(path_to_checkpoint)

if params['use_pretrained'] == 1:
	if load_pretrained_model_from[-1] == '/':
		os.system('cp {0}checkpoint.{1} {2}.'.format(load_pretrained_model_from,params['iter_to_load'],path_to_checkpoint))
		os.system('cp {0}logfile {1}.'.format(load_pretrained_model_from,path_to_checkpoint))
		os.system('cp {0}complete_log {1}.'.format(load_pretrained_model_from,path_to_checkpoint))
	else:
		os.system('cp {0}/checkpoint.{1} {2}.'.format(load_pretrained_model_from,params['iter_to_load'],path_to_checkpoint))
		os.system('cp {0}/logfile {1}.'.format(load_pretrained_model_from,path_to_checkpoint))
		os.system('cp {0}/complete_log {1}.'.format(load_pretrained_model_from,path_to_checkpoint))

print 'Dir: {0}'.format(path_to_checkpoint)
args = ['python','trainGCNN.py']
for k in params.keys():
	args.append('--{0}'.format(k))
	if not isinstance(params[k],list):
		args.append(str(params[k]))
	else:
		for x in params[k]:
			args.append(str(x))

FNULL = open('{0}stdout.txt'.format(path_to_checkpoint),'w')
p=sbp.Popen(args,env=my_env)
pd = p.pid
p.wait()
