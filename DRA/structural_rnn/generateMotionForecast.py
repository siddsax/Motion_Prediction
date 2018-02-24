import subprocess as sbp
import os
import copy
import socket as soc
from datetime import datetime
import sys
import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu1")
my_env = os.environ
# Adding CUDA to path
my_env['PATH'] += ':/usr/local/cuda/bin'
my_env['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32'

params = {}
params['forecast'] = sys.argv[1]
params['checkpoint'] = sys.argv[2]
print(params['checkpoint'])
for iteration in [5000]:
	params['iteration'] = iteration
	params['motion_prefix'] = 50
	params['motion_suffix'] = 100
	args = ['python','forecastTrajectories.py']
	for k in params.keys():
		args.append('--{0}'.format(k))
		if not isinstance(params[k],list):
			args.append(str(params[k]))
		else:
			for x in params[k]:
				args.append(str(x))
	print(args)
	p=sbp.Popen(args)#,shell=False,stderr=sbp.STDOUT)
	p.wait()
