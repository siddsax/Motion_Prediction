import theano
from theano import tensor as T
import numpy as np

def euclidean_loss(y_t,y):
	delta_t_ignore = 0 #50
	scaling = 1
	if y.ndim > 2:
		scaling = (y.shape[0]-delta_t_ignore)*y.shape[2] # = T*D
		#scaling = y.shape[2] # = T
	y_new = y[delta_t_ignore:,:,:].flatten()
	y_t_new = y_t[delta_t_ignore:,:,:].flatten()

	return scaling * T.mean(T.sqr(y_new-y_t_new))


def hinge_euclidean_loss(y_t, y):
	delta_t_ignore = 0  # 50
	if y.ndim > 2:
		scaling = (y.shape[0]-delta_t_ignore)*y.shape[2]  # = T*D
		#scaling = y.shape[2] # = T
	y_new = y[delta_t_ignore:, :, :].flatten()
	y_t_new = y_t[delta_t_ignore:, :, :].flatten()

	diff = y_new-y_t_new
	sqrt = T.sqr(diff)
	mean = T.mean(sqrt)
	flagv = sqrt > 2*mean
	sqrt += 10*sqrt*flagv
	return T.mean(sqrt)


def softmax_loss(p_t,y):
	shape = p_t.shape
	is_tensor3 = p_t.ndim > 2
	t = 1
	if is_tensor3:
		t = shape[0]
		p_t = p_t.reshape((shape[0]*shape[1],shape[2]))
		y = y.flatten()

	return - t*(T.mean(T.log(p_t)[T.arange(y.shape[0]), y]))	

def temporal_loss(y_t,offset):
	delta_t_ignore = 0  # 50
	scaling = 1
	if y_t.ndim > 2:
		scaling = (y_t.shape[0]-delta_t_ignore)*y_t.shape[2]  # = T*D

	# if (y_t.shape[1] > offset):
	y_fut = y_t[:,offset:,:].flatten()
	y = y_t[:,:-offset,:].flatten() 
	return -scaling * T.mean(T.sqr(y_fut-y))

def temp_euc_loss(y_t,y):
	offset = 5
	lbda = 1 
	return euclidean_loss(y_t,y) + lbda*temporal_loss(y_t,offset)


def softmax_decay_loss(p_t,y):

	shape = p_t.shape


	def recurrence(x_t,y_t,log_loss,t):
		log_loss_new = log_loss + T.exp(-t)*T.sum(T.log(x_t)[T.arange(y_t.shape[0]), y_t])
		t_new = t - 1
		return log_loss_new, t_new
	[log_loss_list, cells], ups = theano.scan(fn=recurrence,
					sequences=[p_t, y],
					outputs_info=[theano.shared(value=0.0), p_t.shape[0]-1],
					n_steps=p_t.shape[0]
				)
	return - (1.0/shape[1])*log_loss_list[-1]

