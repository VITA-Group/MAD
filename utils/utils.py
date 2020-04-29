import numpy as np 
import shutil

def create_dir(_path):
	if not os.path.exists(_path):
		os.makedirs(_path)

def delete_dir(_path):
	if os.path.isdir(_path):
		shutil.rmtree(_path)

def trace_distance(trace1, trace2, weighted=False):
	''' Calculate the distance between two traces on a tree.
	Args:
		trace1: list of strings (wnid as string). from leaf to root
		trace2: list of strings (wnid as string). from leaf to root
	
	Returns:
		distance: float
	'''
	# reverse
	trace1.reverse() # now its is root -> leaf  left -> right
	trace2.reverse()

	D1 = 0
	for i, item in enumerate(trace1):
		if item not in trace2:
			if weighted:
				D1 += 2**(-i+1)
			else:
				D1 += 1

	D2 = 0
	for i, item in enumerate(trace2):
		if item not in trace1:
			if weighted:
				D2 += 2**(-i+1)
			else:
				D2 += 1

	distance = D1 + D2

	return distance



import os
import torch
import torch.nn as nn

count_ops = 0
num_ids = 0
def get_feature_hook(self, _input, _output):
	global count_ops, num_ids 
	# print('------>>>>>>')
	# print('{}th node, input shape: {}, output shape: {}, input channel: {}, output channel {}'.format(
	# 	num_ids, _input[0].size(2), _output.size(2), _input[0].size(1), _output.size(1)))
	# print(self)
	delta_ops = self.in_channels * self.out_channels * self.kernel_size[0] * self.kernel_size[1] * _output.size(2) * _output.size(3) / self.groups
	count_ops += delta_ops
	# print('ops is {:.6f}M'.format(delta_ops / 1024.  /1024.))
	num_ids += 1
	# print('')

def measure_model(net, H_in, W_in):
	import torch
	import torch.nn as nn
	_input = torch.randn((1, 3, H_in, W_in))
	#_input, net = _input.cpu(), net.cpu()
	hooks = []
	for module in net.named_modules():
		if isinstance(module[1], nn.Conv2d) or isinstance(module[1], nn.ConvTranspose2d):
			# print(module)
			hooks.append(module[1].register_forward_hook(get_feature_hook))

	_out = net(_input)
	global count_ops
	print('count_ops: {:.6f}M'.format(count_ops / 1024. /1024.)) # in Million
	