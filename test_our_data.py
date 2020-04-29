import os, sys, json, argparse, time
import numpy as np
from skimage.io import imsave
import pandas as pd 
import torch
import torch.nn as nn
import torchvision.models as models

import pretrainedmodels  

from our_data_loader import dataloader, my_effnet_loader

from common_flags import COMMON_FLAGS
json_dir = COMMON_FLAGS.json_dir
dataset_dir = COMMON_FLAGS.dataset_dir
data_download_dir = COMMON_FLAGS.data_download_dir
prediction_dir = COMMON_FLAGS.prediction_dir
compare_dir = COMMON_FLAGS.compare_dir

'''
Note: all grey images occur loading error.
'''

def show_sample_images():
	'''
	Visualize some images we collected.
	'''
	# data loader:
	batch_size = args.batch_size
	my_data_loader = dataloader(batch_size)

	# testing:
	for batch_idx, (images, labels, img_idx) in enumerate(my_data_loader):
		print('batch %d images:' % batch_idx, images.size())
		print('img_idx:', type(img_idx))
		
		N = labels.size()[0]

		# visualize image examples:
		img_list = []
		for i in range(N):
			img = np.moveaxis(images[i,...].numpy(), 0, -1)
			print('img:', img.shape)
			img_list.append(img)
		img_big = np.concatenate(img_list, axis=1)
		temp_list = np.split(img_big, 4, axis=1)
		img_big = np.concatenate(temp_list, axis=0)
		imsave('imagenet_example.png', img_big)
		print('labels:', labels[0:N])
		
		break

def test_mydata(model, model_name, args):
	'''
	Get test acc using search keyword as gnd labels. Not necessarily accurate.
	'''
	# data loader:
	if model_name == 'effnetE7':
		my_data_loader = my_effnet_loader(args.batch_size)
	else:
		my_data_loader = dataloader(args.batch_size, img_size=args.img_size, mean=args.mean, std=args.std)

	# testing:
	print('testing %s our data ... ' % model_name)
	start_time = time.time()
	correct_total, total = 0, 0
	probs_lst, preds_lst, labels_lst = [], [], []
	for batch_idx, (images, labels, img_idx) in enumerate(my_data_loader):
		images = images.cuda()
		labels = labels.cuda()
		img_idx = img_idx.cuda()

		labels_lst.append(labels.cpu().numpy())

		N = labels.size()[0]
		total += N

		logits = model(images)
		probs = nn.functional.softmax(logits)
		probs_lst.append(probs.detach().cpu().numpy())
		preds = torch.argmax(logits, dim=1)
		preds_lst.append(preds.cpu().numpy())
		correct = torch.sum(preds == labels)
		correct_total += correct

		# print
		if batch_idx % 10 == 0:
			print('batch %d images:' % batch_idx, images.size())
			print('correct/batch_size: %d/%d' % (correct, N))
	
	testing_time = time.time()-start_time
	result_str = '%s corect_num/total: %d/%d, acc: %.2f, time: %s' % (model_name, correct_total, total, float(correct_total)/float(total)*100, testing_time) 
	print(result_str)
	f = open(os.path.join(prediction_dir, "%s_evaluate_result.txt" % model_name), "w+")
	f.write(result_str)
	f.close()

	# save npy:
	preds_all = np.concatenate(preds_lst, axis=0)
	probs_all = np.concatenate(probs_lst, axis=0)
	np.save(os.path.join(prediction_dir, '%s_preds.npy' % model_name), preds_all)
	np.save(os.path.join(prediction_dir, '%s_probs.npy' % model_name), probs_all)

def get_predictions():
	# args:
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default='7')
	parser.add_argument('-b', '--batch_size', type=int, default=128)
	args = parser.parse_args()

	print(args)
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	
	# model list:
	model_name_lst = ['vgg16bn',  # 0:1
		'resnet34', 'resnet101', 'resnet152', # 1:4
		'wrn-101-2', 'resnext101_32x4d', 'se_resnet101', # 4:7
		'senet154', # 7:8
		'nasnetalarge', 'pnasnet5large', # 8:10
		'resnext101_32x48d_wsl', 'effnetE7'] # 10:12

	## Test
	for model_name in ['effnetE7']:
		# define model:
		with torch.no_grad():
			if model_name is 'vgg16bn':
				model = models.vgg16_bn(pretrained=True)
			elif model_name is 'vgg19bn':
				model = models.vgg19_bn(pretrained=True)
			elif model_name is 'resnet34':
				model = models.resnet34(pretrained=True)
			elif model_name is 'resnet101':
				model = models.resnet101(pretrained=True)
			elif model_name is 'resnet152':
				model = models.resnet152(pretrained=True)
			elif model_name is 'wrn-101-2':
				model = models.wide_resnet101_2(pretrained=True)
			elif model_name is 'resnext101_32x4d':
				model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
			elif model_name is 'se_resnet101':
				model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
			elif model_name is 'senet154':
				model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')	
			elif model_name is 'nasnetalarge':
				model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
			elif model_name is 'pnasnet5large':
				model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')					
			elif model_name is 'resnext101_32x48d_wsl':
				model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')	
			elif model_name is 'effnetE7':
				from efficientnet_pytorch import EfficientNet
				model = EfficientNet.from_pretrained('efficientnet-b7') 
			else:
				raise Exception('unimplemented model structure %s' % model_name)

			print(type(model))
			if 'pretrainedmodels' in str(type(model)):
				_, _H, _W = model.input_size
				mean = model.mean
				std = model.std
			else:
				_H, _W = 224, 224
				mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
			args.img_size = _H
			args.mean, args.std = mean, std		

			model = nn.DataParallel(model.eval().cuda())

		for param in model.parameters():
			param.requires_grad = False

		# on test set:
		test_mydata(model, model_name, args)

if __name__ == "__main__":
	get_predictions()

	
	