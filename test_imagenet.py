import os, sys, argparse, time
import numpy as np
from skimage.io import imsave

import torch
import torch.nn as nn
import torchvision.models as models

import pretrainedmodels

from imagenet_loader import imagenet_loader, effnet_loader
from utils.utils import measure_model, create_dir

def test_imagenet(model, model_name):

	# data loader:
	imagenet_dir = os.path.join('/hdd3/haotao/imagenet_pytorch_download')
	if model_name == 'effnetE7':
		val_loader = effnet_loader(imagenet_dir, args.batch_size)
	else:
		_, val_loader = imagenet_loader(imagenet_dir, args.batch_size, args.batch_size, 
					img_size=args.img_size, mean=args.mean, std=args.std)
	
	# testing:
	start_time = time.time()
	correct_total, total = 0, 0
	probs_lst, preds_lst, labels_lst = [], [], []
	for batch_idx, (images, labels) in enumerate(val_loader):
		images = images.cuda()
		labels = labels.cuda()

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

	prediction_dir = 'imagenet_predictions'
	create_dir(prediction_dir)

	testing_time = time.time()-start_time
	result_str = '%s corect_num/total: %d/%d, acc: %.2f, time: %s' % (model_name, correct_total, total, float(correct_total)/float(total)*100, testing_time) 
	print(result_str)
	f = open(os.path.join(prediction_dir, "%s_evaluate_result.txt" % model_name), "w+")
	f.write(result_str)
	f.close()
	
	preds_all = np.concatenate(preds_lst, axis=0)
	probs_all = np.concatenate(probs_lst, axis=0)
	labels_all = np.concatenate(labels_lst, axis=0)
	np.save(os.path.join(prediction_dir, '%s_preds.npy' % model_name), preds_all)
	np.save(os.path.join(prediction_dir, '%s_probs.npy' % model_name), probs_all)
	np.save(os.path.join(prediction_dir, 'labels.npy'), labels_all)
	print('labels_all:', labels_all[0:10])

if __name__ == "__main__":
	torch.backends.cudnn.benchmark = False
	# args:
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default='0')
	parser.add_argument('-b', '--batch_size', type=int, default=512)
	args = parser.parse_args()

	print(args)
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	
	# model list:
	# model_name_lst = ['vgg16bn', 
	# 	'resnet34', 'resnet101', 
	# 	'wrn-101-2', 'resnext101_32x4d', 'se_resnet101', 
	# 	'senet154']
	model_name_lst = [
		'nasnetalarge', 'pnasnet5large',
		'resnext101_32x48d_wsl' ]

	## Test
	for model_name in model_name_lst:
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

			print(str(type(model)))
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

		# measure model:
		# measure_model(model, args.img_size, args.img_size)

		print('testing %s on ImageNet ... ' % model_name)
		test_imagenet(model, model_name)
