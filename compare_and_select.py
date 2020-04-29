import numpy as np 
import os, sys, json, scipy
sys.path.append(os.path.join(os.path.expanduser('~'), 'MAD'))
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_ubyte
import pandas as pd
from stat import S_IREAD, S_IRGRP, S_IROTH

from utils.utils import trace_distance, measure_model, create_dir
from common_flags import COMMON_FLAGS
json_dir = COMMON_FLAGS.json_dir
dataset_dir = COMMON_FLAGS.dataset_dir
data_download_dir = COMMON_FLAGS.data_download_dir
prediction_dir = COMMON_FLAGS.prediction_dir
compare_dir = COMMON_FLAGS.compare_dir
select_dir = COMMON_FLAGS.select_dir

def compare(model_name1, model_name2, distance_mode = 'min', weighted=True):
	''' Compare the difference between two models predictions.
	Save results in .txt file and comparison details in a .csv file.
	Filter out the hard examples based these two criterias:
	1. The wordnet tree distance is large;
	2. One or both of them are confidence in their predictions.
	'''
	# create_dir(compare_dir)
	if weighted:
		tree_distance_mode = 'weighted_' + distance_mode
	else:
		tree_distance_mode = 'unweighted_' + distance_mode
	create_dir(os.path.join(compare_dir, tree_distance_mode))

	preds1 = np.load(os.path.join(prediction_dir, '%s_preds.npy' % model_name1))
	preds2 = np.load(os.path.join(prediction_dir, '%s_preds.npy' % model_name2))
	probs1 = np.load(os.path.join(prediction_dir, '%s_probs.npy' % model_name1))
	probs2 = np.load(os.path.join(prediction_dir, '%s_probs.npy' % model_name2))

	labels = np.loadtxt(COMMON_FLAGS.label_path)

	# make the prediction readable:
	with open(os.path.join(json_dir, 'imgnetid2word.json')) as fp:
		imgnetid2word = json.load(fp)
	words1 = []
	for imgnetid in preds1:
		word = imgnetid2word[str(int(imgnetid))]
		words1.append(word)
	words1 = np.array(words1)

	words2 = []
	for imgnetid in preds2:
		word = imgnetid2word[str(int(imgnetid))]
		words2.append(word)
	words2 = np.array(words2)

	# find disagree index:
	disagree_idx = np.argwhere(preds1!=preds2).squeeze()
	print('disagree_idx:', disagree_idx.shape, disagree_idx[0:20])
	# build data frame:
	prediction_df = pd.DataFrame({'disagree_idx': disagree_idx.astype(int), 
		'preds1': preds1[disagree_idx].astype(int), 
		'words1': words1[disagree_idx], 
		'probs1': np.amax(probs1[disagree_idx], axis=1), 
		'preds2': preds2[disagree_idx].astype(int), 
		'words2': words2[disagree_idx], 
		'probs2': np.amax(probs2[disagree_idx], axis=1),
		'gnd': labels[disagree_idx].astype(int)},
		columns=['disagree_idx', 'preds1', 'words1', 'probs1', 'preds2', 'words2', 'probs2', 'gnd']
		)

	# load json:
	with open(os.path.join(json_dir, 'code_with_imgnet_id.json'), 'r') as fp:
		code_with_imgnet_id = json.load(fp)

	# get tree distance and confident number:
	tree_dist_lst = []
	for index, row in prediction_df.iterrows():
		# trace label:
		int_label1 = int(row['preds1'])
		trace_label_lst1 = code_with_imgnet_id[str(int_label1)] # list, of a single or multiple string elements
		int_label2 = int(row['preds2'])
		trace_label_lst2 = code_with_imgnet_id[str(int_label2)]
		# all possible paths from leaf1 to leaf2:
		all_path_dist = []
		for trace_label1 in trace_label_lst1:
			trace_label1 = trace_label1.split(' ')
			for trace_label2 in trace_label_lst2:
				trace_label2 = trace_label2.split(' ')
				# tree dist:
				path_dist = trace_distance(trace_label1, trace_label2, weighted=weighted)
				all_path_dist.append(path_dist)
		# There maybe multiple paths from leaf1 to leaf2. Each path may have different length.
		if distance_mode == 'ave':
			tree_dist = np.mean(all_path_dist)
		elif distance_mode == 'min':
			tree_dist = np.min(all_path_dist)
		else:
			raise Exception('wrong distance_mode %s' % distance_mode)
		
		# append this pair of different predictions to the list:
		tree_dist_lst.append(tree_dist)

	# add new column to the df:
	prediction_df['tree_dist'] = tree_dist_lst

	

	# save csv:
	prediction_df.to_csv(os.path.join(compare_dir, tree_distance_mode, "disagree_%s_%s.csv" % (model_name1, model_name2)), index=False)
	
	# save txt
	result_str = 'disagree_num_total: %d, total: %d' % (len(disagree_idx), preds1.shape[0])
	print(result_str)
	f = open(os.path.join(compare_dir, tree_distance_mode, "compare_result_%s_%s.txt" % (model_name1, model_name2)), "w+")
	f.write(result_str)
	f.close()

def select(model_name1, model_name2, distance_mode = 'min', weighted=True, probs_th = 0.8):
	if weighted:
		tree_distance_mode = 'weighted_' + distance_mode
	else:
		tree_distance_mode = 'unweighted_' + distance_mode
	select_dir = os.path.join(COMMON_FLAGS.select_dir, tree_distance_mode, '%s_vs_%s' % (model_name1, model_name2))
	create_dir(select_dir)
	# load csv:
	prediction_df = pd.read_csv(os.path.join(compare_dir, tree_distance_mode, "disagree_%s_%s.csv" % (model_name1, model_name2)))

	# filter all disagreed images by probs:
	probs_th = probs_th
	conditioned_lines = prediction_df.loc[prediction_df['probs1'] >= probs_th] # prob condition
	conditioned_lines = conditioned_lines.loc[prediction_df['probs2'] >= probs_th] # prob condition
	print('conditioned_lines:', type(conditioned_lines))

	# sort by tree distance:
	sorted_lines = conditioned_lines.sort_values(by=['tree_dist'], ascending=False) 
	# save selected lines:
	sorted_lines.to_csv(os.path.join(select_dir, "%s_sorted_conditioned_disagree_%s_%s.csv" % (tree_distance_mode, model_name1, model_name2)), index=False)
	print('sorted_lines:', sorted_lines.shape)

	# select 50 to plot:
	img_idx_to_plot, preds1_to_plot, preds2_to_plot, crawl_keys_to_plot = [], [], [], []
	selected_num = 0
	pred_pairs = [] # keep track of which pairs of different predictions have already occured. Each element is a set with two elements.
	occurance_num = {} # keep truck of how many times a class has been predicted.
	for i, row in sorted_lines.iterrows():
		img_idx = int(row['disagree_idx'])
		pred1, pred2 = int(row['preds1']), int(row['preds2'])
		crawl_keys = int(row['gnd'])

		pred_pair = {pred1, pred2}
		# keep track of occurance_num:
		for pred in pred_pair:
			if pred not in occurance_num.keys():
				occurance_num[pred] = 1
			else:
				occurance_num[pred] += 1
		# whether to select this image:
		if pred1 not in occurance_num.keys() or occurance_num[pred1] <= 3: # condition 1
			if pred2 not in occurance_num.keys() or occurance_num[pred2] <= 3: # condition 2
				if pred_pair not in pred_pairs: # condition 3
					img_idx_to_plot.append(img_idx)
					preds1_to_plot.append(pred1)
					preds2_to_plot.append(pred2)
					crawl_keys_to_plot.append(crawl_keys)
					# update selected_num and pred_pairs:
					selected_num += 1
					pred_pairs.append(pred_pair)
		if selected_num >= 60: # break condition
			break
	# convert list to ndarray:
	img_idx_to_plot, preds1_to_plot, preds2_to_plot, crawl_keys_to_plot = \
		np.array(img_idx_to_plot), np.array(preds1_to_plot), np.array(preds2_to_plot), np.array(crawl_keys_to_plot)
	assert len(img_idx_to_plot) == 60

	# plot big img:
	if True:
		img_lst = []
		for idx in img_idx_to_plot:
			img_name = os.path.join(dataset_dir, str(idx) + '.jpg')
			img = resize(imread(img_name), (244,244,3))
			img_lst.append(img)

		img_big = np.concatenate(img_lst, axis=1)
		img_lst = np.split(img_big, 6, axis=1)
		img_big = np.concatenate(img_lst, axis=0)

		print('img_big:', img_big.shape)
		imsave(os.path.join(select_dir, "disagree_%s_%s.png" % (model_name1, model_name2)), img_as_ubyte(img_big))

	# save pred results in csv:
	if True:
		with open(os.path.join(json_dir, 'imgnetid2word.json'), 'r') as fp:
			imgnetid2word = json.load(fp)
		preds1_word_to_plot = np.array([imgnetid2word[key] for key in preds1_to_plot.astype(str)]).reshape((6,10))
		preds2_word_to_plot = np.array([imgnetid2word[key] for key in preds2_to_plot.astype(str)]).reshape((6,10))
		crawl_keys_word_to_plot = np.array([imgnetid2word[key] for key in crawl_keys_to_plot.astype(str)]).reshape((6,10))
		print('preds1_word_to_plot:', preds1_word_to_plot.shape)
		print('preds2_word_to_plot:', preds2_word_to_plot.shape)
		pd.DataFrame(preds1_word_to_plot).to_csv(os.path.join(select_dir, '%s_word_to_plot.csv' % model_name1), header=False, index=False)
		pd.DataFrame(preds2_word_to_plot).to_csv(os.path.join(select_dir, '%s_word_to_plot.csv' % model_name2), header=False, index=False)
		pd.DataFrame(crawl_keys_word_to_plot).to_csv(os.path.join(select_dir, 'crawl_keys_word_to_plot.csv'), header=False, index=False)
		pd.DataFrame(img_idx_to_plot).to_csv(os.path.join(select_dir, 'img_idx_to_plot.csv'), header=False, index=False)

		# os.chmod(os.path.join(select_dir, '%s_word_to_plot.csv' % model_name1), S_IREAD|S_IRGRP|S_IROTH)
		# os.chmod(os.path.join(select_dir, '%s_word_to_plot.csv' % model_name2), S_IREAD|S_IRGRP|S_IROTH)
		# os.chmod(os.path.join(select_dir, 'crawl_keys_word_to_plot.csv'), S_IREAD|S_IRGRP|S_IROTH)
		# os.chmod(os.path.join(select_dir, 'img_idx_to_plot.csv'), S_IREAD|S_IRGRP|S_IROTH)

if __name__ == "__main__":

	# ## demo:
	# weighted=True
	# distance_mode='min'
	# model_name1, model_name2 = 'vgg16bn', 'resnet34'
	# # model_name1, model_name2 = 'resnet101', 'wrn-101-2'
	# # model_name1, model_name2 = 'resnet101', 'resnext101_32x4d'
	# # model_name1, model_name2 = 'resnet101', 'se_resnet101'
	# # model_name1, model_name2 = 'resnext101_32x4d', 'se_resnet101'
	# # model_name1, model_name2 = 'wrn-101-2', 'se_resnet101'
	# # model_name1, model_name2 = 'wrn-101-2', 'resnext101_32x4d'
	# compare(model_name1, model_name2, distance_mode=distance_mode, weighted=weighted)
	# select(model_name1, model_name2, distance_mode=distance_mode, weighted=weighted)


	# real:
	model_name_lst = ['vgg16bn',  # 0:1
		'resnet34', 'resnet101', 'resnet152', # 1:4
		'wrn-101-2', 'resnext101_32x4d', 'se_resnet101', # 4:7
		'senet154', # 7:8
		'nasnetalarge', 'pnasnet5large', # 8:10
		'resnext101_32x48d_wsl', 'effnetE7'] # 10:12
	c = 0
	model_num = len(model_name_lst)
	for i in range(0,model_num):
		model_name1 = model_name_lst[i]
		for j in range(i+1,model_num):
			model_name2 = model_name_lst[j]

			# Compare
			compare(model_name1, model_name2, distance_mode = 'min', weighted=True)

			# select
			select(model_name1, model_name2, distance_mode = 'min', weighted=True)

			c += 1

	assert c == scipy.misc.comb(model_num, 2)
