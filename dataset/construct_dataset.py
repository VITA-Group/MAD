'''
1. Merge all downloader images into one folder;
2. Store the corresponding labels (keys used to crawl this image from Internet) in one file.
'''
# mindate='2019-01-01'
# maxdate='2019-08-01'  
dates = ['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', 
        '2019-01-01', '2019-08-01']
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, sys, json
sys.path.append(os.path.join(os.path.expanduser('~'), 'MAD'))
from shutil import copy2
import numpy as np
from skimage.io import imread

from common_flags import COMMON_FLAGS
json_dir = COMMON_FLAGS.json_dir
dataset_dir = COMMON_FLAGS.dataset_dir

with open(os.path.join(json_dir, 'selected_keywords.json'), 'r') as fp:
    selected_keywords_ordered = json.load(fp)
    selected_keywords_ordered.sort()
print('selected_keywords_ordered:', selected_keywords_ordered, type(selected_keywords_ordered))

with open(os.path.join(json_dir, 'invert_imagenet_labels.json'), 'r') as fp:
    invert_imagenet_labels = json.load(fp)

c = 0 # total count of images
# merge_folder_name = dataset_dir + '-%s-%s' % (mindate,maxdate)
merge_folder_name = dataset_dir
if not os.path.isdir(merge_folder_name):
    os.makedirs(merge_folder_name)
labels = []
for i, keyword in enumerate(selected_keywords_ordered):
    for k in range(len(dates)-1):
        mindate, maxdate = dates[k], dates[k+1]
        data_download_dir = COMMON_FLAGS.data_download_dir + '-%s-%s' % (mindate,maxdate)
        folder_name = os.path.join(data_download_dir, keyword)
        file_list = os.listdir(folder_name)
        if len(file_list) == 0: # empty class: no images of this class are crawled from Internet.
            pass
        for filename in file_list:
            img = imread(os.path.join(folder_name, filename))
            if len(img.shape) == 3: # only consider 3 channel images.
                copy2(os.path.join(folder_name, filename), os.path.join(merge_folder_name, '%d.jpg' % c))
                labels.append(invert_imagenet_labels[keyword])
                print(c)
                c += 1


# save labels 
np.savetxt(os.path.join('labels.txt'), labels, fmt='%s')

print('c:', c)
assert len(labels) == c