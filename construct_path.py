import pandas as pd
import numpy as np
import os, json, copy
from common_flags import COMMON_FLAGS
json_dir = 'class_info'

def construct_path():
    '''
    Constract and save a Python dict, called 'path_dict_imgnet_id'.
    keys in path_dict_imgnet_id: str, imgnetid; 
    values in path_dict_imgnet_id: list of strings. Each string is a possible path from leaf to root, containing wnids (str) seperated by space.
    '''
    # get all imagnet wnids from csv file:
    with open(os.path.join(json_dir, 'imgnetid2wnid.json'), 'r') as fp:
        imgnetid2wnid = json.load(fp)


    # get all father-son pairs from the word net structure file: 
    father_lst = []
    son_lst = []
    with open(os.path.join(json_dir, 'wnid_tree.txt'), 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        tmp = line.split(' ')
        father_lst.append(tmp[0])
        son_lst.append(tmp[1])
    print(len(father_lst), len(son_lst), son_lst[0:5])

    # construct father_son_pairs as dictionary: index with son
    father_son_pairs = {}
    for j, (father,son) in enumerate(zip(father_lst, son_lst)):
        if son not in father_son_pairs.keys():
            father_son_pairs[son] = [father]
        else:
            if father not in father_son_pairs[son]:
                # print('son %s has multiple fathers %s' % (son, father))
                father_son_pairs[son].append(father)
            else:
                print('duplicated father-son pair: %s-%s' % (father, son))
    with open(os.path.join(json_dir, 'father_son_pairs.json'), 'w+') as fp:
        json.dump(father_son_pairs, fp, indent=4)


    # initialize path as 1000 imagenet wnids:
    path_lst = [] # the 1st element is for imgnetid=1 class.
    for imgnetid in range(1000):
        wnid = imgnetid2wnid[str(imgnetid)]
        path_lst.append([wnid])
    print(path_lst[0:10])

    # prior knowledge
    root_wnid = "n00001740"

    # loop
    while True:
        path_need_tracking_num = 0
        # loop
        for i, path in enumerate(path_lst):
            father_num = 0
            top = path[-1]
            if top == root_wnid:
                continue
            else:
                path_need_tracking_num += 1
            
            fathers = father_son_pairs[top] # get all fathers using son as index
            father_num = len(fathers) # how many fathers this node have
            path.append(fathers[0]) # append the first father to the path
            if father_num > 1: # more than 1 father found
                for father in fathers[1:]:
                    path_new = copy.deepcopy(path)
                    path_new[-1] = father
                    path_lst.append(path_new) # add new path
            # print
            if i % 100 == 0:
                print('i:', i)

        # break while:
        if path_need_tracking_num == 0:
            break


    # save txt
    with open(os.path.join(json_dir, 'code_with_wnid.txt'), 'w+') as fp:
        for path in path_lst:
            fp.write(" ".join(path) + '\n')

    # construct dict:
    path_dict = {}
    for i, path in enumerate(path_lst): # only need to consider those with index >=1000
        leaf_id = path[0]
        path_str = " ".join(path)
        if i < 1000:
            path_dict[leaf_id] = [path_str]
        else:
            path_dict[leaf_id].append(path_str)

    # save dict as json:
    with open(os.path.join(json_dir, 'code_with_wnid.json'), 'w+') as fp:
        json.dump(path_dict, fp, indent=4)

    # load wnid2imgnetid:
    with open(os.path.join(json_dir, 'wnid2imgnetid.json'), 'r') as fp:
        wnid2imgnetid = json.load(fp)

    # use imagenet id as index:
    path_dict_imgnet_id = {}
    for key in path_dict:
        imgnet_id = wnid2imgnetid[key]
        path_dict_imgnet_id[imgnet_id] = path_dict[key]

    # save dict as json:
    with open(os.path.join(json_dir, 'code_with_imgnet_id.json'), 'w+') as fp:
        json.dump(path_dict_imgnet_id, fp, indent=4)

    ## use word in path: 
    # wnid indexing:
    with open(os.path.join(json_dir, 'wnid2word.json'), 'r') as fp:
        wnid2word = json.load(fp)
    path_dict_wnid_index_readable = {}
    path_dict_imgnetid_index_readable = {}
    for key in path_dict:
        imgnet_id = wnid2imgnetid[key]
        path_dict_wnid_index_readable[key] = [] # init
        path_dict_imgnetid_index_readable[imgnet_id] = [] # init
        path_lst = path_dict[key]
        for path in path_lst:
            wnid_lst = path.split(' ')
            word_lst = []
            for wnid in wnid_lst:
                word_lst.append(wnid2word[wnid])
            path_readable = '; '.join(word_lst)
            path_dict_wnid_index_readable[key].append(path_readable)
            path_dict_imgnetid_index_readable[imgnet_id].append(path_readable)
    with open(os.path.join(json_dir, 'code_with_wnid_readable.json'), 'w+') as fp:
        json.dump(path_dict_wnid_index_readable, fp, indent=4)
    with open(os.path.join(json_dir, 'code_with_imgnet_id_readable.json'), 'w+') as fp:
        json.dump(path_dict_imgnetid_index_readable, fp, indent=4)


if __name__ == '__main__':
    construct_path()