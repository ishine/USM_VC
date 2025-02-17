import numpy as np
# from collections import defaultdict
# import h5py
import os
import pickle
import glob
import random
from scipy.special import softmax

def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.npy'):
                file_list.append(f'{os.path.join(root, file)}')
    return file_list

ppg_path = 'data/libritts/hubert'
f_read = open('data/semantic_dict/hubert/ce_layer_para.pkl', 'rb')
outdir = 'data/semantic_dict/hubert'
os.makedirs(outdir,exist_ok=True)
ce_para = pickle.load(f_read)
f_read.close()
dict_size, bn_dim = ce_para['w2'].shape
# weight of linear layer
w1 = ce_para['w1']
# bias of linear layer
b1 = ce_para['b1']

w2 = ce_para['w2']
# bias of linear layer
b2 = ce_para['b2']
#########################################################
# zero statistic
n0 = np.zeros(dict_size)
# first statistic
n1 = np.zeros([dict_size, 768])

num_utt = 0
files = get_file_list(ppg_path)
random.shuffle(files)
for i, fn in enumerate(files):
    ppg = np.load(f"{fn}")
    ln1 = ppg @ w1.T + b1
    ln1_relu = np.maximum(0, ln1)
    logit = ln1_relu @ w2.T + b2
    logit_soft = softmax(logit, axis=1)#T,4096
    maxv = np.max(logit_soft, axis=1)
    maxi = np.argmax(logit_soft, axis=1)
    n1 += logit_soft.T @ ppg #4096, 768
    n0 += np.sum(logit_soft, axis=0)
    num_utt += 1
    if i % 100 == 0:
        print(f"processed num utts {i}")
    #import ipdb; ipdb.set_trace()
    if i >= 100000:
        print(f"processed num utts {100000}\n")
        break
#import ipdb; ipdb.set_trace()
print(f"{num_utt} feature files in total")
phn_center = n1/n0[:,None]
np.save(f'{outdir}/global_semantic_dict.npy', phn_center)
# np.save(f'{outdir}/n0.npy', n0)

