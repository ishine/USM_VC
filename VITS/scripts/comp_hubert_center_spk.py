import numpy as np
from collections import defaultdict
import h5py
import os
import pickle
from scipy.special import softmax

ppg_path = 'data/vctk/hubert'

fid2spk_file  = 'doc/fid2spk'
spk_map_file = 'doc/speaker_map_vctk'
out_path = 'data/semantic_dict/hubert/'
os.makedirs(out_path, exist_ok=True)

spkid2utts = defaultdict(list)
spk2spkid = {line.split(': ')[0]: int(line.split()[1]) for line in open(spk_map_file,'r')}
for line in open(fid2spk_file):
    terms = line.strip().split()
    spk_id = spk2spkid[terms[1]]
    spkid2utts[spk_id].append(f"{terms[0]}.npy")
######## load linear layer's paras in ce loss ###########    
f_read = open('data/semantic_dict/hubert/ce_layer_para.pkl', 'rb')
ce_para = pickle.load(f_read)
f_read.close()
dict_size, bn_dim = ce_para['w2'].shape
ppg_dim = 768
#import ipdb; ipdb.set_trace()
# weight of linear layer
w1 = ce_para['w1']
# bias of linear layer
b1 = ce_para['b1']

w2 = ce_para['w2']
# bias of linear layer
b2 = ce_para['b2']

#########################################################

for spk_id in spkid2utts.keys():
    if os.path.exists(f"{out_path}/speaker-dependent-dict/phn_center_{spk_id}.npy"):
        continue
    # zero statistic
    n0 = np.zeros(dict_size)
    # first statistic
    n1 = np.zeros([dict_size, ppg_dim])
    num_utt = 0
    for i, fn in enumerate(spkid2utts[spk_id]):
        # try:
        ppg = np.load(f"{ppg_path}/{fn}")
        # except:
        #     continue    
        ln1 = ppg @ w1.T + b1
        ln1_relu = np.maximum(0, ln1)
        logit = ln1_relu @ w2.T + b2
        logit_soft = softmax(logit, axis=1)#T,4096
        maxv = np.max(logit_soft, axis=1)
        maxi = np.argmax(logit_soft, axis=1)
        n1 += logit_soft.T @ ppg #4096, 768
        n0 += np.sum(logit_soft, axis=0)
        num_utt += 1
        #import ipdb; ipdb.set_trace()
    print(f"processed spk {spk_id} with {num_utt} ppg files")
    #import ipdb; ipdb.set_trace()
    phn_center = n1/n0[:,None]
    np.save(f"{out_path}/speaker-dependent-dict/phn_center_{spk_id}.npy", phn_center)
    # np.save(f"{out_path}/speaker-dependent-dict/n0_{spk_id}.npy", n0)

