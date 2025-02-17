import torch
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
class SpkEmbeddingFinder(object):
    def __init__(self,spk_dict_path): 
        if spk_dict_path is not None and os.path.exists(f"{spk_dict_path}/spk2ind.pkl") and os.path.exists(f"{spk_dict_path}/spk_vecs.npy"):
            self.dvec_mat = np.load(f"{spk_dict_path}/spk_vecs.npy")
            with open(f"{spk_dict_path}/spk2ind.pkl", 'rb') as f:
                self.spk2ind = pickle.load(f)
            self.is_map2spk = True
            self.map2spk_style ='fix'
        else:
            self.is_map2spk = False  

    def get_cosine_match_dvec(self, dvec):
        cos_sim = cosine_similarity(dvec.cpu(), self.dvec_mat).squeeze()
        dvec = self.dvec_mat[cos_sim.argmax()]
        return dvec[None, :]    
    
    @torch.no_grad()
    def find_spk_embedding(self,spk,device="cpu",prefix="libritts_"):
        if not isinstance(spk,str):
            spk = self.id2spk_name[int(spk)]
        if not spk.startswith(prefix):
            spk = prefix+spk
        try:
            dvec_ind = self.spk2ind[spk]
            dvec = self.dvec_mat[dvec_ind][None, :]
        except:
            print(f"fix mode: not find spk {spk} in spk2ind, turn to auto style.")
            dvec = self.get_cosine_match_dvec(dvec)
        dvec = torch.from_numpy(dvec).to(device).to(torch.float32)
        dvec = dvec / torch.norm(dvec)
        return dvec