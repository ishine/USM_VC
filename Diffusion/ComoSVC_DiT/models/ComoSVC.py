import os
import torch
import torch.nn as nn
import yaml

from ComoSVC_DiT.models.como import Como
# copied from https://github.com/jaywalnut310/vits/blob/main/commons.py#L121
def sequence_mask(length: torch.Tensor, max_length: int = None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
def load_model_vocoder(
        model_path,
        device='cpu',
        config_path = None,
        total_steps=1,
        teacher=False
        ):
    if config_path is None:
        config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    else:
        config_file = config_path

    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # # load vocoder
    # vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model
    model = ComoSVC(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                total_steps,
                teacher      
                )
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'],strict=False)
    model.eval()
    return model, vocoder, args


class ComoSVC(nn.Module):
    def __init__(
            self,
            input_channel,
            speak_embedding_dim,
            #use_pitch_aug=True,
            # out_dims=128, # define in como
            # n_layers=20, 
            # n_chans=384, 
            # n_hidden=100,
            total_steps=50,
            teacher=True,
            dit_config=None
            ):
        super().__init__()
        n_hidden = dit_config['mel_proj_channels']+dit_config['ppg_proj_channels']
        self.unit_embed = nn.Linear(input_channel, dit_config['ppg_proj_channels'])
        #self.f0_embed = nn.Linear(1, n_hidden)
        self.teacher=teacher
        self.spk_proj = nn.Linear(speak_embedding_dim,n_hidden)
   
#        self.input_porj = nn.Linear(input_channel+speak_embedding_dim, n_hidden)
        # if use_pitch_aug:
        #     self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        # else:
        #     self.aug_shift_embed = None
        # self.n_spk = n_spk
        # if n_spk is not None and n_spk > 1:
        #     self.spk_embed = nn.Embedding(n_spk, n_hidden)
        more_dit_config = {
            "hidden_channels": dit_config['mel_proj_channels']+dit_config['ppg_proj_channels'],
            "out_channels": dit_config['mel_channels'],
            "filter_channels": dit_config['filter_channels'],
            "n_heads": dit_config['n_heads'],
            "n_layers": dit_config['n_dec_layers'],
            "kernel_size": dit_config['kernel_size'],
            "dropout": dit_config['p_dropout'],
            "gin_channels": dit_config['mel_proj_channels']+dit_config['ppg_proj_channels'],#condition的channel
            "mel_proj": True,
            "mel_input_channels":dit_config['mel_channels'],
            "mel_proj_channels": dit_config['mel_proj_channels'],
            
        }
        
        self.n_hidden = n_hidden
        self.decoder = Como(total_steps, teacher,dit_config=more_dit_config) 
        self.input_channel = input_channel
        

    def forward(self, x,  x_lens,gt_spec,gt_spec_lens, dvec,
                infer=True,sample_steps=30):
          
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
            
        forward: x : B,T,256
            
        '''
        #import ipdb; ipdb.set_trace()
        # x : B,T,256
        # devc: B,256
        #import ipdb; ipdb.set_trace()
        gt_spec_lens = gt_spec_lens.int()
        gt_spec_mask = sequence_mask(gt_spec_lens, gt_spec.size(1)).unsqueeze(1).to(gt_spec.dtype)
        x = self.unit_embed(x) 
        #import ipdb ; ipdb.set_trace()
        #f0 = self.f0_embed((1+ f0 / 700).log())  
        dvec = self.spk_proj(dvec).unsqueeze(1)
        condition = dvec.repeat(1, x.size(1), 1)
        # if self.n_spk is not None and self.n_spk > 1:
        #     #import ipdb ; ipdb.set_trace()
        #     x = x + self.spk_embed(spk_id).unsqueeze(1) #torch.Size([1, 740, 256])+torch.Size([1, 1, 256])


        # B,T,_ = x.shape
        # devc = devc.unsqueeze(1).repeat(1, T, 1)
        # x = torch.cat((x, devc), dim=-1) # B,T,256+256
        
        #x = self.unit_embed(units) + self.f0_embed((1+ f0 / 700).log()) + self.volume_embed(volume)
        # x = self.input_porj(x)
        # if self.n_spk is not None and self.n_spk > 1:
        #     if spk_id.shape[1] > 1:
        #         g = spk_id.reshape((spk_id.shape[0], spk_id.shape[1], 1, 1, 1))  # [N, S, B, 1, 1]
        #         g = g * self.speaker_map  # [N, S, B, 1, H]
        #         g = torch.sum(g, dim=1) # [N, 1, B, 1, H]
        #         g = g.transpose(0, -1).transpose(0, -2).squeeze(0) # [B, H, N]
        #         x = x + g
        #     else:
        #         x = x + self.spk_embed(spk_id)

        # if self.aug_shift_embed is not None and aug_shift is not None:
        #     x = x + self.aug_shift_embed(aug_shift / 5) 
        
        if not infer:
            output  = self.decoder(gt_spec,gt_spec_mask,x,condition,infer=False)       
        else:
            output = self.decoder(gt_spec,gt_spec_mask,x,condition,infer=True,sample_steps=sample_steps)

        return output

