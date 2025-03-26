# Infer
Please download VITS weights and config from https://huggingface.co/anonymous-VC-Demo/USM_VC_Models  
Then move them to exp_logs/  
```
######## prepare data #############
bash scripts/0_prepare_data.sh

#######  train model ############
bash launch_train/start.sh


###### infer model trained using hubert feature #############
bash 0_infer_hubert_vctk.sh


###### infer model trained using BNF feature #############
bash 0_infer_ppg_vctk.sh

```

# Reference
The model architecture of VITS model: 
https://github.com/RVC-Project/Retrieval-based-Voice-
Conversion-WebUI