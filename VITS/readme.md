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