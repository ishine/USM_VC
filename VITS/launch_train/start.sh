set -e
# your project dir
proj_dir="/home/project/VITS"
export PYTHONPATH=${proj_dir}:$PYTHONPATH
log_root="/home/project/VITS/exp_logs/"




#pip install pyworld
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1
echo "Train model..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  
############  MLF/USM/USM_{*} BASED ON HUBERT  ###############
python ${proj_dir}/train_usm_hubert.py \
  -c ${proj_dir}/configs/train_vctk_hubert_usm.json \
  -l ${log_root} \
  -m usm_hubert


############  soft unit BASED ON HUBERT  ###############
python ${proj_dir}/train_soft_hubert.py \
  -c ${proj_dir}/configs/train_vctk_hubert_soft.json \
  -l ${log_root} \
  -m soft_hubert  

############  BNF/USM/USM_{*} BASED ON PPG  ###############
python ${proj_dir}/train_usm_ppg.py \
  -c ${proj_dir}/configs/train_vctk_ppg_usm.json \
  -l ${log_root} \
  -m usm_ppg


############  soft unit BASED ON PPG  ###############
python ${proj_dir}/train_soft_ppg.py \
  -c ${proj_dir}/configs/train_vctk_ppg_soft.json \
  -l ${log_root} \
  -m soft_ppg     

