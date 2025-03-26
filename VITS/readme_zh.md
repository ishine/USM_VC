# 推理流程
请从https://huggingface.co/anonymous-VC-Demo/USM_VC_Models下载VITS权重和config  
然后移动至exp_logs/  
```
######## 准备数据 #############
bash scripts/0_prepare_data.sh

####### 训练模型 ############
bash launch_train/start.sh

#使用HuBERT特征训练的模型推理
bash 0_infer_hubert_vctk.sh

#使用BNF特征训练的模型推理
bash 0_infer_ppg_vctk.sh

```

# 引用
The model architecture of VITS model: 
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI