# ComoSVC DiT
使用DiT作为backbone的ComoSVC

## 安装
请从https://huggingface.co/anonymous-VC-Demo/USM_VC_Models下载权重和config  
- 下载vocoder权重和config并放置于pretrained_models/vocoder下. 
- 下载ppg model权重和config并放置于pretrained_models/ppg_model下. 
- 下载ComoSVC DiT权重和config并放置于pretrained_models/ComoSVC_DiT下.
```python
conda create --name comosvc_dit python=3.10
conda activate comosvc_dit
pip install torch==2.0.0 torchaudio==2.0.1
pip install -r requirements.txt
```

## 推理

```python
export PYTHONPATH="${PWD}:${PWD}/ppg:$PYTHONPATH"
python infer.py \
    --ckpt pretrained_models/ComoSVC_DiT/libritts_bnf/model.checkpoint \
    --source_audio example/example.wav \
    --target_spk_name libritts_4948 \
    --output_path example/output.wav

```

## 引用
该项目的实现参考了：  
[StableTTS](https://github.com/KdaiP/StableTTS/tree/main): DiT模型的代码  
[CoMoSVC](https://github.com/Grace9994/CoMoSVC): CoMoSVC的算法实现  