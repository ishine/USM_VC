data_dir=data/vctk
wav_dir=data/vctk/wav24
content_type=hubert
#ppg or hubert
ppg_dir=${data_dir}/${content_type} 
fid2spk=doc/fid2spk
fid2style=doc/fid2style
speaker_map=doc/speaker_map_vctk
style_map=doc/style_map

### extract content feature
echo "Extract content feature ..."
python3 extract_hubert_feature.py
## NOTE: extract content feature from PPG model, you can refer to the infer process of the diffusion model, the same PPG model is used for VITS.




### extract linear spectrogram
echo "Compute linear spectrogram ..."
python3 compute_spec.py \
  --wav_dir ${wav_dir} \
  --output_dir ${data_dir}/spec \
  --nj 30 \
  --config configs/train_vctk_hubert_usm.json


### extract f0
# echo "Compute F0 ..."
# python3 ensemble_f0_detector.py \
#   --wav_dir ${wav_dir} \
#   --output_dir ${data_dir}/f0 \
#   --frame_period_ms 10 \
#   --sampling_rate 24000 \
#   --f0_floor 50 \
#   --f0_ceil 1100 \
#   --num_workers 10

### split data to training and validation sets; then generate corresponding meta lists
echo "Split data into train-validation sets..."
python3 split_data.py \
  --wav_dir ${wav_dir} \
  --spec_dir ${data_dir}/spec \
  --ppg_dir ${ppg_dir} \
  --fid2spk ${fid2spk} \
  --fid2style ${fid2style} \
  --f0_dir ${data_dir}/f0 \
  --speaker_map ${speaker_map} \
  --style_map ${style_map} \
  --val_size 30 \
  --output_dir ${data_dir}/filelists_${content_type}

### Compute global semantic dict or speaker-dependent semantic dict
echo "Compute dict for hubert base model..."
python3 comp_hubert_center_global.py
python3 comp_hubert_center_spk.py

echo "Compute dict for PPG model..."
python3 comp_ppg_center_global.py
python3 comp_ppg_center_spk.py

echo "Successfully finish feature preperation! Please go to the model training stage."
