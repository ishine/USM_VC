
export CUDA_VISIBLE_DEVICES=0

python3 vc_infer_e2e.py \
    -p data/vctk/ppg \
    -c exp_logs/usm_ppg/G.pth \
    -o out/ppg/usm \
    --speaker_map  data/vctk/filelists_ppg/speaker.map \
    --trg_speaker p228




python3 vc_infer_e2e.py \
    -p data/vctk/ppg \
    -c exp_logs/baseline_ppg/G.pth \
    -o out/ppg/base \
    --speaker_map  data/vctk/filelists_ppg/speaker.map \
    --trg_speaker p228



