


export CUDA_VISIBLE_DEVICES=0

python3 vc_infer_e2e.py \
    -p data/vctk/hubert \
    -c exp_logs/baseline_hubert/G.pth \
    -o out/hubert/base \
    --speaker_map  data/vctk/filelists_hubert/speaker.map \
    --is_ssl \
    --trg_speaker P228






python3 vc_infer_e2e.py \
    -p data/vctk/hubert \
    -c exp_logs/usm_hubert/G.pth \
    -o out/hubert/usm \
    --speaker_map  data/vctk/filelists_hubert/speaker.map \
    --is_ssl \
    --trg_speaker P228




