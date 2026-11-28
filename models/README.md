Expert-model weights (`bash scripts/download_models.sh`). Expected layout:

    models/
      OmniShotCut/OmniShotCut_ckpt.pth
      TransVLM/                     HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct
      Q-Align/                      q-future/one-align
      DeQA-Score-Mix3/              zhiyuanyou/DeQA-Score-Mix3 (safetensors only)
      FilmOps/                      Neo961/FilmOps (shot_scale, camera_angle, camera_movement) + shot_scale/dinov2_vitb14_pretrain.pth
      insightface/models/buffalo_l/
      DEVIL/raft-things.pth, dino_deitsmall16_pretrain.pth, dinov2_vitl14_pretrain.pth, ViClip-InternVid-10M-FLT.pth

Also cached by the script (fetched by upstream code on first use): Hugging Face `q-future/one-align` tokenizer files and
`Study-is-happy/neuflow-v2` in `~/.cache/huggingface/hub`, torchvision `resnet18-f37072fd.pth` in `~/.cache/torch/hub/checkpoints`.
