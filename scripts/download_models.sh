#!/usr/bin/env bash
# Download the expert-model weights into ./models (or $MSB_MODELS). About 80 GB in total.
# Needs `pip install -U huggingface_hub` (for `hf`) and wget.
set -euo pipefail
M="${MSB_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}"
mkdir -p "$M" && cd "$M"
hf download uva-cv-lab/OmniShotCut_v1.5 --local-dir OmniShotCut                  # transitions: hard cuts
hf download HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct --local-dir TransVLM          # transitions: gradual intervals
hf download q-future/one-align --local-dir Q-Align                                # Aesthetic Score
hf download zhiyuanyou/DeQA-Score-Mix3 --local-dir DeQA-Score-Mix3                # Perceptual Score
hf download Neo961/FilmOps --local-dir FilmOps                                    # Camera Control
hf download public-data/insightface --include "models/buffalo_l/*" --local-dir insightface   # face consistency
mkdir -p DEVIL                                                                    # Dynamic Degree
wget -nc -P DEVIL https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
wget -nc -P DEVIL https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
hf download OpenGVLab/ViCLIP ViClip-InternVid-10M-FLT.pth --local-dir DEVIL
echo "RAFT: put raft-things.pth (princeton-vl/RAFT, download_models.sh) into $M/DEVIL/"
