#!/usr/bin/env bash
# Download the expert-model weights into ./models (or $MSB_MODELS). About 78 GB in total.
# Needs `pip install -U huggingface_hub` (for `hf`), wget and unzip.
# The last block also fills the user caches with three small files that upstream code otherwise fetches on first use,
# so evaluation can then run on a node without internet access.
set -euo pipefail
M="${MSB_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}"
mkdir -p "$M" && cd "$M"
hf download uva-cv-lab/OmniShotCut_v1.5 --local-dir OmniShotCut                  # transitions: hard cuts
hf download HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct --local-dir TransVLM          # transitions: gradual intervals
hf download q-future/one-align --local-dir Q-Align                                # Aesthetic Score
hf download zhiyuanyou/DeQA-Score-Mix3 --exclude "pytorch_model*.bin" "pytorch_model.bin.index.json" --local-dir DeQA-Score-Mix3  # Perceptual Score (safetensors only)
hf download Neo961/FilmOps --include "shot_scale/*" "camera_angle/*" "camera_movement/*" --local-dir FilmOps  # Camera Control (required operators only)
mkdir -p FilmOps/shot_scale
wget -nc -P FilmOps/shot_scale https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
hf download public-data/insightface --include "models/buffalo_l/*" --local-dir insightface   # face consistency
mkdir -p DEVIL                                                                    # Dynamic Degree
wget -nc -P DEVIL https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
wget -nc -P DEVIL https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
hf download OpenGVLab/VBench_Used_Models ViClip-InternVid-10M-FLT.pth --local-dir DEVIL
if [ ! -f DEVIL/raft-things.pth ]; then                                          # RAFT optical flow (princeton-vl/RAFT)
  wget -O DEVIL/raft_models.zip https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
  unzip -j -o DEVIL/raft_models.zip models/raft-things.pth -d DEVIL && rm DEVIL/raft_models.zip
fi

# Fetched implicitly by upstream code on first use; cache them now for offline nodes.
hf download q-future/one-align --include "*.json" "*.model" "*.py" > /dev/null      # Q-Align: tokenizer id hardcoded in modeling_mplug_owl2.py
hf download Study-is-happy/neuflow-v2 > /dev/null                                   # TransVLM: NeuFlow optical flow
T="${TORCH_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/torch}/hub/checkpoints"            # OmniShotCut: torchvision ResNet-18
mkdir -p "$T" && wget -nc -P "$T" https://download.pytorch.org/models/resnet18-f37072fd.pth
