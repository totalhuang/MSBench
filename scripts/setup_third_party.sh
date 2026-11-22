#!/usr/bin/env bash
# Clone the upstream repositories MSBench calls into ./third_party (or $MSB_THIRD_PARTY).
# Run with the main environment active; Q-Align is installed separately into its own environment (see README).
set -euo pipefail
TP="${MSB_THIRD_PARTY:-$(cd "$(dirname "$0")/.." && pwd)/third_party}"
mkdir -p "$TP" && cd "$TP"
clone() { [ -d "$2" ] || { git clone "$1" "$2" && git -C "$2" checkout -q "$3"; }; }  # tested commits
clone https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git OmniShotCut 4f8033cbf352e3ce183d32567b51b41250105b74
clone https://github.com/heygen-com/TransVLM.git TransVLM 21cf65a3663a8196d6824b296434ee54bd5687d2
clone https://github.com/MingXiangL/DEVIL.git DEVIL 48556c329e6a6f92a587021bc97022494faec69c
clone https://github.com/facebookresearch/dinov2.git dinov2 7764ea0f912e53c92e82eb78a2a1631e92725fc8
clone https://github.com/Neo-yk/FilmOps.git FilmOps ad60f5b0a9481d3c260793d190e9068058b4219e
clone https://github.com/Q-Future/Q-Align.git Q-Align fd0c3def92da21cd124f8301722b6b83bb75eccd
pip install -e OmniShotCut
pip install -e FilmOps
echo "third-party code in $TP"
echo "Q-Align: activate the Q-Align environment and run  pip install --no-deps -e $TP/Q-Align"
