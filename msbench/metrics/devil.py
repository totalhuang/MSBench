"""Dynamic Degree: the seven raw DEVIL components per clean shot, combined by a frozen ridge model.

Components (DEVIL, MingXiangL/DEVIL), frames of the shot at <= 8 fps (``step = int(fps / 8)``):
  flow              mean RAFT optical-flow magnitude (frames resized to a 512 long side, 20 iterations)
  ssim              mean SSIM of consecutive 224x224 frames (sign flipped in the model: more change = more motion)
  phash             mean perceptual-hash distance of consecutive frames
  dino_segm_dist    DINOv2 ViT-L/14 autocorrelation distance over lags >= n/8
  viclip_segm_dist  1 - mean ViCLIP similarity of the four quarter segments
  info_dino         mean DINO ViT-S/16 distance of the frames to their mean feature
  temporal_entropy  DEVIL's compression-based temporal information of the shot clip
Shots shorter than 0.4 s, or with fewer than 3 sampled frames, are not scored.

Video score: equal-weight mean of each component over the scored shots, z-scored with the benchmark-wide statistics of
the paper (``configs/devil_ridge.json``: mu, sd over all 3,950 evaluated videos) and combined by the ridge weights
fitted on the human Dynamic Degree MOS (lambda = 1). The result is on the 1-5 MOS scale.
"""
import json
import os
import subprocess
import sys
import tempfile

from .. import common as C
from .visual import NA

COMPONENTS = ['flow', 'ssim', 'phash', 'dino_segm_dist', 'viclip_segm_dist', 'info_dino', 'temporal_entropy']
TARGET_FPS = 8.0
MIN_SHOT_S = 0.4


def ridge_params():
    p = json.loads((C.CONFIGS / 'devil_ridge.json').read_text())
    assert p['components'] == COMPONENTS
    return p


def dynamic_degree(components, params=None):
    """components: {name: equal-weight shot mean} -> Dynamic Degree on the 1-5 MOS scale (None if a component is missing)."""
    p = params or ridge_params()
    v = [components.get(c) for c in COMPONENTS]
    if any(x is None for x in v):
        return None
    return p['intercept'] + sum(w * sg * (x - m) / sd for w, sg, x, m, sd in zip(p['w'], p['sign'], v, p['mu'], p['sd']))


def frame_range(fps, a, b):
    import math
    return math.ceil(a * fps - 1e-8), math.ceil(b * fps - 1e-8)


def cut_clip(video_path, a, b, out):
    """Lossless FFV1 clip of [a, b] s."""
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '4', '-i', video_path, '-ss', str(a), '-t', str(b - a),
                    '-an', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', '-y', out], check=True)


class Devil:
    min_shot = MIN_SHOT_S

    def __init__(self, device='cuda'):
        import numpy as np
        import timm
        import torch
        from easydict import EasyDict
        devil = C.third_party_dir() / 'DEVIL'
        sys.path[:0] = [str(devil), str(devil / 'vbench' / 'third_party'), str(C.third_party_dir() / 'dinov2')]
        from RAFT.core.raft import RAFT
        from dinov2.hub.backbones import dinov2_vitl14
        from viclip import get_viclip
        W = C.models_dir() / 'DEVIL'
        self.devil_dir, self.dev, self.torch, self.np = devil, device, torch, np
        self.raft = RAFT(EasyDict({'model': str(W / 'raft-things.pth'), 'small': False, 'mixed_precision': False, 'alternate_corr': False}))
        sd = torch.load(W / 'raft-things.pth', map_location='cpu')
        self.raft.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}); self.raft.to(device).eval()
        self.dino = timm.create_model('vit_small_patch16_224_dino', pretrained=False, num_classes=0)
        miss = self.dino.load_state_dict(torch.load(W / 'dino_deitsmall16_pretrain.pth', map_location='cpu'), strict=False)
        assert not miss.missing_keys, miss
        self.dino.to(device).eval()
        self.dinov2 = dinov2_vitl14(pretrained=False)
        self.dinov2.load_state_dict(torch.load(W / 'dinov2_vitl14_pretrain.pth', map_location='cpu')); self.dinov2.to(device).eval()
        self.viclip = get_viclip('l', str(W / 'ViClip-InternVid-10M-FLT.pth'))['viclip'].to(device).eval()

    def _temporal_entropy(self, clip_path):
        # DEVIL's own script; DEVIL_TE_SCRIPT may point to a copy that counts packets instead of decoding every frame
        # (same frame count on constant-frame-rate MP4, faster).
        script = os.environ.get('DEVIL_TE_SCRIPT', str(self.devil_dir / 'cal_temporal_info.sh'))
        with tempfile.TemporaryDirectory(prefix='devil_te_') as tmp:
            r = subprocess.run(['bash', script, clip_path, tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(r.stdout.strip().splitlines()[-1])

    def raw_scores(self, video_path, f0, f1, clip_path):
        import decord
        import imagehash
        import torch.nn.functional as F
        from PIL import Image
        from pytorch_msssim import ssim as torch_ssim
        from torchvision.transforms import Compose, Normalize, Resize
        from RAFT.core.utils_core.utils import InputPadder
        torch, np = self.torch, self.np
        transform = Compose([Resize((224, 224)), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        vr = decord.VideoReader(video_path, num_threads=2); fps = vr.get_avg_fps()
        step = int(fps / TARGET_FPS) if fps > TARGET_FPS else 1
        idx = list(range(f0, f1, max(step, 1)))
        if len(idx) < 3:
            return None
        batch = vr.get_batch(idx)
        org = batch.asnumpy() if hasattr(batch, 'asnumpy') else batch.cpu().numpy()
        vid = torch.from_numpy(org).permute(0, 3, 1, 2).float().div(255)
        h, w = vid.shape[-2:]; s = 512 / max(h, w)
        nh, nw = (int(h * s), 512) if w > h else (512, int(w * s))
        no_resize = F.interpolate(vid.float(), size=(nh, nw), mode='bilinear', align_corners=False).to(self.dev)
        video = transform(vid).to(self.dev); n = video.shape[0]
        with torch.no_grad():
            pad = InputPadder(no_resize.shape); x, y = pad.pad(no_resize[:-1], no_resize[1:])
            flows = torch.cat([self.raft(x[i:i + 20].contiguous(), y[i:i + 20].contiguous(), iters=20, test_mode=True)[1]
                               for i in range(0, x.shape[0], 20)], 0)
            flow = torch.sqrt(flows[:, 0] ** 2 + flows[:, 1] ** 2).flatten(1, 2).mean().item()
            ssim = torch_ssim(video[:-1].clamp(0, 1), video[1:].clamp(0, 1), data_range=1.0, size_average=True).item()
            hashes = [imagehash.phash(Image.fromarray(f)) for f in org]
            phash = float(np.mean([hashes[i + 1] - hashes[i] for i in range(n - 1)]))
            feat = F.normalize(self.dinov2.get_intermediate_layers(video, n=1)[0], dim=-1, p=2)
            acf = [(1 - (feat[:-k] * feat[k:]).sum(-1).mean()).abs().item() for k in range(max(1, n // 8), n)]
            dino_segm = sum(acf) / len(acf) if acf else float('nan')
            seg_len = int(n * 0.25)
            if seg_len <= 0:
                return None

            def pad_seg(seg):
                reps = seg_len // seg.size(0) + 1
                return torch.repeat_interleave(seg, repeats=reps, dim=0)[:seg_len]
            segs = torch.stack([video[i * seg_len:(i + 1) * seg_len] if min((i + 1) * seg_len, n) - i * seg_len == seg_len
                                else pad_seg(video[i * seg_len:]) for i in range(int((n + seg_len - 1) // seg_len))])
            L = segs.size(1)
            segs = segs[:, torch.linspace(0, L - 1, steps=8).long()] if L > 8 else torch.cat((segs, segs[:, -1:].repeat(1, 8 - L, 1, 1, 1)), dim=1)
            sf = F.normalize(self.viclip.get_vid_features(segs), dim=-1, p=2); sims = sf @ sf.T
            viclip_segm = 1 - torch.masked_select(sims, torch.triu(torch.ones_like(sims, dtype=torch.bool))).mean().item()
            f = F.normalize(self.dino(video), dim=-1, p=2)
            info_dino = (1 - F.cosine_similarity(f.mean(0, keepdim=True)[:, None], f[None], dim=-1)).mean(1).item()
        te = self._temporal_entropy(clip_path)
        return {'flow': flow, 'ssim': ssim, 'phash': phash, 'dino_segm_dist': dino_segm, 'viclip_segm_dist': viclip_segm,
                'info_dino': info_dino, 'temporal_entropy': te, 'n_frames_8fps': n}

    def shot(self, path, a, b, fps):
        f0, f1 = frame_range(fps, a, b)
        with tempfile.TemporaryDirectory(prefix='devil_') as tmp:
            clip = f'{tmp}/shot.mkv'
            cut_clip(path, a, b, clip)
            r = self.raw_scores(path, f0, f1, clip)
        if r is None:
            raise NA('fewer than 3 frames at 8 fps')
        return r
