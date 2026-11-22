"""Visual Quality: Aesthetic Score (Q-Align video prompt) and Perceptual Score (DeQA-Score).

Q-Align (``q-future/one-align``): the official "How would you rate the quality of this video?" prompt, frames of each
clean shot sampled at 4 fps (``n = int(len * 4)`` frames at ``a + i / 4``; a shot shorter than 0.25 s uses its middle
frame), square padding, 5-level logit weighting -> 0-1. Needs the Q-Align environment (transformers 4.36.2).

DeQA-Score (``zhiyuanyou/DeQA-Score-Mix3``): frames sampled at 8 fps inside the shot, batch 4, shot raw = frame mean
(1-5); the video score divides the duration-weighted shot mean by 5. Needs the DeQA environment (transformers 4.36.1).
"""
import math
import sys

from .. import common as C


class NA(Exception):
    """The metric does not apply to this shot."""


def load_segment(video_path, start_time, end_time, num_frames=None):
    """Uniformly sample ``num_frames`` RGB frames (default 8 fps) from [start_time, end_time]."""
    import decord
    import numpy as np
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(video_path, num_threads=1)
    fps = vr.get_avg_fps(); total = len(vr)
    f0 = max(0, min(int(start_time * fps), total - 1))
    f1 = max(f0 + 1, min(int(end_time * fps), total))
    n_seg = f1 - f0
    if n_seg <= 0:
        return [], fps
    if num_frames is None:
        num_frames = math.ceil((end_time - start_time) * 8)
    k = min(n_seg, max(1, num_frames))
    if k >= n_seg:
        idx = list(range(f0, f1))
    else:
        step = n_seg / k
        idx = sorted({min(int(f0 + i * step), f1 - 1) for i in range(k)})
    return [f for f in vr.get_batch(idx).asnumpy().astype(np.uint8)], fps


class QAlign:
    min_shot = 0.0
    FPS = 4.0
    PROMPT = "USER: How would you rate the quality of this video?\n<|image|>\nASSISTANT: The quality of the video is"

    def __init__(self, device='cuda:0'):
        import torch
        sys.path.insert(0, str(C.third_party_dir() / 'Q-Align'))
        from q_align.model.builder import load_pretrained_model
        from q_align.constants import IMAGE_TOKEN_INDEX
        from q_align.mm_utils import tokenizer_image_token
        import q_align.model.modeling_llama2 as _ml2
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
        # upstream modeling_llama2.py uses this helper in its SDPA branch without importing it
        _ml2._prepare_4d_causal_attention_mask_for_sdpa = _prepare_4d_causal_attention_mask_for_sdpa
        self.torch = torch
        self.tokenizer, self.model, self.proc, _ = load_pretrained_model(str(C.models_dir() / 'Q-Align'), None, 'mplug_owl2', device=device)
        self.ids = [i[1] for i in self.tokenizer(['excellent', 'good', 'fair', 'poor', 'bad'])['input_ids']]
        self.w = torch.Tensor([1, 0.75, 0.5, 0.25, 0.0]).half().to(self.model.device)
        self.input_ids = tokenizer_image_token(self.PROMPT, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

    def _square(self, im):
        from PIL import Image
        w, h = im.size
        if w == h:
            return im
        s = max(w, h)
        out = Image.new(im.mode, (s, s), tuple(int(x * 255) for x in self.proc.image_mean))
        out.paste(im, (0, (s - h) // 2) if w > h else ((s - w) // 2, 0))
        return out

    def frames(self, path, a, b):
        import decord
        from PIL import Image
        vr = decord.VideoReader(path); vfps = vr.get_avg_fps()
        n = int((b - a) * self.FPS)
        times = [a + i / self.FPS for i in range(n)] if n >= 1 else [(a + b) / 2]
        idx = [min(len(vr) - 1, int(t * vfps)) for t in times]
        fr = vr.get_batch(idx)
        fr = fr.asnumpy() if hasattr(fr, 'asnumpy') else fr.cpu().numpy()
        return [Image.fromarray(f) for f in fr], idx

    def shot(self, path, a, b, fps):
        frames, idx = self.frames(path, a, b)
        with self.torch.inference_mode():
            t = self.proc.preprocess([self._square(f) for f in frames], return_tensors='pt')['pixel_values'].half().to(self.model.device)
            lg = self.model(self.input_ids, images=[t])['logits'][:, -1, self.ids]
            score = float((self.torch.softmax(lg, -1) @ self.w).float().item())
        return {'vqa': score, 'vqa_frame_idx': idx, 'vqa_fps': self.FPS}


class DeQA:
    min_shot = 0.0

    def __init__(self):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
        path = str(C.models_dir() / 'DeQA-Score-Mix3')
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        config.mlp_bias = False; config.use_cache = False
        config._attn_implementation = 'sdpa'  # the remote-code class takes only `config`, so set it here
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(path, config=config, trust_remote_code=True, local_files_only=True,
                                                          torch_dtype=torch.float16, device_map='cuda:0', low_cpu_mem_usage=True).eval()
        module = sys.modules[type(self.model.model.layers[0].self_attn).__module__]
        module._prepare_4d_causal_attention_mask_for_sdpa = _prepare_4d_causal_attention_mask_for_sdpa
        self.model.model._use_sdpa = True

    def shot(self, path, a, b, fps):
        import numpy as np
        from PIL import Image
        frames, _ = load_segment(path, a, b, num_frames=max(1, math.ceil((b - a) * 8)))
        ims = [Image.fromarray(f) for f in frames]; s = []
        with self.torch.inference_mode():
            for j in range(0, len(ims), 4):
                s.extend(self.model.score(ims[j:j + 4]).detach().float().cpu().numpy().reshape(-1).tolist())
        if not s:
            raise NA('no frames')
        raw = float(np.mean(s))
        return {'frame_scores': [float(x) for x in s], 'n_frames': len(s), 'raw': raw, 'deqa': min(1.0, max(0.0, raw / 5))}
