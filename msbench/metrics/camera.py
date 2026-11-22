"""Camera Control: FilmOps shot scale / camera angle / camera movement against the scripted camera of each shot.

Per clean shot, FilmOps (Neo-yk/FilmOps) labels the scale and angle of frames sampled at 2 fps (``n = max(1,
int(len * 2))`` frames at the bin centres) and the movement of a lossless clip of the shot. Planned shot i is
compared with detected shot i (order matching); a planned shot without a detected counterpart scores 0.
FilmOps labels are mapped to the MSBench vocabulary with partial credit for neighbouring classes (tables below,
fixed before looking at any human score). Camera Control = mean of scale, angle and movement similarity.
"""
import sys
import tempfile

from .. import common as C
from .devil import cut_clip
from .visual import NA

SCALE_AXIS = ['extreme-close-up', 'close-up', 'medium-close-up', 'medium', 'medium-long', 'full', 'long']
FO_SCALE_POS = {'Extreme Close-Up': 0.0, 'Close-Up': 1.0, 'Close Shot': 2.0, 'Medium Shot': 3.0, 'Medium Full Shot': 4.0,
                'Full Shot': 5.0, 'Long Shot': 6.0, 'Extreme Long Shot': 6.5}
ANGLE_V = {
    'Eye Level': {'eye-level': 1.0, 'high-angle': 0.5, 'low-angle': 0.5},
    'High Angle': {'high-angle': 1.0, 'eye-level': 0.5, "bird's-eye-view": 0.5},
    'Low Angle': {'low-angle': 1.0, 'eye-level': 0.5},
    "Bird's Eye": {"bird's-eye-view": 1.0, 'high-angle': 0.5},
    'Extreme Low Angle': {'low-angle': 1.0, 'eye-level': 0.25},
    'Extreme High Angle': {'high-angle': 0.75, "bird's-eye-view": 0.75, 'eye-level': 0.0},
}
MOVE = {
    'Static': {'static': 1.0},
    'Pan': {'pan': 1.0, 'tilt': 0.5},
    'Zoom': {'zoom': 1.0, 'dolly': 0.5},
    'Push In': {'dolly': 1.0, 'zoom': 0.5},
    'Pull Out': {'dolly': 1.0, 'zoom': 0.5},
    'Tracking': {'tracking': 1.0},
    'Follow': {'tracking': 0.75, 'handheld': 0.25},
    'Arc': {'tracking': 0.5},
    'Roll': {'handheld': 0.25},
    'Dynamic': {'handheld': 0.5, 'tracking': 0.25},
}


def scale_sim(label, target):
    d = abs(FO_SCALE_POS[label] - SCALE_AXIS.index(target))
    return 1.0 if d < 0.25 else 0.5 if d <= 1.0 else 0.0


def angle_sim(labels, target):
    labels = labels or []
    if target == 'tilted':
        return 1.0 if 'Dutch Angle' in labels else 0.0
    return max([ANGLE_V[l].get(target, 0.0) for l in labels if l in ANGLE_V] or [0.0])


def move_sim(labels, target):
    return max([MOVE[l].get(target, 0.0) for l in (labels or []) if l in MOVE] or [0.0])


def _known_mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def shot_match(row_shot, target):
    """Similarity of one detected shot to a planned camera {scale, angle, movement}; None if the shot has no output."""
    if 'status' in row_shot:
        return None
    fr = row_shot['frames']
    scale = [scale_sim(f['scale'][0], target['scale']) if f['scale'] and f['scale'][0] in FO_SCALE_POS else None for f in fr]
    angle = [angle_sim(f['angle'], target['angle']) if f['angle'] and any(l != 'Unknown' for l in f['angle']) else None for f in fr]
    return {'camera_scale': _known_mean(scale), 'camera_angle': _known_mean(angle),
            'camera_movement': move_sim(row_shot['movement'], target['movement'])}


def video_scores(shots, shot_sequence):
    """shots: FilmOps rows of the detected clean shots; shot_sequence: the script."""
    out = {k: [] for k in ('camera_scale', 'camera_angle', 'camera_movement')}
    for i, s in enumerate(shot_sequence):
        m = shot_match(shots[i], s['camera']) if i < len(shots) else None
        for k in out:
            out[k].append(m[k] if m else 0.0)
    res = {k: sum(v) / len(v) for k, v in out.items()}
    res['camera_control'] = sum(res.values()) / 3
    return res


class FilmOps:
    min_shot = 0.0

    def __init__(self):
        sys.path.insert(0, str(C.third_party_dir() / 'FilmOps'))
        from filmops import FilmOpsConfig, FilmOpsPipeline
        self.p = FilmOpsPipeline(FilmOpsConfig(checkpoint_dir=str(C.models_dir() / 'FilmOps'), device='cuda',
                                               enabled_operators=['shot_scale', 'camera_angle', 'camera_movement']))
        self.p.load()

    def shot(self, path, a, b, fps):
        import decord
        from PIL import Image
        n = max(1, int((b - a) * 2))
        times = [a + (i + 0.5) * (b - a) / n for i in range(n)]
        vr = decord.VideoReader(path)
        idx = [min(len(vr) - 1, int(t * fps)) for t in times]
        fr = vr.get_batch(idx)
        fr = fr.asnumpy() if hasattr(fr, 'asnumpy') else fr.cpu().numpy()
        frames = []
        with tempfile.TemporaryDirectory(prefix='filmops_') as tmp:
            for k, f in enumerate(fr):
                p = f'{tmp}/f{k}.jpg'
                Image.fromarray(f).save(p, quality=95)
                r = self.p.analyse_image(p)
                frames.append({'frame': idx[k], 'scale': r.get('shot_scale', {}).get('labels'), 'angle': r.get('camera_angle', {}).get('labels'),
                               'angle_raw': {k2: [v[0], float(v[1])] for k2, v in r.get('camera_angle', {}).get('raw', {}).items()}})
            clip = f'{tmp}/shot.mkv'
            cut_clip(path, a, b, clip)
            mv = self.p.analyse_video(clip).get('camera_movement', {})
        if 'error' in mv:
            raise NA('camera_movement error: ' + str(mv['error']))
        return {'frames': frames, 'movement': mv.get('labels'), 'movement_raw': mv.get('raw')}

