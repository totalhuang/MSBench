"""Run TransVLM's ``inference/infer_video.py`` (heygen-com/TransVLM) with a decord-based ``torchcodec`` stand-in.

Some torchcodec builds crash (``std::bad_alloc``, not catchable from Python) with the system FFmpeg; TransVLM only
needs ``VideoDecoder(path).metadata`` and ``get_frames_at``, which decord provides. All arguments are passed to
infer_video.py unchanged. The TransVLM checkout is ``$MSB_THIRD_PARTY/TransVLM`` (default ``third_party/TransVLM``).
"""
import importlib.machinery
import os
import runpy
import sys
import types

os.environ.setdefault('FORCE_QWENVL_VIDEO_READER', 'decord')

import decord  # noqa: E402
import torch  # noqa: E402

from msbench import common as C  # noqa: E402


class _Meta:
    def __init__(self, num_frames, average_fps):
        self.num_frames = num_frames
        self.average_fps = average_fps


class _Frames:
    def __init__(self, data):
        self.data = data


class _VideoDecoder:
    def __init__(self, path, num_ffmpeg_threads=0, dimension_order='NCHW', **kw):
        self._vr = decord.VideoReader(str(path), num_threads=num_ffmpeg_threads or 0)
        self._dim = dimension_order
        self.metadata = _Meta(len(self._vr), float(self._vr.get_avg_fps()))

    def get_frames_at(self, indices):
        batch = self._vr.get_batch(list(indices))
        t = torch.from_numpy(batch.asnumpy() if hasattr(batch, 'asnumpy') else batch.cpu().numpy())
        if self._dim == 'NCHW':
            t = t.permute(0, 3, 1, 2).contiguous()
        return _Frames(t)


def _install_shim():
    dec = types.ModuleType('torchcodec.decoders')
    dec.__spec__ = importlib.machinery.ModuleSpec('torchcodec.decoders', loader=None)
    dec.VideoDecoder = _VideoDecoder
    top = types.ModuleType('torchcodec')
    top.__spec__ = importlib.machinery.ModuleSpec('torchcodec', loader=None)
    top.__path__ = []
    top.decoders = dec
    sys.modules['torchcodec'] = top
    sys.modules['torchcodec.decoders'] = dec
    # transformers (>= 4.57) sees the stand-in as an installed torchcodec and reads its package version at import time
    # (audio_utils.TORCHCODEC_VERSION, used only for audio). The torchcodec wheel is deliberately not installed, so
    # report a placeholder version for it; every other package lookup is unchanged.
    import importlib.metadata as md
    _version = md.version

    def version(name):
        return '0.0.0+decordshim' if name == 'torchcodec' else _version(name)
    md.version = version


if __name__ == '__main__':
    _install_shim()
    root = C.third_party_dir() / 'TransVLM' / 'inference'
    if not (root / 'infer_video.py').exists():
        raise SystemExit(f'{root}/infer_video.py not found: clone heygen-com/TransVLM into {C.third_party_dir()}')
    sys.path.insert(0, str(root))
    sys.argv = [str(root / 'infer_video.py')] + sys.argv[1:]
    runpy.run_path(str(root / 'infer_video.py'), run_name='__main__')
