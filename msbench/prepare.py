"""Index the generated videos of one system and trim them to the requested duration.

Videos are named ``<sample_id>.mp4`` after the keys of ``data/benchmark.json``. A video longer than the script
(``duration_s``) is trimmed to its first ``round(duration_s * fps)`` frames (libx264 CRF 18, preset slow, audio
re-encoded to AAC), the same rule used for every system in the paper; the trimmed copy goes to
``<results>/videos/``. Shorter videos are evaluated as they are.
"""
import json
import pathlib
import subprocess

from . import common as C


def probe(path):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                                   '-show_entries', 'stream=avg_frame_rate,nb_read_packets,width,height',
                                   '-show_entries', 'format=duration', '-of', 'json', str(path)], text=True)
    d = json.loads(out)
    s = d['streams'][0]
    a, b = s['avg_frame_rate'].split('/')
    fps = float(a) / float(b)
    frames = int(s['nb_read_packets'])
    return {'fps_float': fps, 'frames': frames, 'duration_s': frames / fps, 'width': s.get('width'), 'height': s.get('height')}


def trim(src, dst, frames):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(src), '-frames:v', str(frames),
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(dst)], check=True)


def run(a):
    run_ = C.Run(a.results, a.benchmark, a.videos)
    if not run_.videos or not run_.videos.is_dir():
        raise SystemExit('--videos must be a folder of <sample_id>.mp4 files')
    bench = run_.bench()
    found = {p.stem: p for p in run_.videos.glob('*.mp4')}
    keys = sorted(k for k in bench if k in found)
    unknown = sorted(set(found) - set(bench))
    print(f'videos {len(found)}, matched to the benchmark {len(keys)}, not in the benchmark {len(unknown)}, '
          f'benchmark scripts without a video {len(set(bench) - set(found))}')
    rows = []
    for k in keys:
        src = found[k].resolve(); info = probe(src); want = float(bench[k]['duration_s'])
        n = round(want * info['fps_float'])
        path = src
        if not a.no_trim and info['frames'] > n:
            path = run_.path('videos', f'{k}.mp4')
            if not path.exists():
                trim(src, path, n)
            info_t = probe(path)
        else:
            info_t = info
        rows.append({'key': k, 'video_path': str(path), 'source_path': str(src), 'duration_s': want,
                     'original': info, 'media': info_t, 'trimmed': path != src})
    C.write_jsonl(run_.inputs_path, rows)
    short = [r['key'] for r in rows if r['media']['duration_s'] < r['duration_s'] - 1 / r['media']['fps_float']]
    print(f'wrote {run_.inputs_path} ({len(rows)} videos, {sum(r["trimmed"] for r in rows)} trimmed, '
          f'{len(short)} shorter than the script)')
