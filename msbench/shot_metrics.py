"""Per-shot raw scores of the GPU expert models on the clean shots of the shot plan.

  msbench shots --method qalign    Aesthetic Score        (Q-Align environment)
  msbench shots --method deqa      Perceptual Score       (DeQA environment)
  msbench shots --method devil     Dynamic Degree comps.  (main environment)
  msbench shots --method filmops   Camera Control         (main environment)

Output: shot_metrics/<method>.jsonl, one row per video with a ``shots`` list. Resumable; ``--shard k/n`` splits the
videos over several processes or GPUs (they may share the output file).
"""
import time
import traceback

from . import common as C
from .metrics.visual import NA

METHODS = ('qalign', 'deqa', 'devil', 'filmops')


def build(method):
    if method == 'qalign':
        from .metrics.visual import QAlign
        return QAlign()
    if method == 'deqa':
        from .metrics.visual import DeQA
        return DeQA()
    if method == 'devil':
        from .metrics.devil import Devil
        return Devil()
    from .metrics.camera import FilmOps
    return FilmOps()


def run(a):
    run_ = C.Run(a.results, a.benchmark, keys=a.keys)
    plan = run_.shot_plan(); inp = run_.inputs()
    out = run_.path('shot_metrics', f'{a.method}.jsonl')
    done = {k for k, r in C.latest_ok(C.read_jsonl(out)).items() if r['status'] in ('ok', 'not_applicable')}
    keys = C.shard([k for k in sorted(inp) if k in plan], a.shard)
    todo = [k for k in keys if k not in done]
    print(a.method, 'videos', len(keys), 'todo', len(todo), flush=True)
    if not todo:
        return
    import numpy as np
    import torch
    torch.manual_seed(2027); np.random.seed(2027)
    t0 = time.monotonic(); model = build(a.method)
    print(a.method, 'model loaded', round(time.monotonic() - t0, 1), 's', flush=True)
    failures = 0
    for k in todo:
        started = time.monotonic(); p = plan[k]
        row = {'key': k, 'method': a.method, 'hard_cut_margin_frames': 2}
        try:
            shots = []
            for i, (s, e) in enumerate(p['clean']):
                rec = {'index': i, 'start': s, 'end': e}
                try:
                    if e - s < model.min_shot:
                        raise NA(f'shot shorter than {model.min_shot} s')
                    rec.update(model.shot(inp[k]['video_path'], s, e, p['fps']))
                except NA as exc:
                    rec.update(status='not_applicable', reason=str(exc))
                shots.append(rec)
            row['shots'] = shots
            row['status'] = 'ok' if any('status' not in s for s in shots) else 'not_applicable'
        except Exception as exc:
            failures += 1; row.update(status='error', error=repr(exc)); traceback.print_exc()
        row['wall_sec'] = time.monotonic() - started
        C.append_jsonl(out, row)
        print(a.method, k, row['status'], round(row['wall_sec'], 1), row.get('error', ''), flush=True)
    if failures:
        raise SystemExit(f'{failures} videos failed; rerun the same command to retry them')
