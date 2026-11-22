"""Transition detection and the shot plan.

  msbench transitions omnishotcut   OmniShotCut on every video (GPU)          -> transition/omnishotcut.jsonl
  msbench transitions transvlm      TransVLM gradual intervals (GPU)          -> transition/transvlm.jsonl
  msbench transitions type          VLM judge types every gradual interval    -> qwen_type/type4.jsonl, type6.jsonl
  msbench transitions plan          fuse events, write the shot plan          -> transitions.jsonl, shot_plan.json
"""
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import common as C
from . import judge as J
from . import shots as S

TYPES4 = ('hard cut', 'dissolve', 'fade', 'wipe')
TYPES6 = TYPES4 + ('other', 'none')


def _omni_rows(run):
    return {r['video_id']: r for r in C.read_jsonl(run.path('transition/omnishotcut.jsonl')) if r.get('status') == 'ok'}


def _transvlm_rows(run):
    return {pathlib.Path(r['video']).stem: r for r in C.read_jsonl(run.path('transition/transvlm.jsonl'))}


# ---- OmniShotCut -------------------------------------------------------------------------------------------------
def omnishotcut(a):
    import torch
    import omnishotcut as O
    run = C.Run(a.results, a.benchmark, keys=a.keys)
    out = run.path('transition/omnishotcut.jsonl')
    done = set(_omni_rows(run))
    todo = [r for k, r in C.shard(sorted(run.inputs().items()), a.shard) if k not in done]
    print('omnishotcut todo', len(todo), flush=True)
    if not todo:
        return
    assert torch.cuda.is_available(), 'OmniShotCut needs a CUDA GPU'
    ckpt = pathlib.Path(a.checkpoint or C.models_dir() / 'OmniShotCut' / 'OmniShotCut_ckpt.pth')
    model = O.load(str(ckpt))
    failures = 0
    for r in todo:
        started = time.monotonic(); fps = r['media']['fps_float']
        row = {'video': r['video_path'], 'video_id': r['key']}
        try:
            with torch.inference_mode():
                ranges, intra, inter = model.inference(r['video_path'], mode='default')
            assert len(ranges) == len(intra) == len(inter) and len(ranges) > 0
            assert all(0 <= x < y for x, y in ranges)
            row.update(status='ok', fps=fps, frame_ranges=ranges, time_ranges=[[x / fps, y / fps] for x, y in ranges],
                       intra_labels=intra, inter_labels=inter)
        except Exception as exc:
            failures += 1; row.update(status='error', error=repr(exc))
        row['wall_sec'] = time.monotonic() - started
        C.append_jsonl(out, row)
        print('omnishotcut', r['key'], row['status'], round(row['wall_sec'], 1), row.get('error', ''), flush=True)
    if failures:
        raise SystemExit(f'{failures} videos failed; rerun the same command to retry them')


# ---- TransVLM ----------------------------------------------------------------------------------------------------
def transvlm(a):
    run = C.Run(a.results, a.benchmark, keys=a.keys)
    out = run.path('transition/transvlm.jsonl')
    done = set(_transvlm_rows(run))
    todo = [r for k, r in C.shard(sorted(run.inputs().items()), a.shard) if k not in done]
    print('transvlm todo', len(todo), flush=True)
    if not todo:
        return
    lst = run.path('transition', f'transvlm_videos_{os.getpid()}.txt')
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text(''.join(r['video_path'] + '\n' for r in todo))
    tmp = run.path('transition', 'transvlm_tmp'); tmp.mkdir(exist_ok=True)
    cmd = [sys.executable, '-m', 'msbench.third_party.transvlm_entry', '--video-list', str(lst),
           '--ckpt-dir', str(a.checkpoint or C.models_dir() / 'TransVLM'), '--backend', 'hf', '--temp-dir', str(tmp),
           '--flow-mini-batch-size', '4', '--output-jsonl', str(out)]
    print('+', ' '.join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, cwd=str(C.REPO))
    finally:
        lst.unlink(missing_ok=True)


# ---- transition types (VLM judge) ---------------------------------------------------------------------------------
def parse_type(answer, classes):
    import re
    m = re.findall(r'final answer\s*:\s*\**\s*(' + '|'.join(classes) + ')', answer.lower())
    if not m:
        raise ValueError('No final type')
    return m[-1]


def _answers(path):
    return {r['key']: r['pred_type'] for r in C.read_jsonl(path) if r.get('status') == 'ok'}


def type_(a):
    run = C.Run(a.results, a.benchmark, keys=a.keys)
    inp = run.inputs(); omni = _omni_rows(run); trans = _transvlm_rows(run)
    missing = [k for k in inp if k not in omni or k not in trans]
    if missing:
        raise SystemExit(f'{len(missing)} videos lack OmniShotCut or TransVLM output, e.g. {missing[:3]}')
    jobs, seen = [], set()
    for k in sorted(inp):
        for e in S.fuse_video(omni[k], trans[k]):
            if e['family'] != 'gradual':
                continue
            key = f'{k}|{S.where(e["start"], e["end"])}'
            if key not in seen:
                seen.add(key); jobs.append({'key': key, 'video': k, 'start': e['start'], 'end': e['end']})
    judge = None if a.dry_run else J.from_args(a)
    for n, name, classes in ((4, 'type4', TYPES4), (6, 'type6', TYPES6)):
        out = run.path('qwen_type', f'{name}.jsonl'); done = _answers(out)
        todo = [j for j in jobs if j['key'] not in done][:a.max_jobs]
        template = C.prompt(f'transition_type{n}')
        print(f'{name}: intervals {len(jobs)}, todo {len(todo)}', flush=True)
        if a.dry_run:
            if todo:
                print(template.format(where=S.where(todo[0]['start'], todo[0]['end'])))
            continue
        lock = threading.Lock(); fails = []

        def work(j, template=template, classes=classes, out=out, name=name):
            text = template.format(where=S.where(j['start'], j['end']))
            rec = {**j, 'status': 'error', 'classes': n}; t0 = time.monotonic()
            try:
                content = [{'type': 'text', 'text': text}, judge.video_part(inp[j['video']]['video_path'], 10)]
                r, _ = judge.chat(content, 'low', 8192, False)
                rec.update(r)
                if rec['finish_reason'] != 'stop':
                    raise ValueError('Incomplete output: ' + str(rec['finish_reason']))
                rec['pred_type'] = parse_type(rec['raw_answer'], classes); rec['status'] = 'ok'
            except Exception as e:
                rec['error'] = str(e)[:1500]
                with lock:
                    fails.append(j['key'])
            rec['elapsed_s'] = time.monotonic() - t0
            C.append_jsonl(out, rec)
            print(name, j['key'], rec['status'], rec.get('pred_type'), round(rec['elapsed_s'], 1), rec.get('error', '')[:120], flush=True)

        with ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(work, todo))
        if fails:
            raise SystemExit(f'{name}: {len(fails)} requests failed; rerun to retry them')


# ---- fusion and shot plan ------------------------------------------------------------------------------------------
def plan(a):
    run = C.Run(a.results, a.benchmark, keys=a.keys)
    inp = run.inputs(); omni = _omni_rows(run); trans = _transvlm_rows(run)
    t4, t6 = _answers(run.path('qwen_type/type4.jsonl')), _answers(run.path('qwen_type/type6.jsonl'))
    recs, plan_ = [], {}
    for k in sorted(inp):
        if k not in omni or k not in trans:
            raise SystemExit(f'{k}: OmniShotCut or TransVLM output missing')
        kept, dropped = S.type_events(k, S.fuse_video(omni[k], trans[k]), t4, t6)
        m = inp[k]['media']; dur, fps = m['duration_s'], m['fps_float']
        kept = S.clip_to_video(kept, dur, fps)
        rec = {'key': k, 'status': 'ok', 'pred': kept, 'dropped': dropped, 'duration_s': dur, 'fps': fps,
               'n_frames': m['frames'], 'shots': S.spans(kept, dur, fps)}
        recs.append(rec); plan_[k] = S.plan_row(rec)
    C.write_jsonl(run.transitions_path, recs)
    run.plan_path.write_text(json.dumps(plan_) + '\n')
    from collections import Counter
    print('videos', len(recs), 'events', sum(len(r['pred']) for r in recs), 'dropped', sum(len(r['dropped']) for r in recs))
    print('shots per video:', sorted(Counter(len(v['clean']) for v in plan_.values()).items()))
