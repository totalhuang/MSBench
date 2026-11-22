"""Video-level sub-metrics, dimension scores and model-level scores.

Sub-metrics (video level, before mapping to [0, 1]):
  aesthetic       Q-Align, duration-weighted mean over clean shots                                    0-1
  perceptual      DeQA, duration-weighted shot mean / 5                                                0-1
  dynamic_degree  DEVIL components -> frozen ridge model                                               1-5
  physical        PP expected score                                                                    1-5
  vqa             soft VQA score                                                                       0-1
  tc / cta        Transition Control / Cut Time Accuracy                                               0-1
  camera_control  FilmOps, mean of scale / angle / movement similarity                                 0-1
  entity / background / face   CSEC / CSBC expected score (1-5), CSFC (0-1), where eval_metrics enables them

1-5 scores are mapped to [0, 1] by (x - 1) / 4. A cross-shot sub-metric is *valid* only when the video has at least two
detected shots (face: and the person is found in two shots); an invalid applicable video counts as 0.

Model level (the numbers of the paper's tables): every sub-metric is averaged over the videos it applies to (so a
cross-shot metric equals validity x score on valid videos), TC = TC-V x TC-A, each dimension is the equal-weight mean of
its available sub-metrics and the Final Score is the equal-weight mean of the four dimensions:
  VQ = {aesthetic, perceptual}, DQ = {dynamic_degree, physical}, SA = {vqa, tc, cta, camera_control},
  CSC = {entity, background, face}.
"""
import json
import math

from . import common as C
from .metrics import camera as CAM
from .metrics import devil as DV
from .metrics import timing as T

ON_1_5 = {'physical', 'entity', 'background', 'dynamic_degree'}
DIMS = {'VQ': ['aesthetic', 'perceptual'], 'DQ': ['dynamic_degree', 'physical'], 'SA': ['vqa', 'tc', 'cta', 'camera_control'],
        'CSC': ['entity', 'background', 'face']}
CSC = ('entity', 'background', 'face')
AGG = ['aesthetic', 'perceptual', 'dynamic_degree', 'physical', 'vqa', 'tc_V', 'tc_A', 'cta', 'camera_control', 'entity', 'background', 'face']


def _valid(shots, field):
    return [s for s in shots if 'status' not in s and s.get(field) is not None]


def dw(shots, field):
    v = _valid(shots, field)
    if not v:
        return None
    w = [s['end'] - s['start'] for s in v]
    return sum(s[field] * x for s, x in zip(v, w)) / sum(w)


def _mean(shots, field):
    v = _valid(shots, field)
    return sum(s[field] for s in v) / len(v) if v else None


def unit(metric, v):
    if v is None:
        return None
    return min(1.0, max(0.0, (v - 1) / 4)) if metric in ON_1_5 else v


def value(row, m):
    """0-1 value of a sub-metric as it enters a dimension (invalid cross-shot -> 0); None if not applicable."""
    if row.get(m) is None:
        return None
    if row.get(m + '_valid') is False:
        return 0.0
    return unit(m, row[m])


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def video_rows(run):
    bench, inp, plan = run.bench(), run.inputs(), run.shot_plan()

    def latest(name, ok=('ok', 'not_applicable')):
        return C.latest_ok(C.read_jsonl(run.path(*name.split('/'))), ok=ok)
    sm = {m: latest(f'shot_metrics/{m}.jsonl') for m in ('qalign', 'deqa', 'devil', 'filmops')}
    jd = {m: latest(f'judge/{m}.jsonl', ok=('ok',)) for m in ('pp', 'csec', 'csbc', 'vqa')}
    face = {r['key']: r for r in C.read_jsonl(run.path('face', 'scores.jsonl'))}
    params = DV.ridge_params()
    rows, missing = [], {}

    def miss(name, k):
        missing.setdefault(name, []).append(k)
    for k in sorted(inp):
        if k not in plan:
            miss('shot_plan', k); continue
        info, p = bench[k], plan[k]; em = info['eval_metrics']
        out = {'key': k, 'duration_s': info['duration_s'], 'n_shots_detected': len(p['clean'])}

        def get(name, src):
            r = src.get(k)
            if r is None or r['status'] == 'error':
                miss(name, k); return None
            return r if r['status'] == 'ok' else None
        x = get('qalign', sm['qalign']); out['aesthetic'] = dw(x['shots'], 'vqa') if x else None
        x = get('deqa', sm['deqa']); raw = dw(x['shots'], 'raw') if x else None
        out['perceptual'] = min(1.0, max(0.0, raw / 5)) if raw is not None else None
        x = get('devil', sm['devil'])
        if x:
            comps = {c: _mean(x['shots'], c) for c in DV.COMPONENTS}
            out['devil_components'] = comps; out['dynamic_degree'] = DV.dynamic_degree(comps, params)
        x = sm['filmops'].get(k)
        if x is None or x['status'] == 'error':
            miss('filmops', k)
        else:
            out.update(CAM.video_scores(x['shots'], info['shot_sequence']))
        for m, name in (('physical', 'pp'), ('entity', 'csec'), ('background', 'csbc')):
            if name in ('csec', 'csbc') and not em.get({'csec': 'entity_consistency', 'csbc': 'background_consistency'}[name]):
                continue
            r = jd[name].get(k)
            if r is None or r.get('score_ev') is None:
                miss(name, k); continue
            out[m] = r['score_ev']
        r = jd['vqa'].get(k)
        if r is None:
            miss('vqa', k)
        else:
            out['vqa'] = r['vqa']
        detected = [{'start_time': e['start'], 'end_time': e['end'], 'type': e['type']} for e in p['events']]
        dur = p['n_frames'] / p['fps']; expected = T.expected_transitions(info['shot_sequence'])
        out['cta'] = float(T.cut_time_accuracy(T.shots_from_transitions(detected, dur), info['shot_sequence'], dur, detected, expected))
        t = T.transition_control(expected, detected)
        out['tc'] = t['tc'] if t else None
        if t:
            out['tc_detail'] = {q: t[q] for q in ('n_expected', 'n_detected', 'n_matched', 'n_correct')}
        if em.get('face_consistency') and any(e.get('is_human') for e in info['entities_info']['entities']):
            f = face.get(k)
            if f is None or f['status'] != 'ok':
                miss('face', k)
            else:
                out['face'] = f['face']; out['face_missing'] = f['missing']
        multi = out['n_shots_detected'] >= 2
        for m in CSC:
            if m in out:
                out[m + '_valid'] = multi and not (m == 'face' and out.get('face_missing'))
        for d, subs in DIMS.items():
            out[d] = _avg([value(out, s) for s in subs])
        out['final'] = _avg([out[d] for d in DIMS])
        rows.append(out)
    return rows, missing


def model_scores(rows):
    """Model-level sub-metrics, dimensions and Final Score (0-1) of a set of video rows."""
    cols = {c: [] for c in AGG}
    for r in rows:
        d = r.get('tc_detail')
        tv = d['n_matched'] / d['n_expected'] if r.get('tc') is not None and d else None
        ta = (d['n_correct'] / d['n_matched'] if d['n_matched'] else None) if tv is not None else None
        for c in AGG:
            v = {'tc_V': tv, 'tc_A': ta}[c] if c in ('tc_V', 'tc_A') else value(r, c)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                cols[c].append(v)
    s = {c: (sum(v) / len(v) if v else None) for c, v in cols.items()}
    s['tc'] = s['tc_V'] * (s['tc_A'] or 0.0) if s['tc_V'] is not None else None
    for d, subs in DIMS.items():
        s[d] = _avg([s[x] for x in subs])
    s['final'] = _avg([s[d] for d in DIMS])
    for m in CSC:  # validity rate and score on valid videos (raw scale)
        app = [r for r in rows if r.get(m) is not None]
        ok = [r[m] for r in app if r.get(m + '_valid')]
        s[m + '_V'] = len(ok) / len(app) if app else None
        s[m + '_S'] = sum(ok) / len(ok) if ok else None
    s['n_videos'] = len(rows)
    return s


COLUMNS = [('aesthetic', 'AS'), ('perceptual', 'PS'), ('VQ', 'VQ'), ('dynamic_degree', 'DD'), ('physical', 'PP'), ('DQ', 'DQ'),
           ('vqa', 'VQA'), ('tc', 'TC'), ('cta', 'CTA'), ('camera_control', 'CC'), ('SA', 'SA'), ('entity', 'CSEC'),
           ('background', 'CSBC'), ('face', 'CSFC'), ('CSC', 'CSC'), ('final', 'Final')]


def run(a):
    run_ = C.Run(a.results, a.benchmark, keys=a.keys)
    rows, missing = video_rows(run_)
    C.write_jsonl(run_.path('final_scores.jsonl'), rows)
    splits = {'all': rows}
    for d in sorted({r['duration_s'] for r in rows}):
        splits[f'{d:g}s'] = [r for r in rows if r['duration_s'] == d]
    base = [r for r in rows if r['duration_s'] in (4, 8)]
    if base and len(base) != len(rows):
        splits['base (4s+8s)'] = base
    summary = {name: model_scores(rs) for name, rs in splits.items() if rs}
    run_.path('summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    md = ['| split | n | ' + ' | '.join(h for _, h in COLUMNS) + ' |', '|---|---:|' + '---:|' * len(COLUMNS)]
    for name, s in summary.items():
        md.append(f"| {name} | {s['n_videos']} | " + ' | '.join('-' if s.get(c) is None else f'{100 * s[c]:.1f}' for c, _ in COLUMNS) + ' |')
    run_.path('summary.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    for name, ks in sorted(missing.items()):
        print(f'MISSING {name}: {len(ks)} videos, e.g. {ks[:3]}')
