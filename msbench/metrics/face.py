"""Cross-Shot Face Consistency (CSFC).

Applies to scripts with ``eval_metrics.face_consistency`` and at least one human entity.

1. ``detect``: InsightFace ``buffalo_l`` (detection + ArcFace) on full frames sampled at 4 fps inside the ``m01`` shots
   of the shot plan (times rounded to 0.01 s); frames in transition gaps are skipped.
2. ``som``: for up to 6 evenly spaced frames with faces per shot, the frame is drawn with numbered face boxes and the
   VLM judge is asked which box is the target person (entity caption, other people's captions as distractors), or 0.
   Done for every human entity, also in single-person videos. Greedy decoding, no thinking, max 16 tokens.
3. ``score``: per entity, picked faces are grouped by shot; for every shot pair, each face takes its best cosine
   similarity to the other shot, calibrated by f(x) = 1 - (1 - x)^beta with beta = ln 0.4 / ln 0.7 (x = 0.7 -> 0.6),
   averaged and symmetrised; the entity score is the mean over shot pairs, the video score the mean over entities.
   A video in which no entity is found in at least two shots scores 0.
"""
import io
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .. import common as C
from .. import judge as J

FPS = 4.0
PER_SHOT = 6
BETA = math.log(0.4) / math.log(0.7)


def humans(info):
    return [e for e in info['entities_info']['entities'] if e.get('is_human')]


def videos(run):
    bench, plan, inp = run.bench(), run.shot_plan(), run.inputs(); out = {}
    for k, r in inp.items():
        info = bench[k]; hs = humans(info)
        if info['eval_metrics'].get('face_consistency') and hs and k in plan:
            shots = [{'start_time': round(s, 2), 'end_time': round(e, 2)} for s, e in plan[k]['m01']]
            out[k] = {'path': r['video_path'], 'humans': hs, 'shots': shots}
    return out


def _shot_id(t, shots):
    for i, s in enumerate(shots):
        if float(s['start_time']) <= t <= float(s['end_time']):
            return i
    return -1


def detect_faces(app, video_id, video_path, shots, fps):
    import cv2
    cap = cv2.VideoCapture(video_path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(vfps / fps)); rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / vfps; sid = _shot_id(t, shots)
        if i % stride == 0 and sid >= 0:
            h, w = frame.shape[:2]
            faces = [{'bbox': [round(float(x), 1) for x in fc.bbox], 'det': round(float(fc.det_score), 4),
                      'area': round(float((fc.bbox[2] - fc.bbox[0]) * (fc.bbox[3] - fc.bbox[1]) / (w * h)), 5),
                      'feat': [round(float(x), 6) for x in fc.normed_embedding]} for fc in app.get(frame)]  # BGR input
            rows.append({'video_id': video_id, 'frame_idx': i, 't': round(t, 3), 'shot_id': sid, 'faces': faces})
        i += 1
    cap.release()
    return rows


def marked_png(video_path, frame_idx, boxes):
    """The frame with numbered red boxes around the detected faces (1-based)."""
    import cv2
    from PIL import Image, ImageDraw
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise ValueError(f'cannot read frame {frame_idx} of {video_path}')
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); draw = ImageDraw.Draw(img)
    lw = max(2, img.width // 400)
    for k, b in enumerate(boxes, 1):
        draw.rectangle(b, outline=(255, 0, 0), width=lw)
        tx, ty = b[0], max(0, b[1] - 12 * lw)
        draw.rectangle([tx, ty, tx + 10 * lw, ty + 12 * lw], fill=(255, 0, 0))
        draw.text((tx + 2 * lw, ty), str(k), fill=(255, 255, 255))
    buf = io.BytesIO(); img.save(buf, 'PNG')
    return buf.getvalue()


def som_frames(rows, n_shots, per_shot):
    out = []
    for sid in range(n_shots):
        fr = [r for r in rows if r['shot_id'] == sid and r['faces']]
        k = min(per_shot, len(fr))
        idx = sorted({round(j * (len(fr) - 1) / max(k - 1, 1)) for j in range(k)})
        out += [fr[i] for i in idx]
    return out


def calibrate(x):
    return 1.0 - (1.0 - x) ** BETA


def _pairwise(feats_a, feats_b):
    import numpy as np
    per = [calibrate(max(max(0.0, float(np.dot(fa, fb))) for fb in feats_b)) for fa in feats_a]
    return sum(per) / len(per) if per else 0.0


def entity_score(groups):
    """Calibrated cross-shot similarity over {shot_id: [embedding]}; None if fewer than 2 shots."""
    import numpy as np
    sids = sorted(s for s, f in groups.items() if f)
    if len(sids) < 2:
        return None
    feats = {s: [np.asarray(x, dtype=np.float32) for x in groups[s]] for s in sids}
    pairs = [(_pairwise(feats[sids[i]], feats[sids[j]]) + _pairwise(feats[sids[j]], feats[sids[i]])) / 2
             for i in range(len(sids)) for j in range(i + 1, len(sids))]
    return sum(pairs) / len(pairs)


def _faces(run):
    rows = {}
    for r in C.read_jsonl(run.path('face', 'faces.jsonl')):
        rows.setdefault(r['video_id'], []).append(r)
    return rows


def detect(a):
    import logging
    from insightface.app import FaceAnalysis
    import onnxruntime as ort
    run = C.Run(a.results, a.benchmark, keys=a.keys)
    vids = videos(run); have = _faces(run)
    keys = C.shard(sorted(k for k in vids if k not in have), a.shard)
    print('face detect videos', len(keys), flush=True)
    if not keys:
        return
    logging.getLogger('insightface').setLevel(logging.ERROR)
    cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
    app = FaceAnalysis(name='buffalo_l', root=os.environ.get('MSB_INSIGHTFACE_ROOT', str(C.models_dir() / 'insightface')),
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider'],
                       allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0 if cuda else -1, det_size=(640, 640))
    out = run.path('face', 'faces.jsonl'); out.parent.mkdir(parents=True, exist_ok=True)
    for k in keys:
        t = time.monotonic(); v = vids[k]
        rows = detect_faces(app, k, v['path'], v['shots'], FPS)
        with out.open('a') as f:  # one video at a time: an interrupted run never leaves a partial video
            f.write(''.join(json.dumps(r) + '\n' for r in rows))
        print(k, sum(bool(r['faces']) for r in rows), '/', len(rows), 'frames with faces', round(time.monotonic() - t, 1), 's', flush=True)


def pick(judge, png, caption, others, n_boxes, retries=2):
    text = C.prompt('face_som').format(caption=caption, others=('Other people in the video (not the target): ' + ' | '.join(others)) if others else '')
    content = [judge.image_part(png), {'type': 'text', 'text': text}]
    last = None
    for _ in range(retries + 1):
        try:
            ans = judge.chat(content, 'greedy', 16, False)[0]['raw_answer']
            m = re.fullmatch(r'\s*(\d+)\s*\.?\s*', ans or '')
            if m and int(m.group(1)) <= n_boxes:
                return int(m.group(1))
            last = ValueError(f'unusable answer {ans!r} for {n_boxes} boxes')
        except Exception as e:
            last = e
    raise RuntimeError(f'Set-of-Mark request failed: {last}')


def som(a):
    run = C.Run(a.results, a.benchmark, keys=a.keys)
    vids = videos(run); det = _faces(run)
    missing = [k for k in vids if k not in det]
    if missing:
        raise SystemExit(f'{len(missing)} videos without face detections, e.g. {missing[:3]}: run `msbench face detect`')
    out = run.path('face', 'som.jsonl')
    done = {(r['video_id'], r['entity'], r['frame_idx']) for r in C.read_jsonl(out)}
    jobs = [(k, v, ent, fr) for k, v in vids.items() for fr in som_frames(det[k], len(v['shots']), PER_SHOT)
            for ent in v['humans'] if (k, ent['id'], fr['frame_idx']) not in done][:a.max_jobs]
    print('som requests', len(jobs), flush=True)
    if a.dry_run:
        return
    judge = J.from_args(a)
    print('judge', judge.describe(), flush=True)
    lock = threading.Lock(); fails = []

    def work(job):
        k, v, ent, fr = job
        boxes = [fc['bbox'] for fc in fr['faces']]
        others = [e['caption'] for e in v['humans'] if e['id'] != ent['id']]
        try:
            p = pick(judge, marked_png(v['path'], fr['frame_idx'], boxes), ent['caption'], others, len(boxes))
        except Exception as e:
            with lock:
                fails.append(str(e)[:200])
            return
        C.append_jsonl(out, {'video_id': k, 'entity': ent['id'], 'frame_idx': fr['frame_idx'], 'shot_id': fr['shot_id'],
                             'n_faces': len(boxes), 'pick': p, 'judge_model': judge.model})

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, jobs))
    if fails:
        raise SystemExit(f'{len(fails)} Set-of-Mark requests failed (e.g. {fails[0]}); rerun to retry them')


def score(a):
    run = C.Run(a.results, a.benchmark, keys=a.keys)
    vids = videos(run); det = _faces(run)
    picks = {(r['video_id'], r['entity'], r['frame_idx']): r['pick'] for r in C.read_jsonl(run.path('face', 'som.jsonl'))}
    rows = []
    for k, v in vids.items():
        need = [(k, e['id'], fr['frame_idx']) for fr in som_frames(det.get(k, []), len(v['shots']), PER_SHOT) for e in v['humans']]
        lost = [n for n in need if n not in picks]
        if k not in det or lost:
            rows.append({'key': k, 'status': 'error', 'error': 'face detections or Set-of-Mark answers missing'}); continue
        ents, scores = {}, []
        for ent in v['humans']:
            groups = {}
            for r in det[k]:
                p = picks.get((k, ent['id'], r['frame_idx']), 0)
                if p:
                    groups.setdefault(r['shot_id'], []).append(r['faces'][p - 1]['feat'])
            s = entity_score(groups)
            ents[ent['id']] = {'frames_per_shot': {str(i): len(x) for i, x in sorted(groups.items())}, 'score': s}
            if s is not None:
                scores.append(s)
        rows.append({'key': k, 'status': 'ok', 'face': sum(scores) / len(scores) if scores else 0.0, 'missing': not scores,
                     'entities': ents, 'n_shots': len(v['shots'])})
    C.write_jsonl(run.path('face', 'scores.jsonl'), rows)
    bad = [r for r in rows if r['status'] != 'ok']
    print('face scored', len(rows) - len(bad), 'errors', len(bad), 'not found in 2 shots -> 0:', sum(r.get('missing', False) for r in rows))
