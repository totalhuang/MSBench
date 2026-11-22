"""Transition events and shot spans (pure Python).

Transition events of a video (paper, Appendix "Shot segmentation and transition recognition"):

* hard cuts: OmniShotCut ``Hard_Cut`` / ``Sudden_Jump`` relations (points);
* gradual intervals: TransVLM intervals, minus those within ``GUARD`` s of an OmniShotCut cut;
* type of a gradual interval: the OmniShotCut dissolve / fade / wipe label of the most-overlapping OmniShotCut gradual
  segment if there is one, else the four-class answer of the VLM judge;
* intervals without an OmniShotCut label are dropped when the judge types them as a hard cut or the six-class judge
  answer is "none" (no shot change inside the interval).

Shot spans derived from the events:

* ``clean``: shots between events, gradual intervals removed, 2 frames trimmed on each side of a hard cut. Read by
  Q-Align, DeQA, DEVIL and FilmOps.
* ``m01``: 0.1 s margin around every event, minimum shot length 0.3 s. Read by VQA (shot windows) and face consistency.
"""

GUARD = 0.2
MARGIN_FRAMES = 2
EPS = 1e-6
OMNI_GRADUAL = {'Dissolve': 'dissolve', 'Fade': 'fade', 'Wipes': 'wipe', 'Push': None, 'Slide': None, 'Zoom': None, 'Doorway': None}


def where(s, e):
    """Location phrase used in the transition-type prompt (also the cache key of the judge answer)."""
    return f'at {s:.1f}s' if s == e else f'between {s:.1f}s and {e:.1f}s'


def omni_events(r):
    """Events of one OmniShotCut output row (time_ranges, intra_labels, inter_labels)."""
    ranges, intra, inter = r['time_ranges'], r['intra_labels'], r['inter_labels']
    assert len(ranges) == len(intra) == len(inter)
    events = []
    for i, ((a, b), label, relation) in enumerate(zip(ranges, intra, inter)):
        assert 0 <= a < b
        if label in OMNI_GRADUAL:
            events.append({'start': a, 'end': b, 'type': OMNI_GRADUAL[label], 'origin': f'intra:{label}'})
        if i and a > 0 and relation in ('Hard_Cut', 'Sudden_Jump'):
            events.append({'start': a, 'end': a, 'type': 'hard cut', 'origin': f'inter:{relation}'})
        elif i and a > 0 and relation in ('Transition', 'Transition_Source'):
            covered = label in OMNI_GRADUAL or (i > 0 and intra[i - 1] in OMNI_GRADUAL)
            if not covered:
                events.append({'start': a, 'end': a, 'type': None, 'origin': f'orphan:{relation}'})
    return sorted(events, key=lambda e: (e['start'], e['end']))


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def fuse_video(omni_row, transvlm_row):
    """OmniShotCut hard cuts + TransVLM gradual intervals (untyped), sorted by time."""
    ev = omni_events(omni_row)
    cuts = [e['start'] for e in ev if e['start'] == e['end'] and e['type'] == 'hard cut']
    grad = [e for e in ev if e['end'] > e['start']]
    events = [{'start': c, 'end': c, 'family': 'abrupt', 'type': 'hard cut', 'source': 'OmniShotCut'} for c in cuts]
    for s in transvlm_row['segments']:
        a, b = float(s['start_time']), float(s['end_time'])
        if any(a - GUARD <= c <= b + GUARD for c in cuts):
            continue
        best = max(grad, key=lambda g: _iou((a, b), (g['start'], g['end'])), default=None)
        lab = best['type'] if best and _iou((a, b), (best['start'], best['end'])) > 0 else None
        events.append({'start': a, 'end': b, 'family': 'gradual', 'omni_label': lab, 'source': 'TransVLM'})
    return sorted(events, key=lambda e: (e['start'], e['end']))


def type_events(key, events, type4, type6):
    """Type the gradual intervals with the judge answers ({'<key>|<where>': type}); returns (kept, dropped)."""
    kept, dropped = [], []
    for e in events:
        if e['family'] == 'gradual':
            k = f'{key}|{where(e["start"], e["end"])}'
            if k not in type4 or k not in type6:
                raise KeyError(f'no judge type for {k}: run `msbench transitions type` first')
            e['type_qwen'] = type4[k]
            e['type'] = e['omni_label'] or e['type_qwen']
            e['qwen6'] = type6[k]
            if not e['omni_label'] and e['type_qwen'] == 'hard cut':
                dropped.append(dict(e, drop_reason='no OmniShotCut label, judge type hard cut')); continue
            if not e['omni_label'] and e['qwen6'] == 'none':
                dropped.append(dict(e, drop_reason='no OmniShotCut label, six-class judge answer none')); continue
        kept.append(e)
    return kept, dropped


def clip_to_video(events, duration, fps):
    """Clip gradual intervals to [0, duration]; drop those left shorter than one frame."""
    for e in events:
        if e['family'] == 'gradual' and (e['start'] < 0 or e['end'] > duration):
            e['raw_interval'] = [e['start'], e['end']]; e['start'] = max(0.0, e['start']); e['end'] = min(duration, e['end'])
    return [e for e in events if e['family'] == 'abrupt' or e['end'] - e['start'] >= 1 / fps]


def spans(events, duration, fps):
    """Shot spans: split at hard cuts, gradual intervals removed (no trimming)."""
    out = []; cur = 0.0
    for e in events:
        if e['start'] > cur:
            out.append((cur, e['start']))
        cur = max(cur, e['end'])
    if duration > cur:
        out.append((cur, duration))
    return [{'start': round(a, 4), 'end': round(b, 4), 'start_frame': round(a * fps), 'end_frame': round(b * fps)}
            for a, b in out if b - a > 1e-6]


def clean_shots(rec):
    """[[start, end], ...] s: spans with MARGIN_FRAMES frames trimmed at each hard-cut side."""
    margin = MARGIN_FRAMES / rec['fps']
    cuts = [e['start'] for e in rec['pred'] if e['family'] == 'abrupt']

    def is_cut(t):
        return any(abs(t - c) < 1e-3 for c in cuts)
    out = []
    for s in rec['shots']:
        a, b = s['start_frame'] / rec['fps'], s['end_frame'] / rec['fps']
        if is_cut(s['start']):
            a += margin
        if is_cut(s['end']):
            b -= margin
        if b > a + EPS:
            out.append([a, b])
    return out


def margin_shots(detected_transitions, video_duration, margin=0.1, min_shot_duration=0.3):
    """Shot intervals between transitions with ``margin`` s kept free at each boundary (``m01`` with margin 0.1)."""
    def _q(t):
        return round(float(t), 3)

    if not detected_transitions:
        return [(0.0, _q(video_duration))]
    tr = sorted(detected_transitions, key=lambda x: x['start_time'])
    shots = []
    first_start = tr[0]['start_time']
    first_end = _q(max(0.0, first_start - margin))
    if first_end >= min_shot_duration:
        shots.append((0.0, first_end))
    elif first_start > 0.0:
        shots.append((0.0, _q(min(video_duration, max(first_start, min_shot_duration)))))
    for i in range(len(tr) - 1):
        prev_end, next_start = tr[i]['end_time'], tr[i + 1]['start_time']
        s, e = _q(prev_end + margin), _q(next_start - margin)
        if e - s < min_shot_duration:
            mid = (float(prev_end) + float(next_start)) / 2.0; half = min_shot_duration / 2.0
            s, e = _q(max(0.0, mid - half)), _q(min(video_duration, mid + half))
        shots.append((s, e))
    last_end = tr[-1]['end_time']
    last_start = _q(last_end + margin)
    if video_duration - last_start >= min_shot_duration:
        shots.append((last_start, _q(video_duration)))
    elif last_end < video_duration:
        start = _q(max(0.0, video_duration - min_shot_duration))
        if start < video_duration:
            shots.append((start, _q(video_duration)))
    shots = [(max(0.0, _q(s)), min(_q(e), _q(video_duration))) for (s, e) in shots if e - s > 0]
    return shots or [(0.0, video_duration)]


def detected_transitions(rec):
    return [{'start_time': p['start'], 'end_time': p['end'], 'type': p.get('type')} for p in rec['pred']]


def plan_row(rec):
    """Shot plan of one fused-transition record."""
    m01 = [[round(s, 3), round(e, 3)] for s, e in margin_shots(detected_transitions(rec), float(rec['duration_s']), 0.1)]
    return {'fps': rec['fps'], 'n_frames': rec['n_frames'], 'duration_s': float(rec['duration_s']), 'clean': clean_shots(rec), 'm01': m01,
            'events': [{'start': e['start'], 'end': e['end'], 'type': e['type'], 'family': e['family']} for e in rec['pred']]}
