"""Cut Time Accuracy (CTA) and Transition Control (TC), computed from the detected transition events.

Requested transitions come from the script: a hard cut (or match cut) is a point at the end of its shot, a gradual
transition spans [shot end, shot end + duration].

CTA: detected shots (between detected events) are matched to the scripted shots by index; the error sums the start-
and end-time differences of matched shots, the durations of extra or missing shots with an increasing multiplier
(x2, x3, ...), and the same terms for transition durations. CTA = exp(-error / video duration).

TC: requested and detected transitions are aligned one-to-one in temporal order, minimising the summed distance of
their centres (dynamic programming over min(n, m) pairs, no distance cap). TC-V = matched / requested,
TC-A = correct type / matched, TC = correct / requested = TC-V x TC-A (unmatched requests count as failures).
"""
import math


def expected_transitions(shot_sequence):
    out = []
    for i, shot in enumerate(shot_sequence):
        tr = shot.get('transition', {}); kind = tr.get('type', 'None')
        if kind == 'None' or i == len(shot_sequence) - 1:
            continue
        end = float(shot.get('end_time', 0)); dur = float(tr.get('duration', '0.0'))
        s, e = (end, end) if kind in ('hard cut', 'match cut') else (end, end + dur)
        out.append({'start_time': s, 'end_time': e, 'type': kind, 'duration': dur})
    return out


def shots_from_transitions(detected, video_duration):
    if not detected:
        return [(0.0, video_duration)]
    shots = []
    if detected[0]['start_time'] > 0:
        shots.append((0.0, detected[0]['start_time']))
    for i in range(len(detected)):
        end = detected[i]['end_time']
        if i < len(detected) - 1:
            nxt = detected[i + 1]['start_time']
            if nxt > end:
                shots.append((end, nxt))
        elif end < video_duration:
            shots.append((end, video_duration))
    if not shots:
        shots = [(detected[0]['end_time'], video_duration)] if detected[0]['end_time'] < video_duration else [(0.0, video_duration)]
    return shots


def _count_penalty(a, b):
    """Matched |a_i - b_i| sum, and the penalised extra / missing tail (multiplier 2, 3, ...)."""
    n = min(len(a), len(b))
    diff = sum(abs(a[i] - b[i]) for i in range(n))
    tail = a[n:] if len(a) > len(b) else b[n:]
    return diff, sum(d * (2 + i) for i, d in enumerate(tail))


def cut_time_accuracy(detected_shots, script_shots, video_duration, detected, expected):
    es = [float(s['start_time']) for s in script_shots]; ee = [float(s['end_time']) for s in script_shots]
    ds = [s[0] for s in detected_shots]; de = [s[1] for s in detected_shots]
    n = min(len(es), len(ds))
    err = sum(abs(ds[i] - es[i]) + abs(de[i] - ee[i]) for i in range(n))
    dd = [b - a for a, b in zip(ds, de)]; ed = [b - a for a, b in zip(es, ee)]
    err += sum(d * (2 + i) for i, d in enumerate(dd[n:] if len(dd) > len(ed) else ed[n:]))
    t_diff, t_tail = _count_penalty([t['end_time'] - t['start_time'] for t in detected],
                                    [t['end_time'] - t['start_time'] for t in expected])
    err += t_diff + t_tail
    return math.exp(-err / video_duration) if video_duration > 0 else 0.0


def _norm(t):
    if t is None:
        return None
    t = t.strip().lower().replace('_', ' ')
    return 'hard cut' if t == 'match cut' else t


def align(expected, detected):
    n, m = len(expected), len(detected); k = min(n, m)
    if k == 0:
        return []
    ce = [(float(e['start_time']) + float(e['end_time'])) / 2 for e in expected]
    cd = [(float(d['start_time']) + float(d['end_time'])) / 2 for d in detected]
    INF = float('inf')
    f = [[[INF] * (k + 1) for _ in range(m + 1)] for _ in range(n + 1)]
    back = {}
    for i in range(n + 1):
        for j in range(m + 1):
            f[i][j][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for c in range(1, k + 1):
                best, arg = f[i - 1][j][c], 'skip_e'
                if f[i][j - 1][c] < best:
                    best, arg = f[i][j - 1][c], 'skip_d'
                v = f[i - 1][j - 1][c - 1] + abs(ce[i - 1] - cd[j - 1])
                if v < best:
                    best, arg = v, 'pair'
                f[i][j][c], back[(i, j, c)] = best, arg
    pairs, i, j, c = [], n, m, k
    while c > 0:
        a = back[(i, j, c)]
        if a == 'pair':
            pairs.append((i - 1, j - 1)); i, j, c = i - 1, j - 1, c - 1
        elif a == 'skip_e':
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def transition_control(expected, detected):
    if not expected:
        return None
    pairs = align(expected, detected)
    correct = sum(_norm(expected[i]['type']) == _norm(detected[j]['type']) for i, j in pairs)
    return {'tc': correct / len(expected), 'n_expected': len(expected), 'n_detected': len(detected),
            'n_matched': len(pairs), 'n_correct': correct}
