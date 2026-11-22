"""Sub-metrics scored by the Qwen3.8 VLM judge.

  pp    Physical Plausibility        failure checklist (body, object continuity, motion, gravity/contact), 1-5
  csec  Cross-Shot Entity Consistency   only scripts with eval_metrics.entity_consistency, 1-5
  csbc  Cross-Shot Background Consistency  only scripts with eval_metrics.background_consistency, 1-5
  vqa   VQA Score                    script yes/no questions, one request per video

pp / csec / csbc: the prompt text first, then the full-resolution video at 10 fps; ``off_official`` decoding, max 8192
tokens; the score is the expected value of the final 1-5 score digit under its top-5 logprobs.

vqa: the whole video at 4 fps, every (question, shot window) slot as one numbered line; windowed lines are prefixed with
"(Judge only the time window A s to B s of the video.)". A question tied to shots gets one slot per shot (the ``m01``
span of the detected shot with that index, dropped if shorter than 1 s; a missing shot counts as "no") plus a
whole-video slot. Soft rule per question: (sum of shot P(yes) + max(whole-video P(yes), max shot P(yes))) / (#slots);
a question without shot slots takes the whole-video P(yes). The video score is the mean over questions.
"""
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .. import common as C
from .. import judge as J
from .. import shots as S

FLAG = {'csec': 'entity_consistency', 'csbc': 'background_consistency'}
VQA_FPS = 4.0
VQA_MIN_SHOT = 1.0


# ---- prompt context ----------------------------------------------------------------------------------------------
def entity_info(info):
    ents = info.get('entities_info', {}).get('entities', [])
    if not ents:
        return '(No entity information available)'
    return '\n'.join(f"- {e.get('id', '?')} ({'human' if e.get('is_human', False) else 'object'}): {e.get('caption', '')}" for e in ents)


def scene_structure_info(info):
    import re
    lines = []
    setting = (info.get('global_context') or {}).get('Setting', '')
    scenes = {}
    if setting:
        for m in re.finditer(r'\[Scene(\d+)\]\s*:\s*', setting):
            start = m.end(); nxt = re.search(r'\[Scene\d+\]\s*:', setting[start:])
            scenes[f'Scene{m.group(1)}'] = setting[start:start + nxt.start() if nxt else len(setting)].strip().rstrip('.')
        if scenes:
            lines.append('Scenes in this video:')
            lines += [f'- {s}: {scenes[s]}' for s in sorted(scenes)]
            lines.append('')
    seq = info.get('shot_sequence', [])
    if seq:
        lines.append('Intended shot-to-scene mapping:')
        for shot in seq:
            m = re.search(r'\[Scene(\d+)\]', shot.get('description', ''))
            lines.append(f"- Shot {shot.get('shot_id', 0)} -> {'Scene' + m.group(1) if m else 'unknown'}")
        lines.append('')
    bg = info.get('background_group')
    if bg and isinstance(bg, list):
        lines.append('Background groups (shots intended to share the same scene):')
        lines += [f'- Group {i + 1}: ' + ', '.join(f'Shot {s}' for s in g) for i, g in enumerate(bg)]
    return '\n'.join(lines) if lines else '(No scene structure information available)'


def fill(template, info):
    fields = {'duration_s': f"{float(info['duration_s']):g}", 't2v_prompt': info['t2v_prompt'],
              'entity_info': entity_info(info), 'scene_structure_info': scene_structure_info(info)}
    for k, v in fields.items():
        template = template.replace('{' + k + '}', v)
    return template


# ---- VQA plan and score ------------------------------------------------------------------------------------------
def vqa_questions(info):
    qs = []
    for q in info['VQA_questions'].get('general', []):
        s = q.get('shot_ids', q.get('shot_id'))
        qs.append({'question': q['question'], 'shot_ids': ([s] if isinstance(s, int) else list(s)) if s is not None else None})
    return qs


def vqa_plan(info, transition_rec):
    """(slots, lines): slots[qi] = [(kind, line_index | None, interval | None)] with the whole-video slot last;
    lines = [(interval | None, question)] in prompt order (windows chronologically, then whole-video questions)."""
    shots = [(round(s, 3), round(e, 3)) for s, e in
             S.margin_shots(S.detected_transitions(transition_rec), float(info['duration_s']), 0.1)]
    raw_slots, jobs = [], []
    for qi, q in enumerate(vqa_questions(info)):
        row = []
        for s in (q['shot_ids'] or []):
            if s < len(shots):
                row.append(('shot', len(jobs), list(shots[s]))); jobs.append((list(shots[s]), q['question']))
            else:
                row.append(('oob', None, None))
        row.append(('full', len(jobs), None)); jobs.append((None, q['question']))
        raw_slots.append(row)
    order = sorted([i for i, j in enumerate(jobs) if j[0] is not None], key=lambda i: (jobs[i][0][0], jobs[i][0][1])) + \
        [i for i, j in enumerate(jobs) if j[0] is None]
    pos = {old: new for new, old in enumerate(order)}
    slots = [[(kind, pos[ji] if ji is not None else None, iv) for kind, ji, iv in row] for row in raw_slots]
    return slots, [jobs[i] for i in order]


def vqa_text(info, lines):
    qs = []
    for i, (iv, q) in enumerate(lines):
        pre = f'(Judge only the time window {iv[0]:.2f} s to {iv[1]:.2f} s of the video.) ' if iv else ''
        qs.append(f'{i + 1}. {pre}{q}')
    return C.prompt('vqa').format(duration=f"{float(info['duration_s']):g}", questions='\n'.join(qs))


def vqa_score(slots, answers, probs, min_shot=VQA_MIN_SHOT):
    """Soft VQA score of one video from the per-line answers / P(yes)."""
    total = 0.0
    for row in slots:
        f = row[-1][1]
        fa = answers[f]; fp = probs[f] if probs and probs[f] is not None else float(fa)
        local = []
        for kind, li, iv in row[:-1]:
            if kind == 'oob':
                local.append(0.0); continue
            if min_shot and iv[1] - iv[0] < min_shot:
                continue
            p = probs[li] if probs and probs[li] is not None else float(answers[li])
            local.append(p)
        total += fp if not local else (sum(local) + max([fp] + local)) / (len(local) + 1)
    return total / len(slots)


# ---- runner ------------------------------------------------------------------------------------------------------
def run(a):
    run_ = C.Run(a.results, a.benchmark, keys=a.keys)
    bench, inp = run_.bench(), run_.inputs()
    out = run_.path('judge', f'{a.metric}.jsonl')
    done = {k for k, r in C.latest_ok(C.read_jsonl(out), ok=('ok',)).items()}
    tr = {r['key']: r for r in C.read_jsonl(run_.transitions_path)} if a.metric == 'vqa' else {}
    keys = [k for k in sorted(inp) if a.metric not in FLAG or bench[k]['eval_metrics'].get(FLAG[a.metric])]
    if a.metric == 'vqa':
        missing = [k for k in keys if k not in tr]
        if missing:
            raise SystemExit(f'{len(missing)} videos have no transitions yet: run the transition stages first')
    todo = [k for k in C.shard(keys, a.shard) if k not in done][:a.max_jobs]
    template = C.prompt(a.metric)
    print(f'{a.metric}: videos {len(keys)}, todo {len(todo)}', flush=True)
    if a.dry_run:
        if todo:
            k = todo[0]
            print(vqa_text(bench[k], vqa_plan(bench[k], tr[k])[1]) if a.metric == 'vqa' else fill(template, bench[k]))
        return
    judge = J.from_args(a)
    print('judge', judge.describe(), flush=True)
    lock = threading.Lock(); counts = {'ok': 0, 'error': 0}

    def work(k):
        info = bench[k]; rec = {'key': k, 'status': 'error'}; t0 = time.monotonic()
        try:
            if a.metric == 'vqa':
                slots, lines = vqa_plan(info, tr[k]); text = vqa_text(info, lines)
                content = [judge.video_part(inp[k]['video_path'], VQA_FPS), {'type': 'text', 'text': text}]
                r, tokens = judge.chat(content, 'off_official', 2048, True)
            else:
                text = fill(template, info)
                content = [{'type': 'text', 'text': text}, judge.video_part(inp[k]['video_path'], 10)]
                r, tokens = judge.chat(content, 'off_official', 8192, True)
            rec.update(r)
            if rec['finish_reason'] != 'stop':
                raise ValueError('Incomplete output: ' + str(rec['finish_reason']))
            if a.metric == 'vqa':
                rec['answers'] = J.parse_yesno(rec['raw_answer'], len(lines))
                rec['yes_probs'] = J.yes_probs(tokens, len(lines)) if tokens else None
                rec['lines'] = [[iv, q] for iv, q in lines]
                rec['vqa'] = vqa_score(slots, rec['answers'], rec['yes_probs'])
            else:
                rec['score'], rec['parsed'] = J.parse_score(rec['raw_answer'])
                rec['score_ev'], rec['score_probs'] = J.expected_score(tokens) if tokens else (None, None)
            rec['prompt_sha256'] = hashlib.sha256(template.encode()).hexdigest()
            rec['status'] = 'ok'
        except Exception as e:
            rec['error'] = str(e)[:2000]
        rec['elapsed_s'] = time.monotonic() - t0
        C.append_jsonl(out, rec)
        with lock:
            counts[rec['status']] += 1
        print(a.metric, k, rec['status'], round(rec['elapsed_s'], 1), rec.get('error', '')[:120], flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(work, todo))
    print(a.metric, counts, flush=True)
    if counts['error']:
        raise SystemExit(f"{counts['error']} requests failed; rerun the same command to retry them")
