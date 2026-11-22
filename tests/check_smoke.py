"""Check a smoke-test result folder against tests/reference_scores.json (scores of the paper run).

Fails if a video lacks any sub-metric or the detected shot count differs. Score differences are reported: expert-model
metrics should be close (GPU / library versions cause small changes); VLM-judge metrics vary with sampling and with a
self-hosted judge, so they only produce warnings.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERT = {'aesthetic': 0.03, 'perceptual': 0.03, 'dynamic_degree': 0.15, 'tc': 0.01, 'cta': 0.05, 'camera_control': 0.1}
JUDGE = {'physical': 0.5, 'entity': 0.5, 'background': 0.5, 'vqa': 0.1, 'face': 0.1}
DIMS = ('VQ', 'DQ', 'SA', 'CSC', 'final')


def main(results):
    ref = json.loads((ROOT / 'tests' / 'reference_scores.json').read_text())['videos']
    got = {}
    p = pathlib.Path(results) / 'final_scores.jsonl'
    if p.exists():
        got = {json.loads(l)['key']: json.loads(l) for l in p.read_text().splitlines() if l.strip()}
    fails, warns = [], []
    for k, r in ref.items():
        g = got.get(k)
        if g is None:
            fails.append(f'{k[:8]}: no scores'); continue
        if g['n_shots_detected'] != r['n_shots_detected']:
            fails.append(f"{k[:8]}: {g['n_shots_detected']} shots detected, paper run {r['n_shots_detected']}")
        print(f'\n{k[:8]}  {"metric":15s} {"ours":>8s} {"paper":>8s}')
        for m, want in r['scores'].items():
            v = g.get(m)
            if v is None:
                fails.append(f'{k[:8]}: {m} missing'); print(f'          {m:15s} {"-":>8s} {want:8.3f}  MISSING'); continue
            tol = EXPERT.get(m, JUDGE.get(m))
            flag = ''
            if tol is not None and abs(v - want) > tol:
                flag = 'differs'
                (warns if m in JUDGE else fails).append(f'{k[:8]}: {m} {v:.3f} vs {want:.3f}')
            print(f'          {m:15s} {v:8.3f} {want:8.3f}  {flag}')
    print()
    for w in warns:
        print('WARN', w, '(VLM judge: sampling / deployment differences are expected)')
    for f in fails:
        print('FAIL', f)
    print('SMOKE TEST', 'FAILED' if fails else 'PASSED')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'results/smoke_example'))
