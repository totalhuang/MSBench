"""Paths, benchmark loading, result files and small helpers shared by every stage.

Layout of one evaluation run (``--results results/<model>``)::

    results/<model>/
      inputs.jsonl                  one row per video found in --videos (written by ``msbench prepare``)
      transition/                   OmniShotCut and TransVLM outputs, gradual-interval candidates
      qwen_type/                    transition-type answers of the VLM judge
      transitions.jsonl             fused transition events per video
      shot_plan.json                shot spans every shot-level metric reads
      shot_metrics/<method>.jsonl   per-shot raw scores (qalign, deqa, devil, filmops)
      face/                         face detections, Set-of-Mark picks, face scores
      judge/<metric>.jsonl          VLM-judge answers (pp, csec, csbc, vqa)
      final_scores.jsonl            video-level scores
      summary.json / summary.md     model-level scores

Every runner appends to its output file and skips finished videos, so an interrupted stage can be restarted.
"""
import fcntl
import hashlib
import json
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
PROMPTS = pathlib.Path(__file__).resolve().parent / 'prompts'
CONFIGS = REPO / 'configs'
DEFAULT_BENCHMARK = REPO / 'data' / 'benchmark.json'


def models_dir():
    """Root of the downloaded weights (``MSB_MODELS``, default ``<repo>/models``)."""
    return pathlib.Path(os.environ.get('MSB_MODELS', REPO / 'models'))


def third_party_dir():
    """Root of the cloned upstream repositories (``MSB_THIRD_PARTY``, default ``<repo>/third_party``)."""
    return pathlib.Path(os.environ.get('MSB_THIRD_PARTY', REPO / 'third_party'))


def prompt(name):
    return (PROMPTS / f'{name}.txt').read_text()


def read_jsonl(p):
    p = pathlib.Path(p)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # torn last line of a killed writer
                continue
    return out


def append_jsonl(p, row):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=False, default=float) + '\n')
        f.flush()


def write_jsonl(p, rows):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def latest_ok(rows, k='key', ok=('ok', 'not_applicable')):
    """{key: row}; a finished row wins over errors, the last finished row wins."""
    out = {}
    for r in rows:
        if r.get('status') in ok or out.get(r[k], {}).get('status') not in ok:
            out[r[k]] = r
    return out


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def shard(items, spec):
    """spec 'k/n' -> every n-th item starting at k (0-based); None -> all."""
    if not spec:
        return list(items)
    k, n = map(int, spec.split('/'))
    return [x for i, x in enumerate(items) if i % n == k]


class Run:
    """One evaluated system: its benchmark file, video folder and result folder."""

    def __init__(self, results, benchmark=None, videos=None, keys=None):
        self.dir = pathlib.Path(results)
        self.benchmark_path = pathlib.Path(benchmark or DEFAULT_BENCHMARK)
        self.videos = pathlib.Path(videos) if videos else None
        self._bench = None
        self._keys = set(keys) if keys else None

    # ---- files -----------------------------------------------------------------------------------------------
    def path(self, *parts):
        return self.dir.joinpath(*parts)

    @property
    def inputs_path(self):
        return self.path('inputs.jsonl')

    @property
    def transitions_path(self):
        return self.path('transitions.jsonl')

    @property
    def plan_path(self):
        return self.path('shot_plan.json')

    # ---- data ------------------------------------------------------------------------------------------------
    def bench(self):
        if self._bench is None:
            self._bench = json.loads(self.benchmark_path.read_text())
        return self._bench

    def inputs(self):
        """{key: input row}, restricted to --keys when given. key = benchmark sample id."""
        rows = {r['key']: r for r in read_jsonl(self.inputs_path)}
        if not rows:
            raise SystemExit(f'{self.inputs_path} is missing: run `msbench prepare` first')
        if self._keys is not None:
            rows = {k: v for k, v in rows.items() if k in self._keys}
        return rows

    def shot_plan(self):
        if not self.plan_path.exists():
            raise SystemExit(f'{self.plan_path} is missing: run the transition stages first')
        return json.loads(self.plan_path.read_text())
