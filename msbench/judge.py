"""Qwen3.8 VLM judge client (OpenAI-compatible chat API, standard library only).

Two ways to reach the judge:

* ``dashscope`` (default, used for every number in the paper): Alibaba Cloud Model Studio (百炼) serving
  ``qwen3.8-27b``. Set ``DASHSCOPE_API_KEY``; ``DASHSCOPE_BASE_URL`` / ``MSB_JUDGE_MODEL`` override the endpoint
  and the model name. Videos up to 7.2 MB are sent inline as base64; larger ones go through DashScope's temporary
  file upload (``oss://`` URL, cached for 40 h in ``~/.cache/msbench/oss_urls.jsonl``).
* ``openai``: any OpenAI-compatible server you deployed yourself, e.g. vLLM serving Qwen3.8-27B at
  ``http://<ip>:<port>/v1`` (``--judge-url`` / ``MSB_JUDGE_URL``, model ``--judge-model`` / ``MSB_JUDGE_MODEL``,
  optional ``MSB_JUDGE_API_KEY``). Videos are always sent inline as base64; the frame rate is passed through the
  vLLM-specific ``media_io_kwargs`` / ``mm_processor_kwargs`` fields (``MSB_JUDGE_PLAIN=1`` sends plain OpenAI fields
  only, for servers that reject them; the frame rate is then the server's default). Scores from a self-hosted judge are close to,
  but not bit-identical with, the DashScope ones.

Decoding settings (identical for both backends):

* ``off_official``: no thinking, temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 (model-card setting for
  instruct mode). Used by PP, CSEC, CSBC and VQA, with top-5 logprobs for expected scores / P(yes).
* ``low``: thinking with low effort, temperature 1.0, top_p 0.95, top_k 20 (streamed). Used for transition types.
* ``greedy``: temperature 0. Used for the Set-of-Mark face picks.

All requests use seed 2027.
"""
import base64
import json
import math
import os
import pathlib
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

DASHSCOPE_BASE = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
DASHSCOPE_MODEL = 'qwen3.8-27b'
MAX_B64_BYTES = 7.2e6  # DashScope: the base64 string must stay under 10 MB
TOP_LOGPROBS = 5
SEED = 2027
SAMPLING = {
    'off_official': {'enable_thinking': False, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'presence_penalty': 1.5},
    'low': {'enable_thinking': True, 'reasoning_effort': 'low', 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'presence_penalty': 0.0},
    'greedy': {'enable_thinking': False, 'temperature': 0.0},
}
TRANSIENT = ('429', '500', '502', '503', '504', 'timed out', 'Connection reset', 'Remote end closed', 'RemoteDisconnected',
             'Temporary failure in name resolution')


def _load_dotenv():
    """Fill os.environ from a .env file in the working directory or the repository root (never overrides)."""
    for p in (pathlib.Path.cwd() / '.env', pathlib.Path(__file__).resolve().parents[1] / '.env'):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    if v.strip() and not os.environ.get(k.strip()):
                        os.environ[k.strip()] = v.strip().strip('"\'')


class Judge:
    def __init__(self, backend=None, url=None, model=None, api_key=None, timeout=1800):
        _load_dotenv()
        self.backend = backend or os.environ.get('MSB_JUDGE_BACKEND', 'dashscope')
        if self.backend == 'dashscope':
            self.base = (url or os.environ.get('DASHSCOPE_BASE_URL') or DASHSCOPE_BASE).rstrip('/')
            self.model = model or os.environ.get('MSB_JUDGE_MODEL') or DASHSCOPE_MODEL
            self.key = api_key or os.environ.get('DASHSCOPE_API_KEY')
            if not self.key:
                raise SystemExit('DASHSCOPE_API_KEY is not set (see .env.example), or use --judge-backend openai --judge-url ...')
        elif self.backend == 'openai':
            self.base = (url or os.environ.get('MSB_JUDGE_URL') or '').rstrip('/')
            if not self.base:
                raise SystemExit('--judge-backend openai needs --judge-url http://<ip>:<port>/v1 (or MSB_JUDGE_URL)')
            self.model = model or os.environ.get('MSB_JUDGE_MODEL') or 'Qwen3.8-27B'
            self.key = api_key or os.environ.get('MSB_JUDGE_API_KEY') or 'EMPTY'
        else:
            raise SystemExit(f'unknown judge backend {self.backend!r} (dashscope | openai)')
        self.timeout = timeout
        self._lock = threading.Lock()
        self._oss = None

    def describe(self):
        return {'backend': self.backend, 'base_url': self.base, 'model': self.model}

    # ---- content items --------------------------------------------------------------------------------------
    def video_part(self, path, fps):
        """Full-resolution video at ``fps`` frames per second."""
        p = pathlib.Path(path)
        if self.backend == 'dashscope' and p.stat().st_size > MAX_B64_BYTES:
            url = self._upload(p)
        else:
            url = 'data:video/mp4;base64,' + base64.b64encode(p.read_bytes()).decode()
        return {'type': 'video_url', 'video_url': {'url': url}, 'fps': float(fps)}

    @staticmethod
    def image_part(raw, mime='image/png'):
        if len(raw) > MAX_B64_BYTES:
            raise ValueError(f'image of {len(raw)} bytes is over the inline limit')
        return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,' + base64.b64encode(raw).decode()}}

    # ---- requests ---------------------------------------------------------------------------------------------
    def chat(self, content, mode='off_official', max_tokens=8192, logprobs=True):
        """One user turn with ``content`` (list of text / video / image items).

        Returns ``(record, tokens)``: record has raw_answer, finish_reason, usage, model; tokens are the answer
        tokens with their top logprobs (empty if not requested or not available)."""
        def make(lp):
            return self._body(content, mode, max_tokens, lp)
        for attempt in range(3):
            try:
                return self._send(make, logprobs)
            except Exception as e:
                if attempt == 2 or not any(x in f'{type(e).__name__} {e}' for x in TRANSIENT):
                    raise
                time.sleep(5 * 2 ** attempt)

    def _body(self, content, mode, max_tokens, logprobs):
        s = SAMPLING[mode]
        b = {'model': self.model, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_tokens, 'seed': SEED}
        if self.backend == 'dashscope':
            b.update(s)
        else:  # vLLM: thinking switches go through the chat template, the frame rate through the media kwargs
            b.update({k: v for k, v in s.items() if k not in ('enable_thinking', 'reasoning_effort')})
            kw = {'enable_thinking': s['enable_thinking']}
            if s.get('reasoning_effort'):
                kw['reasoning_effort'] = s['reasoning_effort']
            b['chat_template_kwargs'] = kw
            fps = [c['fps'] for c in content if c.get('type') == 'video_url']
            if os.environ.get('MSB_JUDGE_PLAIN') == '1':  # non-vLLM server: plain OpenAI fields only
                b.pop('chat_template_kwargs'); b.pop('top_k', None)
                b['messages'] = [{'role': 'user', 'content': [{k: v for k, v in c.items() if k != 'fps'} for c in content]}]
            elif fps:
                b['media_io_kwargs'] = {'video': {'fps': fps[0], 'num_frames': -1}}
                b['mm_processor_kwargs'] = {'fps': fps[0], 'do_sample_frames': False}
                b['messages'] = [{'role': 'user', 'content': [{k: v for k, v in c.items() if k != 'fps'} for c in content]}]
        if logprobs:
            b.update(logprobs=True, top_logprobs=TOP_LOGPROBS)
        if s['enable_thinking']:
            b.update(stream=True, stream_options={'include_usage': True})
        return b

    def _open(self, payload):
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key}
        uses_oss = any(str((c.get(f) or {}).get('url', '')).startswith('oss://')
                       for m in payload['messages'] for c in (m['content'] if isinstance(m['content'], list) else [])
                       for f in ('video_url', 'image_url'))
        if uses_oss:
            headers['X-DashScope-OssResourceResolve'] = 'enable'
        req = urllib.request.Request(self.base + '/chat/completions', data=json.dumps(payload).encode(), headers=headers)
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=self.timeout)

    def _post(self, payload):
        if not payload.get('stream'):
            with self._open(payload) as res:
                return json.loads(res.read())
        content, reason, tokens, usage, finish, model = [], [], [], None, None, None
        with self._open(payload) as res:
            for line in res:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    break
                if not data:
                    continue
                chunk = json.loads(data)
                if chunk.get('error'):
                    raise ValueError(str(chunk['error']))
                usage = chunk.get('usage') or usage; model = chunk.get('model') or model
                for ch in chunk.get('choices', []):
                    d = ch.get('delta') or {}
                    content.append(d.get('content') or ''); reason.append(d.get('reasoning_content') or d.get('reasoning') or '')
                    if d.get('content'):
                        tokens += (ch.get('logprobs') or {}).get('content') or []
                    finish = ch.get('finish_reason') or finish
        return {'model': model, 'usage': usage, 'choices': [{'finish_reason': finish, 'logprobs': {'content': tokens},
                'message': {'content': ''.join(content), 'reasoning_content': ''.join(reason)}}]}

    def _send(self, make, logprobs):
        try:
            d = self._post(make(logprobs))
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors='replace')
            if logprobs and e.code == 400 and 'logprob' in msg.lower():  # endpoint without logprobs: retry without
                d = self._post(make(False))
            else:
                raise RuntimeError(f'HTTP {e.code}: {msg[:800]}')
        ch = d['choices'][0]
        tokens = ((ch.get('logprobs') or {}).get('content')) or []
        rec = {'raw_answer': ch['message'].get('content') or '', 'finish_reason': ch.get('finish_reason'), 'usage': d.get('usage'),
               'model': d.get('model'), 'backend': self.backend, 'logprobs_available': bool(tokens)}
        if ch['message'].get('reasoning_content'):
            rec['reasoning_content'] = ch['message']['reasoning_content']
        return rec, tokens

    # ---- DashScope temporary upload for videos over the inline limit ----------------------------------------
    def _upload(self, path):
        cache = pathlib.Path(os.environ.get('MSB_CACHE', pathlib.Path.home() / '.cache' / 'msbench')) / 'oss_urls.jsonl'
        ck = (str(path.resolve()), path.stat().st_size, self.model)
        with self._lock:
            if self._oss is None:
                self._oss = {}
                for line in (cache.read_text().splitlines() if cache.exists() else []):
                    try:
                        r = json.loads(line); self._oss[(r['path'], r['size'], r['model'])] = (r['url'], r['t'])
                    except (json.JSONDecodeError, KeyError):
                        continue
            hit = self._oss.get(ck)
        if hit and time.time() - hit[1] < 40 * 3600:
            return hit[0]
        req = urllib.request.Request(f'https://dashscope.aliyuncs.com/api/v1/uploads?action=getPolicy&model={self.model}',
                                     headers={'Authorization': 'Bearer ' + self.key})
        pol = json.loads(urllib.request.urlopen(req, timeout=60).read())['data']
        key = f"{pol['upload_dir']}/{path.name}"
        cmd = ['curl', '-sS', '-o', '/dev/null', '-w', '%{http_code}', '-X', 'POST', pol['upload_host'],
               '--form-string', f"OSSAccessKeyId={pol['oss_access_key_id']}", '--form-string', f"Signature={pol['signature']}",
               '--form-string', f"policy={pol['policy']}", '--form-string', f"x-oss-object-acl={pol['x_oss_object_acl']}",
               '--form-string', f"x-oss-forbid-overwrite={pol['x_oss_forbid_overwrite']}", '--form-string', f'key={key}',
               '--form-string', 'success_action_status=200', '-F', f'file=@{path.resolve()}']
        code = subprocess.run(cmd, capture_output=True, text=True).stdout
        if code != '200':
            raise RuntimeError(f'DashScope temporary upload failed (http {code})')
        url, t = 'oss://' + key, time.time()
        with self._lock:
            self._oss[ck] = (url, t)
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, 'a') as f:
                f.write(json.dumps({'path': ck[0], 'size': ck[1], 'model': self.model, 'url': url, 't': t}) + '\n')
        return url


def add_args(ap):
    g = ap.add_argument_group('VLM judge')
    g.add_argument('--judge-backend', choices=['dashscope', 'openai'], default=None,
                   help='dashscope (default; DASHSCOPE_API_KEY) or openai (your own OpenAI-compatible server)')
    g.add_argument('--judge-url', help='openai backend: base URL, e.g. http://10.0.0.2:8000/v1 (or MSB_JUDGE_URL)')
    g.add_argument('--judge-model', help='model name on the server (default qwen3.8-27b on DashScope, Qwen3.8-27B otherwise)')
    g.add_argument('--workers', type=int, default=4, help='concurrent judge requests')


def from_args(a):
    return Judge(a.judge_backend, a.judge_url, a.judge_model)


# ---- answer parsing --------------------------------------------------------------------------------------------
SCORE_KEY = re.compile(r'score"?\s*\**\s*[:=]\s*\**\s*\[?\s*$', re.I)


def parse_score(answer):
    """Score from the last JSON object containing "score"; else the last 'score: x' line."""
    d = json.JSONDecoder(); found = None
    for i, c in enumerate(answer):
        if c != '{':
            continue
        try:
            v, _ = d.raw_decode(answer[i:])
        except ValueError:
            continue
        if isinstance(v, dict) and 'score' in v:
            found = v
    if found is not None:
        return float(found['score']), found
    m = re.findall(r'score"?\s*[:=]\s*([1-5](?:\.\d+)?)', answer, re.I)
    if m:
        return float(m[-1]), None
    raise ValueError('No score')


def expected_score(tokens):
    """Probability-weighted 1-5 score at the last digit token that follows 'score:'."""
    text, hit = '', None
    for t in tokens:
        if t['token'].strip() in {'1', '2', '3', '4', '5'} and SCORE_KEY.search(text):
            hit = t
        text += t['token']
    if hit is None:
        return None, None
    probs = {}
    for c in hit['top_logprobs']:
        d = c['token'].strip()
        if d in {'1', '2', '3', '4', '5'}:
            probs[d] = probs.get(d, 0.0) + math.exp(c['logprob'])
    total = sum(probs.values())
    return sum(int(d) * p for d, p in probs.items()) / total, {d: round(p, 5) for d, p in sorted(probs.items())}


def parse_yesno(answer, n):
    """n booleans from numbered 'k. yes/no' lines (or the last bare yes/no when n == 1)."""
    got = {int(k): v.lower() == 'yes' for k, v in re.findall(r'^\W*(\d+)\W+(yes|no)\b', answer, re.I | re.M)}
    if n > 1 or got:
        if sorted(got) != list(range(1, n + 1)):
            raise ValueError(f'Expected {n} numbered yes/no answers, got {sorted(got)}')
        return [got[k] for k in range(1, n + 1)]
    words = re.findall(r'\b(yes|no)\b', answer, re.I)
    if not words:
        raise ValueError('No yes/no answer')
    return [words[-1].lower() == 'yes']


def yes_probs(tokens, n):
    """P(yes) / (P(yes) + P(no)) at the last n yes/no answer tokens."""
    out = []
    for t in tokens:
        if t['token'].strip().lower() in {'yes', 'no'}:
            p = {'yes': 0.0, 'no': 0.0}
            for c in t['top_logprobs']:
                w = c['token'].strip().lower()
                if w in p:
                    p[w] += math.exp(c['logprob'])
            out.append(p['yes'] / (p['yes'] + p['no']) if p['yes'] + p['no'] else None)
    return out[-n:] if len(out) >= n else None
