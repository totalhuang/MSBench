"""MSBench command line.

    msbench prepare     --videos videos/<model> --results results/<model>
    msbench transitions omnishotcut|transvlm|type|plan --results results/<model>
    msbench shots       --method qalign|deqa|devil|filmops --results results/<model>
    msbench face        detect|som|score --results results/<model>
    msbench judge       --metric pp|csec|csbc|vqa --results results/<model>
    msbench aggregate   --results results/<model>

Stages that call the VLM judge (transitions type, face som, judge) accept --judge-backend / --judge-url / --judge-model.
"""
import argparse

from . import judge as J


def _common(p, needs_videos=False):
    p.add_argument('--results', required=True, help='result folder of one system, e.g. results/my_model')
    p.add_argument('--benchmark', help='benchmark JSON (default data/benchmark.json)')
    p.add_argument('--keys', nargs='*', help='only these sample ids (smoke tests)')
    p.add_argument('--shard', help="k/n: the k-th of n interleaved parts of the videos (0-based)")
    if needs_videos:
        p.add_argument('--videos', required=True, help='folder of <sample_id>.mp4 files')


def _judge_opts(p):
    J.add_args(p)
    p.add_argument('--max-jobs', type=int, help='cap the number of requests (smoke tests)')
    p.add_argument('--dry-run', action='store_true', help='count requests and print one prompt; no API call')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='msbench', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)

    p = sp.add_parser('prepare', help='index (and trim) the videos of one system')
    _common(p, needs_videos=True)
    p.add_argument('--no-trim', action='store_true', help='evaluate videos longer than the script without trimming')

    p = sp.add_parser('transitions', help='transition detection and shot plan')
    p.add_argument('step', choices=['omnishotcut', 'transvlm', 'type', 'plan'])
    _common(p); _judge_opts(p)
    p.add_argument('--checkpoint', help='omnishotcut: .pth file; transvlm: checkpoint folder')

    p = sp.add_parser('shots', help='per-shot expert models (GPU)')
    _common(p)
    p.add_argument('--method', required=True, choices=['qalign', 'deqa', 'devil', 'filmops'])

    p = sp.add_parser('face', help='cross-shot face consistency')
    p.add_argument('step', choices=['detect', 'som', 'score'])
    _common(p); _judge_opts(p)

    p = sp.add_parser('judge', help='VLM-judge sub-metrics')
    _common(p); _judge_opts(p)
    p.add_argument('--metric', required=True, choices=['pp', 'csec', 'csbc', 'vqa'])

    p = sp.add_parser('aggregate', help='video- and model-level scores')
    _common(p)

    a = ap.parse_args(argv)
    if a.cmd == 'prepare':
        from . import prepare
        return prepare.run(a)
    if a.cmd == 'transitions':
        from . import transitions as T
        return {'omnishotcut': T.omnishotcut, 'transvlm': T.transvlm, 'type': T.type_, 'plan': T.plan}[a.step](a)
    if a.cmd == 'shots':
        from . import shot_metrics
        return shot_metrics.run(a)
    if a.cmd == 'face':
        from .metrics import face
        return {'detect': face.detect, 'som': face.som, 'score': face.score}[a.step](a)
    if a.cmd == 'judge':
        from .metrics import vlm
        return vlm.run(a)
    from . import aggregate
    return aggregate.run(a)


if __name__ == '__main__':
    main()
