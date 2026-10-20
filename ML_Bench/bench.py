"""Benchmark Core ML GPT latency across precisions (fp16/fp32) and compute units.

Run (from repo root):  python -m ML_Bench.bench [--precision fp16 fp32] [--warmup 10] [--iters 100]
Outputs: ML_Bench/results/bench_<model>.{txt,csv,json}
"""
import argparse
import csv
import io
import json
import platform
import subprocess
import time
from contextlib import redirect_stdout
from datetime import datetime

import coremltools as ct
import numpy as np

import config
from ML_Bench.plan import RESULTS_DIR, print_table, section

# The ANE only runs fp16, so CPU_AND_NE on an fp32 model would just fall back to CPU.
COMPUTE_MAP = {
    'fp16': [ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_NE],
    'fp32': [ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.CPU_AND_GPU],
}
STATS = ['mean', 'std', 'min', 'p50', 'p90', 'p99', 'max']


def bench_one(path, units, x, warmup, iters):
    """Load (includes on-device compile), warm up, then time `iters` predicts. Returns (load_ms, times_ms)."""
    t0 = time.perf_counter()
    model = ct.models.MLModel(str(path), compute_units=units)
    load_ms = (time.perf_counter() - t0) * 1000

    # Warmup removes first-call effects (kernel compilation, memory allocation, caches)
    for _ in range(warmup):
        model.predict(x)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(x)
        times.append((time.perf_counter() - t0) * 1000)

    return load_ms, np.array(times)


def summarize(times):
    return {
        'mean': times.mean(), 'std': times.std(), 'min': times.min(),
        'p50': np.percentile(times, 50), 'p90': np.percentile(times, 90), 'p99': np.percentile(times, 99),
        'max': times.max(),
    }


def machine_info():
    try:
        chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        chip = platform.processor()
    return {'chip': chip, 'macos': platform.mac_ver()[0], 'coremltools': ct.__version__}


# ── Pretty printing ──────────────────────────────────────────────────────────

def report(records, meta):
    print(f"Machine       : {meta['machine']['chip']}  (macOS {meta['machine']['macos']}, coremltools {meta['machine']['coremltools']})")
    print(f"Input         : token_ids (1, {meta['context_len']}) int32")
    print(f"Iterations    : {meta['iters']} timed  (+{meta['warmup']} warmup)")
    print(f"Timestamp     : {meta['timestamp']}")

    for precision in dict.fromkeys(r['precision'] for r in records):
        rows = [r for r in records if r['precision'] == precision]
        cpu = next((r['mean'] for r in rows if r['compute_units'] == 'CPU_ONLY'), None)
        fastest = min(r['mean'] for r in rows)

        section(f"{precision}  —  {rows[0]['model']}  (latency in ms)")
        print_table(
            ['units', *STATS, 'load', 'time / CPU', ''],
            [[r['compute_units'], *(f'{r[s]:.2f}' for s in STATS), f"{r['load_ms']:.0f}",
              f"{r['mean'] / cpu:.2f}x" if cpu else '—',  # <1 = faster than CPU_ONLY
              '← fastest' if r['mean'] == fastest else '']
             for r in rows],
            'l' + 'r' * (len(STATS) + 2) + 'l',
        )

    if len({r['precision'] for r in records}) > 1:
        section('fp16 vs fp32  (mean ms, same compute units)')
        by_key = {(r['precision'], r['compute_units']): r['mean'] for r in records}
        shared = [u for u in dict.fromkeys(r['compute_units'] for r in records)
                  if ('fp16', u) in by_key and ('fp32', u) in by_key]
        print_table(
            ['units', 'fp16', 'fp32', 'fp32 / fp16'],
            [[u, f"{by_key['fp16', u]:.2f}", f"{by_key['fp32', u]:.2f}",
              f"{by_key['fp32', u] / by_key['fp16', u]:.2f}x"] for u in shared],
            'lrrr',
        )


# ── Saving ───────────────────────────────────────────────────────────────────

def save(records, raw_times, meta, text):
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / 'bench_gpt_shakespeare'

    stem.with_suffix('.txt').write_text(text)

    with open(stem.with_suffix('.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['precision', 'compute_units', 'model', *STATS, 'load_ms'])
        w.writeheader()
        for r in records:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    with open(stem.with_suffix('.json'), 'w') as f:
        json.dump({**meta,
                   'results': [{**r, 'times_ms': t.round(4).tolist()} for r, t in zip(records, raw_times)]},
                  f, indent=2)

    return stem


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--precision', nargs='+', default=list(COMPUTE_MAP), choices=COMPUTE_MAP)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    x = {'token_ids': np.zeros((1, config.CONTEXT_LEN), dtype=np.int32)}
    meta = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'machine': machine_info(),
        'context_len': config.CONTEXT_LEN,
        'warmup': args.warmup,
        'iters': args.iters,
    }

    records, raw_times = [], []
    for precision in args.precision:
        path = config.MLPACKAGE_PATHS[precision]
        if not path.exists():
            parser.error(f'{path} not found — export it first: python -m export.export_coreml --precision {precision}')

        for units in COMPUTE_MAP[precision]:
            print(f'  running {precision} / {units.name} …', flush=True)
            load_ms, times = bench_one(path, units, x, args.warmup, args.iters)
            records.append({'precision': precision, 'compute_units': units.name, 'model': path.stem,
                            **{k: float(v) for k, v in summarize(times).items()}, 'load_ms': load_ms})
            raw_times.append(times)

    # Capture the report so the exact same text goes to the terminal and the .txt file
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(records, meta)
    text = buf.getvalue()
    print('\n' + text, end='')

    stem = save(records, raw_times, meta, text)
    print(f'\nSaved → {stem.relative_to(config.ROOT)}.{{txt,csv,json}}')
