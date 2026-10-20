"""Inspect Core ML's compute plan: which device each op runs on, where it could run, and its estimated cost.

Run (from repo root):  python -m ML_Bench.plan [--precision fp16|fp32] [--units ALL|CPU_AND_NE|CPU_AND_GPU|CPU_ONLY] [--show-inputs]
Outputs: ML_Bench/results/plan_<model>_<units>.{txt,csv,json}
"""
import argparse
import csv
import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

import config

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / 'results'

DEVICE_NAMES = {
    'MLCPUComputeDevice': 'CPU',
    'MLGPUComputeDevice': 'GPU',
    'MLNeuralEngineComputeDevice': 'ANE',
}
DEVICE_ORDER = ['CPU', 'GPU', 'ANE']


def device_name(device):
    return DEVICE_NAMES.get(type(device).__name__, type(device).__name__)


def collect_ops(plan):
    """One record per compute op (consts are skipped; they're weights/literals, not work)."""
    main = plan.model_structure.program.functions['main']
    records, n_const = [], 0

    for op in main.block.operations:
        if op.operator_name == 'const':
            n_const += 1
            continue

        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        cost = plan.get_estimated_cost_for_mlprogram_operation(op)

        records.append({
            'idx': len(records),
            'op': op.operator_name.split('.')[-1],  # 'ios16.linear' -> 'linear'
            'output': ', '.join(o.name for o in op.outputs),
            'device': device_name(usage.preferred_compute_device) if usage else None,
            'supported': [device_name(d) for d in usage.supported_compute_devices] if usage else [],
            'cost': cost.weight if cost else None,
            'inputs': {k: [b.name for b in arg.bindings if b.name] for k, arg in op.inputs.items()},
        })

    return records, n_const


# ── Pretty printing ──────────────────────────────────────────────────────────

def print_table(headers, rows, align):
    """align: string of 'l'/'r' per column."""
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

    def fmt(row):
        return '  '.join(
            str(c).ljust(w) if a == 'l' else str(c).rjust(w)
            for c, w, a in zip(row, widths, align)
        )

    print(fmt(headers))
    print('  '.join('─' * w for w in widths))
    for row in rows:
        print(fmt(row))


def section(title):
    print(f'\n{title}\n{"═" * len(title)}')


def pct(x):
    return '—' if x is None else f'{100 * x:6.2f}%'


def supported_flags(supported):
    return ' '.join(d if d in supported else '·' * len(d) for d in DEVICE_ORDER)


def truncate(s, n):
    return s if len(s) <= n else s[:n - 1] + '…'


def report(records, n_const, model_name, units, show_inputs):
    total_cost = sum(r['cost'] or 0 for r in records)

    print(f'Model         : {model_name}')
    print(f'Compute units : {units}')
    print(f'Ops           : {len(records)} compute  (+{n_const} const, hidden)')
    print(f'Total cost    : {total_cost:.4f}  (relative units; percentages below are of this total)')

    # 1. Per-op table
    section('Operations')
    headers = ['#', 'op', 'output', 'device', 'supported', 'cost']
    align = 'rllllr'
    if show_inputs:
        headers.append('inputs')
        align += 'l'
    rows = []
    for r in records:
        row = [r['idx'], r['op'], truncate(r['output'], 28), r['device'] or '—',
               supported_flags(r['supported']), pct(r['cost'] / total_cost if r['cost'] else None)]
        if show_inputs:
            row.append(truncate(', '.join(f'{k}={"|".join(v)}' for k, v in r['inputs'].items()), 70))
        rows.append(row)
    print_table(headers, rows, align)

    # 2. By device
    section('By device')
    by_dev = defaultdict(lambda: [0, 0.0])
    for r in records:
        by_dev[r['device'] or '—'][0] += 1
        by_dev[r['device'] or '—'][1] += r['cost'] or 0
    print_table(
        ['device', 'ops', 'cost share'],
        [[d, n, pct(c / total_cost)] for d, (n, c) in sorted(by_dev.items(), key=lambda kv: -kv[1][1])],
        'lrr',
    )

    # 3. By op type
    section('By op type')
    by_op = defaultdict(lambda: {'n': 0, 'cost': 0.0, 'devices': set(), 'supported': set()})
    for r in records:
        agg = by_op[r['op']]
        agg['n'] += 1
        agg['cost'] += r['cost'] or 0
        agg['devices'].add(r['device'] or '—')
        agg['supported'].update(r['supported'])
    print_table(
        ['op', 'count', 'cost share', 'ran on', 'supported'],
        [[op, a['n'], pct(a['cost'] / total_cost), '/'.join(sorted(a['devices'])), supported_flags(a['supported'])]
         for op, a in sorted(by_op.items(), key=lambda kv: -kv[1]['cost'])],
        'lrrll',
    )

    # 4. ANE-eligible ops placed elsewhere
    section('ANE-eligible but not on ANE')
    missed = [r for r in records if 'ANE' in r['supported'] and r['device'] != 'ANE']
    missed_cost = sum(r['cost'] or 0 for r in missed)
    no_ane = [r for r in records if 'ANE' not in r['supported']]
    print(f'{len(missed)} / {len(records)} ops could run on ANE but were placed elsewhere '
          f'({pct(missed_cost / total_cost).strip()} of cost)')
    if no_ane:
        print(f'{len(no_ane)} ops have no ANE support: '
              + ', '.join(sorted({r["op"] for r in no_ane})))


# ── Saving ───────────────────────────────────────────────────────────────────

def save(records, n_const, text, model_name, units):
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / f'plan_{model_name}_{units}'

    stem.with_suffix('.txt').write_text(text)

    with open(stem.with_suffix('.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['idx', 'op', 'output', 'device', 'supported', 'cost', 'inputs'])
        w.writeheader()
        for r in records:
            w.writerow({**r,
                        'supported': '|'.join(r['supported']),
                        'inputs': json.dumps(r['inputs'])})

    with open(stem.with_suffix('.json'), 'w') as f:
        json.dump({'model': model_name, 'compute_units': units, 'n_const': n_const, 'ops': records}, f, indent=2)

    return stem


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--precision', default='fp16', choices=config.MLPACKAGE_PATHS,
                        help='which exported model to inspect (paths from config.MLPACKAGE_PATHS)')
    parser.add_argument('--model', type=Path, help='explicit .mlpackage path; overrides --precision')
    parser.add_argument('--units', default='ALL', choices=[u.name for u in ct.ComputeUnit])
    parser.add_argument('--show-inputs', action='store_true', help='add an inputs column to the op table')
    args = parser.parse_args()

    units = ct.ComputeUnit[args.units]
    model_path = args.model or config.MLPACKAGE_PATHS[args.precision]
    if not model_path.exists():
        parser.error(f'{model_path} not found — export it first: python -m export.export_coreml --precision {args.precision}')
    model_name = model_path.stem

    # Load model, compile, and get the compute plan
    mlmodel = ct.models.MLModel(str(model_path), compute_units=units)
    plan = MLComputePlan.load_from_path(path=mlmodel.get_compiled_model_path(), compute_units=units)

    records, n_const = collect_ops(plan)

    # Capture the report so the exact same text goes to the terminal and the .txt file
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(records, n_const, model_name, args.units, args.show_inputs)
    text = buf.getvalue()
    print(text, end='')

    stem = save(records, n_const, text, model_name, args.units)
    print(f'\nSaved → {stem.relative_to(config.ROOT)}.{{txt,csv,json}}')
