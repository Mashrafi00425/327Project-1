#!/usr/bin/env python3
"""Strict CPU vs GPU frequent-itemset equality plus independent small-case oracle.

Requires an actual GPU-built ./fp_growth_cuda by default. Use --cuda-bin
./fp_growth_cuda_sim solely to test the orchestration without CUDA hardware.
"""
import argparse
from collections import Counter
import csv
from itertools import combinations
import json
from pathlib import Path
import random
import subprocess
import tempfile

BASE = Path(__file__).resolve().parents[1]


def run(exe, filename, threshold, directory, tag):
    outfile, summary = directory / f'{tag}.csv', directory / f'{tag}.json'
    proc = subprocess.run([str(exe), str(filename), str(threshold),
                           '--output', str(outfile), '--summary', str(summary), '--top', '5'],
                          text=True, capture_output=True, timeout=180)
    if proc.returncode:
        raise RuntimeError(f'{exe.name} failed on {filename.name}: {proc.stderr}')
    with outfile.open(newline='', encoding='utf-8') as f:
        parsed = list(csv.DictReader(f))
    patterns = {row['itemset']: int(row['support']) for row in parsed}
    if len(parsed) != len(patterns):
        raise AssertionError(f'Duplicate patterns in {exe.name}')
    return patterns, json.loads(summary.read_text())


def oracle(rows, support):
    counts = Counter()
    for row in rows:
        tokens = sorted(set(row))
        for size in range(1, len(tokens) + 1):
            counts.update(';'.join(items) for items in combinations(tokens, size))
    return {k: v for k, v in counts.items() if v >= support}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda-bin', default='./fp_growth_cuda')
    ap.add_argument('--random-cases', type=int, default=20)
    args = ap.parse_args()
    cpu, gpu = BASE / 'fp_growth', (BASE / args.cuda_bin).resolve()
    if not cpu.exists() or not gpu.exists():
        raise SystemExit('Build first: make && make cuda')
    rng = random.Random(217)
    tests = [ (BASE / 'data/toy_transactions.txt', '3', None),
              (BASE / 'data/uci_wdbc_569.txt', '50%', None),
              (BASE / 'data/uci_digits_1797.txt', '80%', None),
              (BASE / 'data/synthetic_retail_50000.txt', '7%', None)]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for i in range(args.random_cases):
            items = [f'x{j}' for j in range(rng.randint(2, 8))]
            rows = [rng.sample(items, rng.randint(1, len(items)))
                    for _ in range(rng.randint(4, 15))]
            threshold = rng.randint(1, len(rows) + 1)
            infile = work / f'random_{i}.txt'
            infile.write_text('\n'.join(' '.join(row) for row in rows) + '\n')
            tests.append((infile, str(threshold), oracle(rows, threshold)))
        for idx, (source, threshold, expected) in enumerate(tests):
            if not source.exists():
                print(f'SKIP missing data: {source.name}')
                continue
            cpu_items, cpu_summary = run(cpu, source, threshold, work, f'cpu_{idx}')
            gpu_items, gpu_summary = run(gpu, source, threshold, work, f'gpu_{idx}')
            if cpu_items != gpu_items:
                missed = set(cpu_items.items()) ^ set(gpu_items.items())
                raise AssertionError(f'Mismatch {source.name}: {list(missed)[:8]}')
            if expected is not None and cpu_items != expected:
                raise AssertionError(f'Independent oracle mismatch on {source.name}')
            if gpu_summary['backend'] != 'cuda-hybrid-all-level-counting':
                raise AssertionError('Incorrect backend')
            assert gpu_summary['support_root_calls'] == 1
            assert gpu_summary['cuda_kernel_calls'] == (
                gpu_summary['support_root_calls'] + gpu_summary['support_conditional_calls'])
            if gpu_summary['support_conditional_calls'] == 0 and len(gpu_items) > 1:
                raise AssertionError('Expected conditional-tree GPU histograms')
            print(f'PASS {source.name:<35} support={threshold:<4} '
                  f'patterns={len(cpu_items):>5}, '
                  f'GPU kernels={gpu_summary["cuda_kernel_calls"]}, '
                  f'device={gpu_summary["cuda_device"]}')
    print(f'PASS: {len(tests)} CPU/CUDA dataset comparisons; all exact supports matched.')

if __name__ == '__main__':
    main()
