#!/usr/bin/env python3
"""Reproducible CPU/CUDA comparison on same machine, written to results/.

No fake GPU measurements: requires an actual CUDA build and GPU runtime.
"""
import argparse
import csv
import json
import platform
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIG = [
    ('toy','toy_transactions.txt','3'),
    ('WDBC','uci_wdbc_569.txt','50%'),
    ('Digits','uci_digits_1797.txt','80%'),
    ('Digits','uci_digits_1797.txt','70%'),
    ('Digits x30 (replicated)','uci_digits_repeated_53910.txt','80%'),
    ('Retail synthetic 50k','synthetic_retail_50000.txt','7%'),
    ('Retail synthetic 50k','synthetic_retail_50000.txt','5%'),
    ('UCI mushroom original','uci_mushroom_original.txt','90%'),
    ('UCI chess original','uci_chess_original.txt','90%'),
    ('UCI connect4 original','uci_connect4_original.txt','95%'),
]


def run(exe, infile, minimum, timeout):
    p = subprocess.run([str(exe), str(infile), minimum, '--no-output', '--summary', '-', '--top', '0'],
                       text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'{exe.name}: {p.stderr.strip()}')
    return json.loads(p.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--timeout', type=int, default=300)
    a = ap.parse_args()
    if a.repeats <= 0:
        raise SystemExit('--repeats must be positive')
    cpu, gpu = ROOT / 'fp_growth', ROOT / 'fp_growth_cuda'
    if not cpu.exists() or not gpu.exists():
        raise SystemExit('First build BOTH binaries: make && make cuda')
    records = []
    for title, filename, support in CONFIG:
        source = ROOT / 'data' / filename
        if not source.exists():
            print('Skipping missing:', filename)
            continue
        cr, gr = [], []
        for i in range(a.repeats):
            # Alternate order between repetitions to reduce thermal bias.
            if i % 2:
                g, c = run(gpu, source, support, a.timeout), run(cpu, source, support, a.timeout)
            else:
                c, g = run(cpu, source, support, a.timeout), run(gpu, source, support, a.timeout)
            if c['frequent_itemsets'] != g['frequent_itemsets'] or c['max_pattern_length'] != g['max_pattern_length']:
                raise AssertionError(f'CPU-GPU summary mismatch: {filename} {support}')
            cr.append(c); gr.append(g)
        cmed = statistics.median(x['runtime_seconds'] for x in cr)
        gmed = statistics.median(x['runtime_seconds'] for x in gr)
        r = dict(dataset=title, filename=filename, min_support=support,
                 transactions=cr[0]['transactions'], patterns=cr[0]['frequent_itemsets'],
                 cpu_median_seconds=round(cmed,6), cuda_median_seconds=round(gmed,6),
                 cpu_peak_rss_mib=round(statistics.median(x['peak_rss_kib'] for x in cr)/1024,3),
                 cuda_peak_rss_mib=round(statistics.median(x['peak_rss_kib'] for x in gr)/1024,3),
                 cuda_buffer_max_mib=round(max(x['cuda_buffer_high_water_bytes'] for x in gr)/1024**2,3),
                 cuda_kernels=gr[0]['cuda_kernel_calls'],
                 cuda_conditional_kernels=gr[0]['support_conditional_calls'],
                 gpu_device=gr[0]['cuda_device'], speedup_cpu_over_gpu=round(cmed/gmed,3))
        records.append(r)
        print(f'{title:<27} {support:<4} patterns {r["patterns"]:<7} '
              f'CPU {cmed:.4f}s CUDA {gmed:.4f}s CPU/CUDA {r["speedup_cpu_over_gpu"]:.3f}x')
    if not records: raise RuntimeError('No data files found')
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    with (out / 'cuda_benchmark_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (out / 'cuda_benchmark_results.json').write_text(json.dumps({
        'platform': platform.platform(), 'description': 'Fresh CPU/CUDA SAME-MACHINE actual measurements',
        'repeats': a.repeats, 'datasets': records}, indent=2) + '\n')
    print('Saved results/cuda_benchmark_results.csv and results/cuda_benchmark_results.json')

if __name__ == '__main__': main()
