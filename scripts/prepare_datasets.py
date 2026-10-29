#!/usr/bin/env python3
"""Prepare offline, UCI-derived samples and reproducible synthetic stress tests.

Depends on scikit-learn and NumPy only for offline data preparation.
The C++ miner itself has no third-party dependency.
"""
from pathlib import Path
import argparse
import csv
import random
from sklearn.datasets import load_breast_cancer, load_digits
import numpy as np

BASE = Path(__file__).resolve().parents[1] / 'data'


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as out:
        for row in rows:
            out.write(' '.join(row) + '\n')


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    # Feature-wise nonzero intensities, deliberately omit the target label.
    # Only pixel-positive features are included; absence is represented by no token.
    rows = [[f'pixel{j:02d}=ink' for j, px in enumerate(sample) if px > 0]
            for sample in digits.data]
    assert len(rows) == 1797
    write_rows(BASE / 'uci_digits_1797.txt', rows)
    print('Created UCI Optical Recognition (test copy):', len(rows), 'transactions')
    # Scaling test by repeating the EXACT UCI-derived observations (no new information).
    # If support fraction is unchanged, pattern types should match and counts scale x30.
    write_rows(BASE / 'uci_digits_repeated_53910.txt', rows * 30)
    print('Created replicated UCI digits scaling set:', len(rows) * 30)

    cancer = load_breast_cancer()
    # Convert continuous attributes into binary, per-feature median bins.
    medians = np.median(cancer.data, axis=0)
    features = [[f'feature{j:02d}=' + ('high' if x > medians[j] else 'low')
                 for j, x in enumerate(sample)] for sample in cancer.data]
    assert len(features) == 569
    write_rows(BASE / 'uci_wdbc_569.txt', features)
    print('Created UCI Wisconsin Diagnostic (copy):', len(features), 'transactions')

    # Independent synthetic market baskets: reproducible, NOT from UCI.
    rng = random.Random(3582026)
    vocab = [f'product{j:03d}' for j in range(120)]
    weights = [1.0 / ((j + 4) ** 0.86) for j in range(len(vocab))]
    retail = []
    for _ in range(50000):
        count = rng.randint(7, 13)
        # weighted draws may repeat; de-duplicate to enforce set semantics
        row = sorted(set(rng.choices(vocab, weights=weights, k=count)))
        retail.append(row)
    write_rows(BASE / 'synthetic_retail_50000.txt', retail)
    print('Created synthetic retail stress test:', len(retail), 'transactions')

    with (BASE / 'DATA_SOURCES.txt').open('w', encoding='utf-8') as f:
        f.write('uci_digits_1797.txt: sklearn.datasets.load_digits(): a UCI Optical Recognition of Handwritten Digits test-set copy; 1,797 rows. Features transformed to pixelNN=ink for nonzero pixels; class label excluded.\n')
        f.write('uci_digits_repeated_53910.txt: 30 repeats of same 1,797 rows, for synthetic scaling only; not 53,910 independent UCI samples.\n')
        f.write('uci_wdbc_569.txt: sklearn.datasets.load_breast_cancer(): UCI Wisconsin Diagnostic Breast Cancer copy; 569 rows. Per-feature above-median/high vs at-or-below-median/low; class label excluded.\n')
        f.write('synthetic_retail_50000.txt: independently generated 50k synthetic baskets, seed 3582026; NOT an actual UCI dataset.\n')
        f.write('Sources: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic and https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits ; https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html\n')

if __name__ == '__main__':
    main()
