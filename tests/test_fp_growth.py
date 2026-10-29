#!/usr/bin/env python3
"""End-to-end independent exhaustive verifier for FP-Growth."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import csv
import json
import random
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / 'fp_growth'


def oracle(transactions, minimum):
    counts = Counter()
    for t in transactions:
        unique = sorted(set(t))
        for k in range(1, len(unique)+1):
            counts.update(combinations(unique, k))
    return {k: v for k, v in counts.items() if v >= minimum}


def run_miner(transactions, minimum, threshold=None):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / 'transactions.txt'
        source.write_text('\n'.join(' '.join(row) for row in transactions) + '\n', encoding='utf-8')
        cmd = [str(EXEC), str(source), str(threshold or minimum),
               '--output', str(tmp / 'out.csv'), '--summary', str(tmp / 'out.json')]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        if proc.returncode != 0:
            raise AssertionError(f'Miner exited {proc.returncode}: {proc.stderr}')
        summary = json.loads((tmp / 'out.json').read_text())
        with (tmp / 'out.csv').open(newline='', encoding='utf-8') as fp:
            lines = list(csv.DictReader(fp))
        patterns = {tuple(row['itemset'].split(';')): int(row['support']) for row in lines}
        if len(patterns) != len(lines):
            raise AssertionError('Duplicate itemset emitted')
        return patterns, summary


class TestFPGrowth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EXEC.exists():
            subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra',
                            str(ROOT / 'src/fp_growth.cpp'),'-o',str(EXEC)], check=True)

    def test_five_baskets_exact_patterns(self):
        baskets = [['bread','milk'],['bread','diaper','beer','eggs'],
                   ['milk','diaper','beer','cola'],['bread','milk','diaper','beer'],
                   ['bread','milk','diaper','cola']]
        observed, summary = run_miner(baskets, 3)
        self.assertEqual(observed, oracle(baskets,3))
        self.assertEqual(len(observed), 8)
        self.assertEqual(observed[('beer','diaper')], 3)
        self.assertEqual(summary['transactions'], 5)

    def test_duplicate_tokens_are_ignored(self):
        baskets = [['a','a','b'],['a','b','b'],['b','c'],['a','c']]
        obs, _ = run_miner(baskets,2)
        self.assertEqual(obs,oracle(baskets,2))

    def test_disjoint_singletons_and_no_patterns(self):
        baskets = [['a'],['b'],['c']]
        self.assertEqual(run_miner(baskets,1)[0],oracle(baskets,1))
        self.assertEqual(run_miner(baskets,4)[0],{})

    def test_fraction_and_percent_support(self):
        rows = [['a','b'],['a','c'],['a','b'],['b'],['a','b']]
        reference = oracle(rows,3)
        self.assertEqual(run_miner(rows,3,'0.6')[0], reference)
        self.assertEqual(run_miner(rows,3,'60%')[0], reference)

    def test_all_items_same_transaction(self):
        rows = [['a','b','c','d']]*5
        self.assertEqual(run_miner(rows,5)[0],oracle(rows,5))

    def test_random_exhaustive_oracle(self):
        rng = random.Random(217)
        for case in range(100):
            vocabulary = [f'i{x}' for x in range(rng.randint(2,8))]
            rows = []
            for _ in range(rng.randint(3,18)):
                row = rng.sample(vocabulary, rng.randint(1,len(vocabulary)))
                rng.shuffle(row)
                rows.append(row)
            minimum = rng.randint(1, len(rows)+1)
            got, summary = run_miner(rows,minimum)
            self.assertEqual(got,oracle(rows,minimum), f'random case {case}')
            self.assertEqual(summary['frequent_itemsets'],len(got))

    def test_repetition_metaphorphic_property(self):
        rows = [['x','y','z'],['x','y'],['x','z'],['y','z'],['x','y','z']]
        original,_ = run_miner(rows,3)
        triple,_ = run_miner(rows*3,9)
        self.assertEqual({items: n*3 for items,n in original.items()},triple)

if __name__ == '__main__':
    unittest.main(verbosity=2)
