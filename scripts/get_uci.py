#!/usr/bin/env python3
"""Download ORIGINAL UCI categorical datasets and prepare baskets for FP-Growth.
Requires internet connectivity. These datasets are intentionally separate from
already-bundled offline UCI-derived benchmark copies.
"""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from io import BytesIO
import csv
import zipfile
import argparse
import subprocess

BASE = Path(__file__).resolve().parents[1] / 'data'
DATASETS = {
    'mushroom': dict(id=73, zipname='mushroom.zip', filename='agaricus-lepiota.data',
                     url_alt='https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
                     ncols=23, rows=8124),
    'chess': dict(id=22, zipname='chess+king+rook+vs+king+pawn.zip', filename='kr-vs-kp.data',
                  url_alt='https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data',
                  ncols=37, rows=3196),
    'connect4': dict(id=26, zipname='connect+4.zip', filename='connect-4.data.Z',
                     url_alt='https://archive.ics.uci.edu/ml/machine-learning-databases/connect-4/connect-4.data.Z',
                     ncols=43, rows=67557),
}


def download(url):
    request = Request(url, headers={'User-Agent': 'FP-Growth educational experiment/1.0'})
    with urlopen(request, timeout=50) as response:
        return response.read()


def get_source(key, spec):
    zipurl = f'https://archive.ics.uci.edu/static/public/{spec["id"]}/{spec["zipname"]}'
    errors = []
    for url in [zipurl, spec['url_alt']]:
        try:
            raw = download(url)
            if raw[:2] == b'PK':
                with zipfile.ZipFile(BytesIO(raw)) as z:
                    files = [f for f in z.namelist() if not f.endswith('/')]
                    matches = [x for x in files if Path(x).name == spec['filename']]
                    if not matches:
                        # Some archives contain uncompressed data rather than .Z
                        matches = [x for x in files if Path(x).name == spec['filename'].replace('.Z','')]
                    if not matches: raise ValueError(f'Expected file not found; archive has {files}')
                    data = z.read(matches[0])
                    filename = matches[0]
            else:
                data, filename = raw, spec['filename']
            if filename.endswith('.Z'):
                p = subprocess.run(['uncompress','-c'], input=data, capture_output=True)
                if p.returncode: raise RuntimeError('uncompress -c failed; use Linux/macOS uncompress or 7-Zip')
                data = p.stdout
            return data.decode('utf-8'), url
        except (URLError,HTTPError,OSError,ValueError,RuntimeError,zipfile.BadZipFile) as e:
            errors.append(f'{url}: {e}')
    raise RuntimeError('Could not download '+key+'; '+ '\n'.join(errors))


def convert(key):
    spec = DATASETS[key]
    text, url = get_source(key,spec)
    parsed = list(csv.reader(text.splitlines()))
    parsed = [row for row in parsed if row]
    if len(parsed) != spec['rows']:
        raise ValueError(f'Expected {spec["rows"]} rows, got {len(parsed)}; not saving incorrect file')
    out_path = BASE / f'uci_{key}_original.txt'
    with out_path.open('w',encoding='utf-8') as fp:
        for i,row in enumerate(parsed):
            if len(row) != spec['ncols']:
                raise ValueError(f'Row {i} has {len(row)} columns; expected {spec["ncols"]}')
            # Exclude last column (target) to avoid label leakage.
            # Position prefix makes identical attribute values distinct items.
            # '?' in mushroom is preserved as a categorical missing-value marker.
            fp.write(' '.join(f'f{j:02d}={value.strip()}' for j,value in enumerate(row[:-1]))+'\n')
    print(f'Wrote {out_path} ({len(parsed)} rows) from {url}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('datasets',nargs='*',choices=list(DATASETS)+['all'],default=['all'])
    a = p.parse_args()
    keys = list(DATASETS) if 'all' in a.datasets else a.datasets
    BASE.mkdir(parents=True, exist_ok=True)
    for key in keys:
        try: convert(key)
        except Exception as e:
            print(f'ERROR [{key}]: {e}')
    print('Example: ./fp_growth data/uci_mushroom_original.txt 70% --no-output')

if __name__ == '__main__':
    main()
