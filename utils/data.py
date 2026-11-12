"""Explicit-label ImageNet-compatible preparation without model-based filtering."""
import csv
import random
from pathlib import Path

from PIL import Image

from .io import read_json, sha256, write_json

SPLIT_SEED = 20261027


def split_ids(ids, seed=SPLIT_SEED, development=300, pilot=24):
    ordered = sorted(ids)
    if len(ordered) < development or not 0 <= pilot <= development:
        raise ValueError('Dataset must contain at least the requested development size')
    random.Random(seed).shuffle(ordered)
    return {identity: ('pilot' if index < pilot else 'dev' if index < development else 'holdout')
            for index, identity in enumerate(ordered)}


def read_labels(path, label_offset, id_column=None, label_column=None):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = {name.lower().replace('_', '').replace(' ', ''): name for name in reader.fieldnames or []}
        def column(explicit, candidates):
            if explicit:
                if explicit not in (reader.fieldnames or []):
                    raise ValueError(f'CSV has no column {explicit}')
                return explicit
            for candidate in candidates:
                if candidate in columns:
                    return columns[candidate]
            raise ValueError(f'Cannot recognize CSV columns {reader.fieldnames}; specify columns explicitly')
        id_column = column(id_column, ['imageid', 'id', 'filename', 'image'])
        label_column = column(label_column, ['truelabel', 'label', 'classid', 'class'])
        result = []
        seen = set()
        for row in reader:
            identity = Path(row[id_column].strip()).stem
            if not identity or identity in seen or any(c in identity for c in '/\\:') or identity in {'.', '..'}:
                raise ValueError(f'Duplicate or invalid image ID: {identity!r}')
            original = int(row[label_column])
            label = original - label_offset
            if not 0 <= label < 1000:
                raise ValueError(f'Invalid 0-based label {label} for {identity}; check --label-offset')
            result.append((identity, original, label))
            seen.add(identity)
    if not result:
        raise ValueError('Empty labels CSV')
    return sorted(result)


def prepare(images_dir, labels_csv, output, label_offset, id_column=None, label_column=None,
            expected_count=1000, development=300, pilot=24):
    images_dir, output = Path(images_dir).resolve(), Path(output).resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError('Preparation already exists; choose a new output directory')
    labels = read_labels(labels_csv, label_offset, id_column, label_column)
    if len(labels) != expected_count:
        raise ValueError(f'Expected {expected_count} images, got {len(labels)}')
    files = {}
    for path in sorted(images_dir.rglob('*')):
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            if path.stem in files:
                raise ValueError(f'Ambiguous image ID {path.stem}')
            files[path.stem] = path
    partitions = split_ids([entry[0] for entry in labels], development=development, pilot=pilot)
    records = []
    for identity, original_label, label in labels:
        if identity not in files:
            raise FileNotFoundError(f'No image for {identity}')
        source = files[identity]
        destination = output / 'clean' / f'{identity}.png'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            original_size = list(image.size)
            image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR).save(destination)
        records.append({'id': identity, 'original_path': str(source), 'canonical_path': str(destination),
                        'original_sha256': sha256(source), 'canonical_sha256': sha256(destination),
                        'original_size': original_size, 'original_label': original_label,
                        'label': label, 'split': partitions[identity]})
    manifest = {'schema_version': 1, 'seed': SPLIT_SEED, 'label_offset': label_offset,
                'labels_csv': str(Path(labels_csv).resolve()), 'labels_sha256': sha256(labels_csv),
                'preprocessing': {'resize': [224, 224], 'library': 'Pillow', 'interpolation': 'bilinear',
                                  'quantization': 'PIL RGB uint8 resize output saved losslessly as PNG',
                                  'protocol': 'shared local canonical224; not original paper 299 protocol'},
                'counts': {split: sum(r['split'] == split for r in records) for split in ('pilot', 'dev', 'holdout')},
                'records': records}
    write_json(output / 'manifest.json', manifest)
    return manifest


def select_records(manifest_path, split, allow_holdout=False, limit=None, ids=None):
    splits = split.split(',')
    if not splits or any(s not in {'pilot', 'dev', 'holdout'} for s in splits):
        raise ValueError('Specify split from pilot,dev,holdout (comma-separated)')
    if 'holdout' in splits and not allow_holdout:
        raise ValueError('Holdout access requires --allow-holdout')
    records = [r for r in read_json(manifest_path)['records'] if r['split'] in splits]
    if ids:
        wanted = set(ids.split(','))
        selected = [r for r in records if r['id'] in wanted]
        if {r['id'] for r in selected} != wanted:
            raise ValueError('Requested IDs are absent from selected split')
        records = selected
    if limit is not None:
        if limit <= 0:
            raise ValueError('--limit must be positive')
        records = records[:limit]
    if not records:
        raise ValueError('No selected records')
    return records
