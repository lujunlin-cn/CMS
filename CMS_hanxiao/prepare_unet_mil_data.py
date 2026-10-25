#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get('CMS_DATA_ROOT', '/home/zhaona/CMS_hanxiao/cxr_first'))
OUT = ROOT.parent / 'unet_mil_data'
EXCLUDE = ROOT.parent / 'audit' / 'exclude_train_images.csv'


def parse_labels(value):
    labels = set()
    for token in str(value or '').split(';'):
        token = token.strip()
        if not token:
            continue
        label = int(token)
        if not 1 <= label <= 10:
            raise ValueError(f'invalid class id: {value}')
        labels.add(label)
    return sorted(labels)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_exclusions():
    if not EXCLUDE.exists():
        return set()
    with EXCLUDE.open(newline='', encoding='utf-8') as f:
        return {(r['Subject_id'], r['Study_id'], r['Dicom_id']) for r in csv.DictReader(f) if r.get('action') == 'exclude_from_train_baseline_only'}


def prepare_split(split, excluded):
    with (ROOT / f'{split}.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    labels_by_study = defaultdict(set)
    subject_by_study = {}
    for row in rows:
        labels_by_study[row['Study_id']].update(parse_labels(row.get('Predict_class', '')))
        old = subject_by_study.setdefault(row['Study_id'], row['Subject_id'])
        if old != row['Subject_id']:
            raise RuntimeError(f'Study spans subjects: {split} {row["Study_id"]}')
    image_rows = []
    skipped = []
    seen = set()
    for row in rows:
        key = (row['Subject_id'], row['Study_id'], row['Dicom_id'])
        if key in seen:
            raise RuntimeError(f'duplicate key: {key}')
        seen.add(key)
        path = ROOT / split / row['Subject_id'] / row['Study_id'] / f"{row['Dicom_id']}.jpg"
        if not path.exists():
            raise FileNotFoundError(path)
        if key in excluded:
            skipped.append(key)
            continue
        image_rows.append({
            'subject_id': row['Subject_id'],
            'study_id': row['Study_id'],
            'dicom_id': row['Dicom_id'],
            'relative_path': str(path.relative_to(ROOT)),
            'view_position': row.get('ViewPosition', ''),
            'patient_orientation': row.get('PatientOrientation', ''),
            'rows': row.get('Rows', ''),
            'columns': row.get('Columns', ''),
            'labels': ';'.join(str(x) for x in sorted(labels_by_study[row['Study_id']])),
        })
    studies = defaultdict(list)
    for row in image_rows:
        studies[row['study_id']].append(row)
    study_rows = []
    for study_id in sorted(studies):
        group = studies[study_id]
        study_rows.append({
            'subject_id': group[0]['subject_id'],
            'study_id': study_id,
            'labels': [int(x) for x in group[0]['labels'].split(';') if x],
            'images': [x['relative_path'] for x in group],
            'dicom_ids': [x['dicom_id'] for x in group],
            'view_positions': [x['view_position'] for x in group],
            'patient_orientations': [x['patient_orientation'] for x in group],
        })
    return rows, image_rows, study_rows, skipped


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    excluded = load_exclusions()
    summary = {
        'created_at_epoch': time.time(),
        'root': str(ROOT),
        'external_medical_data_used': False,
        'test_used': False,
        'exclude_manifest': str(EXCLUDE),
        'splits': {},
    }
    for split in ('train', 'valid'):
        raw, image_rows, study_rows, skipped = prepare_split(split, excluded if split == 'train' else set())
        with (OUT / f'{split}_images.csv').open('w', newline='', encoding='utf-8') as f:
            fields = ['subject_id','study_id','dicom_id','relative_path','view_position','patient_orientation','rows','columns','labels']
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(image_rows)
        write_jsonl(OUT / f'{split}_studies.jsonl', study_rows)
        summary['splits'][split] = {
            'source_csv_rows': len(raw),
            'manifest_image_rows': len(image_rows),
            'manifest_studies': len(study_rows),
            'manifest_subjects': len({r['subject_id'] for r in image_rows}),
            'excluded_images': len(skipped),
            'excluded_examples': [list(x) for x in skipped[:20]],
            'label_source': 'official Study-level Predict_class',
        }
    train_subjects = {r['subject_id'] for r in json_lines(OUT / 'train_studies.jsonl')}
    valid_subjects = {r['subject_id'] for r in json_lines(OUT / 'valid_studies.jsonl')}
    train_studies = {r['study_id'] for r in json_lines(OUT / 'train_studies.jsonl')}
    valid_studies = {r['study_id'] for r in json_lines(OUT / 'valid_studies.jsonl')}
    summary['cross_split'] = {
        'subject_overlap': len(train_subjects & valid_subjects),
        'study_overlap': len(train_studies & valid_studies),
    }
    for name in ('train_images.csv','valid_images.csv','train_studies.jsonl','valid_studies.jsonl'):
        summary[name] = {'path': str(OUT / name), 'sha256': sha256(OUT / name)}
    (OUT / 'manifest_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def json_lines(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == '__main__':
    main()
