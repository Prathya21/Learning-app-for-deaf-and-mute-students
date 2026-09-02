#!/usr/bin/env python3
"""
Generate synthetic landmark dataset for ML pipeline development.

This is a DEVELOPMENT FIXTURE ONLY. It contains NO real ISL data.
Generated using NumPy with a fixed seed for reproducibility.

Usage:
    python backend/scripts/generate_synthetic_landmarks.py

Output:
    backend/data/synthetic_landmarks/
        train/
        val/
        test/
        manifest.csv
        class_map.json
        signer_info.json
        dataset_metadata.json
"""

import numpy as np
import json
from pathlib import Path

# Configuration
CLASSES = [
    'HELLO', 'THANK_YOU', 'PLEASE', 'YES', 'NO', 'STUDENT', 'TEACHER',
    'LEARN', 'BOOK', 'WATER', 'GOOD', 'BAD', 'HELP', 'UNDERSTAND',
    'QUESTION', 'READ', 'WRITE', 'WHAT', 'WHERE', 'HOW',
    'GOODBYE', 'SORRY', 'OKAY', 'ME', 'YOU', 'HE', 'SHE',
    'MOTHER', 'FATHER', 'BROTHER', 'SISTER', 'FRIEND',
    'SCHOOL', 'HOME', 'HOSPITAL', 'MARKET', 'EAT', 'DRINK',
    'FOOD', 'TEA', 'COME', 'GO', 'SIT', 'STAND', 'WHEN', 'TODAY'
]

SIGNERS = [f'Signer{i:02d}' for i in range(1, 16)]  # 15 signers like ISL500
SEED = 42
T = 64  # temporal frames
DIM = 126  # 2 hands x 21 landmarks x 3 coords

# Sample counts per class (higher for priority classes)
SAMPLES_PER_CLASS = {}
for cls in CLASSES:
    if cls in ['HELLO', 'THANK_YOU', 'TEACHER', 'STUDENT', 'WATER', 'YES', 'NO', 'PLEASE']:
        SAMPLES_PER_CLASS[cls] = 25
    elif cls in ['GOOD', 'BAD', 'HELP', 'UNDERSTAND', 'SCHOOL', 'HOME', 'FRIEND']:
        SAMPLES_PER_CLASS[cls] = 20
    else:
        SAMPLES_PER_CLASS[cls] = 15

# Train/Val/Test split ratios
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}


def generate_landmarks_for_class(class_name: str, class_idx: int, signer: str, sample_idx: int):
    """Generate a single synthetic landmark sequence for a class."""
    T = 64
    np.random.seed(SEED + hash(f"{class_name}_{signer}_{sample_idx}") % 10000)

    # Temporal pattern
    t = np.linspace(0, 2 * np.pi, 64)
    base_motion = np.sin(t)[:, None] * 0.1

    # Hand 1 (dominant) - more movement
    hand1 = np.zeros((64, 21, 3))
    for landmark in range(21):
        base_x = 0.5 + (landmark % 5) * 0.03 + class_idx * 0.001
        base_y = 0.3 + (landmark // 5) * 0.04 + class_idx * 0.001
        base_z = 0.0
        hand1[:, landmark, 0] = base_x + base_motion[:, 0] + np.random.normal(0, 0.01, 64)
        hand1[:, landmark, 1] = base_y + base_motion[:, 0] * 0.5 + np.random.normal(0, 0.01, 64)
        hand1[:, landmark, 2] = base_z + np.random.normal(0, 0.02, 64)

    # Hand 2 (non-dominant) - less movement
    hand2 = np.zeros((64, 21, 3))
    for landmark in range(21):
        base_x = 0.5 + (landmark % 5) * 0.03
        base_y = 0.7 + (landmark // 5) * 0.04
        base_z = 0.0
        hand2[:, landmark, 0] = base_x + np.random.normal(0, 0.005, 64)
        hand2[:, landmark, 1] = base_y + np.random.normal(0, 0.005, 64)
        hand2[:, landmark, 2] = base_z + np.random.normal(0, 0.02, 64)

    # Combine: [T, 126] = [T, 2*21*3]
    landmarks = np.concatenate([hand1, hand2], axis=1).reshape(64, 126)

    # Wrist-centric normalization
    wrist_pos = landmarks[:, 0:3]  # landmark 0 is wrist
    landmarks = landmarks - np.repeat(wrist_pos, 42, axis=1)

    # Scale normalization
    hand_bbox = np.max(np.abs(landmarks), axis=1, keepdims=True)
    landmarks = landmarks / (hand_bbox + 1e-6)

    return landmarks.astype(np.float32)


def main():
    base_dir = Path(__file__).parent.parent / "data" / "synthetic_landmarks"
    for split in ['train', 'val', 'test']:
        (base_dir / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    class_idx_map = {cls: idx for idx, cls in enumerate(CLASSES)}
    signer_stats = {s: {'total_samples': 0} for s in SIGNERS}

    np.random.seed(42)

    for class_name in CLASSES:
        class_idx = class_idx_map[class_name]
        n_samples = SAMPLES_PER_CLASS[class_name]

        for sample_idx in range(SAMPLES_PER_CLASS[class_name]):
            signer = np.random.choice(SIGNERS)

            # Determine split
            r = np.random.random()
            if r < 0.7:
                split = 'train'
            elif r < 0.85:
                split = 'val'
            else:
                split = 'test'

            landmarks = generate_landmarks_for_class(class_name, class_idx, signer, sample_idx)

            filename = f'{class_name}_{signer}_{split}_{np.random.randint(1000):04d}.npy'
            filepath = Path('data/synthetic_landmarks') / split / filename
            full_path = Path(__file__).parent.parent / 'data' / 'synthetic_landmarks' / split / filename
            np.save(full_path, landmarks)

            manifest.append({
                'sample_id': filename.replace('.npy', ''),
                'class': class_name,
                'class_idx': class_idx_map[class_name],
                'signer': signer,
                'split': split,
                'filepath': f'{split}/{filename}',
                'T': 64,
                'dims': 126,
                'source': 'synthetic'
            })

            signer_stats[signer]['total_samples'] = signer_stats[signer].get('total_samples', 0) + 1

    # Save manifest
    manifest_path = Path(__file__).parent.parent / 'data' / 'synthetic_landmarks' / 'manifest.csv'
    with open(manifest_path, 'w') as f:
        f.write(','.join(manifest[0].keys()) + '\n')
        for m in manifest:
            f.write(','.join(str(m[k]) for k in m.keys()) + '\n')

    # Save class map
    class_map = {cls: idx for idx, cls in enumerate(CLASSES)}
    with open(Path(__file__).parent.parent / 'data' / 'synthetic_landmarks' / 'class_map.json', 'w') as f:
        json.dump(class_map, f, indent=2)

    # Save signer info (with explicit nominal label type)
    signer_info = {s: {"total_samples": stats['total_samples'], "type": "nominal_label"} for s, stats in signer_stats.items()}
    with open(Path(__file__).parent.parent / 'data' / 'synthetic_landmarks' / 'signer_info.json', 'w') as f:
        json.dump(signer_info, f, indent=2)

    # Save dataset metadata
    metadata = {
        "dataset_type": "synthetic",
        "intended_use": "pipeline_development_only",
        "real_isl_recognition_valid": False,
        "contains_human_biomechanics": False,
        "contains_media_pipe_landmarks": False,
        "class_differentiation_method": "artificial_spatial_offset_per_class",
        "signer_modeling": "nominal_labels_only_no_biomechanical_variation",
        "train_val_test_split_method": "random_assignment",
        "signer_split_independence": False,
        "noise_model": "isotropic_gaussian_per_landmark",
        "temporal_model": "sinusoidal_base_plus_gaussian_noise",
        "generated_by": "numpy.random with fixed seed 42",
        "generated_at": "2025-09-03",
        "classes": len(CLASSES),
        "samples": sum(SAMPLES_PER_CLASS.values()),
        "signers": len(SIGNERS),
        "note": "This is a development fixture for ML pipeline plumbing only. It contains NO real ISL data, NO real MediaPipe landmarks, and NO human biomechanical data. Do not use for ISL recognition evaluation."
    }
    with open(Path(__file__).parent.parent / 'data' / 'synthetic_landmarks' / 'dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {len(SAMPLES_PER_CLASS)} classes, {sum(SAMPLES_PER_CLASS.values())} samples")
    print(f"Train: {sum(1 for m in manifest if m['split']=='train')}")
    print(f"Val: {sum(1 for m in manifest if m['split']=='val')}")
    print(f"Test: {sum(1 for m in manifest if m['split']=='test')}")


if __name__ == '__main__':
    main()