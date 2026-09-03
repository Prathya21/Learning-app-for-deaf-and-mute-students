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

CANONICAL LANDMARK CONTRACT:
============================
Frame shape: [126]
Layout:
  LEFT_HAND:  21 landmarks × (x, y, z) = 63 values
  RIGHT_HAND: 21 landmarks × (x, y, z) = 63 values

Ordering:
  1. LEFT_HAND always first (indices 0-62)
  2. RIGHT_HAND always second (indices 63-125)

Landmark ordering: MediaPipe indices 0-20 (0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky)
Coordinate format: x, y, z per landmark (MediaPipe normalized [0,1] coordinates)
Missing hand: All 63 values = 0.0

Normalization: Done in preprocessing pipeline (wrist_centric_scale), NOT in generator.
Generator produces raw MediaPipe-style coordinates in [0,1] range.

Raw frame: [126]
Raw sequence: [N, 126]
Model input: [64, 126] after temporal sampling
"""

import numpy as np
import json
from pathlib import Path
from enum import Enum


class HandSide(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


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

# Probability of missing hand per sample (for realism)
MISSING_HAND_PROB = 0.15


def generate_hand_landmarks(
    class_idx: int,
    hand_side: HandSide,
    is_dominant: bool,
    sample_seed: int
) -> np.ndarray:
    """
    Generate raw landmark coordinates for one hand.
    
    Returns: [64, 63] array (64 frames, 21 landmarks × 3 coords)
    Coordinates in [0, 1] range (MediaPipe normalized format).
    NO normalization applied - raw MediaPipe-style coordinates.
    """
    np.random.seed(seed=sample_seed)
    
    T = 64
    landmarks = np.zeros((T, 21, 3), dtype=np.float32)
    
    # Base positions differ by hand side (anatomically plausible)
    if hand_side == HandSide.LEFT:
        base_x_center = 0.3  # Left hand more to the left
        base_y_center = 0.5
    else:
        base_x_center = 0.7  # Right hand more to the right
        base_y_center = 0.5
    
    base_z = 0.0
    
    # Dominant hand has more movement
    motion_amplitude = 0.15 if is_dominant else 0.08
    
    t = np.linspace(0, 2 * np.pi, 64)
    base_motion = np.sin(t)[:, None] * motion_amplitude
    
    for landmark_idx in range(21):
        # Anatomically plausible landmark positions
        # Wrist at center, fingers spread out
        if landmark_idx == 0:  # Wrist
            lx, ly = base_x_center, base_y_center
        elif landmark_idx <= 4:  # Thumb
            lx = base_x_center + (landmark_idx - 2) * 0.03
            ly = base_y_center - 0.05
        elif landmark_idx <= 8:  # Index
            lx = base_x_center + 0.05
            ly = base_y_center - (landmark_idx - 4) * 0.04
        elif landmark_idx <= 12:  # Middle
            lx = base_x_center + 0.08
            ly = base_y_center - (landmark_idx - 8) * 0.04
        elif landmark_idx <= 16:  # Ring
            lx = base_x_center + 0.10
            ly = base_y_center - (landmark_idx - 12) * 0.04
        else:  # Pinky
            lx = base_x_center + 0.12
            ly = base_y_center - (landmark_idx - 16) * 0.04
        
        landmarks[:, landmark_idx, 0] = lx + base_motion[:, 0] + np.random.normal(0, 0.008, 64)
        landmarks[:, landmark_idx, 1] = ly + base_motion[:, 0] * 0.5 + np.random.normal(0, 0.008, 64)
        landmarks[:, landmark_idx, 2] = base_z + np.random.normal(0, 0.015, 64)
    
    # Ensure coordinates stay in [0, 1] range
    landmarks = np.clip(landmarks, 0.0, 1.0)
    
    return landmarks.reshape(64, 63)  # [T, 63]


def generate_missing_hand() -> np.ndarray:
    """Generate all-zero hand (63 zeros)."""
    return np.zeros((64, 63), dtype=np.float32)


def generate_landmarks_for_class(
    class_idx: int,
    signer: str,
    sample_idx: int,
    class_seed: int
) -> np.ndarray:
    """
    Generate a complete landmark sequence for a class sample.
    
    Returns: [64, 126] array (LEFT_HAND_63 + RIGHT_HAND_63)
    Coordinates in [0, 1] range, NO normalization applied.
    """
    np.random.seed(SEED + hash(f"{class_idx}_{signer}_{sample_idx}") % 10000)
    
    # Determine which hands are present
    has_left = np.random.random() > 0.1  # 90% chance left hand present
    has_right = np.random.random() > 0.1  # 90% chance right hand present
    
    # At least one hand must be present
    if not has_left and not has_right:
        has_left = True
    
    # Determine dominant hand (for movement amplitude)
    dominant_is_left = np.random.random() > 0.5
    
    # Generate left hand
    if has_left:
        left_seed = SEED + hash(f"{signer}_{sample_idx}_LEFT") % 10000
        left_hand = generate_hand_landmarks(
            class_idx=0,
            hand_side=HandSide.LEFT,
            is_dominant=dominant_is_left,
            sample_seed=left_seed
        )
    else:
        left_hand = generate_missing_hand()
    
    # Generate right hand
    if has_right:
        right_seed = SEED + hash(f"{signer}_{sample_idx}_RIGHT") % 10000
        right_hand = generate_hand_landmarks(
            class_idx=0,
            hand_side=HandSide.RIGHT,
            is_dominant=not dominant_is_left,
            sample_seed=right_seed
        )
    else:
        right_hand = generate_missing_hand()
    
    # Combine: LEFT_HAND first (63), RIGHT_HAND second (63) = 126
    landmarks = np.concatenate([left_hand, right_hand], axis=1)  # [64, 126]
    
    # Ensure valid range
    landmarks = np.clip(landmarks, 0.0, 1.0)
    
    return landmarks.astype(np.float32)


def main():
    from pathlib import Path
    import json
    
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
        
        # Deterministic class seed
        class_seed = hash(class_name) % 10000
        
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
            
            landmarks = generate_landmarks_for_class(
                class_idx, signer, sample_idx, class_seed
            )
            
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
        "landmark_contract_version": "1.0",
        "landmark_format": "LEFT_HAND_63 + RIGHT_HAND_63",
        "hand_ordering": "LEFT_HAND_first_RIGHT_HAND_second",
        "landmark_indices": "MediaPipe_0_to_20",
        "coordinate_format": "x_y_z_per_landmark_normalized_0_to_1",
        "missing_hand_representation": "zeros_63",
        "normalization_location": "preprocessing_pipeline_only",
        "note": "This is a development fixture for ML pipeline plumbing only. It contains NO real ISL data, NO real MediaPipe landmarks, and NO human biomechanical data. Do not use for ISL recognition evaluation."
    }
    with open(Path(__file__).parent.parent / 'data' / 'synthetic_landmarks' / 'dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {len(CLASSES)} classes, {sum(SAMPLES_PER_CLASS.values())} samples")
    print(f"Train: {sum(1 for m in manifest if m['split']=='train')}")
    print(f"Val: {sum(1 for m in manifest if m['split']=='val')}")
    print(f"Test: {sum(1 for m in manifest if m['split']=='test')}")


if __name__ == '__main__':
    main()