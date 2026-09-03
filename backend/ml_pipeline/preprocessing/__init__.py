"""
Preprocessing pipeline for landmark sequences.

This module provides reusable preprocessing functions for landmark sequences.
All functions are designed to be composable and configurable.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from enum import Enum


class PaddingMode(str, Enum):
    """Padding mode for temporal sequences."""
    ZERO = "zero"
    REPEAT = "repeat"
    REFLECT = "reflect"


class TruncationMode(str, Enum):
    """Truncation mode for temporal sequences."""
    START = "start"      # Keep first T frames
    END = "end"          # Keep last T frames
    MIDDLE = "middle"    # Keep middle T frames
    UNIFORM = "uniform"  # Uniformly sample T frames


class NormalizationMode(str, Enum):
    """Normalization mode for landmark coordinates."""
    WRIST_CENTRIC = "wrist_centric"
    SCALE = "scale"
    WRIST_CENTRIC_SCALE = "wrist_centric_scale"
    NONE = "none"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    # Temporal processing
    target_temporal_length: int = 64  # Target T frames
    padding_mode: PaddingMode = PaddingMode.ZERO
    truncation_mode: TruncationMode = TruncationMode.END
    uniform_sampling: bool = True  # If True, uniformly sample when truncating
    
    # Normalization
    normalization_mode: NormalizationMode = NormalizationMode.WRIST_CENTRIC_SCALE
    wrist_landmark_idx: int = 0  # Index of wrist landmark (0-20)
    
    # Missing hand handling
    missing_hand_value: float = 0.0  # Value for missing hand landmarks
    interpolate_missing: bool = False  # Whether to interpolate missing hands
    
    # Validation
    validate_input: bool = True
    reject_nan: bool = True
    expected_feature_dim: int = 126
    
    # Optional interpolation for missing frames
    interpolate_missing_frames: bool = False
    max_consecutive_missing: int = 5  # Max consecutive missing frames to interpolate


@dataclass
class PreprocessingResult:
    """Result of preprocessing a sequence."""
    sequence: np.ndarray  # Processed sequence [T, D]
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_shape": list(self.original_shape),
            "processed_shape": list(self.processed_shape),
            "metadata": self.metadata,
        }


from dataclasses import field


def validate_sequence(
    sequence: np.ndarray,
    expected_feature_dim: int = 126,
    reject_nan: bool = True
) -> np.ndarray:
    """
    Validate a landmark sequence.
    
    Args:
        sequence: Input sequence [T, D]
        expected_feature_dim: Expected feature dimension
        reject_nan: Whether to reject NaN/inf values
        
    Returns:
        Validated (and potentially cleaned) sequence
        
    Raises:
        ValueError: If validation fails and reject_nan=True
    """
    if not isinstance(sequence, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(sequence)}")
    
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D array [T, D], got {sequence.ndim}D array")
    
    if sequence.shape[1] != expected_feature_dim:
        raise ValueError(f"Expected feature dim {expected_feature_dim}, got {sequence.shape[1]}")
    
    if sequence.shape[0] == 0:
        raise ValueError("Sequence has zero temporal length")
    
    if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
        if reject_nan:
            raise ValueError("Sequence contains NaN or infinity")
        else:
            # Replace with zeros
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
    
    return sequence


def wrist_centric_normalize(
    sequence: np.ndarray,
    wrist_landmark_idx: int = 0,
    num_landmarks_per_hand: int = 21,
    num_hands: int = 2
) -> np.ndarray:
    """
    Apply wrist-centric normalization to landmark sequence.
    
    Translates all landmarks so that the wrist (landmark 0) of each hand
    is at the origin.
    
    Args:
        sequence: Input sequence [T, 126] (2 hands * 21 landmarks * 3 coords)
        wrist_landmark_idx: Index of wrist landmark within each hand (0-20)
        num_landmarks_per_hand: Number of landmarks per hand
        num_hands: Number of hands
        
    Returns:
        Wrist-centric normalized sequence [T, 126]
    """
    T, D = sequence.shape
    num_landmarks = D // 3  # 42 landmarks total (2 hands * 21)
    landmarks_per_hand = num_landmarks // 2
    
    normalized = sequence.copy().reshape(-1, 2, num_landmarks_per_hand, 3)
    
    for hand_idx in range(2):
        wrist_pos = normalized[:, hand_idx, 0:1, :]  # [T, 1, 3]
        normalized[:, hand_idx, :, :] -= wrist_pos
    
    return normalized.reshape(-1, D)


def scale_normalize(
    sequence: np.ndarray,
    scale_factor: Optional[float] = None,
    method: str = "hand_bbox"
) -> Tuple[np.ndarray, float]:
    """
    Apply scale normalization to landmark sequence.
    
    Args:
        sequence: Input sequence [T, D]
        scale_factor: Optional precomputed scale factor
        method: Scaling method ("hand_bbox", "max_abs", "std")
        
    Returns:
        Tuple of (normalized_sequence, scale_factor_used)
    """
    if scale_factor is None:
        if method == "hand_bbox":
            # Compute hand bounding box diagonal per frame, use median
            T, D = sequence.shape
            landmarks = sequence.reshape(T, -1, 3)  # [T, 42, 3]
            hand1 = landmarks[:, :21, :]  # [T, 21, 3]
            hand2 = landmarks[:, 21:, :]  # [T, 21, 3]
            
            # Bounding box diagonal for each hand
            hand1_min = np.min(hand1, axis=1)  # [T, 3]
            hand1_max = np.max(hand1, axis=1)
            hand1_diag = np.linalg.norm(hand1_max - hand1_min, axis=1)  # [T]
            
            hand2_min = np.min(hand2, axis=1)
            hand2_max = np.max(hand2, axis=1)
            hand2_diag = np.linalg.norm(hand2_max - hand2_min, axis=1)
            
            # Use max of both hands, median over time
            scale_factor = float(np.median(np.maximum(hand1_diag, hand2_diag)))
            
        elif method == "max_abs":
            scale_factor = float(np.max(np.abs(sequence)))
        elif method == "std":
            scale_factor = float(np.std(sequence))
        else:
            raise ValueError(f"Unknown scale method: {method}")
    
    if scale_factor == 0 or np.isnan(scale_factor) or np.isinf(scale_factor):
        scale_factor = 1.0
    
    normalized = sequence / scale_factor
    return normalized, scale_factor


def temporal_sample(
    sequence: np.ndarray,
    target_length: int,
    mode: TruncationMode = TruncationMode.END,
    uniform: bool = True
) -> np.ndarray:
    """
    Sample or pad sequence to target temporal length.
    
    Args:
        sequence: Input sequence [T, D]
        target_length: Target temporal length
        mode: Truncation mode if sequence is longer than target
        uniform: If True and truncating, use uniform sampling
        
    Returns:
        Sequence of length target_length
    """
    T, D = sequence.shape
    
    if T == target_length:
        return sequence.copy()
    
    if T < target_length:
        # Pad
        pad_length = target_length - T
        if D == 126:  # Landmark data
            padding = np.zeros((pad_length, 126), dtype=sequence.dtype)
        else:
            padding = np.zeros((pad_length, sequence.shape[1]), dtype=sequence.dtype)
        return np.concatenate([sequence, padding], axis=0)
    
    # T > target_length: Truncate
    if mode == TruncationMode.START:
        return sequence[:target_length].copy()
    elif mode == TruncationMode.END:
        return sequence[-target_length:].copy()
    elif mode == TruncationMode.MIDDLE:
        start = (T - target_length) // 2
        return sequence[start:start + target_length].copy()
    elif mode == TruncationMode.UNIFORM:
        if uniform:
            indices = np.linspace(0, T - 1, target_length, dtype=int)
            return sequence[indices].copy()
        else:
            return sequence[:target_length].copy()
    else:
        raise ValueError(f"Unknown truncation mode: {mode}")


def handle_missing_hands(
    sequence: np.ndarray,
    missing_value: float = 0.0,
    interpolate: bool = False
) -> np.ndarray:
    """
    Handle missing hand landmarks (all zeros).
    
    Args:
        sequence: Input sequence [T, 126]
        missing_value: Value to use for missing landmarks
        interpolate: Whether to interpolate missing values from adjacent frames
        
    Returns:
        Sequence with missing hands handled
    """
    T, D = sequence.shape
    if D != 126:
        return sequence
    
    result = sequence.copy().reshape(T, 2, 21, 3)
    
    # Detect missing hands (all zeros for a hand in a frame)
    hand1_missing = np.all(result[:, 0] == 0, axis=(1, 2))  # [T]
    hand2_missing = np.all(result[:, 1] == 0, axis=(1, 2))  # [T]
    
    if not interpolate:
        # Just ensure missing hands have the missing_value
        result[:, 0][hand1_missing] = missing_value
        result[:, 1][hand2_missing] = missing_value
    else:
        # Interpolate missing frames from adjacent non-missing frames
        for hand_idx in range(2):
            missing = hand1_missing if hand_idx == 0 else hand2_missing
            if np.any(missing) and not np.all(missing):
                # Find non-missing frames
                valid = ~missing
                if np.any(valid):
                    # Simple linear interpolation
                    valid_indices = np.where(valid)[0]
                    missing_indices = np.where(missing)[0]
                    for idx in missing_indices:
                        # Find nearest valid frames
                        before = valid_indices[valid_indices <= idx]
                        after = valid_indices[valid_indices > idx]
                        if len(before) > 0 and len(after) > 0:
                            # Linear interpolation
                            t = (idx - before[-1]) / (after[0] - before[-1])
                            result[idx, hand_idx] = (
                                (1 - t) * result[before[-1], hand_idx] +
                                t * result[after[0], hand_idx]
                            )
                        elif len(before) > 0:
                            result[idx, hand_idx] = result[before[-1], hand_idx]
                        elif len(after) > 0:
                            result[idx, hand_idx] = result[after[0], hand_idx]
    
    return result.reshape(-1, 126)


def preprocess_sequence(
    sequence: np.ndarray,
    config: PreprocessingConfig
) -> PreprocessingResult:
    """
    Apply full preprocessing pipeline to a landmark sequence.
    
    Args:
        sequence: Input sequence [T, 126]
        config: Preprocessing configuration
        
    Returns:
        PreprocessingResult with processed sequence and metadata
    """
    original_shape = sequence.shape
    seq = sequence.copy()
    
    metadata = {
        "original_T": sequence.shape[0],
        "original_D": sequence.shape[1],
        "steps_applied": [],
    }
    
    # 1. Input validation
    if config.validate_input:
        seq = validate_sequence(sequence, config.expected_feature_dim, config.reject_nan)
        metadata["steps_applied"].append("validation")
    
    # 2. Handle missing hands
    seq = handle_missing_hands(seq, config.missing_hand_value, config.interpolate_missing)
    metadata["steps_applied"].append("missing_hand_handling")
    
    # 3. Normalization
    if config.normalization_mode == NormalizationMode.WRIST_CENTRIC:
        seq = wrist_centric_normalize(seq)
        metadata["steps_applied"].append("wrist_centric_normalize")
    elif config.normalization_mode == NormalizationMode.SCALE:
        seq, scale = scale_normalize(seq)
        metadata["steps_applied"].append("scale_normalize")
        metadata["scale_factor"] = scale
    elif config.normalization_mode == NormalizationMode.WRIST_CENTRIC_SCALE:
        seq = wrist_centric_normalize(seq)
        seq, scale = scale_normalize(seq)
        metadata["steps_applied"].extend(["wrist_centric_normalize", "scale_normalize"])
        metadata["scale_factor"] = scale
    elif config.normalization_mode == NormalizationMode.NONE:
        pass
    else:
        raise ValueError(f"Unknown normalization mode: {config.normalization_mode}")
    
    # 4. Temporal sampling to target length
    seq = temporal_sample(
        seq,
        config.target_temporal_length,
        config.truncation_mode,
        config.uniform_sampling
    )
    metadata["steps_applied"].append(f"temporal_sampling_to_{config.target_temporal_length}")
    
    processed_shape = seq.shape
    
    return PreprocessingResult(
        sequence=seq.astype(np.float32),
        original_shape=original_shape,
        processed_shape=processed_shape,
        metadata=metadata
    )


def batch_preprocess(
    sequences: List[np.ndarray],
    config: PreprocessingConfig
) -> List[PreprocessingResult]:
    """Preprocess a batch of sequences."""
    return [preprocess_sequence(seq, config) for seq in sequences]