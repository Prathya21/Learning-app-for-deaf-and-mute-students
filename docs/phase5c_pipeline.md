# Phase 5C: ML Pipeline Prototype Documentation

**Status**: Development Fixture Only — NOT Real ISL Recognition  
**Last Updated**: 2025-09-03

---

## Overview

This document describes the ML pipeline prototype built for EduSign Phase 5C. The pipeline is designed to be **dataset-agnostic** so that the synthetic fixture can be replaced with a real landmark dataset without changing the core architecture.

**Critical**: All models and metrics in this phase are trained/evaluated on **synthetic data only**. They do NOT represent real ISL gesture recognition capability.

---

## 1. Dataset Abstraction

### Core Interface: `LandmarkDataset`

All datasets implement the `LandmarkDataset` abstract base class:

```python
class LandmarkDataset(ABC):
    def __init__(self, root_dir: Path, split: Split):
        ...
    
    @abstractmethod
    def _load_metadata(self) -> None:
        """Load class mapping, stats, etc."""
    
    @abstractmethod
    def _load_samples(self) -> None:
        """Load sample list for current split."""
    
    @abstractmethod
    def _load_sequence(self, sample: SampleMetadata) -> np.ndarray:
        """Load raw [T, 126] sequence."""
    
    def __getitem__(self, idx) -> Tuple[np.ndarray, int, SampleMetadata]:
        ...
    
    def _validate_sequence(self, sequence, sample):
        """Validates: numeric, no NaN/inf, shape [T, 126], T > 0"""
        ...
```

### SampleMetadata

```python
@dataclass
class SampleMetadata:
    sample_id: str           # Unique identifier
    class_name: str          # Gloss (e.g., "HELLO")
    class_idx: int           # Integer class index
    signer: Optional[str]    # Signer ID (nominal for synthetic)
    split: Split             # train/val/test
    source_dataset: str      # "synthetic", "INCLUDE", etc.
    filepath: str            # Relative path to .npy
    temporal_length: int     # Original T frames
    feature_dim: int         # Should be 126
    source: str              # "synthetic", "real", "augmented"
```

### Dataset Metadata

```python
@dataclass
class DatasetMetadata:
    dataset_type: str                    # "synthetic", "real", "mixed"
    intended_use: str                    # "development", "production"
    real_isl_recognition_valid: bool    # CRITICAL: False for synthetic
    feature_dimension: int = 126
    classes: int
    samples: int
    signers: int
    class_names: List[str]
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]
```

### Factory Function

```python
def create_dataset(dataset_type: str, root_dir: Path, split: Split) -> LandmarkDataset:
    if dataset_type == "synthetic":
        return SyntheticLandmarkDataset(root_dir, split)
    # Future: "include", "isl500", etc.
```

---

## 2. Sequence Contract

### Raw Sequence Format

- **Shape**: `[T, 126]` where `T` = temporal frames, `126 = 2 hands × 21 landmarks × 3 (x,y,z)`
- **Layout**: `[hand1_landmark_0_x, hand1_landmark_0_y, hand1_landmark_0_z, ..., hand2_landmark_20_z]`
- **Hand order**: Hand 0 (dominant/first detected), Hand 1 (non-dominant/second)
- **Landmark indices**: MediaPipe HandLandmarker convention (0=wrist, 1-20 fingers)
- **Coordinate system**: Normalized [0, 1] relative to image, z = depth

### Variable Length Support

- Raw sequences can have variable `T` (1–200+ frames)
- Preprocessing handles padding/truncation to configurable `T`
- Dataset validation ensures `T > 0` and `D == 126`

### Variable-Length Support in Training

The `PyTorchDataset` wrapper applies preprocessing on-the-fly, ensuring all sequences fed to the model have consistent shape `[T_target, 126]`.

---

## 3. Preprocessing Pipeline

### Configuration

```python
@dataclass
class PreprocessingConfig:
    target_temporal_length: int = 64
    padding_mode: PaddingMode = PaddingMode.ZERO      # "zero", "repeat", "reflect"
    truncation_mode: TruncationMode = TruncationMode.END  # "start", "end", "middle", "uniform"
    uniform_sampling: bool = True
    
    normalization_mode: NormalizationMode = NormalizationMode.WRIST_CENTRIC_SCALE
    # "wrist_centric", "scale", "wrist_centric_scale", "none"
    
    wrist_landmark_idx: int = 0
    missing_hand_value: float = 0.0
    interpolate_missing: bool = False
    validate_input: bool = True
    reject_nan: bool = True
    expected_feature_dim: int = 126
```

### Pipeline Steps

1. **Validation** — Check numeric, shape `[T, 126]`, no NaN/inf, `T > 0`
2. **Missing Hand Handling** — Detect all-zero hands, replace with `missing_value` or interpolate
3. **Normalization**:
   - `wrist_centric`: Translate so wrist (landmark 0) is at origin per hand
   - `scale`: Divide by hand bounding box diagonal (median over time)
   - `wrist_centric_scale`: Both (default)
4. **Temporal Sampling** — Pad/truncate to `target_temporal_length` (default 64)
   - Padding: zeros at end (configurable)
   - Truncation: from end (configurable: start/end/middle/uniform)

### Output

```python
@dataclass
class PreprocessingResult:
    sequence: np.ndarray          # [T_target, 126], float32
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    metadata: Dict[str, Any]      # Steps applied, scale factor, etc.
```

### Preprocessing Metadata Example

```json
{
  "original_shape": [82, 126],
  "processed_shape": [64, 126],
  "steps_applied": [
    "validation",
    "missing_hand_handling",
    "wrist_centric_normalize",
    "scale_normalize",
    "temporal_sampling_to_64"
  ],
  "scale_factor": 0.156
}
```

---

## 4. Baseline Architecture: `SyntheticPipelineBaseline`

### Architecture

```
Input [B, T, 126]
    ↓
BiLSTM (2 layers, hidden=128, bidirectional, dropout=0.3)
    ↓
Dropout(0.3)
    ↓
Global Average Pooling (over temporal dim)
    ↓
Dropout(0.3)
    ↓
Linear [hidden*2 → num_classes]
    ↓
Logits [B, num_classes]
```

### Configuration

```python
@dataclass
class ModelConfig:
    input_dim: int = 126
    hidden_dim: int = 128
    num_layers: int = 2
    num_classes: int = 46
    dropout: float = 0.3
    bidirectional: bool = True
    sequence_length: int = 64
    batch_first: bool = True
```

### Model Info

```python
model.get_model_info()  # Returns dict with config, parameter counts
```

**Typical Size**: ~150K parameters (lightweight for CPU inference)

### Naming Convention

**Always** refer to this model as `SyntheticPipelineBaseline` — never `ISLRecognizer` or similar.

---

## 5. Training Workflow

### Configuration

```python
@dataclass
class TrainingConfig:
    # Data
    dataset_type: str = "synthetic"
    dataset_root: str = "backend/data/synthetic_landmarks"
    batch_size: int = 32
    
    # Model
    input_dim: int = 126
    hidden_dim: int = 128
    num_layers: int = 2
    num_classes: int = 46
    dropout: float = 0.3
    bidirectional: bool = True
    sequence_length: int = 64
    
    # Training
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    save_best_only: bool = True
    seed: int = 42
    device: str = "cpu"
```

### Training Loop

```python
model, metadata = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    checkpoint_dir=Path("checkpoints"),
    class_mapping=idx_to_class,
    preprocessing_config=preprocessing_config,
)
```

### Features

- **Deterministic seeding** via `set_seed()`
- **Early stopping** with configurable patience/delta
- **Learning rate scheduling** (ReduceLROnPlateau, Cosine, None)
- **Checkpointing** with full metadata
- **Best model saving** (configurable)
- **Training history** tracked in metadata

### Checkpoint Format

```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'metadata': CheckpointMetadata(...),
    'epoch': int
}
```

### CheckpointMetadata (Critical Fields)

```python
@dataclass
class CheckpointMetadata:
    # Dataset info (PRESERVED from dataset metadata)
    dataset_type: str = "synthetic"
    source_dataset: str = "synthetic_fixture"
    real_isl_recognition_valid: bool = False  # CRITICAL
    feature_dimension: int = 126
    temporal_length: int = 64
    class_mapping: Dict[int, str]  # idx -> gloss
    
    # Training state
    epoch: int
    best_val_acc: float
    training_history: List[TrainingMetrics]
    
    # Provenance
    training_timestamp: str
    random_seed: int
    pytorch_version: str
```

**Rule**: If `dataset_metadata.real_isl_recognition_valid == False`, checkpoint **must** preserve `real_isl_recognition_valid: False`.

---

## 6. Checkpoint Format

### Saved File: `best_model.pt`

```python
{
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': {...},
    'metadata': {
        'model_name': 'SyntheticPipelineBaseline',
        'dataset_type': 'synthetic',
        'real_isl_recognition_valid': False,
        'class_mapping': {0: 'HELLO', 1: 'THANK_YOU', ...},
        'training_history': [...],
        'epoch': 15,
        'best_val_acc': 0.923,
        ...
    },
    'epoch': 15
}
```

### Loading

```python
model, optimizer, metadata = load_checkpoint(
    Path("checkpoints/best_model.pt"),
    model,
    optimizer,
    device="cpu"
)
```

---

## 7. Inference Contract

### InferenceConfig

```python
@dataclass
class InferenceConfig:
    num_classes: int = 46
    input_dim: int = 126
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    sequence_length: int = 64
    target_temporal_length: int = 64
    normalization_mode: str = "wrist_centric_scale"
    checkpoint_path: Optional[str] = None  # None = untrained model
    device: str = "cpu"
    return_probabilities: bool = True
    top_k: int = 5
```

### InferenceEngine

```python
engine = InferenceEngine(
    config=InferenceConfig(checkpoint_path="checkpoints/best_model.pt"),
    class_mapping=idx_to_class,  # idx -> gloss
    idx_to_class=idx_to_class,
)

result = engine.predict(landmark_sequence)  # [T, 126] numpy array
```

### InferenceResult

```python
@dataclass
class InferenceResult:
    predicted_class_index: int
    predicted_gloss: str          # e.g., "HELLO"
    confidence: float             # 0.0–1.0
    top_k: List[Dict]             # [{"class_index", "gloss", "probability"}, ...]
    probabilities: Dict[str, float]  # Full distribution (optional)
    model_metadata: Dict          # Includes "trained_on_synthetic_data": True
```

### Output Example

```json
{
  "predicted_class_index": 0,
  "predicted_gloss": "HELLO",
  "confidence": 0.923,
  "top_k": [
    {"class_index": 0, "gloss": "HELLO", "probability": 0.923},
    {"class_index": 1, "gloss": "THANK_YOU", "probability": 0.045},
    {"class_index": 2, "gloss": "TEACHER", "probability": 0.012}
  ],
  "probabilities": {"HELLO": 0.923, "THANK_YOU": 0.045, ...},
  "model_metadata": {
    "model_name": "SyntheticPipelineBaseline",
    "trained_on_synthetic_data": true,
    "real_isl_recognition_valid": false
  }
}
```

---

## 8. Synthetic Data Limitations

### What the Synthetic Fixture Is

- **805 samples**, 46 classes, 15 nominal "signers"
- Generated by `backend/scripts/generate_synthetic_landmarks.py` with `np.random.seed(42)`
- `dataset_metadata.json` explicitly: `"real_isl_recognition_valid": false`
- Signers are **nominal labels only** — no biomechanical variation
- Classes differentiated by **artificial spatial offsets** (`class_idx * 0.001`)

### What It Is NOT

- ❌ Real ISL gesture data
- ❌ Real MediaPipe landmarks
- ❌ Real human biomechanical variation
- ❌ Suitable for accuracy claims
- ❌ Suitable for signer-independent evaluation

### Safe Uses

✅ Pipeline plumbing (loading, batching, preprocessing)  
✅ Model architecture development  
✅ Training loop debugging  
✅ Hyperparameter exploration (synthetic only)  
✅ Overfitting detection on synthetic data  

### Unsafe Uses

❌ Real ISL accuracy claims  
❌ Signer-independent generalization claims  
❌ Architecture comparison (all get ~100% on synthetic)  
❌ Production deployment  

---

## 9. Replacing Synthetic with Real Data

### Exact Steps to Replace Synthetic Fixture

1. **Acquire Real Dataset**
   - Download INCLUDE-50 or equivalent
   - Extract videos to `backend/data/real_videos/`

2. **Extract Landmarks**
   - Run MediaPipe Hands on all videos
   - Save as `.npy` files: `[T, 126]` float32
   - Organize by class: `backend/data/real_landmarks/{class}/{signer}_{split}.npy`

2. **Create Manifest**
   - Generate `manifest.csv` with columns matching synthetic format
   - Include `source: "real"`, real signer IDs, real class names

3. **Generate Metadata**
   - Create `dataset_metadata.json` with:
     ```json
     {
       "dataset_type": "real",
       "intended_use": "production",
       "real_isl_recognition_valid": true,
       "feature_dimension": 126,
       "classes": N,
       "samples": M,
       "signers": K,
       "class_names": [...],
       "class_to_idx": {...},
       "signer_info": {...}
     }
     ```

4. **Implement Real Dataset Loader**
   ```python
   class RealLandmarkDataset(LandmarkDataset):
       def _load_metadata(self): ...
       def _load_samples(self): ...
       def _load_sequence(self, sample): ...
   ```

4. **Register in Factory**
   ```python
   def create_dataset(dataset_type, root_dir, split):
       if dataset_type == "synthetic":
           return SyntheticLandmarkDataset(root_dir, split)
       elif dataset_type == "real":
           return RealLandmarkDataset(root_dir, split)
   ```

5. **Update Training Config**
   ```python
   config = TrainingConfig(
       dataset_type="real",
       dataset_root="backend/data/real_landmarks",
       num_classes=len(real_class_names),
       class_mapping=real_class_mapping,
   )
   ```

6. **Retrain**
   - Run training pipeline with `dataset_type="real"`
   - Verify `real_isl_recognition_valid: True` in checkpoint metadata
   - Evaluate on held-out real test set with signer-independent split

6. **Update Inference**
   ```python
   engine = InferenceEngine(
       config=InferenceConfig(checkpoint_path="real_best_model.pt"),
       class_mapping=real_idx_to_class,
       idx_to_class=real_idx_to_class,
   )
   ```

---

## Testing

Run tests with:

```bash
cd backend
python -m pytest ml_pipeline/tests/ -v
```

### Test Coverage

| Module | Tests |
|--------|-------|
| `dataset` | Loading, validation, class mapping, signer splits |
| `preprocessing` | Padding, truncation, normalization, NaN handling, edge cases |
| `model` | Forward pass, config variants, parameter count, feature extraction |
| `training` | Seed, early stopping, checkpoint save/load, metrics |
| `inference` | Engine creation, output schema, metadata propagation |
| `integration` | Full pipeline smoke test |

---

## Known Limitations

| Limitation | Impact | Resolution |
|------------|--------|------------|
| Synthetic data only | No real accuracy metrics | Replace with real dataset |
| No MediaPipe integration in pipeline | Can't test end-to-end | Phase 5D |
| No signer-independent split on synthetic | Can't validate generalization | Real dataset needed |
| No class weighting | Imbalance not handled | Add `get_class_weights` in training |
| No data augmentation | Limited robustness | Add augmentations for real data |
| CPU only | Slow for large datasets | Add CUDA/MPS support |

---

## Next Phase (5D)

1. Connect `HandTracker` → `InferenceEngine` for live inference
2. Add gesture-to-text API endpoint
3. Implement real-time preprocessing (streaming)
4. Add confidence thresholding and smoothing
5. Build gesture-to-text UI in React