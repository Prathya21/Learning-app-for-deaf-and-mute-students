"""
Dataset contract and base classes for landmark sequence datasets.

This module defines the abstract interface that all landmark sequence datasets
must implement. It is designed to be dataset-agnostic so that synthetic fixtures
can be replaced with real datasets without changing the training pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
from enum import Enum


class Split(str, Enum):
    """Dataset split enumeration."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class SampleMetadata:
    """Metadata for a single dataset sample."""
    sample_id: str
    class_name: str
    class_idx: int
    signer: Optional[str] = None
    split: Split = Split.TRAIN
    source_dataset: str = "unknown"
    filepath: Optional[str] = None
    temporal_length: Optional[int] = None
    feature_dim: Optional[int] = None
    source: str = "unknown"  # "synthetic", "real", "augmented"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetMetadata:
    """Global dataset metadata."""
    dataset_type: str  # "synthetic", "real", "mixed"
    intended_use: str  # "development", "production", "benchmark"
    real_isl_recognition_valid: bool = False
    feature_dimension: int = 126
    classes: int = 0
    samples: int = 0
    signers: int = 0
    class_names: List[str] = field(default_factory=list)
    class_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_class: Dict[int, str] = field(default_factory=dict)
    signer_info: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class DatasetValidationError(Exception):
    """Raised when dataset validation fails."""
    pass


class LandmarkDataset(ABC):
    """
    Abstract base class for landmark sequence datasets.
    
    All landmark sequence datasets (synthetic, real, etc.) must implement
    this interface. This allows the training pipeline to be dataset-agnostic.
    """
    
    def __init__(self, root_dir: Path, split: Split = Split.TRAIN):
        self.root_dir = Path(root_dir)
        self.split = split
        self._metadata: Optional[DatasetMetadata] = None
        self._samples: List[SampleMetadata] = []
        self._class_to_idx: Dict[str, int] = {}
        self._idx_to_class: Dict[int, str] = {}
        self._load_metadata()
        self._load_samples()
    
    @abstractmethod
    def _load_metadata(self) -> None:
        """Load dataset metadata (class mapping, stats, etc.)."""
        pass
    
    @abstractmethod
    def _load_samples(self) -> None:
        """Load sample metadata for the current split."""
        pass
    
    @abstractmethod
    def _load_sequence(self, sample: SampleMetadata) -> np.ndarray:
        """
        Load the raw landmark sequence for a sample.
        
        Args:
            sample: Sample metadata
            
        Returns:
            Raw sequence array of shape [T, D] where D=126 (2*21*3)
        """
        pass
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, SampleMetadata]:
        """
        Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (sequence, class_idx, sample_metadata)
        """
        sample = self._samples[idx]
        sequence = self._load_sequence(sample)
        self._validate_sequence(sequence, sample)
        return sequence, sample.class_idx, sample
    
    def _validate_sequence(self, sequence: np.ndarray, sample: SampleMetadata) -> None:
        """
        Validate a loaded sequence.
        
        Args:
            sequence: Loaded sequence array
            sample: Sample metadata
            
        Raises:
            DatasetValidationError: If validation fails
        """
        # Check it's numeric
        if not np.issubdtype(sequence.dtype, np.number):
            raise DatasetValidationError(
                f"Sample {sample.sample_id}: sequence must be numeric, got {sequence.dtype}"
            )
        
        # Check for NaN or infinity
        if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
            raise DatasetValidationError(
                f"Sample {sample.sample_id}: sequence contains NaN or infinity"
            )
        
        # Check feature dimension
        if sequence.ndim != 2:
            raise DatasetValidationError(
                f"Sample {sample.sample_id}: sequence must be 2D [T, D], got {sequence.ndim}D"
            )
        
        feature_dim = sequence.shape[1]
        expected_dim = self.metadata.feature_dimension if self.metadata else 126
        if feature_dim != expected_dim:
            raise DatasetValidationError(
                f"Sample {sample.sample_id}: feature dimension mismatch. "
                f"Expected {expected_dim}, got {feature_dim}"
            )
        
        # Check temporal length
        if sequence.shape[0] == 0:
            raise DatasetValidationError(
                f"Sample {sample.sample_id}: sequence has zero temporal length"
            )
    
    @property
    def metadata(self) -> Optional[DatasetMetadata]:
        return self._metadata
    
    @property
    def samples(self) -> List[SampleMetadata]:
        return self._samples
    
    @property
    def class_to_idx(self) -> Dict[str, int]:
        return self._class_to_idx
    
    @property
    def idx_to_class(self) -> Dict[int, str]:
        return self._idx_to_class
    
    def get_class_name(self, class_idx: int) -> Optional[str]:
        return self._idx_to_class.get(class_idx)
    
    def get_class_idx(self, class_name: str) -> Optional[int]:
        return self._class_to_idx.get(class_name)
    
    def get_samples_by_class(self, class_idx: int) -> List[SampleMetadata]:
        return [s for s in self._samples if s.class_idx == class_idx]
    
    def get_samples_by_signer(self, signer: str) -> List[SampleMetadata]:
        return [s for s in self._samples if s.signer == signer]
    
    def get_class_distribution(self) -> Dict[int, int]:
        dist = {}
        for sample in self._samples:
            dist[sample.class_idx] = dist.get(sample.class_idx, 0) + 1
        return dist
    
    def get_signer_distribution(self) -> Dict[str, int]:
        dist = {}
        for sample in self._samples:
            if sample.signer:
                dist[sample.signer] = dist.get(sample.signer, 0) + 1
        return dist


class SyntheticLandmarkDataset(LandmarkDataset):
    """
    Synthetic landmark dataset loader.
    
    Loads the synthetic landmark fixture generated by generate_synthetic_landmarks.py.
    """
    
    def _load_metadata(self) -> None:
        import json
        
        # Load class map
        class_map_path = self.root_dir / "class_map.json"
        with open(class_map_path) as f:
            class_to_idx = json.load(f)
        self._class_to_idx = class_to_idx
        self._idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Load dataset metadata if available
        metadata_path = self.root_dir / "dataset_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            self._metadata = DatasetMetadata(
                dataset_type=meta.get("dataset_type", "synthetic"),
                intended_use=meta.get("intended_use", "pipeline_development_only"),
                real_isl_recognition_valid=meta.get("real_isl_recognition_valid", False),
                feature_dimension=meta.get("feature_dimension", 126),
                classes=meta.get("classes", len(self._class_to_idx)),
                samples=meta.get("samples", 0),
                signers=meta.get("signers", 0),
                class_names=list(self._class_to_idx.keys()),
                class_to_idx=self._class_to_idx,
                idx_to_class=self._idx_to_class,
                signer_info=meta.get("signer_info", {}),
                extra={k: v for k, v in meta.items() if k not in [
                    "dataset_type", "intended_use", "real_isl_recognition_valid",
                    "feature_dimension", "classes", "samples", "signers",
                    "class_names", "class_to_idx", "signer_info"
                ]}
            )
        else:
            # Fallback metadata
            self._metadata = DatasetMetadata(
                dataset_type="synthetic",
                intended_use="pipeline_development_only",
                real_isl_recognition_valid=False,
                feature_dimension=126,
                classes=len(self._class_to_idx),
                class_names=list(self._class_to_idx.keys()),
                class_to_idx=self._class_to_idx,
                idx_to_class=self._idx_to_class,
            )
    
    def _load_samples(self) -> None:
        import csv
        
        manifest_path = self.root_dir / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
        self._samples = []
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = SampleMetadata(
                    sample_id=row["sample_id"],
                    class_name=row["class"],
                    class_idx=int(row["class_idx"]),
                    signer=row["signer"] if row["signer"] else None,
                    split=Split(row["split"]),
                    source_dataset=row.get("source", "synthetic"),
                    filepath=row.get("filepath"),
                    temporal_length=int(row.get("T", 64)),
                    feature_dim=int(row.get("dims", 126)),
                    source=row.get("source", "synthetic"),
                )
                if sample.split == self.split:
                    self._samples.append(sample)
        
        if not self._samples:
            raise ValueError(f"No samples found for split {self.split}")
    
    def _load_sequence(self, sample: SampleMetadata) -> np.ndarray:
        if not sample.filepath:
            raise ValueError(f"Sample {sample.sample_id} has no filepath")
        
        filepath = self.root_dir / sample.filepath
        if not filepath.exists():
            raise FileNotFoundError(f"Sequence file not found: {filepath}")
        
        sequence = np.load(filepath)
        return sequence.astype(np.float32)


def create_dataset(
    dataset_type: str,
    root_dir: Path,
    split: Split = Split.TRAIN,
    **kwargs
) -> LandmarkDataset:
    """
    Factory function to create dataset instances.
    
    Args:
        dataset_type: Type of dataset ("synthetic", "real", etc.)
        root_dir: Root directory of the dataset
        split: Dataset split
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        LandmarkDataset instance
        
    Raises:
        ValueError: If dataset_type is unknown
    """
    if dataset_type == "synthetic":
        return SyntheticLandmarkDataset(root_dir, split)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")