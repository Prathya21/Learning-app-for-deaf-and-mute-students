"""
Inference interface for landmark sequence classifiers.

This module provides a reusable inference interface that can be used
both offline and (in future phases) connected to live inference.
"""

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import json

from ..model import SyntheticPipelineBaseline, ModelConfig, create_model
from ..preprocessing import (
    PreprocessingConfig,
    preprocess_sequence,
    NormalizationMode,
    PaddingMode,
    TruncationMode,
)
from ..dataset import SampleMetadata
from ..training import CheckpointMetadata, load_checkpoint


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    # Model
    num_classes: int = 46
    input_dim: int = 126
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    sequence_length: int = 64
    
    # Preprocessing
    target_temporal_length: int = 64
    normalization_mode: str = "wrist_centric_scale"
    padding_mode: str = "zero"
    truncation_mode: str = "end"
    
    # Inference
    device: str = "cpu"  # "cpu", "cuda", "mps"
    return_probabilities: bool = True
    top_k: int = 5  # Return top-k predictions
    
    # Model source
    checkpoint_path: Optional[str] = None
    model_config: Optional[ModelConfig] = None


@dataclass
class InferenceResult:
    """Result of a single inference."""
    predicted_class_index: int
    predicted_gloss: str
    confidence: float
    top_k: List[Dict[str, Any]] = field(default_factory=list)
    probabilities: Optional[Dict[str, float]] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InferenceEngine:
    """
    Reusable inference engine for landmark sequence classification.
    
    This engine handles:
    - Model loading from checkpoint
    - Preprocessing of input sequences
    - Inference with configurable options
    - Result formatting with metadata
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        class_mapping: Dict[int, str],
        idx_to_class: Dict[int, str]
    ):
        self.config = config
        self.class_mapping = class_mapping  # idx -> gloss
        self.idx_to_class = idx_to_class  # idx -> gloss
        
        # Device
        self.device = torch.device(config.device)
        
        # Preprocessing config
        self.preprocessing_config = PreprocessingConfig(
            target_temporal_length=config.target_temporal_length,
            normalization_mode=NormalizationMode(config.normalization_mode),
            padding_mode=PaddingMode(config.padding_mode),
            truncation_mode=TruncationMode(config.truncation_mode),
        )
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Model metadata
        self.model_metadata = self._get_model_metadata()
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint or create new."""
        if self.config.checkpoint_path:
            # Load from checkpoint
            checkpoint_path = Path(self.config.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Create model with config
            if self.config.model_config:
                model = SyntheticPipelineBaseline(self.config.model_config)
            else:
                model = create_model(
                    num_classes=self.config.num_classes,
                    input_dim=self.config.input_dim,
                    hidden_dim=self.config.hidden_dim,
                    num_layers=self.config.num_layers,
                    dropout=self.config.dropout,
                    bidirectional=self.config.bidirectional,
                    sequence_length=self.config.sequence_length,
                )
            
            # Load checkpoint
            checkpoint = torch.load(
                self.config.checkpoint_path,
                map_location=self.device
            )
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.config.device)
            return model
        else:
            # Create new model (untrained)
            model = create_model(
                num_classes=self.config.num_classes,
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                bidirectional=self.config.bidirectional,
                sequence_length=self.config.sequence_length,
            )
            model = model.to(self.config.device)
            return model
    
    def _get_model_metadata(self) -> Dict[str, Any]:
        """Extract model metadata for inference results."""
        metadata = {
            "model_name": "SyntheticPipelineBaseline",
            "architecture": "BiLSTM + GlobalAvgPool + Linear",
            "trained_on_synthetic_data": True,
            "real_isl_recognition_valid": False,
            "num_classes": self.config.num_classes,
            "input_dim": self.config.input_dim,
            "sequence_length": self.config.sequence_length,
        }
        
        # Add checkpoint metadata if available
        if self.config.checkpoint_path:
            try:
                checkpoint = torch.load(
                    self.config.checkpoint_path,
                    map_location='cpu'
                )
                if 'metadata' in checkpoint:
                    metadata.update(checkpoint['metadata'])
            except Exception:
                pass
        
        return metadata
    
    def preprocess(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single landmark sequence.
        
        Args:
            sequence: Raw landmark sequence [T, 126] or [126] for single frame
            
        Returns:
            Preprocessed tensor [1, T, 126]
        """
        # Handle single frame
        if sequence.ndim == 1:
            sequence = sequence.reshape(1, -1)
        
        # Apply preprocessing
        result = preprocess_sequence(sequence, self.preprocessing_config)
        processed = result.sequence  # [T, 126]
        
        # Add batch dimension
        tensor = torch.from_numpy(processed).float().unsqueeze(0)  # [1, T, 126]
        return tensor.to(self.device)
    
    def predict(
        self,
        sequence: np.ndarray,
        return_probabilities: bool = None,
        top_k: int = None
    ) -> InferenceResult:
        """
        Run inference on a single landmark sequence.
        
        Args:
            sequence: Input landmark sequence [T, 126]
            return_probabilities: Whether to return class probabilities
            top_k: Number of top predictions to return
            
        Returns:
            InferenceResult with predictions and metadata
        """
        if return_probabilities is None:
            return_probabilities = self.config.return_probabilities
        if top_k is None:
            top_k = self.config.top_k
        
        # Preprocess
        x = self.preprocess(sequence)
        
        # Inference
        with torch.no_grad():
            logits = self.model(x)  # [1, num_classes]
            
            if return_probabilities:
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [num_classes]
            else:
                probs = None
            
            # Get top-k predictions
            top_k_indices = torch.topk(logits, k=min(top_k, self.config.num_classes), dim=1).indices[0].cpu().numpy()
            top_k_probs = torch.topk(logits, k=min(top_k, self.config.num_classes), dim=1).values[0].cpu().numpy()
            
            # Build result
            predicted_idx = int(top_k_indices[0])
            predicted_gloss = self.idx_to_class.get(predicted_idx, f"UNKNOWN_{predicted_idx}")
            confidence = float(top_k_probs[0])
            
            top_k_results = []
            for idx, prob in zip(top_k_indices, top_k_probs):
                gloss = self.idx_to_class.get(int(idx), f"UNKNOWN_{idx}")
                top_k_results.append({
                    "class_index": int(idx),
                    "gloss": gloss,
                    "probability": float(prob),
                })
            
            # Full probabilities dict if requested
            prob_dict = None
            if return_probabilities:
                prob_dict = {
                    self.idx_to_class.get(i, f"UNKNOWN_{i}"): float(p)
                    for i, p in enumerate(logits.softmax(dim=1).cpu().numpy()[0])
                }
            
            return InferenceResult(
                predicted_class_index=predicted_idx,
                predicted_gloss=predicted_gloss,
                confidence=confidence,
                top_k=top_k_results,
                probabilities=prob_dict,
                model_metadata=self.model_metadata.copy(),
            )
    
    def predict_batch(self, sequences: List[np.ndarray]) -> List[InferenceResult]:
        """Run inference on a batch of sequences."""
        results = []
        for seq in sequences:
            results.append(self.predict(seq))
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        info.update(self.model_metadata)
        return info


def create_inference_engine(
    checkpoint_path: str,
    class_mapping: Dict[int, str],
    idx_to_class: Dict[int, str],
    **kwargs
) -> InferenceEngine:
    """
    Factory function to create an inference engine from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        class_mapping: Dict mapping class index to gloss
        idx_to_class: Dict mapping index to class name
        **kwargs: Additional InferenceConfig parameters
        
    Returns:
        InferenceEngine instance
    """
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    return InferenceEngine(config, class_mapping, idx_to_class)


def load_model_for_inference(
    checkpoint_path: str,
    class_mapping: Dict[int, str],
    idx_to_class: Dict[int, str],
    device: str = "cpu"
) -> InferenceEngine:
    """
    Load a model checkpoint and create an inference engine.
    
    This is the main entry point for loading a trained model for inference.
    """
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return InferenceEngine(config, class_mapping, idx_to_class)


def predict_from_landmarks(
    engine: InferenceEngine,
    landmarks: np.ndarray,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Convenience function for simple inference.
    
    Args:
        engine: InferenceEngine instance
        landmarks: Landmark sequence [T, 126]
        top_k: Number of top predictions
        
    Returns:
        Dictionary with prediction results
    """
    result = engine.predict(landmarks, top_k=top_k)
    return result.to_dict()


if __name__ == "__main__":
    # Quick test
    from ..model import create_model
    from ..dataset import SyntheticLandmarkDataset, Split
    from ..dataset import create_dataset
    from pathlib import Path
    
    # Load synthetic dataset for class mapping
    dataset = SyntheticLandmarkDataset(
        Path("backend/data/synthetic_landmarks"),
        Split.TRAIN
    )
    
    class_mapping = {v: k for k, v in dataset.class_to_idx.items()}
    idx_to_class = dataset.idx_to_class
    
    # Create dummy model for testing
    model = create_model(num_classes=46)
    
    # Test inference engine (without checkpoint)
    config = InferenceConfig(
        num_classes=46,
        checkpoint_path=None,
    )
    
    engine = InferenceEngine(config, class_mapping, idx_to_class)
    print(f"Model info: {engine.get_model_info()}")
    
    # Test with random sequence
    test_seq = np.random.randn(64, 126).astype(np.float32)
    result = engine.predict(test_seq, top_k=3)
    print(f"\nTest inference:")
    print(f"  Predicted: {result.predicted_gloss} (conf={result.confidence:.4f})")
    top3 = [(r['gloss'], f"{r['probability']:.4f}") for r in result.top_k]
    print(f"  Top-3: {top3}")