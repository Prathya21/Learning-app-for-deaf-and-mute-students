"""
Training infrastructure for landmark sequence classifiers.

This module provides a reusable training pipeline with checkpointing,
early stopping, and metadata tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np

from ..dataset import LandmarkDataset, SampleMetadata, Split
from ..model import SyntheticPipelineBaseline, ModelConfig
from ..preprocessing import PreprocessingConfig, preprocess_sequence


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    # Data
    dataset_type: str = "synthetic"
    dataset_root: str = "backend/data/synthetic_landmarks"
    batch_size: int = 32
    num_workers: int = 0  # 0 for main thread, >0 for multiprocessing
    
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
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    scheduler: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "none"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Preprocessing
    target_temporal_length: int = 64
    normalization_mode: str = "wrist_centric_scale"
    padding_mode: str = "zero"
    truncation_mode: str = "end"
    
    # Training control
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    save_best_only: bool = True
    save_every_n_epochs: int = 0  # 0 = disabled
    
    # Logging
    log_every_n_batches: int = 50
    log_every_n_epochs: int = 1
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Device
    device: str = "cpu"  # "cpu", "cuda", "mps"


@dataclass
class TrainingMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    learning_rate: float
    epoch_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointMetadata:
    """Metadata saved with each checkpoint."""
    # Model info
    model_name: str = "SyntheticPipelineBaseline"
    architecture: str = "BiLSTM + GlobalAvgPool + Linear"
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)
    
    # Dataset info
    dataset_type: str = "synthetic"
    source_dataset: str = "synthetic_fixture"
    real_isl_recognition_valid: bool = False
    feature_dimension: int = 126
    temporal_length: int = 64
    class_mapping: Dict[str, int] = field(default_factory=dict)
    
    # Training state
    epoch: int = 0
    best_val_acc: float = 0.0
    best_val_loss: float = float('inf')
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    training_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    random_seed: int = 42
    pytorch_version: str = torch.__version__
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        return cls(**data)


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class PyTorchDataset(Dataset):
    """PyTorch Dataset wrapper for LandmarkDataset."""
    
    def __init__(
        self,
        dataset,
        preprocessing_config: Optional[PreprocessingConfig] = None
    ):
        self.dataset = dataset
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sequence, label, metadata = self.dataset[idx]
        
        # Apply preprocessing
        if self.preprocessing_config:
            from ..preprocessing import preprocess_sequence
            result = preprocess_sequence(sequence, self.preprocessing_config)
            sequence = result.sequence
        
        return torch.from_numpy(sequence).float(), label


def collate_fn(batch):
    """Custom collate function for variable-length sequences (if needed)."""
    sequences, labels = zip(*batch)
    # All sequences should be same length after preprocessing
    sequences = torch.stack(sequences)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences, labels


def create_dataloaders(
    config: TrainingConfig,
    train_dataset,
    val_dataset,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.device != "cpu",
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.device != "cpu",
    )
    
    return train_loader, val_loader


def create_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Create optimizer from config."""
    if config.optimizer.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def create_scheduler(optimizer: optim.Optimizer, config: TrainingConfig):
    """Create learning rate scheduler from config."""
    if config.scheduler.lower() == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            verbose=True,
        )
    elif config.scheduler.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )
    elif config.scheduler.lower() == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().sum()
    return (correct / targets.numel()).item()


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.best_score = -float('inf')
            self._is_better = lambda x, y: x > y + self.min_delta
        else:
            self.best_score = float('inf')
            self._is_better = lambda x, y: x < y - self.min_delta
    
    def __call__(self, score: float) -> bool:
        if self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int = 50
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (sequences, labels) in enumerate(loader):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
        
        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            print(f"  Batch {batch_idx + 1}/{len(loader)}: "
                  f"loss={avg_loss:.4f}, acc={avg_acc:.4f}")
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            logits = model(sequences)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    checkpoint_dir: Path,
    class_mapping: Dict[str, int],
    preprocessing_config: Optional[PreprocessingConfig] = None,
) -> Tuple[nn.Module, CheckpointMetadata]:
    """
    Main training loop.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        checkpoint_dir: Directory to save checkpoints
        class_mapping: Class to index mapping
        preprocessing_config: Preprocessing configuration
        
    Returns:
        Tuple of (trained_model, checkpoint_metadata)
    """
    device = torch.device(config.device)
    model = model.to(device)
    
    # Setup
    set_seed(config.seed, config.deterministic)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        mode='max'
    )
    
    # Checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata
    metadata = CheckpointMetadata(
        config=asdict(config),
        preprocessing_config=asdict(preprocessing_config) if preprocessing_config else {},
        class_mapping={v: k for k, v in class_mapping.items()},  # idx -> class
        feature_dimension=config.input_dim,
        temporal_length=config.sequence_length,
        dataset_type="synthetic",
        source_dataset="synthetic_fixture",
        real_isl_recognition_valid=False,
        random_seed=config.seed,
    )
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    history = []
    
    print(f"Starting training on {device}")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            log_every=config.log_every_n_batches
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Record metrics
        epoch_time = time.time() - epoch_start
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            learning_rate=optimizer.param_groups[0]['lr'],
            epoch_time=epoch_time,
        )
        history.append(metrics.to_dict())
        
        # Print epoch summary
        if epoch % config.log_every_n_epochs == 0:
            print(f"Epoch {epoch}/{config.epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                  f"lr={metrics.learning_rate:.2e}, time={epoch_time:.1f}s")
        
        # Check for improvement
        is_best = val_acc > best_val_acc + 1e-4
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
        
        # Update metadata
        metadata.best_val_acc = best_val_acc
        metadata.best_val_loss = min(best_val_loss, metadata.best_val_loss)
        metadata.epoch = epoch
        metadata.training_history = history
        
        # Save checkpoint
        if config.save_best_only:
            if is_best:
                save_checkpoint(model, metadata, checkpoint_dir / "best_model.pt")
                print(f"  -> Saved best model (val_acc={val_acc:.4f})")
        else:
            if config.save_every_n_epochs > 0 and epoch % config.save_every_n_epochs == 0:
                save_checkpoint(model, metadata, checkpoint_dir / f"epoch_{epoch}.pt")
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    best_path = checkpoint_dir / "best_model.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['metadata'].epoch}")
    
    return model, metadata


def save_checkpoint(
    model: nn.Module,
    metadata: CheckpointMetadata,
    path: Path,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: Optional[int] = None
):
    """Save model checkpoint with metadata."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metadata': metadata.to_dict(),
        'epoch': epoch or metadata.epoch,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: str = "cpu"
) -> Tuple[nn.Module, Optional[optim.Optimizer], CheckpointMetadata]:
    """Load model checkpoint with metadata."""
    checkpoint = torch.load(path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    metadata = CheckpointMetadata.from_dict(checkpoint['metadata'])
    return model, optimizer, metadata


def get_class_weights(dataset, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    from collections import Counter
    labels = []
    for _, label, _ in dataset:
        labels.append(label)
    
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        if i in counts:
            weights.append(total / counts[i])
        else:
            weights.append(1.0)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    return weights