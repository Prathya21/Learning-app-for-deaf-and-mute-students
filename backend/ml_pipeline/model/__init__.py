"""
Baseline sequence classifier for landmark sequences.

This module implements a lightweight BiLSTM baseline classifier for
landmark sequence classification. It is designed for CPU-compatible
training and inference, with configurable dimensions.

IMPORTANT: This is a SyntheticPipelineBaseline trained on synthetic data.
It does NOT perform real ISL gesture recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Configuration for the baseline model."""
    input_dim: int = 126
    hidden_dim: int = 128
    num_layers: int = 2
    num_classes: int = 46
    dropout: float = 0.3
    bidirectional: bool = True
    sequence_length: int = 64
    batch_first: bool = True

    @property
    def lstm_output_dim(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)


class SyntheticPipelineBaseline(nn.Module):
    """
    Lightweight BiLSTM baseline for landmark sequence classification.

    This is a DEVELOPMENT BASELINE trained on synthetic data.
    It does NOT perform real ISL gesture recognition.

    Architecture:
        Input [B, T, 126]
        -> BiLSTM [B, T, hidden_dim*2]
        -> Dropout
        -> Global Average Pooling [B, hidden_dim*2]
        -> Dropout
        -> Linear [B, num_classes]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=config.batch_first,
        )

        self.dropout1 = nn.Dropout(config.dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout2 = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.lstm_output_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    nn.init.orthogonal_(param)
                elif "classifier" in name:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout1(lstm_out)
        pooled = self.global_pool(lstm_out.transpose(1, 2)).squeeze(-1)
        pooled = self.dropout2(pooled)
        logits = self.classifier(pooled)
        return logits

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout1(lstm_out)
        pooled = self.global_pool(lstm_out.transpose(1, 2)).squeeze(-1)
        return pooled

    def get_model_size(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lstm_params = sum(p.numel() for n, p in self.named_parameters() if "lstm" in n)
        classifier_params = sum(p.numel() for n, p in self.named_parameters() if "classifier" in n)
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "lstm_params": lstm_params,
            "classifier_params": classifier_params,
        }

    def get_model_info(self) -> Dict[str, Any]:
        size_info = self.get_model_size()
        return {
            "model_name": "SyntheticPipelineBaseline",
            "architecture": "BiLSTM + GlobalAvgPool + Linear",
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "num_classes": self.config.num_classes,
                "dropout": self.config.dropout,
                "bidirectional": self.config.bidirectional,
                "sequence_length": self.config.sequence_length,
            },
            **size_info,
            "note": "SyntheticPipelineBaseline - trained on synthetic data only. Not for real ISL recognition.",
        }


def create_model(
    num_classes: int,
    input_dim: int = 126,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = True,
    sequence_length: int = 64,
) -> SyntheticPipelineBaseline:
    """Factory function to create a SyntheticPipelineBaseline model."""
    config = ModelConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
        sequence_length=sequence_length,
    )
    return SyntheticPipelineBaseline(config)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    """Generate a human-readable model summary."""
    info = model.get_model_info() if hasattr(model, "get_model_info") else {}

    lines = [
        "=" * 60,
        f"Model: {info.get('model_name', model.__class__.__name__)}",
        "=" * 60,
        f"Architecture: {info.get('architecture', 'Unknown')}",
        "",
        "Configuration:",
    ]

    for key, value in info.get("config", {}).items():
        lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "Parameter Counts:",
        f"  Total:       {info.get('total_params', 0):,}",
        f"  Trainable:   {info.get('trainable_params', 0):,}",
        f"  LSTM:        {info.get('lstm_params', 0):,}",
        f"  Classifier:  {info.get('classifier_params', 0):,}",
        "",
        f"Note: {info.get('note', '')}",
        "=" * 60,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    model = create_model(num_classes=46)
    print(model_summary(model))

    x = torch.randn(2, 64, 126)
    logits = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: [2, 46]")