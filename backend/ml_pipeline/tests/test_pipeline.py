"""
Unit tests for ML pipeline components.

Tests cover:
- Dataset loading and validation
- Preprocessing functions
- Model architecture
- Training pipeline
- Inference interface
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.dataset import (
    LandmarkDataset,
    SyntheticLandmarkDataset,
    SampleMetadata,
    Split,
    DatasetMetadata,
    create_dataset,
    DatasetValidationError,
)
from ml_pipeline.preprocessing import (
    PreprocessingConfig,
    preprocess_sequence,
    validate_sequence,
    wrist_centric_normalize,
    scale_normalize,
    temporal_sample,
    handle_missing_hands,
    PaddingMode,
    TruncationMode,
    NormalizationMode,
    PreprocessingResult,
)
from ml_pipeline.model import (
    SyntheticPipelineBaseline,
    ModelConfig,
    create_model,
    count_parameters,
    model_summary,
)
from ml_pipeline.training import (
    TrainingConfig,
    CheckpointMetadata,
    PyTorchDataset,
    create_dataloaders,
    set_seed,
    EarlyStopping,
    train_one_epoch,
    validate,
    save_checkpoint,
    load_checkpoint,
)
from ml_pipeline.inference import (
    InferenceConfig,
    InferenceEngine,
    InferenceResult,
    create_inference_engine,
)
from ml_pipeline.dataset import create_dataset as create_dataset_fn


class TestDatasetContract:
    """Tests for dataset contract and validation."""
    
    @pytest.fixture
    def synthetic_dataset(self):
        """Create a synthetic dataset for testing."""
        return SyntheticLandmarkDataset(
            Path(__file__).parent.parent.parent / "data" / "synthetic_landmarks",
            Split.TRAIN
        )
    
    def test_dataset_loads(self, synthetic_dataset):
        """Test that synthetic dataset loads correctly."""
        assert len(synthetic_dataset) > 0
        assert synthetic_dataset.metadata is not None
        assert synthetic_dataset.metadata.classes == 46
    
    def test_sample_metadata(self, synthetic_dataset):
        """Test sample metadata structure."""
        sample = synthetic_dataset[0]
        sequence, label, metadata = sample
        
        assert isinstance(sequence, np.ndarray)
        assert sequence.ndim == 2
        assert sequence.shape[1] == 126
        assert isinstance(label, int)
        assert 0 <= label < 46
        assert isinstance(metadata, SampleMetadata)
        assert metadata.class_idx == label
    
    def test_class_mapping(self, synthetic_dataset):
        """Test class mapping consistency."""
        assert len(synthetic_dataset.class_to_idx) == 46
        assert len(synthetic_dataset.idx_to_class) == 46
        
        for idx, name in synthetic_dataset.idx_to_class.items():
            assert synthetic_dataset.class_to_idx[name] == idx
    
    def test_get_samples_by_class(self, synthetic_dataset):
        """Test filtering samples by class."""
        class_0_samples = synthetic_dataset.get_samples_by_class(0)
        assert all(s.class_idx == 0 for s in class_0_samples)
    
    def test_get_samples_by_signer(self, synthetic_dataset):
        """Test filtering samples by signer."""
        signer_samples = synthetic_dataset.get_samples_by_signer("Signer01")
        assert all(s.signer == "Signer01" for s in signer_samples)
    
    def test_class_distribution(self, synthetic_dataset):
        """Test class distribution."""
        dist = synthetic_dataset.get_class_distribution()
        assert len(dist) == 46
        assert sum(dist.values()) == len(synthetic_dataset)


class TestSequenceValidation:
    """Tests for sequence validation."""
    
    def test_valid_sequence(self):
        """Test validation passes for valid sequence."""
        seq = np.random.randn(64, 126).astype(np.float32)
        validate_sequence(seq)  # Should not raise
    
    def test_invalid_dimension(self):
        """Test rejection of wrong feature dimension."""
        seq = np.random.randn(64, 63).astype(np.float32)
        with pytest.raises(ValueError, match="Expected feature dim 126, got 63"):
            validate_sequence(seq)
    
    def test_nan_rejection(self):
        """Test rejection of NaN values."""
        seq = np.random.randn(64, 126).astype(np.float32)
        seq[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or infinity"):
            validate_sequence(seq)
    
    def test_inf_rejection(self):
        """Test rejection of infinity values."""
        seq = np.random.randn(64, 126).astype(np.float32)
        seq[0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or infinity"):
            validate_sequence(seq)
    
    def test_zero_length_rejection(self):
        """Test rejection of zero-length sequence."""
        seq = np.zeros((0, 126), dtype=np.float32)
        with pytest.raises(ValueError, match="zero temporal length"):
            validate_sequence(seq)
    
    def test_wrong_ndim(self):
        """Test rejection of wrong number of dimensions."""
        seq = np.random.randn(64, 126, 1).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 2D array"):
            validate_sequence(seq)


class TestPreprocessing:
    """Tests for preprocessing functions."""
    
    def test_wrist_centric_normalize(self):
        """Test wrist-centric normalization."""
        seq = np.random.randn(64, 126).astype(np.float32)
        # Set wrist to non-zero
        seq[:, :3] = 1.0
        normalized = wrist_centric_normalize(seq)
        # Wrist should now be at origin
        assert np.allclose(normalized[:, :3], 0, atol=1e-6)
    
    def test_scale_normalize(self):
        """Test scale normalization."""
        seq = np.random.randn(64, 126).astype(np.float32) * 10.0
        normalized, scale = scale_normalize(seq)
        # Scale should be around the original scale
        assert scale > 0
        # Normalized should have smaller magnitude
        assert np.max(np.abs(normalized)) < np.max(np.abs(seq))
    
    def test_temporal_sampling_pad(self):
        """Test padding short sequences."""
        seq = np.random.randn(32, 126).astype(np.float32)
        result = temporal_sample(seq, 64)
        assert result.shape == (64, 126)
        # Last 32 frames should be zeros (padding)
        assert np.allclose(result[32:], 0)
    
    def test_temporal_sampling_truncate_start(self):
        """Test truncation from start."""
        seq = np.random.randn(128, 126).astype(np.float32)
        result = temporal_sample(seq, 64, TruncationMode.START)
        assert result.shape == (64, 126)
        assert np.allclose(result, seq[:64])
    
    def test_temporal_sampling_truncate_end(self):
        """Test truncation from end."""
        seq = np.random.randn(128, 126).astype(np.float32)
        result = temporal_sample(seq, 64, TruncationMode.END)
        assert result.shape == (64, 126)
        assert np.allclose(result, seq[-64:])
    
    def test_temporal_sampling_uniform(self):
        """Test uniform sampling."""
        seq = np.random.randn(128, 126).astype(np.float32)
        result = temporal_sample(seq, 64, TruncationMode.UNIFORM, uniform=True)
        assert result.shape == (64, 126)
        # Should sample uniformly
        indices = np.linspace(0, 127, 64, dtype=int)
        assert np.allclose(result, seq[indices])
    
    def test_handle_missing_hands_no_interp(self):
        """Test missing hand handling without interpolation."""
        seq = np.random.randn(64, 126).astype(np.float32)
        # Set hand 1 to zero for some frames
        seq[10:20, :63] = 0
        result = handle_missing_hands(seq, interpolate=False)
        # Missing frames should be zeros (already were)
        assert np.allclose(result[10:20, :63], 0)
    
    def test_full_preprocessing_pipeline(self):
        """Test full preprocessing pipeline."""
        config = PreprocessingConfig(
            target_temporal_length=64,
            normalization_mode=NormalizationMode.WRIST_CENTRIC_SCALE,
        )
        
        # Test with various input lengths
        for T in [32, 64, 128]:
            seq = np.random.randn(T, 126).astype(np.float32)
            result = preprocess_sequence(seq, config)
            
            assert isinstance(result, PreprocessingResult)
            assert result.sequence.shape == (64, 126)
            assert result.original_shape == (T, 126)
            assert result.processed_shape == (64, 126)
            assert len(result.metadata["steps_applied"]) > 0
    
    def test_short_sequence(self):
        """Test preprocessing with very short sequence."""
        config = PreprocessingConfig(target_temporal_length=64)
        seq = np.random.randn(10, 126).astype(np.float32)
        result = preprocess_sequence(seq, config)
        assert result.processed_shape == (64, 126)
    
    def test_long_sequence(self):
        """Test preprocessing with long sequence."""
        config = PreprocessingConfig(target_temporal_length=64)
        seq = np.random.randn(200, 126).astype(np.float32)
        result = preprocess_sequence(seq, config)
        assert result.processed_shape == (64, 126)
    
    def test_exact_length(self):
        """Test preprocessing with exact target length."""
        config = PreprocessingConfig(target_temporal_length=64)
        seq = np.random.randn(64, 126).astype(np.float32)
        result = preprocess_sequence(seq, config)
        assert result.processed_shape == (64, 126)
    
    def test_nan_handling(self):
        """Test NaN handling in preprocessing."""
        config = PreprocessingConfig(reject_nan=False)
        seq = np.random.randn(64, 126).astype(np.float32)
        seq[0, 0] = np.nan
        result = preprocess_sequence(seq, config)
        assert not np.any(np.isnan(result.sequence))
    
    def test_invalid_shape_rejection(self):
        """Test rejection of invalid shapes."""
        config = PreprocessingConfig(validate_input=True)
        seq = np.random.randn(64, 63).astype(np.float32)
        with pytest.raises(ValueError):
            preprocess_sequence(seq, config)


class TestModelArchitecture:
    """Tests for model architecture."""
    
    def test_model_creation(self):
        """Test model creation with default config."""
        model = create_model(num_classes=46)
        assert isinstance(model, SyntheticPipelineBaseline)
    
    def test_model_configurable_classes(self):
        """Test model with different number of classes."""
        for num_classes in [10, 20, 46, 100]:
            model = create_model(num_classes=num_classes)
            assert model.config.num_classes == num_classes
            logits = model(torch.randn(2, 64, 126))
            assert logits.shape == (2, num_classes)
    
    def test_model_configurable_hidden_dim(self):
        """Test model with different hidden dimensions."""
        for hidden_dim in [64, 128, 256]:
            model = create_model(num_classes=10, hidden_dim=hidden_dim)
            assert model.config.hidden_dim == hidden_dim
    
    def test_model_configurable_layers(self):
        """Test model with different number of layers."""
        for num_layers in [1, 2, 3]:
            model = create_model(num_classes=10, num_layers=num_layers)
            assert model.config.num_layers == num_layers
    
    def test_model_forward_pass(self):
        """Test forward pass shape."""
        model = create_model(num_classes=46)
        x = torch.randn(4, 64, 126)
        logits = model(x)
        assert logits.shape == (4, 46)
    
    def test_model_parameter_count(self):
        """Test parameter counting."""
        model = create_model(num_classes=46)
        total = count_parameters(model)
        assert total > 0
        assert total < 1_000_000  # Should be lightweight
    
    def test_model_summary(self):
        """Test model summary generation."""
        model = create_model(num_classes=46)
        summary = model_summary(model)
        assert "SyntheticPipelineBaseline" in summary
        assert "total_params" in model.get_model_info()
    
    def test_feature_extraction(self):
        """Test feature extraction before classification head."""
        model = create_model(num_classes=46)
        x = torch.randn(2, 64, 126)
        features = model.get_feature_vector(x)
        assert features.shape == (2, 256)  # 128 * 2 (bidirectional)


class TestTrainingInfrastructure:
    """Tests for training infrastructure."""
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        a = np.random.rand()
        torch.rand(1)
        
        set_seed(42)
        b = np.random.rand()
        torch.rand(1)
        
        assert a == b
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode."""
        es = EarlyStopping(patience=3, mode='max')
        assert not es(0.5)
        assert not es(0.6)
        assert not es(0.7)
        assert es(0.6)  # Worse
        assert es(0.6)
        assert es(0.6)  # Should trigger early stop after patience
    
    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode."""
        es = EarlyStopping(patience=2, mode='min')
        assert not es(1.0)
        assert not es(0.9)
        assert es(0.9)  # Not improving
        assert es(0.9)  # Should trigger
    
    def test_checkpoint_metadata_serialization(self):
        """Test checkpoint metadata serialization."""
        metadata = CheckpointMetadata(
            epoch=10,
            best_val_acc=0.95,
            class_mapping={0: "HELLO", 1: "THANK_YOU"},
        )
        
        data = metadata.to_dict()
        assert data["epoch"] == 10
        assert data["best_val_acc"] == 0.95
        assert "HELLO" in str(data["class_mapping"])
        
        restored = CheckpointMetadata.from_dict(data)
        assert restored.epoch == 10
        assert restored.best_val_acc == 0.95
    
    def test_checkpoint_save_load(self):
        """Test checkpoint save/load."""
        model = create_model(num_classes=10)
        metadata = CheckpointMetadata(epoch=5, best_val_acc=0.9)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_checkpoint.pt"
            save_checkpoint(model, metadata, path)
            
            # Load back
            model2 = create_model(num_classes=10)
            model2, _, metadata2 = load_checkpoint(path, model)
            
            assert metadata2.epoch == 5  # epoch is saved in metadata
            # Model weights should match
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)


class TestInferenceEngine:
    """Tests for inference engine."""
    
    @pytest.fixture
    def class_mappings(self):
        """Create class mappings for testing."""
        class_mapping = {i: f"CLASS_{i}" for i in range(46)}
        idx_to_class = {i: f"CLASS_{i}" for i in range(46)}
        return class_mapping, idx_to_class
    
    def test_inference_engine_creation(self, class_mappings):
        """Test inference engine creation."""
        class_mapping, idx_to_class = class_mappings
        config = InferenceConfig(num_classes=46, checkpoint_path=None)
        engine = InferenceEngine(config, class_mapping, idx_to_class)
        
        assert engine.config.num_classes == 46
        assert engine.device.type == "cpu"
    
    def test_inference_output_schema(self, class_mappings):
        """Test inference output schema."""
        class_mapping, idx_to_class = class_mappings
        config = InferenceConfig(num_classes=46, checkpoint_path=None)
        engine = InferenceEngine(config, class_mapping, idx_to_class)
        
        # Test with random sequence
        test_seq = np.random.randn(64, 126).astype(np.float32)
        result = engine.predict(test_seq, top_k=3)
        
        assert isinstance(result, InferenceResult)
        assert isinstance(result.predicted_class_index, int)
        assert 0 <= result.predicted_class_index < 46
        assert isinstance(result.predicted_gloss, str)
        assert 0 <= result.confidence <= 1
        assert len(result.top_k) == 3
        assert all("gloss" in r and "probability" in r for r in result.top_k)
    
    def test_inference_result_serialization(self, class_mappings):
        """Test inference result serialization."""
        class_mapping, idx_to_class = class_mappings
        config = InferenceConfig(num_classes=46, checkpoint_path=None)
        engine = InferenceEngine(config, class_mapping, idx_to_class)
        
        test_seq = np.random.randn(64, 126).astype(np.float32)
        result = engine.predict(test_seq, top_k=3)
        
        # Test serialization
        result_dict = result.to_dict()
        assert "predicted_class_index" in result_dict
        assert "predicted_gloss" in result_dict
        assert "confidence" in result_dict
        assert "top_k" in result_dict
    
    def test_model_metadata_propagation(self, class_mappings):
        """Test that model metadata is propagated to results."""
        class_mapping, idx_to_class = class_mappings
        config = InferenceConfig(
            num_classes=46,
            checkpoint_path=None,
        )
        engine = InferenceEngine(config, class_mapping, idx_to_class)
        
        test_seq = np.random.randn(64, 126).astype(np.float32)
        result = engine.predict(test_seq)
        
        assert "model_name" in result.model_metadata
        assert "trained_on_synthetic_data" in result.model_metadata
        assert result.model_metadata["trained_on_synthetic_data"] is True
        assert "real_isl_recognition_valid" in result.model_metadata
        assert result.model_metadata["real_isl_recognition_valid"] is False


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_smoke_test(self):
        """Smoke test of the full pipeline."""
        # 1. Create dataset
        from ml_pipeline.dataset import create_dataset
        from pathlib import Path
        
        data_root = Path(__file__).parent.parent.parent / "data" / "synthetic_landmarks"
        train_dataset = create_dataset("synthetic", data_root, Split.TRAIN)
        val_dataset = create_dataset("synthetic", data_root, Split.VAL)
        
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        
        # 2. Create model
        model = create_model(num_classes=46)
        
        # 3. Create dataloaders
        from ml_pipeline.training import PyTorchDataset, create_dataloaders
        from ml_pipeline.preprocessing import PreprocessingConfig
        
        train_pt_dataset = PyTorchDataset(train_dataset)
        val_pt_dataset = PyTorchDataset(val_dataset)
        
        train_loader, val_loader = create_dataloaders(
            TrainingConfig(batch_size=4),
            train_pt_dataset,
            val_pt_dataset
        )
        
        # 3. Quick training step
        device = torch.device("cpu")
        model = create_model(num_classes=46).to("cpu")
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # One training step
        model.train()
        for sequences, labels in train_loader:
            sequences = sequences.to("cpu")
            labels = labels.to("cpu")
            
            optimizer.zero_grad()
            logits = model(sequences)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            break  # Just one batch
        
        # 4. Validation step
        model.eval()
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to("cpu")
                labels = labels.to("cpu")
                logits = model(sequences)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                break
        
        assert True  # If we get here, pipeline works


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])