from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import torch.serialization
import numpy as np
from pathlib import Path

from app.services.video_lookup import video_lookup_service
from ml_pipeline.inference import InferenceConfig, InferenceEngine, create_inference_engine
from ml_pipeline.preprocessing import PreprocessingConfig, preprocess_sequence, PaddingMode, TruncationMode, NormalizationMode

# Register safe globals for PyTorch 2.6+ checkpoint loading
torch.serialization.add_safe_globals([
    PaddingMode, TruncationMode, NormalizationMode,
    torch.torch_version.TorchVersion
])

router = APIRouter(prefix="/api", tags=["inference"])

# Global inference engine (initialized on startup)
_inference_engine = None


class GestureInferenceRequest(BaseModel):
    frames: List[List[float]] = Field(
        ...,
        description="List of frames, each containing 126 landmark coordinates (LEFT_HAND 63 + RIGHT_HAND 63)",
        min_length=1,
        max_length=200,
    )


class TopKPrediction(BaseModel):
    class_index: int
    gloss: str
    probability: float


class GestureInferenceResponse(BaseModel):
    gloss: str
    confidence: float
    top_k: List[TopKPrediction]
    probabilities: Optional[Dict[str, float]] = None
    model_metadata: Dict[str, Any]


def get_inference_engine() -> InferenceEngine:
    """Get or create the inference engine singleton."""
    global _inference_engine
    if _inference_engine is None:
        # Load the latest checkpoint
        checkpoint_path = Path("models/best_model.pt")
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=503,
                detail="No trained model checkpoint available. Please train a model first."
            )
        
        # Load class mapping from synthetic dataset
        import json
        class_map_path = Path("data/synthetic_landmarks/class_map.json")
        with open(class_map_path) as f:
            class_map = json.load(f)
        
        idx_to_class = {v: k for k, v in class_map.items()}
        
        config = InferenceConfig(
            num_classes=46,
            input_dim=126,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            sequence_length=64,
            target_temporal_length=64,
            normalization_mode="wrist_centric_scale",
            padding_mode="zero",
            truncation_mode="end",
            checkpoint_path=str(Path("models/best_model.pt").resolve()),
            device="cpu",
            return_probabilities=True,
            top_k=5,
        )
        
        _inference_engine = create_inference_engine(
            checkpoint_path=str(Path("models/best_model.pt").resolve()),
            class_mapping={v: k for k, v in enumerate([k for k in json.load(open("data/synthetic_landmarks/class_map.json")).keys()])},
            idx_to_class={v: k for k, v in enumerate([k for k in json.load(open("data/synthetic_landmarks/class_map.json")).keys()])},
        )
    
    return _inference_engine


def _load_class_mappings():
    """Load class mappings from the synthetic dataset."""
    import json
    class_map_path = Path("data/synthetic_landmarks/class_map.json")
    with open(class_map_path) as f:
        class_map = json.load(f)
    idx_to_class = {v: k for k, v in class_map.items()}
    class_mapping = {v: k for k, v in class_map.items()}  # idx -> gloss
    return idx_to_class, class_mapping


@router.post("/inference/gesture", response_model=GestureInferenceResponse)
async def infer_gesture(request: GestureInferenceRequest):
    """
    Run gesture inference on a sequence of landmark frames.
    
    Input: List of frames, each frame is a list of 126 floats (LEFT_HAND 63 + RIGHT_HAND 63)
    Output: Predicted gloss, confidence, top-k predictions, and model metadata
    """
    # Validate input
    if not request.frames:
        raise HTTPException(status_code=400, detail="No frames provided")
    
    for i, frame in enumerate(request.frames):
        if len(frame) != 126:
            raise HTTPException(
                status_code=400,
                detail=f"Frame {i} has {len(frame)} values, expected 126 (LEFT_HAND 63 + RIGHT_HAND 63)"
            )
    
    # Convert to numpy array
    frames = np.array(request.frames, dtype=np.float32)  # [T, 126]
    
    # Validate shape
    if frames.ndim != 2 or frames.shape[1] != 126:
        raise HTTPException(
            status_code=400,
            detail=f"Expected frames of shape [T, 126], got {frames.shape}"
        )
    
    # Validate temporal length
    if len(request.frames) > 200:
        raise HTTPException(
            status_code=400,
            detail="Too many frames. Maximum 200 frames allowed."
        )
    
    try:
        # Get inference engine
        engine = get_inference_engine()
        
        # Run inference on the full sequence
        # The inference engine's preprocess will handle temporal sampling to T=64
        # We need to preprocess the full sequence first
        from ml_pipeline.preprocessing import preprocess_sequence, PreprocessingConfig
        
        # Convert frames to the format expected by preprocessing
        sequence = np.array(request.frames, dtype=np.float32)  # [T, 126]
        
        # Apply preprocessing (temporal sampling to T=64, normalization, etc.)
        preprocessing_config = PreprocessingConfig(target_temporal_length=64)
        preprocessed = preprocess_sequence(sequence, preprocessing_config)
        processed_sequence = preprocessed.sequence  # [64, 126]
        
        # Run inference
        tensor = torch.from_numpy(processed_sequence).float().unsqueeze(0)  # [1, 64, 126]
        
        # Load class mappings
        import json
        class_map_path = "data/synthetic_landmarks/class_map.json"
        with open(class_map_path) as f:
            class_map = json.load(f)
        idx_to_class = {v: k for k, v in json.load(open("data/synthetic_landmarks/class_map.json")).items()}
        
        # Run inference using the model directly (bypassing InferenceEngine for now)
        from ml_pipeline.model import create_model
        from ml_pipeline.inference import InferenceConfig
        from ml_pipeline.model import create_model as create_model_fn
        
        # Load model
        model = create_model(num_classes=46)
        checkpoint_path = Path("models/best_model.pt")
        if not checkpoint_path.exists():
            raise HTTPException(status_code=503, detail="No trained model checkpoint available")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Prepare input tensor [1, T, 126]
        sequence_tensor = torch.from_numpy(processed_sequence).float().unsqueeze(0)
        
        with torch.no_grad():
            logits = model(sequence_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get top-k predictions
            top_k_indices = np.argsort(probs)[::-1][:5]
            top_k_probs = probs[top_k_indices]
            
            # Load class mapping
            import json
            class_map_path = "data/synthetic_landmarks/class_map.json"
            with open(class_map_path) as f:
                class_map = json.load(f)
            idx_to_class = {v: k for k, v in class_map.items()}
            
            predicted_idx = int(top_k_indices[0])
            predicted_gloss = top_k_indices[0]  # We need to map index to gloss
            
            # Build top_k results
            top_k = []
            for idx, prob in zip(top_k_indices, top_k_probs):
                gloss = idx_to_class.get(int(idx), f"UNKNOWN_{idx}")
                top_k.append({
                    "class_index": int(idx),
                    "gloss": gloss,
                    "probability": float(prob)
                })
            
            predicted_gloss = top_k[0]["gloss"]
            confidence = float(top_k_probs[0])
            
            # Full probabilities if requested
            prob_dict = {idx_to_class.get(i, f"UNKNOWN_{i}"): float(p) for i, p in enumerate(probs)}
            
            # Model metadata
            model_metadata = {
                "model_name": "SyntheticPipelineBaseline",
                "architecture": "BiLSTM + GlobalAvgPool + Linear",
                "trained_on_synthetic_data": True,
                "real_isl_recognition_valid": False,
                "num_classes": 46,
                "input_dim": 126,
                "sequence_length": 64,
            }
            
            return GestureInferenceResponse(
                gloss=predicted_gloss,
                confidence=float(confidence),
                top_k=top_k,
                probabilities=prob_dict,
                model_metadata=model_metadata
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/inference/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        import json
        class_map_path = "data/synthetic_landmarks/class_map.json"
        with open(class_map_path) as f:
            class_map = json.load(f)
        
        checkpoint_path = Path("models/best_model.pt")
        if not checkpoint_path.exists():
            return {"status": "no_model", "message": "No trained model checkpoint found"}
        
        return {
            "model_name": "SyntheticPipelineBaseline",
            "architecture": "BiLSTM + GlobalAvgPool + Linear",
            "num_classes": 46,
            "input_dim": 126,
            "sequence_length": 64,
            "trained_on_synthetic_data": True,
            "real_isl_recognition_valid": False,
            "classes": list(json.load(open("data/synthetic_landmarks/class_map.json")).keys()),
            "checkpoint_path": str(Path("models/best_model.pt").resolve()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")