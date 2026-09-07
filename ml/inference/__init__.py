"""
Real-time inference engine for voice cloning detection.
"""

from .cnn import CNNInference, CNNInferenceError, CNNInferenceResult

__all__ = ["CNNInference", "CNNInferenceError", "CNNInferenceResult"]
