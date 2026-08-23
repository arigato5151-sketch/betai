"""Re-export the legacy MLModelPipeline singleton for backward compatibility.

All new code should use ``app.prediction.ml.model_router`` or
``app.prediction.ml.ml_pipeline`` instead.
"""

from app.prediction.ml._legacy_model import *  # noqa: F401,F403
from app.prediction.ml._legacy_model import MLModelPipeline, ml_pipeline

__all__ = ["MLModelPipeline", "ml_pipeline"]
