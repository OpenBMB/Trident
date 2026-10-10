"""mmemb: an extensible multimodal embedding training framework.

All three extension points are registry-based and need no framework changes:
    models   -> mmemb/models/            @MODELS.register("name")
    losses   -> mmemb/losses/            @LOSSES.register("name")
    poolers  -> mmemb/models/pooling.py  @POOLERS.register("name")
"""

__version__ = "0.1.0"

from .config import load_config
from .registry import DATASETS, LOSSES, MODELS, POOLERS

__all__ = ["load_config", "MODELS", "LOSSES", "POOLERS", "DATASETS", "__version__"]
