"""
Single source of truth for filesystem paths used across the artifact
scripts. All paths are overridable via environment variables so the
package runs unmodified on a fresh machine, not just this sandbox.

Previously these paths were hardcoded as string literals independently
inside collect_real_data.py, apply_model_to_real_data.py, and
aggregate_real_world.py (e.g. "/home/user/models"). That meant moving
the artifacts to a new machine, or even just a new directory, required
find-and-replace across three files with real risk of missing one.
"""
import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("STARVATION_ARTIFACTS_DIR", Path(__file__).resolve().parent))
MODELS_DIR = Path(os.environ.get("STARVATION_MODELS_DIR", BASE_DIR / "models"))
DATA_DIR = Path(os.environ.get("STARVATION_DATA_DIR", BASE_DIR))
OUTPUT_DIR = Path(os.environ.get("STARVATION_OUTPUT_DIR", BASE_DIR))

for _d in (MODELS_DIR, DATA_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
