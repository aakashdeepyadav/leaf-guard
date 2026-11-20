"""Central configuration for the LeafGuard project."""

from pathlib import Path

SEED = 42
IMG_SIZE = 160
AE_IMG_SIZE = 64
BATCH_SIZE = 32
QUICK_LIMIT = 2000
NUM_CLASSES = 38
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
BASELINE_EPOCHS = 15
TRANSFER_HEAD_EPOCHS = 5
TRANSFER_FINE_TUNE_EPOCHS = 5
AE_EPOCHS = 10
ATTACK_LIMIT = 1000
ATTACK_EPSILONS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)
ATTACK_DEFAULT_EPSILON = 0.02
CONFIDENCE_WARNING_THRESHOLD = 0.6
DROPOUT_RATE = 0.3
AE_BATCH_SIZE = BATCH_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REAL_WORLD_DIR = DATA_DIR / "real_world_samples"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
PLANTVILLAGE_DATASET = "plant_village"

AUTOTUNE = -1


def ensure_directories() -> None:
    """Create directories used for generated project artifacts."""
    for directory in (DATA_DIR, REAL_WORLD_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
