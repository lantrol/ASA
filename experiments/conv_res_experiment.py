import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from predictive_model_pinn.model_factory import get_model_by_name
from predictive_model_pinn.trainer_utils import CosineLoss, train

# --- CONFIGURATION ---
EXPERIMENT_NAME = "res_conv_model"
MODEL_NAME = "ResNetPhasePredictor"
MODEL_KWARGS = {}  # e.g., {"hidden_dim": 2048}

IMAGE_DIR = "data/emnist"
LABEL_DIR = "data/val_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 0.0001
EPOCHS = 250
BATCH_SIZE = 64
VAL_SPLIT = 0.2
# ---------------------


def run_experiment():
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/{EXPERIMENT_NAME}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Starting experiment: {EXPERIMENT_NAME}")
    print(f"Logging to: {log_dir}")

    # Initialize model
    model = get_model_by_name(MODEL_NAME, **MODEL_KWARGS)
    criterion = CosineLoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_NAME} | Params: {total_params}")

    # Train
    model_save_path = f"checkpoints/{EXPERIMENT_NAME}_{timestamp}_best.pth"
    train(
        model=model,
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        criterion=criterion,
        device=DEVICE,
        val_split=VAL_SPLIT,
        writer=writer,
        experiment_name=EXPERIMENT_NAME,
        save_path=model_save_path,
        flatten=False,
    )

    # Log hyperparameters to TensorBoard
    writer.add_text("hyperparams", f"Model: {MODEL_NAME}, LR: {LR}, Epochs: {EPOCHS}")

    writer.close()
    print(f"Experiment finished. Model saved to {model_save_path}")


if __name__ == "__main__":
    run_experiment()
