import os
import shutil
import torch
from ultralytics import YOLO


def train_model():
    # Select Apple Silicon GPU (MPS), NVIDIA CUDA, or CPU automatically
    if torch.cuda.is_available():
        device = "cuda"
        print("Training on NVIDIA GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Training on Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("Training on CPU")

    os.makedirs("models", exist_ok=True)

    # Load pretrained YOLO11n base model (downloads automatically if missing)
    model = YOLO("models/yolo11n.pt")

    # Train on the Alzheimer's dataset
    model.train(
        data="data/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project="outputs",
        name="alzheimers_train",
        exist_ok=True,
    )

    # Copy the best checkpoint to models/best.pt for inference and deployment
    best_weights = os.path.join("outputs", "alzheimers_train", "weights", "best.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, "models/best.pt")
        print("Training complete! Saved best weights to models/best.pt")


if __name__ == "__main__":
    train_model()