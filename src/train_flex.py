from dataset import ReverbRoomDataset
from model_flex import ReverbCNN
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
import argparse

from seed import set_seeds


def train():
    # Parse command line arguments for backbone selection
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50_places365",
        choices=[
            "resnet50_places365",
            "efficientnet_b4",
            "convnext_base",
            "swin_base",
            "densenet169",
        ],
        help="Backbone architecture to use",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="exV6",
        help="Experiment name for output files",
    )
    args = parser.parse_args()

    set_seeds(42)

    params = {
        "synth_epochs": 100,  # Epochs for training on synthetic data
        "real_epochs": 50,  # Epochs for fine-tuning on real data
        "batch_size": 32,
        "lr": 0.001,
        "fine_tune_lr": 0.0001,  # Lower learning rate for fine-tuning
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "backbone": args.backbone,
        "experiment": args.experiment,
        "synth_model_out": f"output/{args.experiment}-reverbcnn_{args.backbone}_synth.pt",
        "final_model_out": f"output/{args.experiment}-reverbcnn_{args.backbone}.pt",
    }

    os.makedirs(os.path.dirname(params["synth_model_out"]), exist_ok=True)

    synth_train = ReverbRoomDataset(
        ["data/train/synth/hybrid", "data/train/synth/non-hybrid"],
        freqs=params["freqs"],
        augment=True,
    )
    synth_val = ReverbRoomDataset(
        "data/val/synth", freqs=params["freqs"], augment=False
    )

    real_train = ReverbRoomDataset(
        "data/train/real", freqs=params["freqs"], augment=True
    )
    real_val = ReverbRoomDataset("data/val/real", freqs=params["freqs"], augment=False)

    # Print dataset sizes for information
    print(f"Using backbone: {params['backbone']}")
    print(f"Synthetic training set: {len(synth_train)} samples")
    print(f"Synthetic validation set: {len(synth_val)} samples")
    print(f"Real training set: {len(real_train)} samples")
    print(f"Real validation set: {len(real_val)} samples")

    synth_train_loader = DataLoader(
        synth_train, batch_size=params["batch_size"], shuffle=True, num_workers=4
    )
    synth_val_loader = DataLoader(
        synth_val, batch_size=params["batch_size"], shuffle=False, num_workers=4
    )

    real_train_loader = DataLoader(
        real_train, batch_size=params["batch_size"], shuffle=True, num_workers=4
    )
    real_val_loader = DataLoader(
        real_val, batch_size=params["batch_size"], shuffle=False, num_workers=4
    )

    print(f"\n=== STAGE 1: Training on synthetic data with {params['backbone']} ===")

    model = ReverbCNN(
        num_frequencies=len(params["freqs"]),
        learning_rate=params["lr"],
        backbone=params["backbone"],
    )

    checkpoint_callback_synth = ModelCheckpoint(
        dirpath=f"checkpoints/synth/{params['backbone']}",
        filename=f"{params['experiment']}-reverbcnn-{params['backbone']}-synth-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback_synth = EarlyStopping(
        monitor="val_loss", patience=5, mode="min"
    )

    logger_synth = TensorBoardLogger(
        "logs", name=f"{params['experiment']}-reverbcnn_{params['backbone']}_synth"
    )

    trainer_synth = pl.Trainer(
        max_epochs=params["synth_epochs"],
        accelerator="auto",  # Automatically use GPU if available
        callbacks=[checkpoint_callback_synth, early_stop_callback_synth],
        logger=logger_synth,
        log_every_n_steps=10,
    )

    trainer_synth.fit(model, synth_train_loader, synth_val_loader)

    # Save the model after synthetic training
    torch.save(model.state_dict(), params["synth_model_out"])
    print(f"Synthetic model saved to {params['synth_model_out']}")

    print(f"\n=== STAGE 2: Fine-tuning on real data with {params['backbone']} ===")

    # Update the learning rate for fine-tuning (lower learning rate)
    model.learning_rate = params["fine_tune_lr"]

    checkpoint_callback_real = ModelCheckpoint(
        dirpath=f"checkpoints/real/{params['backbone']}",
        filename=f"{params['experiment']}-reverbcnn-{params['backbone']}-real-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback_real = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",  # More patience for fine-tuning
    )

    logger_real = TensorBoardLogger(
        "logs",
        name=f"{params['experiment']}-reverbcnn_{params['backbone']}_real_finetune",
    )

    trainer_real = pl.Trainer(
        max_epochs=params["real_epochs"],
        accelerator="auto",
        callbacks=[checkpoint_callback_real, early_stop_callback_real],
        logger=logger_real,
        log_every_n_steps=5,
    )

    trainer_real.fit(model, real_train_loader, real_val_loader)

    # Save the final fine-tuned model
    torch.save(model.state_dict(), params["final_model_out"])
    print(f"Final fine-tuned model saved to {params['final_model_out']}")


def train_all_backbones():
    """Train all backbone architectures for comparison"""
    backbones = [
        "resnet50_places365",
        "efficientnet_b4",
        "convnext_base",
        "swin_base",
        "densenet169",
    ]

    for backbone in backbones:
        print(f"\n{'=' * 60}")
        print(f"TRAINING WITH BACKBONE: {backbone.upper()}")
        print(f"{'=' * 60}")

        try:
            # Simulate command line args
            import sys

            original_argv = sys.argv.copy()
            sys.argv = [
                "train.py",
                "--backbone",
                backbone,
                "--experiment",
                "comparison",
            ]

            train()

            # Restore original args
            sys.argv = original_argv

        except Exception as e:
            print(f"Error training {backbone}: {e}")
            continue


if __name__ == "__main__":
    # Check if we want to train all backbones
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--all":
        train_all_backbones()
    else:
        train()
