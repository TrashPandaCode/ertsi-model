from dataset import ReverbRoomDataset
from model import ReverbCNN
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
import numpy as np

from seed import set_seeds


def overfit_test():
    set_seeds(42)

    torch.set_float32_matmul_precision('medium')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Overfitting test will use: {device}")

    params = {
        "epochs": 200,  # More epochs for overfitting
        "batch_size": 32,  # Smaller batch size
        "lr": 0.001,  # Standard learning rate
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "subset_size": 3562,  # Number of samples to overfit on
        "model_out": "output/overfit_test_model.pt",
    }

    os.makedirs(os.path.dirname(params["model_out"]), exist_ok=True)

    # Load the full training dataset
    full_train_dataset = ReverbRoomDataset(
        ["data/train/synth", "data/train/real"],
        freqs=params["freqs"],
        augment=False,  # Disable augmentation for overfitting test
    )

    print(f"Full training dataset size: {len(full_train_dataset)} samples")

    # Create a small subset for overfitting
    subset_indices = list(range(min(params["subset_size"], len(full_train_dataset))))
    subset_dataset = Subset(full_train_dataset, subset_indices)

    print(f"Overfitting on subset of {len(subset_dataset)} samples")

    # Use the same subset for both training and validation to ensure overfitting
    train_loader = DataLoader(
        subset_dataset, 
        batch_size=params["batch_size"], 
        shuffle=True, 
        num_workers=2
    )
    
    # Use the same data for validation (this will show perfect overfitting)
    val_loader = DataLoader(
        subset_dataset, 
        batch_size=params["batch_size"], 
        shuffle=False, 
        num_workers=2
    )

    print("\n=== OVERFITTING TEST ===")
    print("Training and validation on the same small subset")
    print("Expecting validation loss to approach zero")

    model = ReverbCNN(
        num_frequencies=len(params["freqs"]), 
        learning_rate=params["lr"]
    )

    # Override the configure_optimizers method to remove the scheduler for overfitting
    def configure_optimizers_overfit(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )
        return optimizer
    
    # Replace the method
    model.configure_optimizers = configure_optimizers_overfit.__get__(model, ReverbCNN)

    # Disable dropout during overfitting test
    def disable_dropout(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0  # Set dropout probability to 0
    
    disable_dropout(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/overfit",
        filename="overfit-test-{epoch:02d}-{val_loss:.6f}",
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        every_n_epochs=10,  # Save checkpoints more frequently
    )

    logger = TensorBoardLogger("logs", name="overfit_test")

    trainer = pl.Trainer(
        max_epochs=params["epochs"],
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=1,  # Log every step
        check_val_every_n_epoch=1,  # Validate every epoch for overfitting test
        enable_progress_bar=True,
    )

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")

    # Print sample data info
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch info:")
    print(f"Input shape: {sample_batch[0].shape}")
    print(f"Target shape: {sample_batch[1].shape}")
    print(f"Target example: {sample_batch[1][0].numpy()}")

    trainer.fit(model, train_loader, val_loader)

    # Save the overfitted model
    torch.save(model.state_dict(), params["model_out"])
    print(f"\nOverfitted model saved to {params['model_out']}")

    # Test final performance on the subset
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            inputs, targets = batch
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
                model = model.cuda()
            
            outputs = model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
            
            # Print first few predictions vs targets
            if num_batches == 1:
                print(f"\nFinal predictions vs targets (first 3 samples):")
                for i in range(min(3, len(outputs))):
                    pred = outputs[i].cpu().numpy()
                    target = targets[i].cpu().numpy()
                    print(f"Sample {i+1}:")
                    print(f"  Predicted: {pred}")
                    print(f"  Target:    {target}")
                    print(f"  Abs diff:  {np.abs(pred - target)}")
        
        avg_loss = total_loss / num_batches
        print(f"\nFinal average MSE loss on subset: {avg_loss:.8f}")
        
        if avg_loss < 0.001:
            print("✅ SUCCESS: Model successfully overfitted (loss < 0.001)")
        elif avg_loss < 0.01:
            print("⚠️  PARTIAL: Model partially overfitted (loss < 0.01)")
        else:
            print("❌ FAILED: Model did not overfit well (loss >= 0.01)")


if __name__ == "__main__":
    overfit_test()