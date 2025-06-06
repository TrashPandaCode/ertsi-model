from dataset import ReverbRoomDataset
from model import ReverbCNNComparison
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch
import json
from seed import set_seeds


def train_backbone_comparison():
    """Train and compare different backbone architectures"""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "42"

    set_seeds(42)
    
    # Backbone architectures to compare
    backbones = [
        "resnet50_places365",
        "dinov2", 
        "clip",  # OpenCLIP ViT-B-32
        "clip_vit_l",  # OpenCLIP ViT-L-14 
        "efficientnet_b4",
        "clip_convnext",
        # "sam_b", # does not work
    ]
    
    # Training parameters
    params = {
        "synth_epochs": 60,
        "real_epochs": 30,
        "batch_size": 16,
        "lr": 0.0008,
        "fine_tune_lr": 0.00005,
        "freqs": [250, 500, 1000, 2000, 4000, 8000],
        "dropout_rate": 0.15,
        "warmup_epochs": 5,
    }
    
    # Load datasets once
    print("Loading datasets...")
    synth_train = ReverbRoomDataset("data/train/synth", freqs=params["freqs"], augment=True)
    synth_val = ReverbRoomDataset("data/val/synth", freqs=params["freqs"], augment=False)
    real_train = ReverbRoomDataset("data/train/real", freqs=params["freqs"], augment=True)
    real_val = ReverbRoomDataset("data/val/real", freqs=params["freqs"], augment=False)
    
    # Create data loaders
    synth_train_loader = DataLoader(synth_train, batch_size=params["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    synth_val_loader = DataLoader(synth_val, batch_size=params["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    real_train_loader = DataLoader(real_train, batch_size=params["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    real_val_loader = DataLoader(real_val, batch_size=params["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    # Results storage
    results = {}
    
    # Train each backbone
    for backbone_name in backbones:
        print(f"\n{'='*80}")
        print(f"Training with backbone: {backbone_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Adjust batch size for memory-intensive models
            current_batch_size = params["batch_size"]
            if backbone_name in ["clip_vit_l", "clip_convnext", "sam_b"]:
                if backbone_name == "sam_b":
                    current_batch_size = 4  # Very small batch size for SAM
                else:
                    current_batch_size = max(8, params["batch_size"] // 2)
                
                print(f"Adjusted batch size to {current_batch_size} for {backbone_name}")
                
                # Create new data loaders with adjusted batch size
                current_synth_train_loader = DataLoader(synth_train, batch_size=current_batch_size, shuffle=True, num_workers=4, pin_memory=True)
                current_synth_val_loader = DataLoader(synth_val, batch_size=current_batch_size, shuffle=False, num_workers=4, pin_memory=True)
                current_real_train_loader = DataLoader(real_train, batch_size=current_batch_size, shuffle=True, num_workers=4, pin_memory=True)
                current_real_val_loader = DataLoader(real_val, batch_size=current_batch_size, shuffle=False, num_workers=4, pin_memory=True)
            else:
                current_synth_train_loader = synth_train_loader
                current_synth_val_loader = synth_val_loader
                current_real_train_loader = real_train_loader
                current_real_val_loader = real_val_loader
            
            model = ReverbCNNComparison(
                backbone_name=backbone_name,
                num_frequencies=len(params["freqs"]),
                learning_rate=params["lr"],
                frequencies=params["freqs"],
                dropout_rate=params["dropout_rate"],
                use_scheduler=True,
                warmup_epochs=params["warmup_epochs"],
            )
            
            # Train the model
            result = train_single_backbone(
                model,
                current_synth_train_loader,
                current_synth_val_loader,
                current_real_train_loader,
                current_real_val_loader,
                params,
                backbone_name,
            )
            
            results[backbone_name] = result
            
        except Exception as e:
            print(f"Failed to train {backbone_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[backbone_name] = {"error": str(e)}
            continue
    
    # Save comparison results
    os.makedirs("comparison_results", exist_ok=True)
    with open("comparison_results/backbone_comparison.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BACKBONE COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for backbone, result in results.items():
        if "error" in result:
            print(f"{backbone:<20}: FAILED - {result['error']}")
        else:
            print(f"{backbone:<20}: Val Loss: {result.get('best_val_loss', 'N/A'):.4f}, "
                  f"Val R²: {result.get('best_val_r2', 'N/A'):.4f}")


def train_single_backbone(
    model,
    synth_train_loader,
    synth_val_loader,
    real_train_loader,
    real_val_loader,
    params,
    backbone_name,
):
    """Train a single backbone architecture"""
    
    # Handle deterministic algorithms for transformer models and SAM
    use_deterministic = backbone_name not in ["dinov2", "clip", "clip_vit_l", "clip_convnext", "sam_b"]
    
    # Special settings for SAM
    precision = "16-mixed" if backbone_name == "sam_b" else "32-true"
    accumulate_grad_batches = 4 if backbone_name == "sam_b" else 1
    
    # Stage 1: Synthetic data training
    print(f"\n=== STAGE 1: Training {backbone_name} on synthetic data ===")
    
    checkpoint_callback_synth = ModelCheckpoint(
        dirpath=f"checkpoints/comparison/{backbone_name}/synth",
        filename=f"{backbone_name}-synth-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=2,
        monitor=f"val_loss_{backbone_name}",
        mode="min",
        save_last=True,
        verbose=True,
    )
    
    early_stop_callback_synth = EarlyStopping(
        monitor=f"val_loss_{backbone_name}",
        patience=10,
        mode="min",
        verbose=True,
        min_delta=0.0001,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    logger_synth = TensorBoardLogger(
        "logs/comparison",
        name=f"{backbone_name}_synth",
        version=None,
    )
    
    trainer_synth = pl.Trainer(
        max_epochs=params["synth_epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback_synth, early_stop_callback_synth, lr_monitor],
        logger=logger_synth,
        log_every_n_steps=10,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        deterministic=use_deterministic,
        enable_model_summary=True,
        precision=precision,  # Use mixed precision for SAM
        accumulate_grad_batches=accumulate_grad_batches,  # Gradient accumulation for SAM
    )
    
    trainer_synth.fit(model, synth_train_loader, synth_val_loader)
    
    # Save synthetic model
    synth_path = f"comparison_results/{backbone_name}_synth.pt"
    os.makedirs(os.path.dirname(synth_path), exist_ok=True)
    torch.save(model.state_dict(), synth_path)
    
    # Stage 2: Real data fine-tuning
    print(f"\n=== STAGE 2: Fine-tuning {backbone_name} on real data ===")
    
    # Adjust learning rate for fine-tuning
    original_lr = model.learning_rate
    model.learning_rate = params["fine_tune_lr"]
    
    checkpoint_callback_real = ModelCheckpoint(
        dirpath=f"checkpoints/comparison/{backbone_name}/real",
        filename=f"{backbone_name}-real-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=2,
        monitor=f"val_loss_{backbone_name}",
        mode="min",
        save_last=True,
        verbose=True,
    )
    
    early_stop_callback_real = EarlyStopping(
        monitor=f"val_loss_{backbone_name}",
        patience=15,
        mode="min",
        verbose=True,
        min_delta=0.00005,
    )
    
    logger_real = TensorBoardLogger(
        "logs/comparison",
        name=f"{backbone_name}_real_finetune",
        version=None,
    )
    
    trainer_real = pl.Trainer(
        max_epochs=params["real_epochs"],
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback_real, early_stop_callback_real, lr_monitor],
        logger=logger_real,
        log_every_n_steps=5,
        val_check_interval=1.0,
        gradient_clip_val=0.5,
        deterministic=use_deterministic,
        enable_model_summary=False,
        precision=precision,  # Use mixed precision for SAM
        accumulate_grad_batches=accumulate_grad_batches,  # Gradient accumulation for SAM
    )
    
    trainer_real.fit(model, real_train_loader, real_val_loader)
    
    # Save final model
    final_path = f"comparison_results/{backbone_name}_final.pt"
    torch.save(model.state_dict(), final_path)
    
    # Restore original learning rate
    model.learning_rate = original_lr
    
    # Get best metrics
    try:
        best_val_loss = trainer_real.callback_metrics.get(f"val_loss_{backbone_name}", "N/A")
        best_val_r2 = trainer_real.callback_metrics.get(f"val_r2_{backbone_name}", "N/A")
        
        result = {
            "best_val_loss": float(best_val_loss) if best_val_loss != "N/A" else None,
            "best_val_r2": float(best_val_r2) if best_val_r2 != "N/A" else None,
            "synth_model_path": synth_path,
            "final_model_path": final_path,
        }
        
        print(f"{backbone_name} training completed!")
        print(f"Best validation loss: {best_val_loss}")
        print(f"Best validation R²: {best_val_r2}")
        
        return result
        
    except Exception as e:
        print(f"Error retrieving metrics for {backbone_name}: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    train_backbone_comparison()