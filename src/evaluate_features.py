import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from model_features import RoomSpecificReverbCNN  # Updated import
from dataset_features import RoomAwareReverbDataset  # Updated import

def evaluate():
    # Configuration
    MODEL_PATH = "output/reverbcnn.pt"  # Path to your trained model
    TEST_DATA_DIR = "data/test/real"  # Test dataset directory
    FREQS = [250, 500, 1000, 2000, 4000, 8000]  # Frequencies used during training
    BATCH_SIZE = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset with room awareness
    test_dataset = RoomAwareReverbDataset(
        TEST_DATA_DIR, 
        freqs=FREQS, 
        augment=False, 
        include_room_id=True
    )
    
    if len(test_dataset) == 0:
        print(f"No test data found in {TEST_DATA_DIR}")
        return
    
    # Get number of rooms for model initialization
    num_rooms = len(test_dataset.room_to_id) if hasattr(test_dataset, 'room_to_id') else None
    print(f"Number of rooms in test set: {num_rooms}")
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Custom collate function to handle room IDs
    def custom_collate_fn(batch):
        if len(batch[0]) == 3:  # includes room_id
            images, rt60s, room_ids = zip(*batch)
            return torch.stack(images), torch.stack(rt60s), torch.stack(room_ids)
        else:
            images, rt60s = zip(*batch)
            return torch.stack(images), torch.stack(rt60s), None
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    # Load model
    model = RoomSpecificReverbCNN(num_frequencies=len(FREQS), num_rooms=num_rooms)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Evaluation
    all_predictions = []
    all_targets = []
    all_room_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, targets, room_ids = batch
                images = images.to(device)
                room_ids = room_ids.to(device)
                predictions = model(images, room_ids)
                all_room_ids.append(room_ids.cpu())
            else:
                images, targets = batch
                images = images.to(device)
                predictions = model(images)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets)
    
    # Concatenate all results
    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()
    
    if all_room_ids:
        room_ids = torch.cat(all_room_ids).numpy()
        id_to_room = {v: k for k, v in test_dataset.room_to_id.items()}
    else:
        room_ids = None
    
    # Create evaluation directory
    os.makedirs("evaluation", exist_ok=True)
    
    # Calculate metrics
    overall_mse = mean_squared_error(targets, predictions)
    overall_mae = mean_absolute_error(targets, predictions)
    overall_r2 = r2_score(targets, predictions)
    
    print(f"\n=== Overall Performance ===")
    print(f"MSE: {overall_mse:.6f}")
    print(f"MAE: {overall_mae:.6f}")
    print(f"R² Score: {overall_r2:.6f}")
    print(f"RMSE: {np.sqrt(overall_mse):.6f}")
    
    # Per-frequency analysis
    freq_names = [f"{freq}Hz" for freq in FREQS]
    
    print(f"\n=== Per-Frequency Performance ===")
    for i, freq_name in enumerate(freq_names):
        freq_mse = mean_squared_error(targets[:, i], predictions[:, i])
        freq_mae = mean_absolute_error(targets[:, i], predictions[:, i])
        freq_r2 = r2_score(targets[:, i], predictions[:, i])
        print(f"{freq_name:>8s}: MSE={freq_mse:.6f}, MAE={freq_mae:.6f}, R²={freq_r2:.6f}")
    
    # Room-specific analysis if available
    if room_ids is not None:
        print(f"\n=== Per-Room Performance ===")
        unique_rooms = np.unique(room_ids)
        
        for room_id in unique_rooms:
            room_mask = room_ids == room_id
            room_name = id_to_room[room_id]
            room_preds = predictions[room_mask]
            room_targets = targets[room_mask]
            
            if len(room_preds) > 0:
                room_mse = mean_squared_error(room_targets, room_preds)
                room_mae = mean_absolute_error(room_targets, room_preds)
                room_r2 = r2_score(room_targets, room_preds)
                print(f"{room_name:>15s} ({len(room_preds):>3d} samples): MSE={room_mse:.6f}, MAE={room_mae:.6f}, R²={room_r2:.6f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Overall scatter plot
    axes[0, 0].scatter(targets.flatten(), predictions.flatten(), alpha=0.5, s=1)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Ground Truth RT60 (s)')
    axes[0, 0].set_ylabel('Predicted RT60 (s)')
    axes[0, 0].set_title(f'Overall Performance\nR² = {overall_r2:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Per-frequency performance
    freq_r2_scores = []
    for i in range(len(FREQS)):
        freq_r2 = r2_score(targets[:, i], predictions[:, i])
        freq_r2_scores.append(freq_r2)
    
    axes[0, 1].bar(freq_names, freq_r2_scores)
    axes[0, 1].set_xlabel('Frequency Band')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('Per-Frequency Performance')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    errors = predictions - targets
    axes[0, 2].hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Prediction Error (s)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Error Distribution')
    axes[0, 2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Frequency correlation heatmap
    correlation_matrix = np.corrcoef(errors.T)
    im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 0].set_xticks(range(len(freq_names)))
    axes[1, 0].set_yticks(range(len(freq_names)))
    axes[1, 0].set_xticklabels(freq_names, rotation=45)
    axes[1, 0].set_yticklabels(freq_names)
    axes[1, 0].set_title('Error Correlation Between Frequencies')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 5. Room-specific performance (if available)
    if room_ids is not None:
        room_r2_scores = []
        room_names = []
        
        for room_id in unique_rooms:
            room_mask = room_ids == room_id
            room_name = id_to_room[room_id]
            room_preds = predictions[room_mask]
            room_targets = targets[room_mask]
            
            if len(room_preds) > 0:
                room_r2 = r2_score(room_targets, room_preds)
                room_r2_scores.append(room_r2)
                room_names.append(room_name)
        
        axes[1, 1].bar(range(len(room_names)), room_r2_scores)
        axes[1, 1].set_xlabel('Room')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Per-Room Performance')
        axes[1, 1].set_xticks(range(len(room_names)))
        axes[1, 1].set_xticklabels(room_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No room information\navailable', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Per-Room Performance')
    
    # 6. Prediction vs Ground Truth for each frequency
    colors = plt.cm.tab10(np.linspace(0, 1, len(FREQS)))
    for i, (freq_name, color) in enumerate(zip(freq_names, colors)):
        axes[1, 2].scatter(targets[:, i], predictions[:, i], 
                          alpha=0.5, s=15, color=color, label=freq_name)
    
    axes[1, 2].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=1)
    axes[1, 2].set_xlabel('Ground Truth RT60 (s)')
    axes[1, 2].set_ylabel('Predicted RT60 (s)')
    axes[1, 2].set_title('Per-Frequency Predictions')
    axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("evaluation/comprehensive_evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results to file
    with open("evaluation/evaluation_results.txt", "w") as f:
        f.write("=== Model Evaluation Results ===\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Test dataset: {TEST_DATA_DIR}\n")
        f.write(f"Number of test samples: {len(predictions)}\n")
        f.write(f"Number of frequencies: {len(FREQS)}\n")
        if num_rooms:
            f.write(f"Number of rooms: {num_rooms}\n")
        f.write("\n")
        
        f.write("=== Overall Performance ===\n")
        f.write(f"MSE: {overall_mse:.6f}\n")
        f.write(f"MAE: {overall_mae:.6f}\n")
        f.write(f"RMSE: {np.sqrt(overall_mse):.6f}\n")
        f.write(f"R² Score: {overall_r2:.6f}\n\n")
        
        f.write("=== Per-Frequency Performance ===\n")
        for i, freq_name in enumerate(freq_names):
            freq_mse = mean_squared_error(targets[:, i], predictions[:, i])
            freq_mae = mean_absolute_error(targets[:, i], predictions[:, i])
            freq_r2 = r2_score(targets[:, i], predictions[:, i])
            f.write(f"{freq_name}: MSE={freq_mse:.6f}, MAE={freq_mae:.6f}, R²={freq_r2:.6f}\n")
        
        if room_ids is not None:
            f.write("\n=== Per-Room Performance ===\n")
            for room_id in unique_rooms:
                room_mask = room_ids == room_id
                room_name = id_to_room[room_id]
                room_preds = predictions[room_mask]
                room_targets = targets[room_mask]
                
                if len(room_preds) > 0:
                    room_mse = mean_squared_error(room_targets, room_preds)
                    room_mae = mean_absolute_error(room_targets, room_preds)
                    room_r2 = r2_score(room_targets, room_preds)
                    f.write(f"{room_name} ({len(room_preds)} samples): MSE={room_mse:.6f}, MAE={room_mae:.6f}, R²={room_r2:.6f}\n")
    
    print(f"\nEvaluation complete! Results saved to evaluation/")

if __name__ == "__main__":
    evaluate()