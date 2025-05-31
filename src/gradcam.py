import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from model import ReverbCNN
from dataset import ReverbRoomDataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Configuration
MODEL_PATH = "output/exV4-reverbcnn.pt"  # Path to your trained model
DATA_DIR = "data/test/real"  # Root directory of your dataset
NUM_FREQUENCIES = 6  # Should match your model's configuration
NUM_RANDOM_SAMPLES = 3  # Number of random images to select
FREQUENCIES_TO_VISUALIZE = [
    0,
    1,
    2,
    3,
    4,
    5,
]  # Indices of frequency bands to visualize (e.g., 0 for 250Hz, 1 for 500Hz, etc.)
OUTPUT_DIR = "exV4-evaluation/gradcam"  # Directory to save Grad-CAM outputs

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def apply_grad_cam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    model = ReverbCNN(num_frequencies=NUM_FREQUENCIES)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the model is trained and saved to the correct path.")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure NUM_FREQUENCIES matches the saved model's architecture.")
        return

    model.to(device)
    model.eval()

    # 2. Define Target Layer for Grad-CAM
    # Typically the last convolutional layer of the backbone.
    # For ResNet-based models like in ReverbCNN, model.backbone[-1] is 'layer4'.
    target_layers = [model.backbone[-1]]

    # 3. Load Sample Image and Preprocess
    # Frequencies used during training, needed for dataset initialization if it filters by them
    # These are the default frequencies from train.py
    training_freqs = [250, 500, 1000, 2000, 4000, 8000]
    dataset = ReverbRoomDataset(DATA_DIR, freqs=training_freqs)

    if len(dataset) == 0:
        print(
            f"Dataset in '{DATA_DIR}' is empty or not found. Cannot perform Grad-CAM."
        )
        return

    # Get random image indices
    num_samples = min(NUM_RANDOM_SAMPLES, len(dataset))
    if num_samples == 0:
        print("No samples to visualize. Exiting.")
        return

    random_image_indices = sorted(random.sample(range(len(dataset)), num_samples))
    print(f"Randomly selected {num_samples} image indices: {random_image_indices}")

    # 4. Define Grad-CAM Target for Regression
    class RegressionOutputTarget:
        def __init__(self, output_index):
            self.output_index = output_index

        def __call__(self, model_output):
            # model_output shape: [batch_size, num_frequencies]
            # Or potentially [num_frequencies] if batch_size is 1 and it gets squeezed
            if model_output.ndim == 1:
                # If 1D, assume it's [num_frequencies] for a single item in batch
                return model_output[self.output_index]
            elif model_output.ndim == 2:
                # If 2D, assume it's [batch_size, num_frequencies]
                return model_output[:, self.output_index]
            else:
                raise ValueError(
                    f"model_output has unexpected number of dimensions: {model_output.ndim}. Expected 1 or 2."
                )

    # 5. Initialize and Run Grad-CAM
    try:
        with GradCAM(model=model, target_layers=target_layers) as cam:
            for image_idx in random_image_indices:
                print(f"\nProcessing image index: {image_idx}")

                # Get the raw PIL image path from dataset entries
                img_path, rt60_values = dataset.entries[image_idx]
                try:
                    raw_pil_image = Image.open(img_path).convert("RGB")
                except FileNotFoundError:
                    print(f"  Error: Image file not found at {img_path}. Skipping.")
                    continue

                # Apply the dataset's transform to get the input tensor for the model
                input_tensor = dataset.transform(raw_pil_image).unsqueeze(0).to(device)

                # For visualization with show_cam_on_image
                vis_image_np = np.array(raw_pil_image.resize((224, 224))) / 255.0
                if vis_image_np.ndim == 2:  # Grayscale
                    vis_image_np = np.stack((vis_image_np,) * 3, axis=-1)
                elif vis_image_np.shape[2] == 4:  # RGBA
                    vis_image_np = vis_image_np[:, :, :3]

                # Create a combined visualization for multiple frequencies
                num_freqs = len(FREQUENCIES_TO_VISUALIZE)
                fig, axes = plt.subplots(
                    1, num_freqs + 1, figsize=(5 * (num_freqs + 1), 5)
                )

                # First column: original image
                axes[0].imshow(vis_image_np)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                # Process each frequency
                for i, freq_idx in enumerate(FREQUENCIES_TO_VISUALIZE):
                    if not (0 <= freq_idx < NUM_FREQUENCIES):
                        print(
                            f"  Error: Frequency index {freq_idx} is out of range. Skipping."
                        )
                        continue

                    target = RegressionOutputTarget(freq_idx)
                    grayscale_cam = cam(input_tensor=input_tensor, targets=[target])

                    if grayscale_cam is None or grayscale_cam.size == 0:
                        print(
                            f"  Error: Grad-CAM computation returned None or empty array for frequency {training_freqs[freq_idx]} Hz. Skipping."
                        )
                        continue

                    grayscale_cam_image = grayscale_cam[0, :]
                    cam_image_overlay = show_cam_on_image(
                        vis_image_np, grayscale_cam_image, use_rgb=True
                    )

                    # Display in the figure
                    axes[i + 1].imshow(cam_image_overlay)
                    axes[i + 1].set_title(
                        f"{training_freqs[freq_idx]} Hz\nRT60: {rt60_values[freq_idx]:.2f}s"
                    )
                    axes[i + 1].axis("off")

                # Save the combined visualization
                plt.tight_layout()
                combined_filename = f"gradcam_combined_img{image_idx}.png"
                plt.savefig(
                    os.path.join(OUTPUT_DIR, combined_filename),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

                print(
                    f"  Combined Grad-CAM visualization saved to {os.path.join(OUTPUT_DIR, combined_filename)}"
                )

                # Also save individual visualizations for each frequency
                for freq_idx in FREQUENCIES_TO_VISUALIZE:
                    if not (0 <= freq_idx < NUM_FREQUENCIES):
                        continue

                    target = RegressionOutputTarget(freq_idx)
                    grayscale_cam = cam(input_tensor=input_tensor, targets=[target])

                    if grayscale_cam is None or grayscale_cam.size == 0:
                        continue

                    grayscale_cam_image = grayscale_cam[0, :]
                    cam_image_overlay = show_cam_on_image(
                        vis_image_np, grayscale_cam_image, use_rgb=True
                    )

                    # Save individual images
                    base_filename = (
                        f"gradcam_img{image_idx}_freq{training_freqs[freq_idx]}Hz"
                    )
                    try:
                        Image.fromarray((vis_image_np * 255).astype(np.uint8)).save(
                            os.path.join(OUTPUT_DIR, f"{base_filename}_original.png")
                        )
                        plt.imsave(
                            os.path.join(OUTPUT_DIR, f"{base_filename}_heatmap.png"),
                            grayscale_cam_image,
                            cmap="jet",
                        )
                        Image.fromarray(cam_image_overlay).save(
                            os.path.join(OUTPUT_DIR, f"{base_filename}_overlay.png")
                        )
                    except Exception as e:
                        print(
                            f"  Error saving individual Grad-CAM images for frequency {training_freqs[freq_idx]} Hz: {e}"
                        )

    except Exception as e:
        print(f"Error during Grad-CAM computation: {e}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    apply_grad_cam()
