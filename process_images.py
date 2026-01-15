import cv2
import numpy as np
import argparse
import os
from pathlib import Path

def apply_darkness(image, alpha=0.4):
    overlay = np.zeros_like(image, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def add_gaussian_noise(image, mean=0, std_dev=25):
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.int16)
    noisy_image = image.astype(np.int16) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def process_images(input_dir, output_dir, alpha, std_dev):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
    count = 0

    if not input_path.exists():
        print(f"Error input directory '{input_dir}' does not exist.")
        return

    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in valid_extensions:
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    continue

                darkened_img = apply_darkness(img, alpha)
                final_img = add_gaussian_noise(darkened_img, mean=0, std_dev=std_dev)

                save_path = output_path / file_path.name
                cv2.imwrite(str(save_path), final_img)
                
                print(f"Processed: {file_path.name}")
                count += 1
            except Exception as e:
                print(f"Failed to process {file_path.name}: {e}")

    print(f"Processing complete. Total images: {count}")
    print(f"Output saved to: {output_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Augmentation Tool")
    parser.add_argument("--input", type=str, required=True, help="Path to input image directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--alpha", type=float, default=0.4, help="Darkness factor (0.0 to 1.0)")
    parser.add_argument("--std", type=int, default=25, help="Standard deviation for Gaussian noise")

    args = parser.parse_args()
    
    process_images(args.input, args.output, args.alpha, args.std)



#     python augment_data.py \
#   --input "" \
#   --output "" \
#   --alpha 0.5 \       0.1->0.3: little darkness; 0.4->0.6: moderate darkness; 0.7->0.9: high darkness
#   --std 30            10->30: low noise; 40->60: moderate noise; 70->100: high noise
