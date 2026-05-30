import argparse
import cv2
import numpy as np
from pathlib import Path

def augment_texture(image, seed):
    """Augment texture by modifying color, saturation, and contrast of non-white pixels."""
    np.random.seed(seed)
    
    # Create mask for non-white pixels (threshold at 250 to catch near-white pixels)
    mask = np.any(image < 150, axis=2)
    # exclude black
    mask &= np.any(image > 5, axis=2)
    
    # Convert to float for processing
    img_float = image.astype(np.float32)
    
    # Color shift: add random values to each channel
    color_shift = np.random.uniform(-256, 256, 3)
    img_float[mask] += color_shift
    
    # Convert to HSV for saturation and contrast adjustments
    img_hsv = cv2.cvtColor(np.clip(img_float, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Saturation adjustment (multiply S channel)
    saturation_factor = np.random.uniform(0.7, 1.3)
    img_hsv[mask, 1] *= saturation_factor
    
    # Contrast adjustment (modify V channel)
    contrast_factor = np.random.uniform(0.8, 1.2)
    mean_v = np.mean(img_hsv[mask, 2])
    img_hsv[mask, 2] = mean_v + contrast_factor * (img_hsv[mask, 2] - mean_v)
    
    # Clip values and convert back to BGR
    img_hsv = np.clip(img_hsv, 0, 255)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
    result = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Augment texture images')
    parser.add_argument('input_file', type=str, help='Input texture file (e.g., amazon_texture_001.png)')
    parser.add_argument('--num_textures', type=int, default=10, help='Number of augmented textures to generate')
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        return
    
    # Read the original image
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    # Extract base name (remove leading dot if present)
    stem = input_path.stem.lstrip('.')
    extension = input_path.suffix
    output_dir = input_path.parent
    
    # Generate augmented textures
    for i in range(args.num_textures):
        augmented = augment_texture(image, seed=i)
        output_name = f"{stem}_{i:03d}{extension}"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), augmented)
        print(f"Generated: {output_path}")
    
    print(f"Successfully generated {args.num_textures} augmented textures")


if __name__ == "__main__":
    main()