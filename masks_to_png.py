import os
import numpy as np
from PIL import Image
import argparse

def convert_masks_to_png(input_dir, output_dir):
    """
    Convert numpy mask files to PNG images with binary values.
    
    Args:
        input_dir (str): Directory containing .npy mask files
        output_dir (str): Directory to save converted PNG files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    npy_files.sort()
    
    print(f"Found {len(npy_files)} .npy files to process")
    
    for npy_file in npy_files:
        try:
            # Load the numpy mask
            mask_path = os.path.join(input_dir, npy_file)
            mask = np.load(mask_path)
            
            # Convert to binary mask: 0 for background, 1 for any foreground object
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Scale to 0-255 for better visibility in PNG
            binary_mask_scaled = binary_mask * 255
            
            # Convert to PIL Image and save as PNG
            img = Image.fromarray(binary_mask_scaled, mode='L')  # 'L' mode for grayscale
            
            # Create output filename
            png_filename = os.path.splitext(npy_file)[0].split('_')[1] + '.png'
            output_path = os.path.join(output_dir, png_filename)
            
            # Save the image
            img.save(output_path)
            
            print(f"Converted {npy_file} -> {png_filename}")
            print(f"  Original mask shape: {mask.shape}, unique values: {np.unique(mask)}")
            print(f"  Binary mask unique values: {np.unique(binary_mask_scaled)}")
            
        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}")
    
    print(f"\nConversion complete! All PNG files saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert numpy mask files to binary PNG images')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing .npy mask files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save converted PNG files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    convert_masks_to_png(args.input_dir, args.output_dir)

if __name__ == "__main__":
    # Example usage if running directly
    # You can modify these paths or use command line arguments
    input_directory = "./outputs/mask_data"
    output_directory = "./outputs/mask_png"
    
    print("Usage examples:")
    print(f"1. Direct execution with hardcoded paths:")
    print(f"   python masks_to_png.py")
    print(f"   (Uses: input='{input_directory}', output='{output_directory}')")
    print(f"")
    print(f"2. Command line arguments:")
    print(f"   python masks_to_png.py --input_dir /path/to/masks --output_dir /path/to/output")
    print(f"")
    
    # Uncomment the following lines to run with hardcoded paths
    # convert_masks_to_png(input_directory, output_directory)
    
    # Or use command line arguments
    main()