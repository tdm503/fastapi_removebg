import cv2
import numpy as np
import os
from glob import glob

def restore_original_colors(mask_folder, original_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    valid_extensions = ('.png', '.jpeg', '.jpg', '.bmp', '.tiff')
    
    mask_files = [f for f in glob(os.path.join(mask_folder, "*.*")) if f.lower().endswith(valid_extensions)]
    
    for mask_path in mask_files:
        try:
            filename_no_ext, _ = os.path.splitext(os.path.basename(mask_path))
            
            original_file = None
            for ext in valid_extensions:
                potential_file = os.path.join(original_folder, f"{filename_no_ext}{ext}")
                if os.path.exists(potential_file):
                    original_file = potential_file
                    break
            
            if original_file is None:
                print(f"can't find original image: {filename_no_ext}")
                continue

            original = cv2.imread(original_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if original is None:
                print(f" read file error (original file): {original_file}")
                continue
                
            if mask is None:
                print(f" read file error (mask file): {mask_path}")
                continue
            
            if mask.shape != original.shape[:2]:
                mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            restored = original.copy()
            restored[binary_mask == 0] = [0, 255, 0]  # Green color in BGR format

            output_path = os.path.join(output_folder, f"{filename_no_ext}.png")
            cv2.imwrite(output_path, restored)
                
            print(f"finish: {filename_no_ext}")
        
        except Exception as e:
            print(f"error {filename_no_ext}: {str(e)}")
