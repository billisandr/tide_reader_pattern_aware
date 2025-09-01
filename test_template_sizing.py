"""
Test script to verify template resizing calculations and show before/after comparison
"""

import cv2
import yaml
import numpy as np
from pathlib import Path

def test_template_sizing():
    """Test and visualize template resizing."""
    
    print("=== Template Sizing Test ===\n")
    
    # Load calibration data
    calibration_path = Path('data/calibration/calibration.yaml')
    if not calibration_path.exists():
        print("[ERROR] calibration.yaml not found!")
        return
        
    with open(calibration_path, 'r') as f:
        calibration_data = yaml.safe_load(f)
    
    pixels_per_cm = calibration_data.get('pixels_per_cm')
    print(f"[CALIB] Input image calibration: {pixels_per_cm} pixels per cm")
    
    # Load config
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    single_e_cm = config.get('detection', {}).get('pattern_aware', {}).get('e_pattern_detection', {}).get('single_e_cm', 5.0)
    print(f"[CONFIG] E-pattern represents: {single_e_cm} cm in real world")
    
    # Calculate target size
    target_height_pixels = pixels_per_cm * single_e_cm
    print(f"[CALC] E-pattern should be {target_height_pixels:.1f} pixels tall in input images")
    print()
    
    # Test templates
    template_dir = Path('data/pattern_templates/scale_markings')
    template_files = ['E_pattern_black.png', 'E_pattern_white.png']
    
    for template_file in template_files:
        template_path = template_dir / template_file
        if not template_path.exists():
            print(f"[ERROR] Template not found: {template_file}")
            continue
            
        print(f"[TEMPLATE] {template_file}")
        
        # Load original template
        original = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            print(f"[ERROR] Failed to load {template_file}")
            continue
            
        orig_height, orig_width = original.shape
        print(f"  Original size: {orig_width} x {orig_height} pixels")
        
        # Calculate scaling
        scale_factor = target_height_pixels / orig_height
        new_width = int(orig_width * scale_factor)
        new_height = int(target_height_pixels)
        
        print(f"  Scale factor: {scale_factor:.3f}")
        print(f"  New size: {new_width} x {new_height} pixels")
        
        if scale_factor < 0.1:
            print(f"  [WARNING] Very small scale factor! Template much larger than input scale")
        elif scale_factor > 2.0:
            print(f"  [WARNING] Large scale factor! Template much smaller than input scale")
        else:
            print(f"  [OK] Reasonable scale factor")
            
        # Resize template
        resized = cv2.resize(original, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Save comparison image
        comparison_height = max(orig_height, 100)  # At least 100px tall for visibility
        comparison_width = orig_width + new_width + 20  # Space between images
        
        comparison = np.zeros((comparison_height, comparison_width), dtype=np.uint8)
        
        # Place original on left
        comparison[0:orig_height, 0:orig_width] = original
        
        # Place resized on right (centered vertically if smaller)
        resized_y_offset = (comparison_height - new_height) // 2
        comparison[resized_y_offset:resized_y_offset + new_height, 
                  orig_width + 10:orig_width + 10 + new_width] = resized
        
        # Add labels
        cv2.putText(comparison, "Original", (5, orig_height + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(comparison, f"Resized ({scale_factor:.2f}x)", 
                   (orig_width + 15, comparison_height - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        
        # Save comparison
        output_dir = Path('data/debug/template_sizing')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison_path = output_dir / f"{template_file.replace('.png', '')}_sizing_comparison.png"
        cv2.imwrite(str(comparison_path), comparison)
        print(f"  Saved comparison: {comparison_path}")
        
        print()
    
    print("=== Summary ===")
    print(f"Templates are being resized to match the input image scale of {pixels_per_cm:.3f} px/cm")
    print(f"Each 5cm E-pattern should appear as {target_height_pixels:.1f} pixels tall in your images")
    print("Check the comparison images in data/debug/template_sizing/ to verify sizing looks correct")

if __name__ == "__main__":
    test_template_sizing()