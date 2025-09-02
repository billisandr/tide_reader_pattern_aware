"""
Debug script for E-pattern matching to see confidence levels and help tune thresholds.
Uses scale-invariant template matching approach (multiple scales tested).
"""

import cv2
import yaml
import numpy as np
from pathlib import Path
from src_pattern_aware.detection_methods.e_pattern_detector import EPatternDetector

def debug_template_matching(image_path):
    """Debug template matching confidence levels."""
    
    # Load configuration
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load calibration data
    calibration_path = Path('data/calibration/calibration.yaml')
    calibration_data = None
    if calibration_path.exists():
        with open(calibration_path, 'r') as f:
            calibration_data = yaml.safe_load(f)
    
    # Initialize detector
    detector = EPatternDetector(config, calibration_data)
    
    # Load and process image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Debugging scale-invariant template matching on: {image_path}")
    print(f"Image size: {image.shape}")
    print(f"Current match threshold: {detector.match_threshold}")
    print(f"Template variants loaded: {len(detector.templates)} (includes multiple scales and orientations)")
    print(f"Scale factors tested: 0.3x to 2.0x (11 different scales)")
    print(f"Pattern types: E-pattern black/white with normal and flipped orientations")
    print()
    
    # Convert to grayscale for template matching
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    # Test each template variant on the entire image (scale-invariant approach)
    # Note: Templates include multiple scales (0.3x to 2.0x) and orientations (normal + flipped)
    for template_name, template_data in detector.templates.items():
        template = template_data['image']
        print(f"Testing template variant: {template_name}")
        print(f"Template size: {template.shape}")
        
        # Extract scale information from template name if available
        if 'scale_' in template_name:
            print(f"  Scale variant: {template_name.split('scale_')[1] if '_scale_' in template_name else 'base'}")
        if 'flipped' in template_name:
            print(f"  Orientation: 180-degree flipped")
        else:
            print(f"  Orientation: normal")
        
        if gray_image.shape[0] < template.shape[0] or gray_image.shape[1] < template.shape[1]:
            print(f"  Image too small for template, skipping")
            continue
        
        try:
            # Perform template matching on entire image
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            
            # Get statistics
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            print(f"  Max confidence: {max_val:.4f} at position {max_loc}")
            print(f"  Min confidence: {min_val:.4f} at position {min_loc}")
            
            # Count matches at different thresholds
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            for thresh in thresholds:
                count = np.sum(result >= thresh)
                print(f"  Matches at threshold {thresh}: {count}")
            
            # Show top 5 matches
            result_flat = result.flatten()
            top_indices = np.argsort(result_flat)[-5:][::-1]  # Top 5 in descending order
            
            print(f"  Top 5 match locations:")
            for i, idx in enumerate(top_indices):
                y, x = np.unravel_index(idx, result.shape)
                confidence = result[y, x]
                print(f"    {i+1}. Position ({x}, {y}), confidence: {confidence:.4f}")
            
        except Exception as e:
            print(f"  Error matching template: {e}")
        
        print()
    
    print("Debug complete. Consider adjusting match_threshold based on these results.")
    print("\nNOTE: This E-pattern detector uses scale-invariant template matching:")
    print("- Templates are tested at multiple scales (0.3x to 2.0x)")
    print("- No template resizing to match calibration ratios (previous approach was limiting)")
    print("- Templates find patterns at their natural size in input images")
    print("- 5cm measurement value used only for final calculation, not template sizing")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python debug_e_pattern_matching.py <image_path>")
        print("Example: python debug_e_pattern_matching.py data/test_images/scale_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    debug_template_matching(image_path)