"""
Debug script to analyze E-pattern template scale vs image calibration mismatch
"""

import cv2
import yaml
import numpy as np
from pathlib import Path

def analyze_template_scale():
    """Analyze the scale mismatch between templates and calibration."""
    
    print("=== E-Pattern Template Scale Analysis ===\n")
    
    # Load calibration data
    calibration_path = Path('data/calibration/calibration.yaml')
    if calibration_path.exists():
        with open(calibration_path, 'r') as f:
            calibration_data = yaml.safe_load(f)
        calibration_px_per_cm = calibration_data.get('pixels_per_cm', 'N/A')
        print(f"[CALIB] Calibration pixels per cm: {calibration_px_per_cm}")
    else:
        print("[ERROR] Calibration file not found!")
        return
    
    # Load config
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    single_e_cm = config.get('detection', {}).get('pattern_aware', {}).get('e_pattern_detection', {}).get('single_e_cm', 5.0)
    print(f"[CONFIG] E-pattern supposed CM value: {single_e_cm} cm\n")
    
    # Analyze templates
    template_dir = Path('data/pattern_templates/scale_markings')
    template_files = ['E_pattern_black.png', 'E_pattern_white.png']
    
    print("=== Template Analysis ===")
    for template_file in template_files:
        template_path = template_dir / template_file
        if template_path.exists():
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                height, width = template.shape
                
                # Calculate what pixel/cm this template implies
                template_px_per_cm = height / single_e_cm
                
                # Calculate scale difference
                scale_ratio = template_px_per_cm / calibration_px_per_cm if calibration_px_per_cm != 'N/A' else 'N/A'
                
                print(f"\n[TEMPLATE] {template_file}:")
                print(f"   Template dimensions: {width} x {height} pixels")
                print(f"   Template height: {height} pixels")
                print(f"   Supposed to represent: {single_e_cm} cm")
                print(f"   Template implies: {template_px_per_cm:.2f} pixels/cm")
                print(f"   Calibration says: {calibration_px_per_cm} pixels/cm")
                
                if scale_ratio != 'N/A':
                    print(f"   [WARNING] Scale mismatch ratio: {scale_ratio:.1f}x")
                    print(f"   Template is {scale_ratio:.1f}x larger than expected!")
                    
                    # Calculate what the template height should be
                    correct_height = calibration_px_per_cm * single_e_cm
                    print(f"   [CORRECT] Template height should be: {correct_height:.1f} pixels")
                    
                    # Calculate what CM value this template actually represents
                    actual_cm = height / calibration_px_per_cm
                    print(f"   [ACTUAL] This template represents: {actual_cm:.1f} cm at current calibration")
            else:
                print(f"[ERROR] Could not load {template_file}")
        else:
            print(f"[ERROR] Template file not found: {template_file}")
    
    print(f"\n=== Solutions ===")
    print(f"1. [OPTION] Resize templates to match calibration scale")
    print(f"2. [OPTION] Update single_e_cm value to match actual template size")
    print(f"3. [OPTION] Create new templates at correct scale")
    print(f"4. [OPTION] Recalibrate with different scale reference")

def suggest_template_resize():
    """Suggest how to resize templates."""
    
    calibration_path = Path('data/calibration/calibration.yaml')
    if not calibration_path.exists():
        return
        
    with open(calibration_path, 'r') as f:
        calibration_data = yaml.safe_load(f)
    calibration_px_per_cm = calibration_data.get('pixels_per_cm')
    
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    single_e_cm = config.get('detection', {}).get('pattern_aware', {}).get('e_pattern_detection', {}).get('single_e_cm', 5.0)
    
    # Calculate target template height
    target_height = calibration_px_per_cm * single_e_cm
    
    print(f"\n=== Template Resize Recommendations ===")
    print(f"Target template height for {single_e_cm}cm: {target_height:.1f} pixels")
    print(f"Calibration reference: {calibration_px_per_cm} pixels/cm")
    
    template_dir = Path('data/pattern_templates/scale_markings')
    for template_file in ['E_pattern_black.png', 'E_pattern_white.png']:
        template_path = template_dir / template_file
        if template_path.exists():
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                current_height, current_width = template.shape
                scale_factor = target_height / current_height
                new_width = int(current_width * scale_factor)
                
                print(f"\n{template_file}:")
                print(f"  Current: {current_width} x {current_height}")
                print(f"  Should be: {new_width} x {int(target_height)}")
                print(f"  Scale factor: {scale_factor:.3f}")
                print(f"  OpenCV resize: cv2.resize(template, ({new_width}, {int(target_height)}))")

if __name__ == "__main__":
    analyze_template_scale()
    suggest_template_resize()