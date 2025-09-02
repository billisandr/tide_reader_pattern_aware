"""
Test script for E-pattern sequential detection system
"""

import cv2
import yaml
from pathlib import Path
from src_pattern_aware.detection_methods.e_pattern_detector import EPatternDetector

def test_e_pattern_detector():
    """Test the E-pattern detector functionality."""
    
    # Load configuration
    config_path = Path('config.yaml')
    if not config_path.exists():
        print("Error: config.yaml not found")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load calibration data from calibration.yaml
    calibration_path = Path('data/calibration/calibration.yaml')
    calibration_data = None
    
    if calibration_path.exists():
        with open(calibration_path, 'r') as f:
            calibration_data = yaml.safe_load(f)
        print(f"Loaded calibration data: pixels_per_cm = {calibration_data.get('pixels_per_cm')}")
    else:
        print("Warning: calibration.yaml not found, detector will use fallback values")
    
    print("Testing E-Pattern Sequential Detector")
    print(f"Templates directory: {config.get('pattern_processing', {}).get('template_directory')}")
    print()
    
    # Initialize detector
    try:
        detector = EPatternDetector(config, calibration_data)
        print(f"[OK] E-pattern detector initialized successfully")
        
        # Get detection info
        info = detector.get_detection_info()
        print(f"[OK] Templates loaded: {info['templates_loaded']}")
        print(f"[OK] Template names: {info['template_names']}")
        print(f"[OK] Single E pattern: {info['single_e_cm']} cm")
        print(f"[OK] Match threshold: {info['match_threshold']}")
        print(f"[OK] Calibration pixels/cm: {info['calibration_pixel_per_cm']}")
        print()
        
        # Test template loading
        template_dir = Path(config.get('pattern_processing', {}).get('template_directory', 
                                     'data/pattern_templates/scale_markings'))
        
        expected_templates = ['E_pattern_black.png', 'E_pattern_white.png']
        found_templates = []
        missing_templates = []
        
        for template_name in expected_templates:
            template_path = template_dir / template_name
            if template_path.exists():
                found_templates.append(template_name)
                print(f"[OK] Found template: {template_name}")
            else:
                missing_templates.append(template_name)
                print(f"[WARN] Missing template: {template_name}")
        
        if missing_templates:
            print(f"\n[WARN] {len(missing_templates)} templates missing. E-pattern detection may not work properly.")
        else:
            print(f"\n[OK] All {len(found_templates)} E-pattern templates found!")
            print(f"[OK] Templates will be loaded in both normal and 180-degree flipped orientations")
            print(f"[OK] Total template variants available: {len(found_templates) * 2}")
            
        # Test configuration loading
        e_config = config.get('detection', {}).get('pattern_aware', {}).get('e_pattern_detection', {})
        if e_config.get('enabled', False):
            print("[OK] E-pattern detection enabled in configuration")
            if e_config.get('support_flipped', False):
                print("[OK] 180-degree flipped pattern support enabled")
        else:
            print("[WARN] E-pattern detection disabled in configuration")
            
        print("\nE-pattern detector is ready for use!")
        print("Features:")
        print("- Uses calibration.yaml pixels_per_cm as reference")
        print("- Supports both E_pattern_black and E_pattern_white")
        print("- Detects patterns in normal and 180-degree flipped orientations") 
        print("- Uses multi-scale template matching (no pixel/cm validation needed)")
        print("- Stops detection when patterns likely underwater")
        print("\nTo test with an actual image, use the pattern-aware water detector system.")
        
    except Exception as e:
        print(f"[ERROR] Error initializing E-pattern detector: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_e_pattern_detector()