#!/usr/bin/env python3
"""
Test script to verify gradient logging functionality.
Processes a single image to test the gradient analysis text file generation.
"""

import os
import sys
import logging
import yaml
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'src_pattern_aware'))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config():
    """Load system configuration."""
    config_path = Path('config.yaml')
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_gradient_logging():
    """Test gradient logging by processing a single image."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Loaded configuration")
        
        # Initialize calibration
        from src.calibration import CalibrationManager
        calibration_manager = CalibrationManager(config)
        pixels_per_cm = calibration_manager.get_pixels_per_cm()
        enhanced_calibration_data = calibration_manager.get_enhanced_calibration_data()
        
        logger.info(f"Calibration: {pixels_per_cm:.2f} pixels/cm")
        
        # Create pattern-aware detector
        from src_pattern_aware.pattern_water_detector import PatternWaterDetector
        detector = PatternWaterDetector(config, pixels_per_cm, enhanced_calibration_data, calibration_manager)
        logger.info("Created PatternWaterDetector")
        
        # Find a test image
        input_dir = Path('data/input')
        test_images = list(input_dir.glob('*.JPG'))[:1]  # Just process one image
        
        if not test_images:
            logger.error("No test images found in data/input")
            return False
        
        test_image = test_images[0]
        logger.info(f"Processing test image: {test_image.name}")
        
        # Process the image
        result = detector.process_image(str(test_image))
        
        if result:
            logger.info(f"Processing successful!")
            logger.info(f"Water level: {result.get('water_level_cm', 'N/A')} cm")
            logger.info(f"Detection method: {result.get('detection_method', 'unknown')}")
            
            # Check for gradient analysis files
            debug_dirs = list(Path('data/debug').glob('pattern_aware_debug_session_*'))
            if debug_dirs:
                latest_debug = max(debug_dirs, key=lambda p: p.stat().st_mtime)
                gradient_dir = latest_debug / 'waterline_gradient_analysis'
                
                if gradient_dir.exists():
                    txt_files = list(gradient_dir.glob('*.txt'))
                    if txt_files:
                        logger.info(f"SUCCESS: Found {len(txt_files)} gradient analysis text files:")
                        for txt_file in txt_files:
                            logger.info(f"  - {txt_file}")
                        return True
                    else:
                        logger.warning("Gradient analysis directory exists but no .txt files found")
                else:
                    logger.warning("Gradient analysis directory not found")
            else:
                logger.warning("No debug session directories found")
        else:
            logger.error("Image processing failed")
        
        return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting gradient logging test...")
    success = test_gradient_logging()
    
    if success:
        logger.info("✓ Test completed successfully!")
    else:
        logger.info("✗ Test failed or incomplete")
        sys.exit(1)