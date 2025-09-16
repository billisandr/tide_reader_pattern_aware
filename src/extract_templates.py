#!/usr/bin/env python3
"""
Template Extraction Script

Extracts scale marking templates from calibration images for pattern-aware detection.
"""

import sys
import yaml
import logging
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import from current directory
from pattern_water_detector import PatternWaterDetector

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config():
    """Load system configuration."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_calibration_data():
    """Load enhanced calibration data if available."""
    calib_path = Path(__file__).parent.parent / 'data' / 'calibration' / 'calibration.yaml'
    if calib_path.exists():
        with open(calib_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def main():
    """Main template extraction function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Calibration image path
    calibration_image = project_root / 'data' / 'calibration' / 'calibration_image.jpg'
    
    # Check if calibration image exists
    if not calibration_image.exists():
        logger.error(f"Calibration image not found: {calibration_image}")
        logger.info("Available images in data/calibration/:")
        calib_dir = project_root / 'data' / 'calibration'
        if calib_dir.exists():
            for img_file in calib_dir.glob('*.jpg'):
                logger.info(f"  - {img_file.name}")
        sys.exit(1)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        
        # Load calibration data
        logger.info("Loading calibration data...")
        calib_data = load_calibration_data()
        
        # Get pixels per cm from calibration
        pixels_per_cm = None
        if calib_data and 'pixels_per_cm' in calib_data:
            pixels_per_cm = calib_data['pixels_per_cm']
            logger.info(f"Using calibrated pixels_per_cm: {pixels_per_cm}")
        else:
            logger.warning("No calibrated pixels_per_cm found, using default")
            pixels_per_cm = 2.0  # Default fallback
        
        # Initialize pattern detector
        logger.info("Initializing pattern detector...")
        detector = PatternWaterDetector(config, pixels_per_cm, calib_data)
        
        # Extract and save templates
        logger.info(f"Extracting templates from: {calibration_image}")
        template_count = detector.extract_and_save_templates(str(calibration_image))
        
        if template_count > 0:
            logger.info(f"Successfully extracted {template_count} templates!")
            template_dir = project_root / 'data' / 'pattern_templates' / 'scale_markings'
            logger.info(f"Templates saved to: {template_dir}")
        else:
            logger.warning("No templates were extracted")
            
    except Exception as e:
        logger.error(f"Template extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()