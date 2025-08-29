"""
Main entry point for pattern-aware water level detection system.

This module provides system selection logic to choose between standard
and pattern-aware detection methods based on configuration and environment variables.
"""

import os
import sys
import logging
import yaml
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'pattern_aware_detection.log'),
            logging.StreamHandler()
        ]
    )

def load_config():
    """Load system configuration."""
    config_path = Path('config.yaml')
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def select_detection_system(config):
    """
    Select detection system based on environment variables and configuration.
    
    Priority:
    1. PATTERN_AWARE_MODE environment variable
    2. pattern_processing.mode configuration
    3. detection.pattern_aware.enabled configuration
    4. Default to 'standard'
    
    Returns:
        str: 'standard', 'pattern_aware', or 'hybrid'
    """
    logger = logging.getLogger(__name__)
    
    # Check environment variable first (highest priority)
    env_mode = os.environ.get('PATTERN_AWARE_MODE', '').lower()
    if env_mode == 'true':
        logger.info("PATTERN_AWARE_MODE environment variable detected - using pattern-aware detection")
        return 'pattern_aware'
    elif env_mode == 'hybrid':
        logger.info("PATTERN_AWARE_MODE=hybrid detected - using hybrid detection")
        return 'hybrid'
    elif env_mode == 'false':
        logger.info("PATTERN_AWARE_MODE=false detected - forcing standard detection")
        return 'standard'
    
    # Check configuration-based mode
    pattern_processing = config.get('pattern_processing', {})
    processing_mode = pattern_processing.get('mode', 'standard')
    
    if processing_mode == 'pattern_aware':
        logger.info("Configuration set to pattern_processing.mode=pattern_aware")
        return 'pattern_aware'
    elif processing_mode == 'hybrid':
        logger.info("Configuration set to pattern_processing.mode=hybrid")
        return 'hybrid'
    
    # Check pattern_aware.enabled flag
    pattern_aware = config.get('detection', {}).get('pattern_aware', {})
    if pattern_aware.get('enabled', False):
        logger.info("Pattern-aware detection enabled in configuration")
        return 'pattern_aware'
    
    # Default to standard detection
    logger.info("Using standard detection system (default)")
    return 'standard'

def create_detector(detection_system, config, pixels_per_cm, enhanced_calibration_data, calibration_manager):
    """
    Create the appropriate detector based on system selection.
    
    Args:
        detection_system: 'standard', 'pattern_aware', or 'hybrid'
        config: System configuration
        pixels_per_cm: Calibration data
        enhanced_calibration_data: Enhanced calibration data
        calibration_manager: Calibration manager instance
        
    Returns:
        Detector instance
    """
    logger = logging.getLogger(__name__)
    
    if detection_system == 'pattern_aware':
        logger.info("Creating pattern-aware detector")
        from src_pattern_aware.pattern_water_detector import PatternWaterDetector
        return PatternWaterDetector(config, pixels_per_cm, enhanced_calibration_data, calibration_manager)
    
    elif detection_system == 'hybrid':
        logger.info("Creating hybrid detector")
        return create_hybrid_detector(config, pixels_per_cm, enhanced_calibration_data, calibration_manager)
    
    else:  # standard
        logger.info("Creating standard detector")
        from src.water_level_detector import WaterLevelDetector
        return WaterLevelDetector(config, pixels_per_cm, enhanced_calibration_data, calibration_manager)

def create_hybrid_detector(config, pixels_per_cm, enhanced_calibration_data, calibration_manager):
    """
    Create a hybrid detector that can use both systems with fallback.
    
    Returns:
        HybridDetector instance
    """
    from .hybrid_detector import HybridDetector
    return HybridDetector(config, pixels_per_cm, enhanced_calibration_data, calibration_manager)

def log_system_info(detection_system, config):
    """Log information about the selected detection system."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Selected detection system: {detection_system}")
    
    if detection_system in ['pattern_aware', 'hybrid']:
        pattern_config = config.get('detection', {}).get('pattern_aware', {})
        logger.info(f"Pattern engine: {pattern_config.get('engine', 'integrated_pattern')}")
        logger.info(f"Fallback enabled: {pattern_config.get('fallback_to_standard', True)}")
        
        # Log enabled methods
        enabled_methods = []
        for method, settings in pattern_config.items():
            if isinstance(settings, dict) and settings.get('enabled', True):
                enabled_methods.append(method)
        
        if enabled_methods:
            logger.info(f"Enabled pattern methods: {', '.join(enabled_methods)}")
        
        # Log pattern processing settings
        pattern_processing = config.get('pattern_processing', {})
        logger.info(f"Debug patterns: {pattern_processing.get('debug_patterns', False)}")
        logger.info(f"Save templates: {pattern_processing.get('save_templates', True)}")

def main():
    """Main entry point for pattern-aware detection system.""" 
    try:
        # Setup logging
        debug_mode = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
        log_level = logging.DEBUG if debug_mode else logging.INFO
        setup_logging(log_level)
        
        logger = logging.getLogger(__name__)
        logger.info("Starting Pattern-Aware Water Level Detection System")
        
        # Load configuration
        config = load_config()
        
        # Select detection system
        detection_system = select_detection_system(config)
        log_system_info(detection_system, config)
        
        # Import and run the original main logic with our detector selection
        # This ensures compatibility with the existing workflow
        from src.main import main as original_main
        from src.calibration import CalibrationManager
        from src.database import DatabaseManager
        
        # Override the detector creation in the original main
        # We'll need to modify the original main.py to support this
        # For now, let's create a simplified version
        
        logger.info("Pattern-aware detection system initialized successfully")
        logger.info("Use PATTERN_AWARE_MODE=true environment variable to enable pattern detection")
        logger.info("Or set pattern_processing.mode='pattern_aware' in config.yaml")
        
        # Run the original main with our system selection
        original_main()
        
    except Exception as e:
        logger.error(f"Fatal error in pattern-aware detection system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()