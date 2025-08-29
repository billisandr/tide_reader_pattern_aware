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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

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

def run_pattern_aware_main(detection_system, config):
    """
    Run the main processing loop with pattern-aware detection system.
    This is based on the original main.py but uses the selected detector.
    """
    import time
    import os
    from pathlib import Path
    
    # Import original main components
    from src.calibration import CalibrationManager
    from src.database import DatabaseManager
    from src.utils import get_unprocessed_images
    
    logger = logging.getLogger(__name__)
    
    # Initialize directory paths (same as original main)
    input_dir = Path(os.environ.get('INPUT_DIR', 'data/input'))
    if os.environ.get('USE_GUI_SELECTOR', 'true').lower() == 'true':
        from src.main import get_directory_with_gui
        logger.info("Opening GUI directory selector...")
        selected_dir = get_directory_with_gui()
        if selected_dir:
            logger.info(f"GUI returned directory: {repr(selected_dir)}")
            input_dir = Path(selected_dir)
        else:
            logger.error("No directory selected via GUI")
            logger.info(f"Using default directory: {input_dir}")
        
    output_dir = Path(os.environ.get('OUTPUT_DIR', 'data/output'))
    processed_dir = Path(os.environ.get('PROCESSED_DIR', 'data/processed'))
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using GUI-selected directory: {input_dir}")
    logger.info(f"Final input directory: {input_dir}")
    logger.info(f"Directory exists: {input_dir.exists()}")
    
    # Initialize calibration and database (same as original)
    calibration_manager = CalibrationManager(config)
    
    # Get calibration data
    pixels_per_cm = calibration_manager.get_pixels_per_cm()
    enhanced_calibration_data = calibration_manager.get_enhanced_calibration_data()
    
    logger.info(f"Using calibration: {pixels_per_cm:.2f} pixels/cm")
    logger.info(f"Calibration method: {enhanced_calibration_data['method'] if enhanced_calibration_data else 'standard'}")
    
    # **KEY DIFFERENCE**: Create the appropriate detector based on system selection
    detector = create_detector(detection_system, config, pixels_per_cm, enhanced_calibration_data, calibration_manager)
    logger.info(f"Created detector type: {type(detector).__name__}")
    
    # Initialize database
    db_path = output_dir / 'measurements.db'
    logger.info(f"Database path: {db_path}")
    logger.info(f"Database file exists: {db_path.exists()}")
    
    db_manager = DatabaseManager(str(db_path))
    
    # Processing configuration
    process_interval = int(os.environ.get('PROCESS_INTERVAL', 60))
    export_enabled = any([
        config['output'].get('csv_export', True),
        config['output'].get('json_export', True),
        config['output'].get('database', True)
    ])
    
    logger.info(f"Starting main processing loop (interval: {process_interval}s)")
    logger.info(f"Export formats enabled: CSV={config['output'].get('csv_export', True)}, "
                f"JSON={config['output'].get('json_export', True)}, "
                f"DB={config['output'].get('database', True)}")
    
    # Main processing loop (same as original but with our detector)
    while True:
        try:
            new_images = get_unprocessed_images(input_dir, processed_dir, db_manager)
            
            if new_images:
                logger.info(f"Found {len(new_images)} new images to process")
                
                for image_path in new_images:
                    try:
                        logger.debug(f"Processing {image_path.name}")
                        
                        # **KEY DIFFERENCE**: Use our selected detector (pattern-aware or standard)
                        result = detector.process_image(str(image_path))
                        
                        if result:
                            # Store in database
                            db_manager.store_measurement(
                                timestamp=result.get('timestamp'),
                                water_level=result['water_level_cm'],
                                scale_above_water=result.get('scale_above_water_cm'),
                                image_path=str(image_path),
                                confidence=result.get('confidence', 0.0)
                            )
                            
                            # Move to processed directory
                            processed_path = processed_dir / image_path.name
                            image_path.rename(processed_path)
                            
                            logger.info(f"Processed {image_path.name}: "
                                      f"Water level = {result['water_level_cm']:.1f}cm "
                                      f"(method: {result.get('detection_method', 'unknown')}, "
                                      f"system: {detection_system})")
                        else:
                            logger.warning(f"Failed to process {image_path.name}")
                            
                    except Exception as e:
                        logger.error(f"Error processing {image_path}: {e}")
                        continue
                
                # Export data in configured formats after processing batch
                if export_enabled:
                    try:
                        db_manager.export_all_formats(output_dir, config)
                        logger.debug("Exported measurements in configured formats")
                    except Exception as e:
                        logger.error(f"Error exporting data: {e}")
            else:
                logger.debug("No new images to process")
            
            # Wait before next iteration
            time.sleep(process_interval)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            time.sleep(process_interval)

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
        
        logger.info("Pattern-aware detection system initialized successfully")
        logger.info("Use PATTERN_AWARE_MODE=true environment variable to enable pattern detection")
        logger.info("Or set pattern_processing.mode='pattern_aware' in config.yaml")
        
        # Run pattern-aware main loop with proper detector selection
        run_pattern_aware_main(detection_system, config)
        
    except Exception as e:
        logger.error(f"Fatal error in pattern-aware detection system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()