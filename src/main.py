#!/usr/bin/env python3
"""
Main application entry point for water level measurement system.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import yaml
import tkinter as tk
from dotenv import load_dotenv

from water_level_detector import WaterLevelDetector
from calibration import CalibrationManager
from database import DatabaseManager
from utils import setup_logging, get_unprocessed_images
from tkinter import filedialog

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_directory_with_gui():
    """Show directory selection dialog."""      
    root = tk.Tk()     
    root.withdraw()  #Hide main window
    directory = filedialog.askdirectory(        
        title="Select input directory for photos"
    )
    root.destroy()     
    return directory 

def main():
    """Main application loop."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Setup logging
    setup_logging(logging.DEBUG)  # Enable debug logging
    logger = logging.getLogger(__name__)
    
    # Ask user to specify the input photos directory
    logger.info(f"USE_GUI_SELECTOR environment variable: {os.environ.get('USE_GUI_SELECTOR', 'not set')}")
    
    if os.environ.get('USE_GUI_SELECTOR', 'false').lower() == 'true':
          logger.info("Opening GUI directory selector...")
          input_dir_str = get_directory_with_gui()
          logger.info(f"GUI returned directory: '{input_dir_str}'")
          
          if not input_dir_str:
              logger.error("No directory selected via GUI")
              sys.exit(1)
          input_dir = Path(input_dir_str)
          logger.info(f"Using GUI-selected directory: {input_dir}")
    else:
          input_dir = Path(os.path.join(os.getcwd(), 'data', 'input'))
          logger.info(f"Using default directory: {input_dir}")
    
    logger.info(f"Final input directory: {input_dir.absolute()}")
    logger.info(f"Directory exists: {input_dir.exists()}")
    
    if input_dir.exists():
        # Use set to avoid counting same files twice (case insensitive on Windows)
        image_files = set()
        for pattern in ["*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"]:
            image_files.update(input_dir.glob(pattern))
        logger.info(f"Found {len(image_files)} unique image files in directory")
        logger.debug(f"Image files: {[f.name for f in image_files]}")
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    db_path = Path(os.environ.get('DB_PATH', 'data/output/measurements.db'))
    db_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure DB directory exists
    
    logger.info(f"Database path: {db_path}")
    logger.info(f"Database file exists: {db_path.exists()}")
    db_manager = DatabaseManager(str(db_path))
    calibration_manager = CalibrationManager(config)
    
    # Check if calibration mode
    if os.environ.get('CALIBRATION_MODE', 'false').lower() == 'true':
        logger.info("Running in calibration mode...")
        calibration_manager.run_calibration()
        sys.exit(0)
    
    # Load calibration data
    if not calibration_manager.is_calibrated():
        logger.error("System not calibrated. Run with CALIBRATION_MODE=true first.")
        sys.exit(1)
    
    # Load enhanced calibration data
    enhanced_calibration_data = calibration_manager.get_enhanced_calibration_data()
    pixels_per_cm = enhanced_calibration_data['pixels_per_cm']
    
    logger.info(f"Using calibration: {pixels_per_cm:.2f} pixels/cm")
    logger.info(f"Calibration method: {enhanced_calibration_data['method']}")
    
    # Initialize detector with enhanced calibration data
    detector = WaterLevelDetector(config, pixels_per_cm, enhanced_calibration_data)
    
    # Processing loop  
    process_interval = int(os.environ.get('PROCESS_INTERVAL', 60))
    #input_dir = Path('data/input')
    processed_dir = Path('data/processed')
    output_dir = Path('data/output')
    
    # Ensure directories exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if any export formats are enabled
    export_enabled = (config.get('output', {}).get('csv_export', False) or 
                     config.get('output', {}).get('json_export', False) or 
                     config.get('output', {}).get('database', False))
    
    logger.info(f"Starting main processing loop (interval: {process_interval}s)")
    logger.info(f"Export formats enabled: CSV={config.get('output', {}).get('csv_export', False)}, "
               f"JSON={config.get('output', {}).get('json_export', False)}, "
               f"DB={config.get('output', {}).get('database', False)}")
    
    while True:
        try:
            logger.debug("Checking for unprocessed images...")
            # Get unprocessed images
            images = get_unprocessed_images(input_dir, processed_dir, db_manager)
            logger.debug(f"get_unprocessed_images returned {len(images) if images else 0} images")
            
            if images:
                logger.info(f"Found {len(images)} new images to process")
                
                for image_path in images:
                    try:
                        # Process image
                        result = detector.process_image(str(image_path))
                        
                        if result:
                            # Store result
                            db_manager.store_measurement(
                                timestamp=result['timestamp'],
                                water_level=result['water_level_cm'],
                                scale_above_water=result['scale_above_water_cm'],
                                image_path=str(image_path),
                                confidence=result.get('confidence', 0.0)
                            )
                            
                            # Move to processed directory
                            processed_path = processed_dir / image_path.name
                            image_path.rename(processed_path)
                            
                            logger.info(f"Processed {image_path.name}: "
                                      f"Water level = {result['water_level_cm']:.1f}cm")
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

if __name__ == "__main__":
    main()