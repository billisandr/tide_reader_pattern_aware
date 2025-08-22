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
    setup_logging()
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
        file_count = len(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.PNG")))
        logger.info(f"Found {file_count} image files in directory")
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    db_path = os.environ.get('DB_PATH', os.path.join(os.getcwd(), 'data', 'measurements.db'))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db_manager = DatabaseManager(db_path)
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
    
    pixels_per_cm = calibration_manager.get_pixels_per_cm()
    logger.info(f"Using calibration: {pixels_per_cm:.2f} pixels/cm")
    
    # Initialize detector
    detector = WaterLevelDetector(config, pixels_per_cm)
    
    # Processing loop
    process_interval = int(os.environ.get('PROCESS_INTERVAL', 60))
    #input_dir = Path(os.path.join(os.getcwd(), 'data', 'input'))
    processed_dir = Path(os.path.join(os.getcwd(), 'data', 'processed'))
    
    logger.info(f"Starting main processing loop (interval: {process_interval}s)")
    
    while True:
        try:
            # Get unprocessed images
            images = get_unprocessed_images(input_dir, processed_dir, db_manager)
            
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