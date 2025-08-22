"""
Calibration module for establishing pixel-to-cm ratio.
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path

class CalibrationManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.calibration_file = Path('/app/data/calibration/calibration.yaml')
    
    def is_calibrated(self):
        """Check if system is calibrated."""
        return self.calibration_file.exists()
    
    def get_pixels_per_cm(self):
        """Load calibration data."""
        if not self.is_calibrated():
            return None
        
        with open(self.calibration_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        return calib_data['pixels_per_cm']
    
    def run_calibration(self):
        """
        Run interactive calibration process.
        """
        self.logger.info("Starting calibration process...")
        
        # Get calibration image
        calib_dir = Path('/app/data/calibration')
        images = list(calib_dir.glob('*.jpg')) + list(calib_dir.glob('*.png'))
        
        if not images:
            self.logger.error("No calibration images found in /app/data/calibration/")
            return False
        
        image_path = images[0]
        self.logger.info(f"Using calibration image: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error("Failed to load calibration image")
            return False
        
        # Method 1: Known scale height
        if self.config['scale']['total_height']:
            pixels_per_cm = self.calibrate_with_known_height(image)
        else:
            # Method 2: Interactive calibration
            pixels_per_cm = self.interactive_calibration(image)
        
        if pixels_per_cm:
            # Save calibration
            calib_data = {
                'pixels_per_cm': float(pixels_per_cm),
                'image_path': str(image_path),
                'scale_height_cm': self.config['scale']['total_height']
            }
            
            with open(self.calibration_file, 'w') as f:
                yaml.dump(calib_data, f)
            
            self.logger.info(f"Calibration saved: {pixels_per_cm:.2f} pixels/cm")
            return True
        
        return False
    
    def calibrate_with_known_height(self, image):
        """
        Calibrate using known scale height.
        """
        height_cm = self.config['scale']['total_height']
        
        # Detect scale in image
        scale_pixels = self.detect_scale_height_pixels(image)
        
        if scale_pixels:
            pixels_per_cm = scale_pixels / height_cm
            self.logger.info(f"Detected scale: {scale_pixels} pixels = {height_cm}cm")
            return pixels_per_cm
        
        self.logger.error("Could not detect scale in calibration image")
        return None
    
    def detect_scale_height_pixels(self, image):
        """
        Detect the scale and measure its height in pixels.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the tallest vertical object (likely the scale)
        max_height = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > w * 2 and h > max_height:  # Vertical object
                max_height = h
        
        return max_height if max_height > 100 else None
    
    def interactive_calibration(self, image):
        """
        Interactive calibration with user input.
        """
        self.logger.info("Interactive calibration: Click top and bottom of scale")
        
        points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Calibration', image)
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)
        cv2.imshow('Calibration', image)
        
        self.logger.info("Click on top of scale, then bottom of scale. Press 'q' when done.")
        
        while len(points) < 2:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        if len(points) >= 2:
            # Calculate pixel distance
            pixel_height = abs(points[1][1] - points[0][1])
            height_cm = float(input("Enter the actual height in cm: "))
            
            pixels_per_cm = pixel_height / height_cm
            return pixels_per_cm
        
        return None