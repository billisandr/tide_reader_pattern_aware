"""
Water level detection module adapted from prateekralhan's approach.
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective, contours
import imutils
from datetime import datetime
import logging

class WaterLevelDetector:
    def __init__(self, config, pixels_per_cm):
        """Initialize the water level detector."""
        self.config = config
        self.pixels_per_cm = pixels_per_cm
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters
        self.edge_low = config['detection']['edge_threshold_low']
        self.edge_high = config['detection']['edge_threshold_high']
        self.blur_kernel = config['detection']['blur_kernel_size']
        
        # Scale parameters
        self.scale_height_cm = config['scale']['total_height']
        self.scale_region = config['scale']['expected_position']
    
    def detect_water_line(self, image):
        """
        Detect the water line in the image.
        Returns y-coordinate of water line.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Focus on scale region if defined
        if self.scale_region:
            roi = gray[
                self.scale_region['y_min']:self.scale_region['y_max'],
                self.scale_region['x_min']:self.scale_region['x_max']
            ]
        else:
            roi = gray
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(roi, (self.blur_kernel, self.blur_kernel), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, self.edge_low, self.edge_high)
        
        # Find horizontal lines (water surface tends to be horizontal)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Find the most prominent horizontal line
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:  # Nearly horizontal
                    horizontal_lines.append((y1 + y2) / 2)
            
            if horizontal_lines:
                # Return median y-coordinate of horizontal lines
                water_line_y = np.median(horizontal_lines)
                if self.scale_region:
                    water_line_y += self.scale_region['y_min']
                return int(water_line_y)
        
        # Fallback: detect using gradient changes
        return self.detect_water_line_gradient(gray)
    
    def detect_water_line_gradient(self, gray_image):
        """
        Fallback method using gradient changes.
        """
        # Calculate vertical gradient
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        
        # Sum gradients horizontally to find strong horizontal edges
        grad_sum = np.sum(np.abs(grad_y), axis=1)
        
        # Find peaks in gradient sum (potential water lines)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(grad_sum, height=np.mean(grad_sum))
        
        if len(peaks) > 0:
            # Return the most prominent peak in expected region
            if self.scale_region:
                valid_peaks = peaks[
                    (peaks >= self.scale_region['y_min']) & 
                    (peaks <= self.scale_region['y_max'])
                ]
                if len(valid_peaks) > 0:
                    return valid_peaks[len(valid_peaks)//2]
            return peaks[len(peaks)//2]
        
        return None
    
    def detect_scale_bounds(self, image):
        """
        Detect the top and bottom of the scale.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find scale
        edges = cv2.Canny(gray, self.edge_low, self.edge_high)
        
        # Find contours
        cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if not cnts:
            return None, None
        
        # Find the largest vertical contour (likely the scale)
        scale_contour = None
        max_height = 0
        
        for c in cnts:
            if cv2.contourArea(c) < self.config['detection']['min_contour_area']:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(c)
            
            # Check if in expected region and is vertical
            if self.scale_region:
                if not (self.scale_region['x_min'] <= x <= self.scale_region['x_max']):
                    continue
            
            if h > w * 2 and h > max_height:  # Vertical object
                max_height = h
                scale_contour = c
        
        if scale_contour is not None:
            x, y, w, h = cv2.boundingRect(scale_contour)
            return y, y + h  # top, bottom
        
        # Fallback to expected region
        if self.scale_region:
            return self.scale_region['y_min'], self.scale_region['y_max']
        
        return None, None
    
    def calculate_water_level(self, water_line_y, scale_top_y, scale_bottom_y):
        """
        Calculate the actual water level based on pixel measurements.
        """
        if water_line_y is None or scale_top_y is None or scale_bottom_y is None:
            return None
        
        # Calculate pixels above water
        pixels_above_water = water_line_y - scale_top_y
        
        # Convert to centimeters
        cm_above_water = pixels_above_water / self.pixels_per_cm
        
        # Calculate water level (scale_height - height_above_water)
        water_level = self.scale_height_cm - cm_above_water
        
        return {
            'water_level_cm': water_level,
            'scale_above_water_cm': cm_above_water,
            'pixels_above_water': pixels_above_water,
            'water_line_y': water_line_y,
            'scale_top_y': scale_top_y,
            'scale_bottom_y': scale_bottom_y
        }
    
    def process_image(self, image_path):
        """
        Main processing function for a single image.
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Resize if configured
            if self.config['processing']['resize_width']:
                height, width = image.shape[:2]
                new_width = self.config['processing']['resize_width']
                new_height = int(height * (new_width / width))
                image = cv2.resize(image, (new_width, new_height))
                
                # Adjust pixels_per_cm for resized image
                scale_factor = new_width / width
                adjusted_pixels_per_cm = self.pixels_per_cm * scale_factor
            else:
                adjusted_pixels_per_cm = self.pixels_per_cm
            
            # Detect water line
            water_line_y = self.detect_water_line(image)
            
            # Detect scale bounds
            scale_top_y, scale_bottom_y = self.detect_scale_bounds(image)
            
            # Calculate water level
            result = self.calculate_water_level(
                water_line_y, scale_top_y, scale_bottom_y
            )
            
            if result:
                # Add metadata
                result['timestamp'] = datetime.now()
                result['image_path'] = image_path
                result['confidence'] = self.estimate_confidence(result)
                
                # Save processed image if configured
                if self.config['processing']['save_processed_images']:
                    self.save_processed_image(image, result)
                
                return result
            else:
                self.logger.warning(f"Could not calculate water level for {image_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def estimate_confidence(self, result):
        """
        Estimate confidence of the measurement.
        """
        # Simple confidence based on expected values
        confidence = 1.0
        
        # Check if water level is within expected range
        if result['water_level_cm'] < 0 or result['water_level_cm'] > self.scale_height_cm:
            confidence *= 0.5
        
        # Check if scale was properly detected
        if result['scale_bottom_y'] - result['scale_top_y'] < 100:  # Too small
            confidence *= 0.7
        
        return confidence
    
    def save_processed_image(self, image, result):
        """
        Save image with annotations showing detected water line and measurements.
        """
        annotated = image.copy()
        
        # Draw water line
        if result['water_line_y']:
            cv2.line(annotated, 
                    (0, result['water_line_y']), 
                    (image.shape[1], result['water_line_y']),
                    (0, 255, 0), 2)
        
        # Draw scale bounds
        if self.scale_region:
            cv2.rectangle(annotated,
                         (self.scale_region['x_min'], self.scale_region['y_min']),
                         (self.scale_region['x_max'], self.scale_region['y_max']),
                         (255, 0, 0), 2)
        
        # Add text with measurements
        text = f"Water Level: {result['water_level_cm']:.1f}cm"
        cv2.putText(annotated, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save annotated image
        output_path = f"/app/data/processed/annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(output_path, annotated)