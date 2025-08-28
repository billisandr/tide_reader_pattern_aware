"""
Water level detection module adapted from prateekralhan's approach.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from scipy.spatial import distance as dist
from imutils import perspective, contours
import imutils
from datetime import datetime
import logging
import time
from debug_visualizer import DebugVisualizer

class WaterLevelDetector:
    def __init__(self, config, pixels_per_cm, enhanced_calibration_data=None):
        """Initialize the water level detector."""
        self.config = config
        self.pixels_per_cm = pixels_per_cm
        self.enhanced_calibration_data = enhanced_calibration_data
        self.logger = logging.getLogger(__name__)
        
        # Log calibration method being used
        if enhanced_calibration_data and enhanced_calibration_data.get('method') == 'enhanced_interactive_waterline':
            self.logger.info(f"Using enhanced waterline-aware calibration (confidence: {enhanced_calibration_data.get('confidence', 'unknown')})")
            if enhanced_calibration_data.get('waterline_reference'):
                waterline_ref = enhanced_calibration_data['waterline_reference']
                self.logger.info(f"Waterline reference at Y={waterline_ref.get('y_average', 'unknown')}")
        else:
            self.logger.info("Using standard calibration method")
        
        # Detection parameters
        self.edge_low = config['detection']['edge_threshold_low']
        self.edge_high = config['detection']['edge_threshold_high']
        self.blur_kernel = config['detection']['blur_kernel_size']
        self.detection_method = config['detection'].get('method', 'edge')
        self.water_hsv_lower = np.array(config['detection'].get('water_hsv_lower', [100, 50, 50]))
        self.water_hsv_upper = np.array(config['detection'].get('water_hsv_upper', [130, 255, 255]))
        
        # Scale parameters
        self.scale_height_cm = config['scale']['total_height']
        self.scale_region = config['scale']['expected_position']
        
        # RGB color detection parameters
        self.color_detection_enabled = config['scale']['color_detection']['enabled']
        self.scale_colors = config['scale']['color_detection']['scale_colors']
        self.morphology_config = config['scale']['color_detection']['morphology']
        self.debug_color_masks = config['scale']['color_detection']['debug_color_masks']
        
        # Initialize debug visualizer with priority: DEBUG_MODE env var > config.debug.enabled > false
        config_debug = config.get('debug', {}).get('enabled', False)
        env_debug_set = 'DEBUG_MODE' in os.environ
        env_debug = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
        
        debug_enabled = env_debug if env_debug_set else config_debug
        
        # Log which debug setting is being used for clarity
        logger = logging.getLogger(__name__)
        if env_debug_set:
            logger.info(f"Debug mode: Environment DEBUG_MODE={env_debug} (overrides config)")
        else:
            logger.info(f"Debug mode: Config debug.enabled={config_debug}")
            
        self.debug_viz = DebugVisualizer(config, enabled=debug_enabled)
    
    def detect_water_line(self, image):
        """
        Detect the water line in the image using the configured detection method.
        Enhanced with waterline reference if available.
        Returns y-coordinate of water line.
        """
        # Log which detection method is being used
        self.logger.debug(f"Using detection method: {self.detection_method}")
        
        # Route to appropriate detection method with waterline enhancement
        if self.detection_method == 'color':
            water_y = self.detect_water_line_color(image)
        elif self.detection_method == 'gradient':
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            water_y = self.detect_water_line_gradient(gray)
        else:  # default to 'edge' method
            water_y = self.detect_water_line_edge(image)
        
        # Apply waterline reference enhancement if available
        if self.enhanced_calibration_data and self.enhanced_calibration_data.get('waterline_reference'):
            water_y = self.enhance_water_detection_with_reference(water_y, image)
        
        return water_y
    
    def enhance_water_detection_with_reference(self, detected_water_y, image):
        """
        Enhance water detection using waterline reference from calibration.
        """
        waterline_ref = self.enhanced_calibration_data['waterline_reference']
        scale_measurements = self.enhanced_calibration_data.get('scale_measurements')
        
        if not waterline_ref or not scale_measurements:
            self.logger.debug("Insufficient waterline reference data, using standard detection")
            return detected_water_y
        
        ref_y = waterline_ref.get('y_average')
        ref_water_level = scale_measurements.get('current_water_level_cm')
        
        if ref_y is None or ref_water_level is None:
            self.logger.debug("Missing waterline reference coordinates, using standard detection")
            return detected_water_y
        
        # Calculate expected water line position based on reference
        # If no detection was made, use reference as fallback
        if detected_water_y is None:
            self.logger.info(f"No water line detected, using calibration reference at Y={ref_y}")
            return ref_y
        
        # Validate detection against reference (within reasonable range)
        pixel_difference = abs(detected_water_y - ref_y)
        cm_difference = pixel_difference / self.pixels_per_cm
        
        # If detected water line is very far from reference, log a warning
        if cm_difference > 50:  # More than 50cm difference
            self.logger.warning(f"Detected water line (Y={detected_water_y}) differs significantly from reference (Y={ref_y}) by {cm_difference:.1f}cm")
            # Could implement fallback logic here if needed
        else:
            self.logger.debug(f"Water detection validated against reference: {cm_difference:.1f}cm difference")
        
        return detected_water_y
    
    def detect_water_line_edge(self, image):
        """
        Detect the water line using edge detection (original method).
        Returns y-coordinate of water line.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply color enhancement if enabled
        enhanced_edges = None
        if self.color_detection_enabled:
            color_mask, color_masks = self.enhance_scale_detection_rgb(image)
            if color_mask is not None:
                # Create color-enhanced edges by combining multiple approaches
                enhanced_edges = self.create_color_enhanced_edges(image, color_mask, color_masks)
        
        # Focus on scale region if defined
        if self.scale_region:
            roi = gray[
                self.scale_region['y_min']:self.scale_region['y_max'],
                self.scale_region['x_min']:self.scale_region['x_max']
            ]
            
            # Also apply to enhanced edges if available
            if enhanced_edges is not None:
                enhanced_roi = enhanced_edges[
                    self.scale_region['y_min']:self.scale_region['y_max'],
                    self.scale_region['x_min']:self.scale_region['x_max']
                ]
            else:
                enhanced_roi = None
            
            # Debug: Show scale region
            scale_rect_annotations = {
                'rectangles': [{
                    'x': self.scale_region['x_min'],
                    'y': self.scale_region['y_min'],
                    'w': self.scale_region['x_max'] - self.scale_region['x_min'],
                    'h': self.scale_region['y_max'] - self.scale_region['y_min'],
                    'color': (255, 0, 0),  # Blue
                    'label': 'Scale Region'
                }]
            }
            self.debug_viz.save_debug_image(
                image, 'scale_detection',
                annotations=scale_rect_annotations,
                info_text=f"Scale region: {self.scale_region}"
            )
        else:
            roi = gray
            enhanced_roi = enhanced_edges
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(roi, (self.blur_kernel, self.blur_kernel), 0)
        
        # Detect edges - use enhanced edges if available, otherwise standard Canny
        if enhanced_roi is not None:
            edges = enhanced_roi
            self.debug_viz.save_debug_image(
                edges, 'edges_color_enhanced',
                info_text="Color-enhanced edge detection"
            )
        else:
            edges = cv2.Canny(blurred, self.edge_low, self.edge_high)
        
        # Debug: Save edge detection result
        self.debug_viz.save_debug_image(
            edges, 'edges',
            info_text=[
                f"Canny edges: low={self.edge_low}, high={self.edge_high}",
                f"Blur kernel: {self.blur_kernel}x{self.blur_kernel}"
            ]
        )
        
        # Find horizontal lines (water surface tends to be horizontal)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                                minLineLength=50, maxLineGap=10)
        
        detected_lines = []
        horizontal_lines = []
        
        if lines is not None:
            # Find the most prominent horizontal line
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Adjust coordinates if using ROI
                if self.scale_region:
                    x1 += self.scale_region['x_min']
                    x2 += self.scale_region['x_min']
                    y1 += self.scale_region['y_min']
                    y2 += self.scale_region['y_min']
                
                detected_lines.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'color': (0, 255, 255) if angle < 10 or angle > 170 else (0, 0, 255),  # Yellow for horizontal, red for others
                    'thickness': 2,
                    'label': f'{angle:.1f}°'
                })
                
                if angle < 10 or angle > 170:  # Nearly horizontal
                    horizontal_lines.append((y1 + y2) / 2)
            
            # Debug: Show detected lines
            lines_annotations = {'lines': detected_lines}
            self.debug_viz.save_debug_image(
                image, 'contours',
                annotations=lines_annotations,
                info_text=[
                    f"Total lines detected: {len(detected_lines)}",
                    f"Horizontal lines: {len(horizontal_lines)}"
                ]
            )
            
            if horizontal_lines:
                # Return median y-coordinate of horizontal lines
                water_line_y = int(np.median(horizontal_lines))
                return water_line_y
        
        # Fallback: detect using gradient changes
        return self.detect_water_line_gradient(gray)
    
    def detect_water_line_color(self, image):
        """
        Detect the water line using color-based water detection.
        Returns y-coordinate of water line.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create water mask using configured HSV ranges
        water_mask = cv2.inRange(hsv, self.water_hsv_lower, self.water_hsv_upper)
        
        # Focus on scale region if defined
        if self.scale_region:
            roi_mask = water_mask[
                self.scale_region['y_min']:self.scale_region['y_max'],
                self.scale_region['x_min']:self.scale_region['x_max']
            ]
            roi_offset_x = self.scale_region['x_min']
            roi_offset_y = self.scale_region['y_min']
        else:
            roi_mask = water_mask
            roi_offset_x = 0
            roi_offset_y = 0
        
        # Debug: Save water mask
        self.debug_viz.save_debug_image(
            water_mask, 'water_color_mask',
            info_text=[
                f"Water HSV range: {self.water_hsv_lower} - {self.water_hsv_upper}",
                f"Mask pixels: {cv2.countNonZero(roi_mask)}"
            ]
        )
        
        # Find horizontal water boundaries
        # Look for horizontal transitions from non-water to water
        horizontal_sums = np.sum(roi_mask, axis=1)  # Sum each row
        
        # Find the topmost significant water region
        threshold = roi_mask.shape[1] * 0.3  # At least 30% of width should be water
        water_rows = np.where(horizontal_sums > threshold)[0]
        
        if len(water_rows) > 0:
            water_line_y = water_rows[0] + roi_offset_y  # Top of water region
            
            # Debug: Show water line detection
            water_annotations = {
                'lines': [{
                    'x1': roi_offset_x, 'y1': water_line_y,
                    'x2': roi_offset_x + roi_mask.shape[1], 'y2': water_line_y,
                    'color': (0, 255, 0), 'label': 'Color-based Water Line'
                }]
            }
            self.debug_viz.save_debug_image(
                image, 'water_detection',
                annotations=water_annotations,
                info_text=[
                    f"Color-based detection",
                    f"Water line at y={water_line_y}",
                    f"Water rows found: {len(water_rows)}"
                ]
            )
            
            self.logger.debug(f"Color-based water detection: y={water_line_y}")
            return water_line_y
        
        # Fallback to gradient method if color detection fails
        self.logger.debug("Color-based detection failed, falling back to gradient")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    
    def enhance_scale_detection_rgb(self, image):
        """
        Enhance scale detection using RGB/HSV color filtering.
        Returns combined mask highlighting potential scale regions.
        """
        if not self.color_detection_enabled:
            return None
        
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Initialize combined mask
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        color_masks = {}
        
        # Apply each enabled color filter
        for color_name, color_config in self.scale_colors.items():
            if not color_config.get('enabled', True):
                continue
            
            # Create color mask
            lower = np.array(color_config['hsv_lower'])
            upper = np.array(color_config['hsv_upper'])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphological operations to clean up mask
            kernel_size = self.morphology_config['kernel_size']
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Close gaps and remove noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                                  iterations=self.morphology_config['close_iterations'])
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                  iterations=self.morphology_config['open_iterations'])
            
            # Store individual mask for debugging
            color_masks[color_name] = mask
            
            # Add to combined mask
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Debug: Save individual and combined color masks
        if self.debug_color_masks:
            self.debug_viz.save_debug_image(
                hsv, 'hsv_conversion',
                info_text="HSV color space conversion"
            )
            
            # Save individual color masks
            for color_name, mask in color_masks.items():
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                self.debug_viz.save_debug_image(
                    mask_colored, f'color_mask_{color_name}',
                    info_text=[
                        f"{color_name.capitalize()} mask",
                        f"HSV range: {self.scale_colors[color_name]['hsv_lower']} - {self.scale_colors[color_name]['hsv_upper']}",
                        f"Pixels detected: {np.sum(mask > 0)}"
                    ]
                )
            
            # Save combined mask
            combined_colored = cv2.applyColorMap(combined_mask, cv2.COLORMAP_JET)
            self.debug_viz.save_debug_image(
                combined_colored, 'color_mask_combined',
                info_text=[
                    "Combined color mask",
                    f"Total pixels detected: {np.sum(combined_mask > 0)}",
                    f"Colors used: {', '.join([name for name, config in self.scale_colors.items() if config.get('enabled', True)])}"
                ]
            )
        
        return combined_mask, color_masks
    
    def create_color_enhanced_edges(self, image, color_mask, color_masks):
        """
        Create enhanced edge map using color information and multiple edge detection approaches.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Standard Canny on color-masked grayscale
        masked_gray = cv2.bitwise_and(gray, gray, mask=color_mask)
        blurred = cv2.GaussianBlur(masked_gray, (self.blur_kernel, self.blur_kernel), 0)
        edges_masked = cv2.Canny(blurred, self.edge_low, self.edge_high)
        
        # Method 2: Multi-channel edge detection
        b, g, r = cv2.split(image)
        edges_combined = np.zeros_like(gray)
        
        for channel in [b, g, r]:
            # Apply color mask to each channel
            masked_channel = cv2.bitwise_and(channel, channel, mask=color_mask)
            blurred_channel = cv2.GaussianBlur(masked_channel, (self.blur_kernel, self.blur_kernel), 0)
            channel_edges = cv2.Canny(blurred_channel, self.edge_low, self.edge_high)
            edges_combined = cv2.bitwise_or(edges_combined, channel_edges)
        
        # Method 3: Color transition edges (detect boundaries between different colors)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Detect edges in hue channel (color transitions)
        hue_edges = cv2.Canny(h, 30, 100)
        # Apply color mask to focus on scale region
        hue_edges = cv2.bitwise_and(hue_edges, hue_edges, mask=color_mask)
        
        # Method 4: Individual color mask edges
        individual_edges = np.zeros_like(gray)
        for color_name, mask in color_masks.items():
            if color_name in ['yellow', 'white']:  # Background colors
                continue  # Skip background, focus on markings
            
            # Detect edges within this color region
            color_roi = cv2.bitwise_and(gray, gray, mask=mask)
            if np.sum(mask > 0) > 100:  # Only if sufficient pixels
                color_edges = cv2.Canny(color_roi, self.edge_low // 2, self.edge_high // 2)
                individual_edges = cv2.bitwise_or(individual_edges, color_edges)
        
        # Combine all edge detection methods
        final_edges = cv2.bitwise_or(edges_masked, edges_combined)
        final_edges = cv2.bitwise_or(final_edges, hue_edges)
        final_edges = cv2.bitwise_or(final_edges, individual_edges)
        
        # Clean up the combined edges
        kernel = np.ones((2, 2), np.uint8)
        final_edges = cv2.morphologyEx(final_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Debug: Save intermediate edge results
        self.debug_viz.save_debug_image(
            edges_masked, 'edges_masked_gray',
            info_text="Canny edges on color-masked grayscale"
        )
        
        self.debug_viz.save_debug_image(
            edges_combined, 'edges_multi_channel',
            info_text="Combined edges from RGB channels"
        )
        
        self.debug_viz.save_debug_image(
            hue_edges, 'edges_hue_transitions',
            info_text="Hue channel color transition edges"
        )
        
        self.debug_viz.save_debug_image(
            individual_edges, 'edges_individual_colors',
            info_text="Edges from individual color masks"
        )
        
        self.debug_viz.save_debug_image(
            final_edges, 'edges_final_combined',
            info_text="Final combined color-enhanced edges"
        )
        
        return final_edges

    def detect_scale_bounds_enhanced(self, image):
        """
        Enhanced scale detection that returns both vertical and horizontal bounds.
        For fixed camera setups, prioritizes configured expected_position.
        Returns: (scale_top_y, scale_bottom_y, scale_x_min, scale_x_max)
        """
        self.logger.debug("Starting enhanced scale bounds detection")
        
        # Check if we should use fixed camera mode (prioritize config position)
        use_fixed_camera_mode = self.config['scale'].get('fixed_camera_mode', True)  # Default to True for fixed setups
        
        if use_fixed_camera_mode:
            self.logger.debug("Using fixed camera mode - prioritizing configured expected_position")
            
            # Use configured position as primary source
            scale_x_min = self.scale_region.get('x_min', 0)
            scale_x_max = self.scale_region.get('x_max', image.shape[1])
            scale_top_y = self.scale_region.get('y_min', 0)
            scale_bottom_y = self.scale_region.get('y_max', image.shape[0])
            
            # Override with enhanced calibration data if available (more accurate)
            if self.enhanced_calibration_data and self.enhanced_calibration_data.get('scale_boundaries'):
                boundaries = self.enhanced_calibration_data['scale_boundaries']
                scale_x_min = boundaries.get('x_min', scale_x_min)
                scale_x_max = boundaries.get('x_max', scale_x_max)
                scale_top_y = boundaries.get('y_min', scale_top_y)
                scale_bottom_y = boundaries.get('y_max', scale_bottom_y)
                self.logger.debug(f"Using enhanced calibration boundaries (fixed camera mode)")
            
            detection_method = "fixed_camera_config"
        else:
            # Use dynamic detection (legacy behavior)
            self.logger.debug("Using dynamic detection mode")
            scale_top_y, scale_bottom_y = self.detect_scale_bounds(image)
            
            if scale_top_y is None or scale_bottom_y is None:
                self.logger.warning("Dynamic detection failed, falling back to config values")
                scale_top_y = self.scale_region.get('y_min', 0)
                scale_bottom_y = self.scale_region.get('y_max', image.shape[0])
                detection_method = "config_fallback"
            else:
                detection_method = "dynamic_detection"
            
            # Horizontal bounds from enhanced calibration or config
            if self.enhanced_calibration_data and self.enhanced_calibration_data.get('scale_boundaries'):
                boundaries = self.enhanced_calibration_data['scale_boundaries']
                scale_x_min = boundaries.get('x_min', self.scale_region.get('x_min', 0))
                scale_x_max = boundaries.get('x_max', self.scale_region.get('x_max', image.shape[1]))
            else:
                scale_x_min = self.scale_region.get('x_min', 0)
                scale_x_max = self.scale_region.get('x_max', image.shape[1])
        
        # Log the final scale bounds
        self.logger.debug(f"Scale bounds ({detection_method}): Y=[{scale_top_y}, {scale_bottom_y}], X=[{scale_x_min}, {scale_x_max}]")
        
        # Debug visualization: Draw scale bounds
        debug_image = image.copy()
        cv2.rectangle(debug_image, (scale_x_min, scale_top_y), (scale_x_max, scale_bottom_y), (0, 255, 0), 3)
        cv2.putText(debug_image, f"Scale Region ({detection_method})", (scale_x_min, scale_top_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        self.debug_viz.save_debug_image(
            debug_image, 'scale_bounds_enhanced',
            info_text=[
                f"Detection method: {detection_method}",
                f"Scale bounds: Y=[{scale_top_y},{scale_bottom_y}], X=[{scale_x_min},{scale_x_max}]",
                f"Fixed camera mode: {use_fixed_camera_mode}"
            ]
        )
        
        return scale_top_y, scale_bottom_y, scale_x_min, scale_x_max

    def detect_water_line_within_scale(self, image, scale_top_y, scale_bottom_y, scale_x_min, scale_x_max):
        """
        Detect waterline only within the detected scale boundaries.
        This is much more efficient than scanning the entire image.
        """
        self.logger.debug(f"Detecting waterline within scale bounds: Y=[{scale_top_y},{scale_bottom_y}], X=[{scale_x_min},{scale_x_max}]")
        
        # Extract the scale region from the image
        scale_region = image[scale_top_y:scale_bottom_y, scale_x_min:scale_x_max].copy()
        
        if scale_region.size == 0:
            self.logger.error("Empty scale region extracted")
            return None
        
        # Debug: Save the extracted scale region
        self.debug_viz.save_debug_image(
            scale_region, 'scale_region_extracted',
            info_text=f"Extracted scale region: {scale_region.shape[1]}x{scale_region.shape[0]}"
        )
        
        # INTEGRATED MULTI-METHOD DETECTION: Run all methods and select best result
        local_water_y = self.detect_water_line_integrated_methods(scale_region)
        
        # Convert local coordinates back to global image coordinates
        if local_water_y is not None:
            global_water_y = scale_top_y + local_water_y
            
            # Apply waterline reference enhancement if available
            if self.enhanced_calibration_data and self.enhanced_calibration_data.get('waterline_reference'):
                global_water_y = self.enhance_water_detection_with_reference(global_water_y, image)
            
            self.logger.debug(f"Detected waterline at global Y={global_water_y} (local Y={local_water_y} + offset {scale_top_y})")
            
            # Debug visualization
            debug_image = image.copy()
            cv2.rectangle(debug_image, (scale_x_min, scale_top_y), (scale_x_max, scale_bottom_y), (0, 255, 0), 2)
            cv2.line(debug_image, (scale_x_min, global_water_y), (scale_x_max, global_water_y), (0, 255, 255), 3)
            cv2.putText(debug_image, f"Waterline Y={global_water_y}", (scale_x_min, global_water_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            self.debug_viz.save_debug_image(
                debug_image, 'waterline_within_scale',
                info_text=f"Waterline detected at Y={global_water_y} within scale bounds"
            )
            
            return global_water_y
        else:
            self.logger.warning("No waterline detected within scale bounds")
            return None

    def detect_water_line_edge_in_region(self, scale_region):
        """
        Detect waterline using edge detection within the scale region.
        Returns local Y coordinate within the region.
        """
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, self.edge_low, self.edge_high)
        
        # Find horizontal lines (potential waterlines)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=min(50, scale_region.shape[1]//4))
        
        if lines is not None:
            horizontal_lines = []
            for line in lines:
                rho, theta = line[0]
                # Filter for nearly horizontal lines
                if abs(theta - np.pi/2) < 0.2:  # Within ~11 degrees of horizontal
                    y_position = rho / np.sin(theta) if np.sin(theta) != 0 else None
                    if y_position is not None and 0 <= y_position < scale_region.shape[0]:
                        horizontal_lines.append(y_position)
            
            if horizontal_lines:
                # Return median Y coordinate
                return int(np.median(horizontal_lines))
        
        # Fallback to gradient method
        return self.detect_water_line_gradient_in_region(scale_region)

    def detect_water_line_gradient_in_region(self, scale_region):
        """
        Detect waterline using gradient analysis within the scale region.
        Returns local Y coordinate within the region.
        """
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate vertical gradient
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        grad_y = np.absolute(grad_y)
        
        # Find the row with maximum gradient (potential waterline)
        row_gradients = np.mean(grad_y, axis=1)
        
        if len(row_gradients) > 0:
            max_gradient_y = np.argmax(row_gradients)
            
            # Validate that it's a significant gradient
            max_gradient_value = row_gradients[max_gradient_y]
            mean_gradient = np.mean(row_gradients)
            
            if max_gradient_value > mean_gradient * 1.5:  # At least 50% above average
                return max_gradient_y
        
        # Fallback: return middle of the region
        return scale_region.shape[0] // 2

    def detect_water_line_color_in_region(self, scale_region):
        """
        Detect waterline using color analysis within the scale region.
        Returns local Y coordinate within the region.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2HSV)
        
        # Create water color mask
        water_mask = cv2.inRange(hsv, self.water_hsv_lower, self.water_hsv_upper)
        
        # Find horizontal contours in water mask
        cnts = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if cnts:
            # Find the topmost water region (waterline)
            min_y = scale_region.shape[0]
            for c in cnts:
                if cv2.contourArea(c) > 50:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(c)
                    if y < min_y:
                        min_y = y
            
            if min_y < scale_region.shape[0]:
                return min_y
        
        # Fallback to gradient method
        return self.detect_water_line_gradient_in_region(scale_region)

    def detect_water_line_gradient_enhanced(self, scale_region):
        """
        Use enhanced calibration gradient data to detect waterline.
        Returns local Y coordinate within the region.
        """
        gradient_data = self.enhanced_calibration_data.get('waterline_gradient')
        if not gradient_data:
            return None
        
        # Use the calibrated color ranges and gradient information
        hsv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        
        # Try to use above/below water color ranges
        if 'detection_ranges' in gradient_data:
            ranges = gradient_data['detection_ranges']
            
            # Create masks for above and below water regions
            above_water_mask = cv2.inRange(hsv, 
                                         np.array(ranges['above_water_hsv']['lower']), 
                                         np.array(ranges['above_water_hsv']['upper']))
            
            below_water_mask = cv2.inRange(hsv,
                                         np.array(ranges['below_water_hsv']['lower']),
                                         np.array(ranges['below_water_hsv']['upper']))
            
            # Find transition between the two regions
            for y in range(scale_region.shape[0] - 1):
                above_pixels = np.sum(above_water_mask[y, :] > 0)
                below_pixels = np.sum(below_water_mask[y + 1, :] > 0)
                
                # If we find a transition from above-water to below-water colors
                if above_pixels > scale_region.shape[1] * 0.3 and below_pixels > scale_region.shape[1] * 0.3:
                    return y
        
        # Fallback: use the difference in grayscale values
        if 'above_water' in gradient_data and 'below_water' in gradient_data:
            above_mean = gradient_data['above_water']['gray_mean']
            below_mean = gradient_data['below_water']['gray_mean']
            threshold_gray = (above_mean + below_mean) / 2
            
            # Find the transition point
            for y in range(scale_region.shape[0]):
                row_mean = np.mean(gray[y, :])
                if (above_mean > below_mean and row_mean < threshold_gray) or \
                   (above_mean < below_mean and row_mean > threshold_gray):
                    return y
        
        return None

    def detect_water_line_integrated_methods(self, scale_region):
        """
        Integrated multi-method detection system that combines edge, color, and gradient methods.
        Uses confidence scoring to select the best result.
        """
        self.logger.debug("Starting integrated multi-method waterline detection")
        
        detection_results = []
        
        # Method 1: Edge Detection
        try:
            edge_result = self.detect_water_line_edge_in_region(scale_region)
            if edge_result is not None:
                confidence = self.calculate_edge_detection_confidence(scale_region, edge_result)
                detection_results.append({
                    'method': 'edge',
                    'y_position': edge_result,
                    'confidence': confidence,
                    'details': f'Edge detection at Y={edge_result}'
                })
                self.logger.debug(f"Edge method: Y={edge_result}, confidence={confidence:.3f}")
        except Exception as e:
            self.logger.warning(f"Edge detection failed: {e}")
        
        # Method 2: Color Detection
        try:
            color_result = self.detect_water_line_color_in_region(scale_region)
            if color_result is not None:
                confidence = self.calculate_color_detection_confidence(scale_region, color_result)
                detection_results.append({
                    'method': 'color',
                    'y_position': color_result,
                    'confidence': confidence,
                    'details': f'Color detection at Y={color_result}'
                })
                self.logger.debug(f"Color method: Y={color_result}, confidence={confidence:.3f}")
        except Exception as e:
            self.logger.warning(f"Color detection failed: {e}")
        
        # Method 3: Gradient Detection
        try:
            gradient_result = self.detect_water_line_gradient_in_region(scale_region)
            if gradient_result is not None:
                confidence = self.calculate_gradient_detection_confidence(scale_region, gradient_result)
                detection_results.append({
                    'method': 'gradient',
                    'y_position': gradient_result,
                    'confidence': confidence,
                    'details': f'Gradient detection at Y={gradient_result}'
                })
                self.logger.debug(f"Gradient method: Y={gradient_result}, confidence={confidence:.3f}")
        except Exception as e:
            self.logger.warning(f"Gradient detection failed: {e}")
        
        # Method 4: Enhanced Gradient (if available)
        if self.enhanced_calibration_data and self.enhanced_calibration_data.get('waterline_gradient'):
            try:
                enhanced_result = self.detect_water_line_gradient_enhanced(scale_region)
                if enhanced_result is not None:
                    confidence = self.calculate_enhanced_detection_confidence(scale_region, enhanced_result)
                    detection_results.append({
                        'method': 'enhanced_gradient',
                        'y_position': enhanced_result,
                        'confidence': confidence,
                        'details': f'Enhanced gradient detection at Y={enhanced_result}'
                    })
                    self.logger.debug(f"Enhanced gradient method: Y={enhanced_result}, confidence={confidence:.3f}")
            except Exception as e:
                self.logger.warning(f"Enhanced gradient detection failed: {e}")
        
        if not detection_results:
            self.logger.warning("No detection methods produced results")
            return None
        
        # Apply consensus analysis for similar results
        consensus_result = self.apply_consensus_analysis(detection_results)
        
        # Select the best result based on confidence and consensus
        best_result = self.select_best_detection_result(detection_results, consensus_result)
        
        if best_result:
            self.logger.info(f"Selected {best_result['method']} method: {best_result['details']}, confidence={best_result['confidence']:.3f}")
            
            # Create visualization of all detection results
            self.visualize_integrated_detection_results(scale_region, detection_results, best_result)
            
            return best_result['y_position']
        else:
            self.logger.warning("Failed to select best detection result")
            return None

    def calculate_edge_detection_confidence(self, scale_region, y_position):
        """Calculate confidence score for edge detection result"""
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.edge_low, self.edge_high)
        
        # Check edge strength at detected position
        if 0 <= y_position < edges.shape[0]:
            row_edges = edges[y_position, :]
            edge_density = np.sum(row_edges > 0) / len(row_edges)
            
            # Check for horizontal continuity
            continuity_score = 1.0 if edge_density > 0.3 else edge_density / 0.3
            
            # Check gradient strength
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_strength = np.mean(np.abs(grad_y[y_position, :]))
            
            # Normalize to 0-1 scale
            normalized_gradient = min(gradient_strength / 50.0, 1.0)
            
            return (continuity_score + normalized_gradient) / 2
        
        return 0.0

    def calculate_color_detection_confidence(self, scale_region, y_position):
        """Calculate confidence score for color detection result"""
        hsv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2HSV)
        water_mask = cv2.inRange(hsv, self.water_hsv_lower, self.water_hsv_upper)
        
        if 0 <= y_position < water_mask.shape[0]:
            # Check water color presence at detection line
            row_water = water_mask[y_position, :]
            water_ratio = np.sum(row_water > 0) / len(row_water)
            
            # Check color consistency in neighboring rows
            consistency_score = 1.0
            for offset in [-2, -1, 1, 2]:
                check_y = y_position + offset
                if 0 <= check_y < water_mask.shape[0]:
                    neighbor_ratio = np.sum(water_mask[check_y, :] > 0) / water_mask.shape[1]
                    if abs(water_ratio - neighbor_ratio) > 0.3:
                        consistency_score *= 0.8
            
            return min(water_ratio * consistency_score, 1.0)
        
        return 0.0

    def calculate_gradient_detection_confidence(self, scale_region, y_position):
        """Calculate confidence score for gradient detection result"""
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        if 0 <= y_position < grad_y.shape[0]:
            # Get gradient strength at detected position
            row_gradient = np.mean(np.abs(grad_y[y_position, :]))
            
            # Compare with neighboring rows
            neighbor_gradients = []
            for offset in range(-3, 4):
                check_y = y_position + offset
                if 0 <= check_y < grad_y.shape[0]:
                    neighbor_gradients.append(np.mean(np.abs(grad_y[check_y, :])))
            
            if neighbor_gradients:
                max_neighbor_gradient = max(neighbor_gradients)
                if max_neighbor_gradient > 0:
                    relative_strength = row_gradient / max_neighbor_gradient
                    return min(relative_strength, 1.0)
        
        return 0.0

    def calculate_enhanced_detection_confidence(self, scale_region, y_position):
        """Calculate confidence score for enhanced gradient detection result"""
        gradient_data = self.enhanced_calibration_data.get('waterline_gradient')
        if not gradient_data:
            return 0.0
        
        # High confidence for enhanced method if it uses calibrated data
        base_confidence = 0.8  # Enhanced method gets higher base confidence
        
        # Validate against expected color differences
        if 'differences' in gradient_data:
            gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
            if 0 <= y_position < gray.shape[0]:
                # Check if observed gradient matches expected
                expected_diff = gradient_data['differences']['gray_diff']
                
                above_region = max(0, y_position - 10)
                below_region = min(gray.shape[0], y_position + 10)
                
                if above_region < y_position < below_region:
                    above_mean = np.mean(gray[above_region:y_position, :])
                    below_mean = np.mean(gray[y_position:below_region, :])
                    observed_diff = abs(above_mean - below_mean)
                    
                    # Compare with expected difference
                    if expected_diff > 0:
                        similarity = 1.0 - min(abs(observed_diff - expected_diff) / expected_diff, 1.0)
                        return base_confidence * similarity
        
        return base_confidence

    def apply_consensus_analysis(self, detection_results):
        """Analyze consensus among detection methods"""
        if len(detection_results) < 2:
            return None
        
        # Group similar results (within 5 pixels)
        consensus_groups = []
        for result in detection_results:
            placed = False
            for group in consensus_groups:
                if any(abs(result['y_position'] - r['y_position']) <= 5 for r in group):
                    group.append(result)
                    placed = True
                    break
            if not placed:
                consensus_groups.append([result])
        
        # Find the largest consensus group
        largest_group = max(consensus_groups, key=len)
        
        if len(largest_group) >= 2:
            # Calculate consensus position and confidence
            avg_position = np.mean([r['y_position'] for r in largest_group])
            avg_confidence = np.mean([r['confidence'] for r in largest_group])
            consensus_boost = len(largest_group) / len(detection_results)
            
            return {
                'y_position': int(avg_position),
                'confidence': avg_confidence * (1 + consensus_boost * 0.2),
                'methods': [r['method'] for r in largest_group],
                'count': len(largest_group)
            }
        
        return None

    def select_best_detection_result(self, detection_results, consensus_result):
        """Select the best detection result considering confidence and consensus"""
        # If we have strong consensus, prefer it
        if consensus_result and consensus_result['confidence'] > 0.6 and consensus_result['count'] >= 2:
            return {
                'method': f"consensus({','.join(consensus_result['methods'])})",
                'y_position': consensus_result['y_position'],
                'confidence': consensus_result['confidence'],
                'details': f"Consensus of {consensus_result['count']} methods at Y={consensus_result['y_position']}"
            }
        
        # Otherwise, select highest confidence individual result
        best_result = max(detection_results, key=lambda x: x['confidence'])
        
        # Boost confidence if enhanced method agrees with others
        if best_result['method'] == 'enhanced_gradient':
            for other in detection_results:
                if other['method'] != 'enhanced_gradient' and abs(other['y_position'] - best_result['y_position']) <= 3:
                    best_result['confidence'] = min(best_result['confidence'] * 1.1, 1.0)
                    break
        
        return best_result

    def visualize_integrated_detection_results(self, scale_region, detection_results, best_result):
        """Create visualization showing all detection results"""
        vis_image = scale_region.copy()
        
        # Draw all detection results
        colors = {
            'edge': (255, 0, 0),      # Blue
            'color': (0, 255, 0),     # Green
            'gradient': (0, 0, 255),  # Red
            'enhanced_gradient': (255, 0, 255)  # Magenta
        }
        
        for result in detection_results:
            y = result['y_position']
            method = result['method'].split('(')[0]  # Remove consensus info
            color = colors.get(method, (128, 128, 128))
            
            # Draw detection line
            cv2.line(vis_image, (0, y), (vis_image.shape[1], y), color, 2)
            
            # Add method label
            cv2.putText(vis_image, f"{method}: {result['confidence']:.2f}", 
                       (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Highlight best result
        best_y = best_result['y_position']
        cv2.line(vis_image, (0, best_y), (vis_image.shape[1], best_y), (0, 255, 255), 4)
        cv2.putText(vis_image, f"BEST: {best_result['method']}", 
                   (5, best_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        self.debug_viz.save_debug_image(
            vis_image, 'integrated_detection_methods',
            info_text=f"Multi-method detection: {len(detection_results)} methods, best={best_result['method']}"
        )
    
    def detect_scale_bounds(self, image):
        """
        Detect the top and bottom of the scale using RGB filtering and edge detection.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply RGB color filtering if enabled
        if self.color_detection_enabled:
            color_mask, color_masks = self.enhance_scale_detection_rgb(image)
            
            if color_mask is not None:
                # Use color mask to focus edge detection on scale regions
                masked_gray = cv2.bitwise_and(gray, gray, mask=color_mask)
                
                # Debug: Show masked grayscale
                self.debug_viz.save_debug_image(
                    masked_gray, 'masked_grayscale',
                    info_text="Grayscale with color mask applied"
                )
                
                # Apply edge detection to masked image
                edges = cv2.Canny(masked_gray, self.edge_low, self.edge_high)
            else:
                # Fallback to standard edge detection
                edges = cv2.Canny(gray, self.edge_low, self.edge_high)
        else:
            # Standard edge detection without color filtering
            edges = cv2.Canny(gray, self.edge_low, self.edge_high)
        
        # Find contours
        cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if not cnts:
            return None, None
        
        # Find the best scale contour considering both size and color information
        scale_contour = None
        max_score = 0
        detected_contours = []
        
        for c in cnts:
            if cv2.contourArea(c) < self.config['detection']['min_contour_area']:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(c)
            
            # Check if in expected region
            in_expected_region = True
            if self.scale_region:
                if not (self.scale_region['x_min'] <= x <= self.scale_region['x_max'] and
                       self.scale_region['y_min'] <= y <= self.scale_region['y_max']):
                    in_expected_region = False
            
            # Calculate score based on multiple factors
            aspect_ratio_score = 1.0 if h > w * 2 else (h / w) / 2  # Prefer vertical objects
            size_score = min(h / 200.0, 1.0)  # Prefer larger objects, cap at 200px
            region_score = 1.0 if in_expected_region else 0.3  # Strong preference for expected region
            
            # If color detection is enabled, boost score for contours in color regions
            color_score = 1.0
            if self.color_detection_enabled and color_mask is not None:
                # Check overlap with color mask
                contour_mask = np.zeros_like(color_mask)
                cv2.drawContours(contour_mask, [c], -1, 255, -1)
                overlap = cv2.bitwise_and(color_mask, contour_mask)
                overlap_ratio = np.sum(overlap > 0) / cv2.contourArea(c)
                color_score = 0.5 + 1.5 * overlap_ratio  # Boost score for color overlap
            
            total_score = aspect_ratio_score * size_score * region_score * color_score
            
            # Store for debugging
            detected_contours.append({
                'contour': c,
                'bbox': (x, y, w, h),
                'scores': {
                    'aspect_ratio': aspect_ratio_score,
                    'size': size_score, 
                    'region': region_score,
                    'color': color_score,
                    'total': total_score
                }
            })
            
            if total_score > max_score:
                max_score = total_score
                scale_contour = c
        
        # Debug: Show all detected contours with scores
        if detected_contours:
            contour_annotations = {
                'rectangles': [],
                'text': []
            }
            
            for i, cont_info in enumerate(detected_contours[:10]):  # Limit to top 10
                x, y, w, h = cont_info['bbox']
                score = cont_info['scores']['total']
                is_best = (cont_info['contour'] is scale_contour)
                
                contour_annotations['rectangles'].append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'color': (0, 255, 0) if is_best else (0, 0, 255),
                    'thickness': 3 if is_best else 1,
                    'label': f'Score: {score:.2f}' + (' [BEST]' if is_best else '')
                })
        
            self.debug_viz.save_debug_image(
                image, 'scale_contours_analysis',
                annotations=contour_annotations,
                info_text=[
                    f"Contours analyzed: {len(detected_contours)}",
                    f"Best score: {max_score:.2f}",
                    "Green = selected scale, Red = other contours"
                ]
            )
        
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
        start_time = time.time()
        
        try:
            # Initialize debug for this image
            self.debug_viz.start_image_debug(image_path)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Debug: Save original image
            self.debug_viz.save_debug_image(
                image, 'original',
                info_text=f"Original image: {image.shape[1]}x{image.shape[0]}"
            )
            
            # Resize if configured
            if self.config['processing']['resize_width']:
                height, width = image.shape[:2]
                new_width = self.config['processing']['resize_width']
                new_height = int(height * (new_width / width))
                image = cv2.resize(image, (new_width, new_height))
                
                # Adjust pixels_per_cm for resized image
                scale_factor = new_width / width
                adjusted_pixels_per_cm = self.pixels_per_cm * scale_factor
                
                # Debug: Save resized image
                self.debug_viz.save_debug_image(
                    image, 'preprocessed',
                    info_text=[
                        f"Resized: {width}x{height} -> {new_width}x{new_height}",
                        f"Scale factor: {scale_factor:.3f}",
                        f"Adjusted pixels/cm: {adjusted_pixels_per_cm:.2f}"
                    ]
                )
            else:
                adjusted_pixels_per_cm = self.pixels_per_cm
                # Debug: Save original as preprocessed
                self.debug_viz.save_debug_image(
                    image, 'preprocessed',
                    info_text=f"No resize applied, pixels/cm: {adjusted_pixels_per_cm:.2f}"
                )
            
            # REDESIGNED WORKFLOW: Scale-first, then waterline within scale bounds
            # Step 1: Detect scale bounds first
            scale_top_y, scale_bottom_y, scale_x_min, scale_x_max = self.detect_scale_bounds_enhanced(image)
            
            # Step 2: Detect water line within scale bounds only
            water_line_y = self.detect_water_line_within_scale(image, scale_top_y, scale_bottom_y, scale_x_min, scale_x_max)
            
            # Calculate water level
            result = self.calculate_water_level(
                water_line_y, scale_top_y, scale_bottom_y
            )
            
            if result:
                # Add metadata
                result['timestamp'] = datetime.now()
                result['image_path'] = image_path
                result['confidence'] = self.estimate_confidence(result)
                result['processing_time'] = time.time() - start_time
                
                # Debug: Create final result visualization
                final_annotations = {
                    'lines': [
                        {
                            'x1': 0, 'y1': water_line_y, 
                            'x2': image.shape[1], 'y2': water_line_y,
                            'color': (0, 255, 0), 'thickness': 3,
                            'label': f'Water Level: {result["water_level_cm"]:.1f}cm'
                        }
                    ],
                    'points': [
                        {
                            'x': image.shape[1] - 50, 'y': scale_top_y,
                            'color': (255, 0, 0), 'radius': 8,
                            'label': 'Scale Top'
                        },
                        {
                            'x': image.shape[1] - 50, 'y': scale_bottom_y,
                            'color': (255, 0, 0), 'radius': 8,
                            'label': 'Scale Bottom'
                        }
                    ],
                    'text': [
                        {
                            'x': 20, 'y': 50,
                            'text': f'Water Level: {result["water_level_cm"]:.1f} cm',
                            'color': (0, 255, 0), 'scale': 1.0
                        },
                        {
                            'x': 20, 'y': 80,
                            'text': f'Scale Above Water: {result["scale_above_water_cm"]:.1f} cm',
                            'color': (255, 255, 0), 'scale': 0.8
                        },
                        {
                            'x': 20, 'y': 110,
                            'text': f'Confidence: {result["confidence"]:.2f}',
                            'color': (255, 255, 255), 'scale': 0.8
                        }
                    ]
                }
                
                self.debug_viz.save_debug_image(
                    image, 'final_result',
                    annotations=final_annotations,
                    info_text=[
                        f"Processing completed in {result['processing_time']:.2f}s",
                        f"Water level: {result['water_level_cm']:.1f}cm",
                        f"Pixels per cm: {adjusted_pixels_per_cm:.2f}"
                    ]
                )
                
                # Create summary image
                self.debug_viz.create_summary_image(result)
                
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
        
        # Save annotated image using configured format
        image_format = self.config['processing'].get('image_format', 'jpg')
        # Ensure format starts with dot
        if not image_format.startswith('.'):
            image_format = '.' + image_format
        
        # Use relative path with proper directory creation
        processed_dir = Path('data/processed')
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}{image_format}"
        success = cv2.imwrite(str(output_path), annotated)
        
        if success:
            self.logger.debug(f"Saved processed image: {output_path}")
        else:
            self.logger.warning(f"Failed to save processed image: {output_path}")