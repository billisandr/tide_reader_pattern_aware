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
    def __init__(self, config, pixels_per_cm, enhanced_calibration_data=None, calibration_manager=None):
        """Initialize the water level detector."""
        self.config = config
        self.pixels_per_cm = pixels_per_cm
        self.enhanced_calibration_data = enhanced_calibration_data
        self.calibration_manager = calibration_manager
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters
        self.edge_low = config['detection']['edge_threshold_low']
        self.edge_high = config['detection']['edge_threshold_high']
        self.blur_kernel = config['detection']['blur_kernel_size']
        self.detection_method = config['detection'].get('method', 'edge')
        self.forced_method = config['detection'].get('forced_method', None)
        self.water_hsv_lower = np.array(config['detection'].get('water_hsv_lower', [100, 50, 50]))
        self.water_hsv_upper = np.array(config['detection'].get('water_hsv_upper', [130, 255, 255]))
        
        # Log calibration method being used
        if enhanced_calibration_data and enhanced_calibration_data.get('method') == 'enhanced_interactive_waterline':
            self.logger.info(f"Using enhanced waterline-aware calibration (confidence: {enhanced_calibration_data.get('confidence', 'unknown')})")
            if enhanced_calibration_data.get('waterline_reference'):
                waterline_ref = enhanced_calibration_data['waterline_reference']
                self.logger.info(f"Waterline reference at Y={waterline_ref.get('y_average', 'unknown')}")
        else:
            self.logger.info("Using standard calibration method")
        
        # Log forced method configuration
        if self.forced_method:
            self.logger.info(f"FORCED METHOD CONFIGURED: Will always use '{self.forced_method}' method when available")
        
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
        Multi-color-space gradient detection optimized for clear water effects.
        Analyzes gradients across RGB, HSV, LAB, and YUV color spaces.
        Returns local Y coordinate within the region.
        """
        gradient_data = self.enhanced_calibration_data.get('waterline_gradient')
        if not gradient_data:
            self.logger.warning("Enhanced gradient detection: No waterline_gradient data available")
            return None
        
        self.logger.debug(f"Enhanced gradient detection: Processing {scale_region.shape[1]}x{scale_region.shape[0]} scale region")
        
        # Convert to multiple color spaces for comprehensive analysis
        hsv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(scale_region, cv2.COLOR_BGR2LAB)
        yuv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2YUV)
        
        # Enhanced multi-color-space detection for clear water effects
        if 'above_water' in gradient_data and 'below_water' in gradient_data:
            above_stats = gradient_data['above_water']
            below_stats = gradient_data['below_water']
            
            # Calculate expected differences across color spaces
            above_brightness = above_stats['hsv_mean'][2]  # V channel
            below_brightness = below_stats['hsv_mean'][2]  # V channel
            expected_hsv_darkening = (above_brightness - below_brightness) / above_brightness
            
            above_gray = above_stats['gray_mean']
            below_gray = below_stats['gray_mean']
            expected_gray_darkening = (above_gray - below_gray) / above_gray
            
            # Calculate expected BGR channel differences
            expected_bgr_darkening = []
            for i in range(3):
                above_bgr = above_stats['bgr_mean'][i]
                below_bgr = below_stats['bgr_mean'][i]
                if above_bgr > 0:
                    expected_bgr_darkening.append((above_bgr - below_bgr) / above_bgr)
                else:
                    expected_bgr_darkening.append(0.0)
            
            self.logger.debug(f"Expected darkening - Gray: {expected_gray_darkening:.3f}, HSV-V: {expected_hsv_darkening:.3f}")
            self.logger.debug(f"Expected BGR darkening: B={expected_bgr_darkening[0]:.3f}, G={expected_bgr_darkening[1]:.3f}, R={expected_bgr_darkening[2]:.3f}")
            
            # Multi-color-space gradient analysis
            best_y = None
            best_score = 0.0
            
            for y in range(5, scale_region.shape[0] - 5):
                # Define analysis windows
                window_above_slice = slice(max(0, y-5), y)
                window_below_slice = slice(y, min(scale_region.shape[0], y+5))
                
                # 1. Grayscale analysis (baseline)
                gray_above = np.mean(gray[window_above_slice, :])
                gray_below = np.mean(gray[window_below_slice, :])
                gray_observed = (gray_above - gray_below) / gray_above if gray_above > 0 else 0
                
                # 2. HSV Value channel analysis
                hsv_above = np.mean(hsv[window_above_slice, :, 2])  # V channel
                hsv_below = np.mean(hsv[window_below_slice, :, 2])
                hsv_observed = (hsv_above - hsv_below) / hsv_above if hsv_above > 0 else 0
                
                # 3. LAB Lightness analysis (often more perceptually accurate)
                lab_above = np.mean(lab[window_above_slice, :, 0])  # L channel
                lab_below = np.mean(lab[window_below_slice, :, 0])
                lab_observed = (lab_above - lab_below) / lab_above if lab_above > 0 else 0
                
                # 4. YUV Luminance analysis
                yuv_above = np.mean(yuv[window_above_slice, :, 0])  # Y channel
                yuv_below = np.mean(yuv[window_below_slice, :, 0])
                yuv_observed = (yuv_above - yuv_below) / yuv_above if yuv_above > 0 else 0
                
                # 5. Individual BGR channel analysis
                bgr_scores = []
                for i in range(3):
                    bgr_above = np.mean(scale_region[window_above_slice, :, i])
                    bgr_below = np.mean(scale_region[window_below_slice, :, i])
                    bgr_observed = (bgr_above - bgr_below) / bgr_above if bgr_above > 0 else 0
                    
                    # Score against expected BGR darkening
                    if expected_bgr_darkening[i] > 0:
                        bgr_score = 1.0 - min(abs(bgr_observed - expected_bgr_darkening[i]) / expected_bgr_darkening[i], 1.0)
                        bgr_scores.append(bgr_score)
                
                # Calculate composite score from all color spaces
                color_space_scores = []
                
                # Gray score (25% weight)
                if expected_gray_darkening > 0:
                    gray_score = 1.0 - min(abs(gray_observed - expected_gray_darkening) / expected_gray_darkening, 1.0)
                    color_space_scores.append(('gray', gray_score, 0.25))
                
                # HSV V-channel score (20% weight)
                if expected_hsv_darkening > 0:
                    hsv_score = 1.0 - min(abs(hsv_observed - expected_hsv_darkening) / expected_hsv_darkening, 1.0)
                    color_space_scores.append(('hsv_v', hsv_score, 0.20))
                
                # LAB L-channel score (25% weight - often better for perceptual differences)
                if expected_gray_darkening > 0:  # Use gray as reference for LAB
                    lab_score = 1.0 - min(abs(lab_observed - expected_gray_darkening) / expected_gray_darkening, 1.0)
                    color_space_scores.append(('lab_l', lab_score, 0.25))
                
                # YUV Y-channel score (15% weight)
                if expected_gray_darkening > 0:
                    yuv_score = 1.0 - min(abs(yuv_observed - expected_gray_darkening) / expected_gray_darkening, 1.0)
                    color_space_scores.append(('yuv_y', yuv_score, 0.15))
                
                # BGR channels composite score (15% weight)
                if bgr_scores:
                    bgr_composite = np.mean(bgr_scores)
                    color_space_scores.append(('bgr', bgr_composite, 0.15))
                
                # Calculate weighted total score
                if color_space_scores:
                    weighted_score = sum(score * weight for _, score, weight in color_space_scores)
                    total_weight = sum(weight for _, _, weight in color_space_scores)
                    
                    if total_weight > 0:
                        final_score = weighted_score / total_weight
                        
                        # Apply tolerance range filtering with scale marking awareness
                        tolerance_min = 0.5  # Allow 50% variation
                        tolerance_max = 1.5
                        
                        # Check if most observed darkenings are within expected range
                        within_range_count = 0
                        total_observations = 0
                        extreme_darkening_count = 0  # Track extreme darkenings (likely scale markings)
                        
                        for obs_val, exp_val in [(gray_observed, expected_gray_darkening), 
                                               (hsv_observed, expected_hsv_darkening),
                                               (lab_observed, expected_gray_darkening),
                                               (yuv_observed, expected_gray_darkening)]:
                            if exp_val > 0:
                                ratio = obs_val / exp_val
                                if tolerance_min <= ratio <= tolerance_max:
                                    within_range_count += 1
                                # Count extreme darkenings (scale markings create 3x+ darkening)
                                elif ratio > 3.0:
                                    extreme_darkening_count += 1
                                total_observations += 1
                        
                        # Reject if too many extreme darkenings (scale marking indicator)
                        extreme_ratio = extreme_darkening_count / total_observations if total_observations > 0 else 0
                        if extreme_ratio > 0.3:  # >30% extreme darkenings = likely scale marking
                            self.logger.debug(f"Y={y}: Rejected due to extreme darkenings ({extreme_ratio:.1%})")
                            continue
                        
                        # Require at least 50% of observations to be within range
                        if total_observations > 0 and (within_range_count / total_observations) >= 0.5:
                            if final_score > best_score:
                                best_score = final_score
                                best_y = y
                                
                                self.logger.debug(f"Y={y}: Multi-color scores - Gray={gray_score:.3f}, HSV={hsv_score if 'hsv_score' in locals() else 0:.3f}, LAB={lab_score if 'lab_score' in locals() else 0:.3f}, YUV={yuv_score if 'yuv_score' in locals() else 0:.3f}, Final={final_score:.3f}")
            
            if best_y is not None and best_score > 0.4:  # Slightly higher threshold for multi-color validation
                # Validate detection against scale marking artifacts
                if self.is_scale_marking_artifact(scale_region, best_y, gradient_data):
                    self.logger.warning(f"Rejected Y={best_y} as scale marking artifact (score={best_score:.3f})")
                    best_y = None
                    best_score = 0.0
                else:
                    self.logger.debug(f"Multi-color-space gradient detection: Y={best_y}, score={best_score:.3f}")
                    return best_y
        
        # Edge-based gradient filtering and texture analysis
        best_y_edge = self.detect_with_edge_gradient_filtering(scale_region, gradient_data)
        if best_y_edge is not None:
            return best_y_edge
        
        # Try HSV-based detection with scale marking filtering
        if 'detection_ranges' in gradient_data:
            ranges = gradient_data['detection_ranges']
            
            # Use calibrated HSV ranges but with moderate expansion for lighting variations
            above_lower = np.array(ranges['above_water_hsv']['lower'])
            above_upper = np.array(ranges['above_water_hsv']['upper'])
            below_lower = np.array(ranges['below_water_hsv']['lower'])
            below_upper = np.array(ranges['below_water_hsv']['upper'])
            
            # More conservative tolerance to avoid scale markings (Hue ±10, Saturation ±20, Value ±30)
            h_tolerance, s_tolerance, v_tolerance = 10, 20, 30
            
            above_lower = np.maximum([0, 0, 0], above_lower - [h_tolerance, s_tolerance, v_tolerance])
            above_upper = np.minimum([179, 255, 255], above_upper + [h_tolerance, s_tolerance, v_tolerance])
            below_lower = np.maximum([0, 0, 0], below_lower - [h_tolerance, s_tolerance, v_tolerance])
            below_upper = np.minimum([179, 255, 255], below_upper + [h_tolerance, s_tolerance, v_tolerance])
            
            # Create masks with expanded ranges
            above_water_mask = cv2.inRange(hsv, above_lower, above_upper)
            below_water_mask = cv2.inRange(hsv, below_lower, below_upper)
            
            # Find transition with additional scale marking validation
            for y in range(scale_region.shape[0] - 1):
                above_pixels = np.sum(above_water_mask[y, :] > 0)
                below_pixels = np.sum(below_water_mask[y + 1, :] > 0)
                
                # More sensitive threshold for subtle color changes
                width_threshold = scale_region.shape[1] * 0.25  # 25% width threshold
                if above_pixels > width_threshold and below_pixels > width_threshold:
                    # Additional validation: check for scale marking characteristics
                    if not self.is_scale_marking_artifact(scale_region, y, gradient_data):
                        self.logger.debug(f"HSV transition detection: Y={y}, above={above_pixels}, below={below_pixels}")
                        return y
                    else:
                        self.logger.debug(f"HSV detection rejected Y={y} as scale marking")
        
        # Final fallback: basic grayscale threshold with improved sensitivity
        if 'above_water' in gradient_data and 'below_water' in gradient_data:
            above_mean = gradient_data['above_water']['gray_mean']
            below_mean = gradient_data['below_water']['gray_mean']
            
            # Use weighted threshold closer to above-water value for clear water
            threshold_gray = above_mean * 0.7 + below_mean * 0.3  # Favor above-water characteristics
            
            # Find the transition point with smoothing
            for y in range(2, scale_region.shape[0] - 2):
                # Use 3-row average for stability
                row_mean = np.mean(gray[y-1:y+2, :])
                if (above_mean > below_mean and row_mean < threshold_gray):
                    self.logger.debug(f"Grayscale fallback detection: Y={y}, threshold={threshold_gray:.1f}")
                    return y
        
        return None

    def is_scale_marking_artifact(self, scale_region, y_position, gradient_data):
        """
        Detect if a gradient detection is actually a scale marking artifact using
        calibration-specific scale marking analysis data when available.
        Scale markings have different characteristics than water interfaces.
        """
        if y_position < 5 or y_position >= scale_region.shape[0] - 5:
            return False
        
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2HSV)
        
        # Get scale marking analysis data from enhanced calibration if available
        calib_data = self.calibration_manager.get_enhanced_calibration_data()
        scale_markings = calib_data.get('scale_markings') if calib_data else None
        
        # Analyze characteristics around detected position
        window_size = 3
        above_slice = slice(max(0, y_position - window_size), y_position)
        below_slice = slice(y_position, min(scale_region.shape[0], y_position + window_size))
        
        above_region = gray[above_slice, :]
        below_region = gray[below_slice, :]
        
        if above_region.size == 0 or below_region.size == 0:
            return False
        
        above_mean = np.mean(above_region)
        below_mean = np.mean(below_region)
        
        # Use calibration-specific thresholds if available, otherwise use defaults
        if scale_markings and scale_markings.get('marking_count', 0) > 0:
            # Characteristic 1: Very dark regions - use calibrated marking darkness
            marking_darkness_threshold = scale_markings.get('marking_darkness_threshold', 80)
            if below_mean < marking_darkness_threshold:
                self.logger.debug(f"Scale marking artifact: Very dark below region ({below_mean:.1f} < {marking_darkness_threshold:.1f} - calibrated threshold)")
                return True
            
            # Characteristic 2: High contrast - use calibrated marking contrast
            contrast_ratio = abs(above_mean - below_mean) / max(above_mean, 1)
            marking_contrast_threshold = scale_markings.get('marking_contrast_threshold', 0.4)
            if contrast_ratio > marking_contrast_threshold:
                self.logger.debug(f"Scale marking artifact: High contrast ({contrast_ratio:.2f} > {marking_contrast_threshold:.2f} - calibrated threshold)")
                return True
            
            # Characteristic 3: Low saturation - use calibrated saturation threshold
            above_hsv = hsv[above_slice, :]
            below_hsv = hsv[below_slice, :]
            
            if above_hsv.size > 0 and below_hsv.size > 0:
                below_saturation = np.mean(below_hsv[:, :, 1])  # S channel
                typical_marking_saturation = scale_markings.get('typical_saturation', 15.0)
                
                if below_saturation < typical_marking_saturation:
                    self.logger.debug(f"Scale marking artifact: Low saturation ({below_saturation:.1f} < {typical_marking_saturation:.1f} - calibrated threshold)")
                    return True
            
            # Characteristic 4: Comparison with background threshold
            background_threshold = scale_markings.get('background_threshold', 120)
            background_vs_marking_diff = scale_markings.get('background_vs_marking_diff', 50.0)
            
            # Check if below region is much darker than typical background
            expected_marking_level = background_threshold - background_vs_marking_diff
            if below_mean < expected_marking_level * 1.2:  # 20% tolerance
                self.logger.debug(f"Scale marking artifact: Below background marking level ({below_mean:.1f} < {expected_marking_level*1.2:.1f})")
                return True
            
            self.logger.debug(f"Using calibrated scale marking thresholds (found {scale_markings['marking_count']} markings in calibration)")
            
        else:
            # Fallback to generic thresholds when no calibration data available
            self.logger.debug("Using generic scale marking thresholds (no calibration data)")
            
            # Characteristic 1: Scale markings create very dark regions (black text/lines)
            if below_mean < 80:
                self.logger.debug(f"Scale marking artifact: Very dark below region ({below_mean:.1f} < 80 - generic threshold)")
                return True
            
            # Characteristic 2: Scale markings have high contrast transitions
            contrast_ratio = abs(above_mean - below_mean) / max(above_mean, 1)
            expected_water_contrast = 0.18  # From calibration data (~18% darkening)
            
            if contrast_ratio > expected_water_contrast * 2.5:  # 45%+ contrast
                self.logger.debug(f"Scale marking artifact: High contrast ({contrast_ratio:.2f} > {expected_water_contrast*2.5:.2f} - generic threshold)")
                return True
            
            # Characteristic 3: HSV analysis - scale markings vs water darkening
            above_hsv = hsv[above_slice, :]
            below_hsv = hsv[below_slice, :]
            
            if above_hsv.size > 0 and below_hsv.size > 0:
                below_saturation = np.mean(below_hsv[:, :, 1])  # S channel
                
                if below_saturation < 20:  # Very low saturation = grayscale marking
                    self.logger.debug(f"Scale marking artifact: Low saturation ({below_saturation:.1f} < 20 - generic threshold)")
                    return True
        
        # Common characteristics regardless of calibration
        
        # Characteristic: Scale markings are typically narrow (vertical lines/text)
        # Water interfaces span the full width of the scale
        below_std = np.std(below_region, axis=0)  # Std deviation across width for each row
        width_consistency = np.mean(below_std)
        
        # High width variation suggests non-uniform marking (text/numbers)
        if width_consistency > 30:  # High variation across width
            self.logger.debug(f"Scale marking artifact: High width variation ({width_consistency:.1f} > 30)")
            return True
        
        # Check against calibrated water characteristics if available
        if gradient_data and 'below_water' in gradient_data:
            expected_below_gray = gradient_data['below_water']['gray_mean']
            
            # If detected "below water" is much darker than calibrated below-water
            if below_mean < expected_below_gray * 0.6:  # 40%+ darker than expected
                self.logger.debug(f"Scale marking artifact: Much darker than calibrated water ({below_mean:.1f} vs expected {expected_below_gray:.1f})")
                return True
        
        return False

    def detect_with_edge_gradient_filtering(self, scale_region, gradient_data):
        """
        Edge-based gradient filtering with texture analysis for clear water detection.
        Combines Sobel, Scharr, and Laplacian edge detection with texture variance analysis.
        """
        if not gradient_data or 'above_water' not in gradient_data:
            return None
            
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Multiple edge detection approaches
        # 1. Sobel edge detection (X and Y gradients)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Larger kernel for Y to catch horizontal waterline
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 2. Scharr edge detection (more sensitive to fine details)
        scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
        
        # 3. Laplacian edge detection (second derivative - good for detecting fine transitions)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian = np.abs(laplacian)
        
        # Expected gradient characteristics from calibration
        expected_gray_diff = gradient_data['differences']['gray_diff']
        
        self.logger.debug(f"Edge gradient filtering - expected gray diff: {expected_gray_diff:.2f}")
        
        best_y = None
        best_score = 0.0
        
        for y in range(5, height - 5):
            # Analyze multiple edge signatures around this position
            window_size = 4
            above_slice = slice(max(0, y - window_size), y)
            below_slice = slice(y, min(height, y + window_size))
            
            # 1. Sobel gradient analysis
            sobel_above = np.mean(sobel_magnitude[above_slice, :])
            sobel_below = np.mean(sobel_magnitude[below_slice, :])
            sobel_at_line = np.mean(sobel_magnitude[y, :])  # Edge strength at waterline itself
            
            # 2. Scharr gradient analysis  
            scharr_above = np.mean(scharr_magnitude[above_slice, :])
            scharr_below = np.mean(scharr_magnitude[below_slice, :])
            scharr_at_line = np.mean(scharr_magnitude[y, :])
            
            # 3. Laplacian analysis (detects transitions)
            laplacian_above = np.mean(laplacian[above_slice, :])
            laplacian_below = np.mean(laplacian[below_slice, :])
            laplacian_at_line = np.mean(laplacian[y, :])
            
            # 4. Texture variance analysis (water often smooths texture)
            above_variance = np.var(gray[above_slice, :]) if above_slice.start < above_slice.stop else 0
            below_variance = np.var(gray[below_slice, :]) if below_slice.start < below_slice.stop else 0
            
            # 5. Directional gradient analysis (horizontal vs vertical)
            horizontal_grad = np.mean(np.abs(sobel_y[y, :]))  # Horizontal edges (good for waterline)
            vertical_grad = np.mean(np.abs(sobel_x[y, :]))    # Vertical edges (scale markings)
            
            # Scoring based on edge characteristics
            edge_scores = []
            
            # Score 1: Strong horizontal gradient at waterline (25% weight)
            if horizontal_grad > 10:  # Minimum gradient threshold
                horizontal_score = min(horizontal_grad / 50.0, 1.0)  # Normalize
                edge_scores.append(('horizontal_gradient', horizontal_score, 0.25))
            
            # Score 2: Consistent edge strength difference above/below (20% weight)
            edge_consistency = 0.0
            if sobel_above + sobel_below > 0:
                sobel_balance = abs(sobel_above - sobel_below) / (sobel_above + sobel_below)
                edge_consistency += (1.0 - sobel_balance)  # Lower difference = higher score
            if scharr_above + scharr_below > 0:
                scharr_balance = abs(scharr_above - scharr_below) / (scharr_above + scharr_below)
                edge_consistency += (1.0 - scharr_balance)
            edge_consistency /= 2.0  # Average
            edge_scores.append(('edge_consistency', edge_consistency, 0.20))
            
            # Score 3: Laplacian transition detection (20% weight)
            if laplacian_at_line > 5:  # Minimum transition strength
                laplacian_score = min(laplacian_at_line / 30.0, 1.0)  # Normalize
                edge_scores.append(('laplacian_transition', laplacian_score, 0.20))
            
            # Score 4: Texture variance change (15% weight) - water often reduces texture
            texture_score = 0.0
            if above_variance > 0:
                variance_ratio = below_variance / above_variance
                # Good if below-water variance is 50-90% of above-water (some texture smoothing)
                if 0.3 <= variance_ratio <= 0.9:
                    texture_score = 1.0 - abs(variance_ratio - 0.7) / 0.4  # Optimal around 70%
            edge_scores.append(('texture_variance', texture_score, 0.15))
            
            # Score 5: Gray level gradient consistency with calibration (20% weight)
            gray_above = np.mean(gray[above_slice, :])
            gray_below = np.mean(gray[below_slice, :])
            observed_gray_diff = abs(gray_above - gray_below)
            
            gray_consistency = 0.0
            if expected_gray_diff > 0:
                gray_ratio = observed_gray_diff / expected_gray_diff
                if 0.5 <= gray_ratio <= 1.5:  # Within 50% of expected
                    gray_consistency = 1.0 - abs(gray_ratio - 1.0) / 0.5
            edge_scores.append(('gray_consistency', gray_consistency, 0.20))
            
            # Calculate composite edge score
            if edge_scores:
                weighted_sum = sum(score * weight for _, score, weight in edge_scores)
                total_weight = sum(weight for _, _, weight in edge_scores)
                
                if total_weight > 0:
                    composite_score = weighted_sum / total_weight
                    
                    # Bonus for strong horizontal edges (waterline characteristic)
                    if horizontal_grad > vertical_grad * 1.5:  # Horizontal dominates
                        composite_score *= 1.1
                    
                    # Penalty for very noisy regions (lots of vertical edges - likely scale markings)
                    if vertical_grad > horizontal_grad * 2.0:
                        composite_score *= 0.8
                    
                    if composite_score > best_score and composite_score > 0.3:  # Minimum threshold
                        best_score = composite_score
                        best_y = y
                        
                        self.logger.debug(f"Y={y}: Edge scores - H-grad={horizontal_score if 'horizontal_score' in locals() else 0:.3f}, "
                                        f"Consistency={edge_consistency:.3f}, Laplacian={laplacian_score if 'laplacian_score' in locals() else 0:.3f}, "
                                        f"Texture={texture_score:.3f}, Gray={gray_consistency:.3f}, Final={composite_score:.3f}")
        
        if best_y is not None and best_score > 0.4:
            # Validate against scale marking artifacts
            if self.is_scale_marking_artifact(scale_region, best_y, gradient_data):
                self.logger.warning(f"Edge detection rejected Y={best_y} as scale marking artifact")
                return None
            else:
                self.logger.debug(f"Edge gradient filtering detection: Y={best_y}, score={best_score:.3f}")
                return best_y
        
        return None

    def detect_water_line_integrated_methods(self, scale_region):
        """
        Integrated multi-method detection system that combines edge, color, and gradient methods.
        Uses confidence scoring to select the best result.
        """
        self.logger.info("Starting integrated multi-method waterline detection")
        
        # Log available detection capabilities
        enhanced_available = bool(self.enhanced_calibration_data and self.enhanced_calibration_data.get('waterline_gradient'))
        self.logger.info(f"Enhanced multi-color-space detection available: {enhanced_available}")
        if enhanced_available:
            gradient_data = self.enhanced_calibration_data.get('waterline_gradient')
            self.logger.info(f"Enhanced calibration method: {self.enhanced_calibration_data.get('method', 'unknown')}")
            if gradient_data and 'above_water' in gradient_data and 'below_water' in gradient_data:
                above_gray = gradient_data['above_water'].get('gray_mean', 'N/A')
                below_gray = gradient_data['below_water'].get('gray_mean', 'N/A')
                self.logger.info(f"Calibration data - Above water: {above_gray}, Below water: {below_gray}")
        
        detection_results = []
        
        # Method 1: Enhanced Multi-Color-Space Detection (if available) - PRIORITIZED
        if self.enhanced_calibration_data and self.enhanced_calibration_data.get('waterline_gradient'):
            self.logger.info("Attempting enhanced multi-color-space gradient detection (Method 1 - Priority)")
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
                    self.logger.info(f"Enhanced multi-color-space method: Y={enhanced_result}, confidence={confidence:.3f}")
                else:
                    self.logger.warning("Enhanced multi-color-space detection returned no result")
            except Exception as e:
                self.logger.error(f"Enhanced gradient detection failed: {e}")
                import traceback
                self.logger.debug(f"Enhanced detection traceback: {traceback.format_exc()}")
        else:
            self.logger.info("Enhanced multi-color-space detection not available - using standard methods")

        # Method 2: Edge Detection
        self.logger.debug("Attempting edge detection (Method 2)")
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
                self.logger.info(f"Edge method: Y={edge_result}, confidence={confidence:.3f}")
            else:
                self.logger.debug("Edge detection returned no result")
        except Exception as e:
            self.logger.warning(f"Edge detection failed: {e}")
        
        # Method 3: Color Detection
        self.logger.debug("Attempting color detection (Method 3)")
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
                self.logger.info(f"Color method: Y={color_result}, confidence={confidence:.3f}")
            else:
                self.logger.debug("Color detection returned no result")
        except Exception as e:
            self.logger.warning(f"Color detection failed: {e}")
        
        # Method 4: Standard Gradient Detection
        self.logger.debug("Attempting standard gradient detection (Method 4)")
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
                self.logger.info(f"Standard gradient method: Y={gradient_result}, confidence={confidence:.3f}")
            else:
                self.logger.debug("Standard gradient detection returned no result")
        except Exception as e:
            self.logger.warning(f"Standard gradient detection failed: {e}")
        
        if not detection_results:
            self.logger.warning("No detection methods produced results")
            return None
        
        # Log all detection results summary
        self.logger.info(f"Detection summary: {len(detection_results)} method(s) found results")
        for result in detection_results:
            self.logger.info(f"  {result['method']}: Y={result['y_position']}, confidence={result['confidence']:.3f}")
        
        # Check if a specific method is forced
        if self.forced_method:
            # Find the forced method result
            forced_result = None
            for result in detection_results:
                if result['method'] == self.forced_method:
                    forced_result = result
                    break
            
            if forced_result:
                self.logger.info(f"FORCED METHOD: Using '{self.forced_method}' method (Y={forced_result['y_position']}, confidence={forced_result['confidence']:.3f})")
                best_result = forced_result
            else:
                self.logger.warning(f"FORCED METHOD: '{self.forced_method}' method did not produce a result, falling back to consensus selection")
                # Apply consensus analysis for similar results
                consensus_result = self.apply_consensus_analysis(detection_results)
                # Select the best result based on confidence and consensus
                best_result = self.select_best_detection_result(detection_results, consensus_result)
        else:
            # Apply consensus analysis for similar results
            consensus_result = self.apply_consensus_analysis(detection_results)
            # Select the best result based on confidence and consensus
            best_result = self.select_best_detection_result(detection_results, consensus_result)
        
        if best_result:
            self.logger.info(f"Selected method: {best_result['method']} at Y={best_result['y_position']} (confidence: {best_result['confidence']:.3f})")
        else:
            self.logger.warning("No valid detection result selected")
        
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
        """Calculate confidence score for enhanced gradient detection result - multi-color-space analysis"""
        gradient_data = self.enhanced_calibration_data.get('waterline_gradient')
        if not gradient_data:
            return 0.0
        
        # Higher base confidence for enhanced method with calibrated data
        base_confidence = 0.95  # Increased further to prioritize multi-color-space method
        
        # Convert to multiple color spaces for comprehensive analysis
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(scale_region, cv2.COLOR_BGR2LAB)
        yuv = cv2.cvtColor(scale_region, cv2.COLOR_BGR2YUV)
        
        if not (0 <= y_position < gray.shape[0]):
            return 0.0
        
        confidence_factors = []
        
        # Factor 1: Multi-color-space darkening consistency (35% weight)
        if 'above_water' in gradient_data and 'below_water' in gradient_data:
            expected_gray_darkening = (gradient_data['above_water']['gray_mean'] - gradient_data['below_water']['gray_mean']) / gradient_data['above_water']['gray_mean']
            expected_hsv_darkening = (gradient_data['above_water']['hsv_mean'][2] - gradient_data['below_water']['hsv_mean'][2]) / gradient_data['above_water']['hsv_mean'][2]
            
            window_size = 4
            above_slice = slice(max(0, y_position - window_size), y_position)
            below_slice = slice(y_position, min(scale_region.shape[0], y_position + window_size))
            
            # Analyze darkening across multiple color spaces
            color_space_consistency = []
            
            # Grayscale consistency
            gray_above = np.mean(gray[above_slice, :])
            gray_below = np.mean(gray[below_slice, :])
            gray_observed = (gray_above - gray_below) / gray_above if gray_above > 0 else 0
            if expected_gray_darkening > 0:
                gray_consistency = 1.0 - min(abs(gray_observed - expected_gray_darkening) / expected_gray_darkening, 1.0)
                color_space_consistency.append(gray_consistency)
            
            # HSV V-channel consistency
            hsv_above = np.mean(hsv[above_slice, :, 2])
            hsv_below = np.mean(hsv[below_slice, :, 2])
            hsv_observed = (hsv_above - hsv_below) / hsv_above if hsv_above > 0 else 0
            if expected_hsv_darkening > 0:
                hsv_consistency = 1.0 - min(abs(hsv_observed - expected_hsv_darkening) / expected_hsv_darkening, 1.0)
                color_space_consistency.append(hsv_consistency)
            
            # LAB L-channel consistency
            lab_above = np.mean(lab[above_slice, :, 0])
            lab_below = np.mean(lab[below_slice, :, 0])
            lab_observed = (lab_above - lab_below) / lab_above if lab_above > 0 else 0
            if expected_gray_darkening > 0:
                lab_consistency = 1.0 - min(abs(lab_observed - expected_gray_darkening) / expected_gray_darkening, 1.0)
                color_space_consistency.append(lab_consistency)
            
            # YUV Y-channel consistency
            yuv_above = np.mean(yuv[above_slice, :, 0])
            yuv_below = np.mean(yuv[below_slice, :, 0])
            yuv_observed = (yuv_above - yuv_below) / yuv_above if yuv_above > 0 else 0
            if expected_gray_darkening > 0:
                yuv_consistency = 1.0 - min(abs(yuv_observed - expected_gray_darkening) / expected_gray_darkening, 1.0)
                color_space_consistency.append(yuv_consistency)
            
            if color_space_consistency:
                multi_color_score = np.mean(color_space_consistency)
                # Bonus if most color spaces agree
                agreement_bonus = 1.0 + (0.2 * (len([x for x in color_space_consistency if x > 0.6]) / len(color_space_consistency)))
                multi_color_score *= agreement_bonus
                confidence_factors.append(('multi_color_darkening', min(multi_color_score, 1.0), 0.35))
        
        # Factor 2: BGR channel analysis (20% weight)
        if 'above_water' in gradient_data:
            expected_bgr_above = gradient_data['above_water']['bgr_mean']
            expected_bgr_below = gradient_data['below_water']['bgr_mean']
            
            bgr_scores = []
            for i in range(3):  # B, G, R channels
                bgr_above = np.mean(scale_region[above_slice, :, i])
                bgr_below = np.mean(scale_region[below_slice, :, i])
                
                # Compare with expected values
                above_match = 1.0 - min(abs(bgr_above - expected_bgr_above[i]) / max(expected_bgr_above[i], 1), 1.0)
                below_match = 1.0 - min(abs(bgr_below - expected_bgr_below[i]) / max(expected_bgr_below[i], 1), 1.0)
                bgr_scores.append((above_match + below_match) / 2)
            
            bgr_composite = np.mean(bgr_scores)
            confidence_factors.append(('bgr_channel_match', bgr_composite, 0.20))
        
        # Factor 3: Edge gradient validation (20% weight)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        horizontal_gradient = np.mean(np.abs(sobel_y[y_position, :]))
        
        # Normalize and score horizontal gradient strength
        gradient_score = min(horizontal_gradient / 30.0, 1.0)  # Normalize to 0-1
        
        # Bonus for strong horizontal edges (characteristic of waterline)
        if horizontal_gradient > 15:
            gradient_score *= 1.2
        
        confidence_factors.append(('horizontal_gradient', min(gradient_score, 1.0), 0.20))
        
        # Factor 4: Spatial width consistency (15% weight)
        if y_position < gray.shape[0]:
            row_above = gray[max(0, y_position-2):y_position, :]
            row_below = gray[y_position:min(gray.shape[0], y_position+2), :]
            
            width_consistency = 0.0
            if row_above.size > 0 and row_below.size > 0:
                # Sample across width to check consistency
                width_samples = min(8, gray.shape[1] // 4)
                consistent_points = 0
                
                for x in range(0, gray.shape[1], max(1, gray.shape[1] // width_samples)):
                    if x < row_above.shape[1] and x < row_below.shape[1]:
                        local_above = np.mean(row_above[:, x])
                        local_below = np.mean(row_below[:, x])
                        if local_above > local_below:  # Expected darkening pattern
                            consistent_points += 1
                
                width_consistency = consistent_points / width_samples
            
            confidence_factors.append(('width_consistency', width_consistency, 0.15))
        
        # Factor 5: Texture variance change (10% weight)
        above_variance = np.var(gray[above_slice, :])
        below_variance = np.var(gray[below_slice, :])
        
        texture_score = 0.5  # Default
        if above_variance > 0:
            variance_ratio = below_variance / above_variance
            # Water often reduces texture variance slightly
            if 0.4 <= variance_ratio <= 0.9:
                texture_score = 1.0 - abs(variance_ratio - 0.65) / 0.25  # Optimal around 65%
        
        confidence_factors.append(('texture_variance', texture_score, 0.10))
        
        # Calculate weighted confidence score
        if confidence_factors:
            weighted_score = 0.0
            total_weight = 0.0
            
            for factor_name, score, weight in confidence_factors:
                weighted_score += score * weight
                total_weight += weight
                self.logger.debug(f"Enhanced confidence factor {factor_name}: {score:.3f} (weight: {weight})")
            
            if total_weight > 0:
                final_confidence = base_confidence * (weighted_score / total_weight)
                
                # Bonus for having comprehensive multi-color-space data
                if 'waterline_gradient' in self.enhanced_calibration_data:
                    final_confidence = min(final_confidence * 1.15, 1.0)  # 15% bonus
                
                self.logger.debug(f"Enhanced multi-color detection confidence: {final_confidence:.3f} (base: {base_confidence}, weighted: {weighted_score/total_weight:.3f})")
                return final_confidence
        
        return base_confidence * 0.6  # Higher fallback for enhanced method

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
            
            # Always save processed image if configured (even for failed detection)
            if self.config['processing']['save_processed_images']:
                self.save_processed_image_enhanced(image, result, water_line_y, scale_top_y, scale_bottom_y, adjusted_pixels_per_cm, time.time() - start_time, image_path)
            
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
    
    def save_processed_image_enhanced(self, image, result, water_line_y, scale_top_y, scale_bottom_y, adjusted_pixels_per_cm, processing_time, image_path):
        """
        Save image with annotations showing detected water line and measurements.
        Enhanced version that handles both successful and failed detections.
        """
        annotated = image.copy()
        
        # Draw water line (green if successful, red if failed)
        if water_line_y is not None:
            color = (0, 255, 0) if result else (0, 0, 255)  # Green for success, red for failure
            cv2.line(annotated, 
                    (0, water_line_y), 
                    (image.shape[1], water_line_y),
                    color, 2)
        
        # Draw scale bounds if detected
        if scale_top_y is not None and scale_bottom_y is not None:
            # Draw horizontal lines for scale bounds
            cv2.line(annotated, (0, scale_top_y), (image.shape[1], scale_top_y), (255, 0, 0), 2)  # Blue for scale top
            cv2.line(annotated, (0, scale_bottom_y), (image.shape[1], scale_bottom_y), (255, 0, 0), 2)  # Blue for scale bottom
        
        # Add text with measurements or error information
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        
        if result:
            # Successful detection
            text = f"Water Level: {result['water_level_cm']:.1f}cm"
            cv2.putText(annotated, text, (10, y_offset), font, font_scale, (0, 255, 0), 2)
            y_offset += 30
            
            text = f"Scale Above Water: {result['scale_above_water_cm']:.1f}cm"
            cv2.putText(annotated, text, (10, y_offset), font, font_scale, (255, 255, 0), 2)
            y_offset += 30
        else:
            # Failed detection - show what was attempted
            status_text = "DETECTION FAILED"
            cv2.putText(annotated, status_text, (10, y_offset), font, font_scale, (0, 0, 255), 2)
            y_offset += 30
            
            # Show what was detected/attempted
            if water_line_y is not None:
                text = f"Water line detected at Y={water_line_y}"
                cv2.putText(annotated, text, (10, y_offset), font, font_scale*0.8, (255, 255, 255), 1)
                y_offset += 25
            else:
                text = "No water line detected"
                cv2.putText(annotated, text, (10, y_offset), font, font_scale*0.8, (255, 255, 255), 1)
                y_offset += 25
                
            if scale_top_y is not None and scale_bottom_y is not None:
                text = f"Scale bounds: {scale_top_y}-{scale_bottom_y}"
                cv2.putText(annotated, text, (10, y_offset), font, font_scale*0.8, (255, 255, 255), 1)
                y_offset += 25
            else:
                text = "Scale bounds not detected"
                cv2.putText(annotated, text, (10, y_offset), font, font_scale*0.8, (255, 255, 255), 1)
                y_offset += 25
        
        # Add processing info
        text = f"Processing time: {processing_time:.2f}s"
        cv2.putText(annotated, text, (10, y_offset), font, font_scale*0.6, (200, 200, 200), 1)
        y_offset += 20
        
        text = f"Pixels/cm: {adjusted_pixels_per_cm:.2f}"
        cv2.putText(annotated, text, (10, y_offset), font, font_scale*0.6, (200, 200, 200), 1)
        
        # Save annotated image using configured format
        image_format = self.config['processing'].get('image_format', 'jpg')
        # Ensure format starts with dot
        if not image_format.startswith('.'):
            image_format = '.' + image_format
        
        # Use output directory for annotated images (not processed directory)
        annotated_dir = Path('data/output/annotated')
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        # Include success/failure status in filename
        status = "success" if result else "failed"
        output_path = annotated_dir / f"annotated_{status}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{image_format}"
        success = cv2.imwrite(str(output_path), annotated)
        
        if success:
            self.logger.debug(f"Saved processed image: {output_path}")
        else:
            self.logger.warning(f"Failed to save processed image: {output_path}")

    def save_processed_image(self, image, result):
        """
        Legacy method - kept for backwards compatibility.
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
        
        # Use output directory for annotated images (not processed directory)
        annotated_dir = Path('data/output/annotated')
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = annotated_dir / f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}{image_format}"
        success = cv2.imwrite(str(output_path), annotated)
        
        if success:
            self.logger.debug(f"Saved processed image: {output_path}")
        else:
            self.logger.warning(f"Failed to save processed image: {output_path}")