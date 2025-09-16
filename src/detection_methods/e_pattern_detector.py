"""
E-Pattern Sequential Scale Measurement Detector

Specialized detector for E-shaped scale markings that implements:
1. Top-to-bottom pattern matching on detected scale
2. Scale-invariant template matching (E_pattern_black/white at multiple scales)
3. Shape-only template matching without size constraints
4. Stopping condition when consecutive pattern matching fails (underwater detection)
5. Debug storage of matched pattern positions
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime

class EPatternDetector:
    """
    Sequential E-pattern detector for scale measurement.
    
    Process:
    1. Load E-shaped templates (E_pattern_black, E_pattern_white) at multiple scales
    2. Start from top of scale, move downwards
    3. Test all template variants (11 scales × 2 orientations) for best match
    4. Use 5cm measurement value for detected E-patterns
    5. Stop when consecutive pattern matching failures exceed threshold
    6. Store debug info and images
    """
    
    def __init__(self, config, calibration_data=None, debug_viz=None):
        """
        Initialize E-pattern detector.
        
        Args:
            config: System configuration
            calibration_data: Enhanced calibration data containing pixels_per_cm
            debug_viz: Optional debug visualizer for saving debug images in pattern-aware session
        """
        self.config = config
        self.debug_viz = debug_viz
        self.logger = logging.getLogger(__name__)
        
        # Get pixels_per_cm from calibration data
        if calibration_data and 'pixels_per_cm' in calibration_data:
            self.pixels_per_cm = calibration_data['pixels_per_cm']
            self.logger.info(f"Using calibration pixels_per_cm: {self.pixels_per_cm}")
        else:
            # Fallback: try to load from calibration.yaml directly
            import yaml
            try:
                calibration_path = Path('data/calibration/calibration.yaml')
                if calibration_path.exists():
                    with open(calibration_path, 'r') as f:
                        calib_data = yaml.safe_load(f)
                    self.pixels_per_cm = calib_data.get('pixels_per_cm', 2.0)
                    self.logger.info(f"Loaded pixels_per_cm from calibration.yaml: {self.pixels_per_cm}")
                else:
                    self.pixels_per_cm = 2.0  # Default fallback
                    self.logger.warning("No calibration data found, using default pixels_per_cm: 2.0")
            except Exception as e:
                self.pixels_per_cm = 2.0
                self.logger.error(f"Failed to load calibration data: {e}, using default: 2.0")
        
        # E-pattern configuration
        e_pattern_config = config.get('detection', {}).get('pattern_aware', {}).get('e_pattern_detection', {})
        self.match_threshold = e_pattern_config.get('match_threshold', 0.6)
        self.max_consecutive_failures = e_pattern_config.get('max_consecutive_failures', 10)
        
        # E-pattern specific settings
        self.single_e_cm = e_pattern_config.get('single_e_cm', 5.0)   # Each E-pattern represents 5 cm
        self.support_flipped = e_pattern_config.get('support_flipped', False)  # Support 180-degree flipped patterns
        
        # Template directory
        self.template_dir = Path(config.get('pattern_processing', {}).get('template_directory', 
                                        'data/pattern_templates/scale_markings'))
        
        # Debug settings
        self.debug_enabled = config.get('debug', {}).get('enabled', False)
        self.debug_dir = Path(config.get('debug', {}).get('debug_output_dir', 'data/debug'))
        
        # Initialize templates
        self.templates = {}
        self.load_e_templates()
        
        # Store matched patterns for debug
        self.matched_patterns = []
        
        self.logger.info(f"E-Pattern detector initialized with {len(self.templates)} templates")
    
    def load_e_templates(self):
        """Load E-shaped templates at multiple scales for scale-invariant matching."""
        # Automatically load all image files from template directory
        if not self.template_dir.exists():
            self.logger.warning(f"Template directory does not exist: {self.template_dir}")
            return
        
        # Find all image files in template directory
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        template_files = {}
        
        for image_file in self.template_dir.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                template_name = image_file.stem  # Use filename without extension as template name
                template_files[template_name] = image_file
        
        if not template_files:
            self.logger.warning(f"No image files found in template directory: {self.template_dir}")
            return
        
        self.logger.info(f"Found {len(template_files)} template files: {list(template_files.keys())}")
        
        # Define multiple scales to test (from very small to large)
        scale_factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
        
        self.logger.info(f"Loading templates at {len(scale_factors)} different scales for scale-invariant matching")
        
        for template_name, template_path in template_files.items():
            if template_path.exists():
                original_template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if original_template is not None:
                    orig_height, orig_width = original_template.shape
                    self.logger.debug(f"Original template {template_name}: {orig_width} x {orig_height} pixels")
                    
                    # Create templates at multiple scales
                    for scale in scale_factors:
                        scaled_width = int(orig_width * scale)
                        scaled_height = int(orig_height * scale)
                        
                        # Skip very small templates that would be unusable
                        if scaled_height < 4 or scaled_width < 4:
                            continue
                            
                        # Resize template
                        if scale == 1.0:
                            scaled_template = original_template.copy()
                        else:
                            scaled_template = cv2.resize(original_template, 
                                                       (scaled_width, scaled_height),
                                                       interpolation=cv2.INTER_CUBIC)
                        
                        # Store normal orientation
                        scale_key = f"{template_name}_scale_{scale:.1f}"
                        self.templates[scale_key] = {
                            'image': scaled_template,
                            'original_image': original_template,
                            'cm_value': self.single_e_cm,  # This will be used ONLY for final measurement
                            'priority': 1,
                            'orientation': 'normal',
                            'path': template_path,
                            'original_size': (orig_width, orig_height),
                            'scaled_size': (scaled_width, scaled_height),
                            'scale_factor': scale
                        }
                        
                        # Create and store 180-degree flipped version (only if enabled)
                        if self.support_flipped:
                            flipped_template = cv2.rotate(scaled_template, cv2.ROTATE_180)
                            flipped_key = f"{template_name}_scale_{scale:.1f}_flipped"
                            self.templates[flipped_key] = {
                                'image': flipped_template,
                                'original_image': cv2.rotate(original_template, cv2.ROTATE_180),
                                'cm_value': self.single_e_cm,
                                'priority': 1,
                                'orientation': 'flipped_180',
                                'path': template_path,
                                'original_size': (orig_width, orig_height),
                                'scaled_size': (scaled_width, scaled_height),
                                'scale_factor': scale
                            }
                    
                    variants_count = len(scale_factors) * (2 if self.support_flipped else 1)
                    self.logger.info(f"Created {variants_count} template variants for {template_name} (flipped: {'enabled' if self.support_flipped else 'disabled'})")
                    
                else:
                    self.logger.warning(f"Failed to load template: {template_path}")
            else:
                self.logger.warning(f"Template file not found: {template_path}")
        
        orientation_info = "multi-scale + flipped variants" if self.support_flipped else "multi-scale variants only"
        self.logger.info(f"Total templates loaded: {len(self.templates)} ({orientation_info})")
    
    def detect_waterline(self, scale_region, image_path=None):
        """
        Detect water line using sequential E-pattern matching.
        
        Args:
            scale_region: Scale region image (BGR)
            image_path: Original image path for debugging
            
        Returns:
            int: Y-coordinate of detected water line (local to scale region)
        """
        if scale_region is None or scale_region.size == 0:
            self.logger.warning("Empty scale region provided")
            return None
        
        if not self.templates:
            self.logger.warning("No E-templates loaded")
            return None
        
        try:
            # Reset matched patterns for this detection
            self.matched_patterns = []
            
            # Convert to grayscale if needed
            if len(scale_region.shape) == 3:
                gray_region = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_region = scale_region.copy()
            
            self.logger.info(f"Starting E-pattern sequential detection on {gray_region.shape} region")
            
            # Perform sequential pattern matching from top to bottom
            water_line_y = self._sequential_pattern_matching(gray_region, scale_region)
            
            # Save debug information if enabled
            if self.debug_enabled and image_path:
                self.logger.info(f"Saving E-pattern debug info for image: {Path(image_path).name}")
                self._save_debug_info(scale_region, water_line_y, image_path)
            elif not self.debug_enabled:
                self.logger.debug("E-pattern debug disabled in config")
            elif not image_path:
                self.logger.warning("No image path provided for E-pattern debug")
            
            if water_line_y is not None:
                self.logger.info(f"E-pattern detection successful: water line at Y={water_line_y}")
                self.logger.info(f"Found {len(self.matched_patterns)} valid pattern matches before water")
                
                # Log pattern size and position summary
                if self.matched_patterns:
                    all_widths = [match.get('pattern_width_px', 0) for match in self.matched_patterns]
                    all_heights = [match.get('pattern_height_px', 0) for match in self.matched_patterns]
                    all_min_x = [match.get('pattern_min_x', 0) for match in self.matched_patterns]
                    all_max_x = [match.get('pattern_max_x', 0) for match in self.matched_patterns]
                    all_min_y = [match.get('pattern_min_y', 0) for match in self.matched_patterns]
                    all_max_y = [match.get('pattern_max_y', 0) for match in self.matched_patterns]
                    
                    if all_widths and all_heights:
                        self.logger.info(f"Pattern size ranges: width {min(all_widths)}-{max(all_widths)}px, "
                                       f"height {min(all_heights)}-{max(all_heights)}px")
                        self.logger.info(f"Pattern position bounds: X[{min(all_min_x)}-{max(all_max_x)}], "
                                       f"Y[{min(all_min_y)}-{max(all_max_y)}]")
            else:
                self.logger.warning("E-pattern detection failed to find water line")
            
            return water_line_y
            
        except Exception as e:
            self.logger.error(f"E-pattern detection failed: {e}")
            
            # Save debug information even on failure
            if self.debug_enabled and image_path:
                try:
                    self.logger.info(f"Saving E-pattern debug info for failed detection: {Path(image_path).name}")
                    self._save_debug_info(scale_region, None, image_path)
                except Exception as debug_error:
                    self.logger.error(f"Failed to save debug info on exception: {debug_error}")
            elif not self.debug_enabled:
                self.logger.debug("E-pattern debug disabled in config")
            elif not image_path:
                self.logger.warning("No image path provided for E-pattern debug")
            
            return None
    
    def _sequential_pattern_matching(self, gray_region, color_region):
        """
        Perform sequential non-overlapping pattern matching from top to bottom.
        
        Returns:
            int: Y-coordinate where water line was detected, or None
        """
        height, width = gray_region.shape
        
        # Current Y position for sequential search
        current_y = 0
        
        # Track the last valid pattern position
        last_valid_y = 0
        consecutive_failures = 0
        
        # Sort templates by priority 
        sorted_templates = sorted(self.templates.items(), 
                                key=lambda x: x[1]['priority'])
        
        self.logger.debug(f"Template matching order: {[name for name, _ in sorted_templates]}")
        self.logger.info(f"Starting sequential pattern matching from Y=0 to Y={height}")
        
        # Sequential search from top to bottom
        while current_y < height - 50:  # Leave space for template matching
            found_pattern = False
            best_match = None
            best_template_height = 0
            
            # Try templates in priority order at current position
            for template_name, template_data in sorted_templates:
                template_img = template_data['image']
                cm_value = template_data['cm_value']
                
                # Check if template fits in remaining region
                if current_y + template_img.shape[0] >= height:
                    continue
                
                # Extract window for matching (add small buffer)
                window_height = min(template_img.shape[0] + 10, height - current_y)
                window = gray_region[current_y:current_y + window_height, :]
                
                # Perform template matching
                matches = self._match_template_in_window(window, template_img, template_name, current_y)
                
                if matches:
                    # Validate matches
                    valid_matches = self._validate_matches_basic(matches, cm_value)
                    
                    if valid_matches:
                        # Take the best match (highest confidence)
                        best_valid_match = max(valid_matches, key=lambda x: x['confidence'])
                        
                        # Validate scale factor continuity to prevent underwater pattern matches
                        if self._validate_scale_continuity(best_valid_match):
                            if best_match is None or best_valid_match['confidence'] > best_match['confidence']:
                                best_match = best_valid_match
                                best_template_height = template_img.shape[0]
                        else:
                            self.logger.warning(f"Rejected {best_valid_match['template_name']} at Y={current_y} due to "
                                              f"unrealistic scale change: {best_valid_match['scale_factor']:.1f}x")
            
            if best_match:
                # Found a valid pattern at current position
                self.matched_patterns.append(best_match)
                last_valid_y = current_y + best_template_height
                found_pattern = True
                consecutive_failures = 0
                
                self.logger.debug(f"Found {best_match['template_name']} at Y={current_y}, "
                                f"confidence={best_match['confidence']:.3f}")
                
                # Move to next position: skip past this template to avoid overlap
                current_y += best_template_height + 2  # Small gap to ensure no overlap
                
            else:
                # No pattern found at current position
                consecutive_failures += 1
                
                # Check if we should stop (likely reached water)
                if consecutive_failures >= self.max_consecutive_failures and last_valid_y > 0:
                    water_line_y = last_valid_y
                    self.logger.info(f"Stopping pattern matching: {consecutive_failures} consecutive failures, "
                                   f"water line estimated at Y={water_line_y}")
                    return water_line_y
                
                # Move forward by small step to continue searching
                current_y += 5  # Small step when no pattern found
        
        # If we processed the entire scale without finding water, use the last pattern position
        if last_valid_y > 0:
            self.logger.info(f"Reached end of scale, water line at last pattern position: Y={last_valid_y}")
            return last_valid_y
        
        self.logger.warning("No valid E-patterns found in scale region")
        return None
    
    def _match_template_in_window(self, window, template, template_name, global_y):
        """
        Match template in a specific window.
        
        Returns:
            list: List of match dictionaries
        """
        matches = []
        
        if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
            return matches
        
        try:
            # Perform template matching
            result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            
            # Get max confidence for logging
            max_confidence = np.max(result) if result.size > 0 else 0
            self.logger.debug(f"Template {template_name} at Y={global_y}: max_confidence={max_confidence:.3f}, threshold={self.match_threshold}")
            
            # Find locations above threshold
            locations = np.where(result >= self.match_threshold)
            
            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                
                # Get scale factor and other template info
                template_data = self.templates.get(template_name, {})
                scale_factor = template_data.get('scale_factor', 1.0)
                original_size = template_data.get('original_size', template.shape[::-1])
                
                match = {
                    'template_name': template_name,
                    'local_x': pt[0],
                    'local_y': pt[1],
                    'global_y': global_y + pt[1],
                    'confidence': confidence,
                    'template_size': template.shape[::-1],  # (w, h) - actual scaled size
                    'scale_factor': scale_factor,
                    'original_template_size': original_size,
                    'center_y': global_y + pt[1] + template.shape[0] // 2,
                    # Pattern size and position metrics (for info/debugging only)
                    'pattern_width_px': template.shape[1],
                    'pattern_height_px': template.shape[0],
                    'pattern_min_x': pt[0],
                    'pattern_max_x': pt[0] + template.shape[1],
                    'pattern_min_y': global_y + pt[1],
                    'pattern_max_y': global_y + pt[1] + template.shape[0]
                }
                matches.append(match)
        
        except Exception as e:
            self.logger.warning(f"Template matching failed for {template_name}: {e}")
        
        return matches
    
    def _validate_matches_basic(self, matches, cm_value):
        """
        Basic validation of matches - templates are already correctly sized.
        
        Returns:
            list: All matches (no pixel per cm validation needed)
        """
        valid_matches = []
        
        for match in matches:
            # Add metadata for debugging
            match['cm_value'] = cm_value
            match['template_correctly_sized'] = True
            match['calibration_pixel_per_cm'] = self.pixels_per_cm
            valid_matches.append(match)
            
            self.logger.debug(f"Valid {match['template_name']} at ({match['local_x']}, {match['global_y']}) "
                            f"confidence: {match['confidence']:.3f}")
        
        return valid_matches
    
    def _validate_scale_continuity(self, candidate_match):
        """
        Validate that the scale factor of a new match is consistent with previous matches.
        This prevents unrealistic scale changes that often indicate underwater artifacts.
        
        Args:
            candidate_match: Match dictionary with scale_factor
            
        Returns:
            bool: True if scale factor is acceptable, False otherwise
        """
        if len(self.matched_patterns) == 0:
            # First match is always valid
            return True
            
        candidate_scale = candidate_match['scale_factor']
        
        # Calculate baseline scale from first few matches
        recent_matches = self.matched_patterns[-3:]  # Last 3 matches
        recent_scales = [match['scale_factor'] for match in recent_matches]
        
        if len(recent_scales) == 0:
            return True
            
        # Calculate average scale of recent matches
        avg_recent_scale = sum(recent_scales) / len(recent_scales)
        
        # Calculate scale change ratio
        scale_change_ratio = abs(candidate_scale - avg_recent_scale) / avg_recent_scale if avg_recent_scale > 0 else 0
        
        # Maximum allowed scale change (30% seems reasonable for legitimate patterns)
        max_scale_change = 0.3
        
        # Special case: reject dramatic scale reductions that often indicate water artifacts
        if candidate_scale < avg_recent_scale * 0.6:  # 40% or more reduction
            self.logger.debug(f"Rejecting match due to dramatic scale reduction: {candidate_scale:.1f}x vs recent avg {avg_recent_scale:.1f}x")
            return False
        
        # General scale change validation
        if scale_change_ratio > max_scale_change:
            self.logger.debug(f"Rejecting match due to excessive scale change: {scale_change_ratio:.2f} (max: {max_scale_change})")
            return False
            
        self.logger.debug(f"Scale continuity OK: {candidate_scale:.1f}x (recent avg: {avg_recent_scale:.1f}x, change: {scale_change_ratio:.2f})")
        return True
    
    def calculate_scale_above_water(self, water_line_y=None):
        """
        Calculate how much of the scale is above water based on matched E-patterns.
        
        This is where we FINALLY use the 5cm per pattern information for measurement.
        
        Args:
            water_line_y: Y-coordinate of detected water line (optional)
            
        Returns:
            dict: Scale measurement information
        """
        if water_line_y is not None:
            # Count patterns above the water line
            patterns_above_water = [
                match for match in self.matched_patterns 
                if match['global_y'] < water_line_y
            ]
        else:
            # Use all detected patterns
            patterns_above_water = self.matched_patterns
        
        # NOW we use the 5cm information - ONLY for measurement calculation
        # Each detected E-pattern represents 5cm of real-world scale
        scale_above_water_cm = len(patterns_above_water) * self.single_e_cm
        
        # Get pattern positions and analyze their scales
        pattern_positions = [match['global_y'] for match in patterns_above_water]
        pattern_positions.sort()
        
        # Analyze the scales that were successfully detected
        detected_scales = {}
        for match in patterns_above_water:
            scale_factor = match.get('scale_factor', 1.0)
            scale_key = f"{scale_factor:.1f}x"
            detected_scales[scale_key] = detected_scales.get(scale_key, 0) + 1
        
        measurement_info = {
            'total_patterns_detected': len(self.matched_patterns),
            'patterns_above_water': len(patterns_above_water),
            'scale_above_water_cm': scale_above_water_cm,
            'single_pattern_cm': self.single_e_cm,
            'pattern_positions_y': pattern_positions,
            'water_line_y': water_line_y,
            'measurement_method': 'e_pattern_count_multi_scale',
            'detected_scales': detected_scales
        }
        
        if pattern_positions:
            measurement_info['highest_pattern_y'] = min(pattern_positions)
            measurement_info['lowest_pattern_y'] = max(pattern_positions)
            measurement_info['pattern_span_pixels'] = max(pattern_positions) - min(pattern_positions)
            
            # Calculate actual pixel per cm based on detected patterns
            if len(patterns_above_water) > 1:
                pixel_span = max(pattern_positions) - min(pattern_positions)
                cm_span = (len(patterns_above_water) - 1) * self.single_e_cm  # Distance between first and last
                if cm_span > 0:
                    measured_pixels_per_cm = pixel_span / cm_span
                    measurement_info['measured_pixels_per_cm'] = measured_pixels_per_cm
                    measurement_info['calibration_pixels_per_cm'] = self.pixels_per_cm
                    measurement_info['scale_accuracy'] = abs(measured_pixels_per_cm - self.pixels_per_cm) / self.pixels_per_cm
        
        self.logger.info(f"Scale measurement: {len(patterns_above_water)} E-patterns × {self.single_e_cm}cm = "
                        f"{scale_above_water_cm}cm above water")
        if detected_scales:
            self.logger.info(f"Successfully detected scales: {detected_scales}")
        
        return measurement_info
    
    def _save_debug_info(self, scale_region, water_line_y, image_path):
        """Save debug information and annotated images."""
        try:
            # Use pattern-aware session directory if debug_viz is available
            if self.debug_viz and hasattr(self.debug_viz, 'session_dir'):
                # Save debug files in the pattern-aware session directory
                e_debug_dir = self.debug_viz.session_dir / 'e_pattern_detection'
                e_debug_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Saving E-pattern debug info to: {e_debug_dir}")
            else:
                # Fallback to standalone debug directory
                e_debug_dir = self.debug_dir / 'e_pattern_detection'
                e_debug_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Saving E-pattern debug info to fallback dir: {e_debug_dir}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_name = Path(image_path).stem  # Get filename without extension
            base_name = f"e_pattern_{image_name}_{timestamp}"
            
            # Save annotated scale region with side panel
            annotated_region = self._create_annotated_region(scale_region, water_line_y)
            annotated_path = e_debug_dir / f"{base_name}_annotated.png"
            cv2.imwrite(str(annotated_path), annotated_region)
            
            # Save detailed match information
            info_path = e_debug_dir / f"{base_name}_matches.txt"
            with open(info_path, 'w') as f:
                f.write(f"E-Pattern Detection Debug Info\n")
                f.write(f"Image: {Path(image_path).name}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Calibration pixel/cm: {self.pixels_per_cm:.3f}\n")
                f.write(f"Water line Y: {water_line_y}\n")
                f.write(f"Total valid matches: {len(self.matched_patterns)}\n\n")
                
                f.write("Multi-Scale Template Matching Information:\n")
                f.write(f"Approach: Scale-invariant template matching\n")
                f.write(f"Templates tested at multiple scales to find natural size in input image\n")
                f.write(f"5cm per pattern used ONLY for final measurement, not template sizing\n\n")
                
                # Group templates by base name and scale
                base_templates = {}
                for name, data in self.templates.items():
                    if 'original_size' in data:
                        base_name = name.split('_scale_')[0].replace('_flipped', '')
                        if base_name not in base_templates:
                            base_templates[base_name] = {
                                'original_size': data['original_size'],
                                'scales': []
                            }
                        scale_factor = data.get('scale_factor', 1.0)
                        if scale_factor not in base_templates[base_name]['scales']:
                            base_templates[base_name]['scales'].append(scale_factor)
                
                f.write("Template Scale Variants:\n")
                for base_name, info in base_templates.items():
                    orig_w, orig_h = info['original_size']
                    f.write(f"  {base_name}:\n")
                    f.write(f"    Original size: {orig_w} x {orig_h} pixels\n")
                    f.write(f"    Tested scales: {sorted(info['scales'])}\n")
                    f.write(f"    Total variants: {len(info['scales']) * 2} (normal + flipped)\n\n")
                
                # Add scale measurement calculation
                if self.matched_patterns:
                    measurement_info = self.calculate_scale_above_water(water_line_y)
                    f.write("Scale Measurement (Final Phase):\n")
                    f.write(f"  Patterns above water: {measurement_info['patterns_above_water']}\n")
                    f.write(f"  Single pattern represents: {measurement_info['single_pattern_cm']} cm (used for measurement only)\n")
                    f.write(f"  Scale above water: {measurement_info['scale_above_water_cm']} cm\n")
                    f.write(f"  Total patterns detected: {measurement_info['total_patterns_detected']}\n")
                    
                    if 'detected_scales' in measurement_info:
                        f.write(f"  Successfully detected scales: {measurement_info['detected_scales']}\n")
                    
                    if 'measured_pixels_per_cm' in measurement_info:
                        measured_pxcm = measurement_info['measured_pixels_per_cm']
                        calib_pxcm = measurement_info['calibration_pixels_per_cm']
                        accuracy = measurement_info['scale_accuracy']
                        f.write(f"  Measured px/cm from patterns: {measured_pxcm:.3f}\n")
                        f.write(f"  Calibration px/cm: {calib_pxcm:.3f}\n")
                        f.write(f"  Scale measurement accuracy: {accuracy:.1%} difference\n")
                    
                    f.write("\n")
                
                f.write("Matched Patterns (Detection Details):\n")
                for i, match in enumerate(self.matched_patterns):
                    f.write(f"  Match {i+1}:\n")
                    f.write(f"    Template: {match['template_name']}\n")
                    f.write(f"    Position: ({match['local_x']}, {match['global_y']})\n")
                    f.write(f"    Confidence: {match['confidence']:.3f}\n")
                    f.write(f"    Template size: {match['template_size']}\n")
                    
                    # Pattern size and position metrics
                    width = match.get('pattern_width_px', 'N/A')
                    height = match.get('pattern_height_px', 'N/A')
                    min_x = match.get('pattern_min_x', 'N/A')
                    max_x = match.get('pattern_max_x', 'N/A')
                    min_y = match.get('pattern_min_y', 'N/A')
                    max_y = match.get('pattern_max_y', 'N/A')
                    f.write(f"    Pattern dimensions: {width} x {height} pixels\n")
                    f.write(f"    Pattern X bounds: {min_x} to {max_x} pixels\n")
                    f.write(f"    Pattern Y bounds: {min_y} to {max_y} pixels\n")
                    
                    # Show scale information if available
                    if 'scale_factor' in match:
                        scale_factor = match['scale_factor']
                        f.write(f"    Detected scale: {scale_factor:.1f}x original template\n")
                        
                        # Calculate what size this represents in original template
                        template_data = self.templates.get(match['template_name'], {})
                        if 'original_size' in template_data:
                            orig_w, orig_h = template_data['original_size']
                            f.write(f"    Original template: {orig_w}x{orig_h} -> Scaled to: {template_data.get('scaled_size', 'unknown')}\n")
                    
                    f.write(f"    CM value (for measurement): {match.get('cm_value', self.single_e_cm)}\n")
                    f.write(f"    Template correctly sized: {match.get('template_correctly_sized', 'No pre-sizing')}\n")
                    f.write("\n")
            
            self.logger.info(f"Saved E-pattern debug info: {annotated_path}, {info_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save E-pattern debug info: {e}")
    
    def _create_annotated_region(self, scale_region, water_line_y):
        """Create annotated scale region with side panel showing matched patterns and water line."""
        h, w = scale_region.shape[:2]
        
        # Use purple color for better visibility
        purple_color = (128, 0, 128)  # Purple in BGR format
        
        # Create image with only pattern outlines and water line (no text overlays)
        annotated_image = scale_region.copy()
        
        # Draw matched patterns (outlines only)
        for match in self.matched_patterns:
            x, y = match['local_x'], match['global_y']
            w_match, h_match = match['template_size']
            
            # Draw rectangle around match with thicker purple lines
            cv2.rectangle(annotated_image, (x, y), (x + w_match, y + h_match), purple_color, 3)
        
        # Draw water line if detected
        if water_line_y is not None:
            cv2.line(annotated_image, (0, water_line_y), (annotated_image.shape[1], water_line_y), 
                    (0, 0, 255), 4)  # Red line for water level
        
        # Prepare info text for side panel
        info_lines = []
        info_lines.append("E-Pattern Detection Results")
        info_lines.append("")
        
        # Pattern summary
        pattern_counts = {}
        for match in self.matched_patterns:
            pattern_type = match['template_name'].replace('_flipped', '')
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        for pattern_type, count in pattern_counts.items():
            pattern_name = pattern_type.replace('E_pattern_', '').replace('_', ' ').title()
            info_lines.append(f"{pattern_name}: {count} patterns (Purple)")
        
        info_lines.append("")
        info_lines.append(f"Total patterns found: {len(self.matched_patterns)}")
        
        # Scale measurement
        total_patterns = len(self.matched_patterns)
        scale_above_water_cm = total_patterns * self.single_e_cm
        info_lines.append(f"Scale above water: {scale_above_water_cm} cm")
        info_lines.append(f"Each E-pattern = {self.single_e_cm} cm")
        
        if water_line_y is not None:
            info_lines.append("")
            info_lines.append(f"Water line detected at Y = {water_line_y}")
        
        # Pattern size and position summary
        if self.matched_patterns:
            # Calculate overall pattern metrics
            all_widths = [match.get('pattern_width_px', 0) for match in self.matched_patterns]
            all_heights = [match.get('pattern_height_px', 0) for match in self.matched_patterns]
            all_min_x = [match.get('pattern_min_x', 0) for match in self.matched_patterns]
            all_max_x = [match.get('pattern_max_x', 0) for match in self.matched_patterns]
            all_min_y = [match.get('pattern_min_y', 0) for match in self.matched_patterns]
            all_max_y = [match.get('pattern_max_y', 0) for match in self.matched_patterns]
            
            if all_widths and all_heights:
                info_lines.append("")
                info_lines.append("Pattern Size & Position Summary:")
                info_lines.append(f"  Width range: {min(all_widths)}-{max(all_widths)} px")
                info_lines.append(f"  Height range: {min(all_heights)}-{max(all_heights)} px")
                info_lines.append(f"  X bounds: {min(all_min_x)} to {max(all_max_x)} px")
                info_lines.append(f"  Y bounds: {min(all_min_y)} to {max(all_max_y)} px")
        
        # Match details
        if self.matched_patterns:
            info_lines.append("")
            info_lines.append("Pattern Match Details:")
            for i, match in enumerate(self.matched_patterns[:6]):  # Reduced to 6 to make room for size info
                template_name = match['template_name'].replace('E_pattern_', '').replace('_', ' ')
                info_lines.append(f"  {i+1}. {template_name} at Y={match['global_y']}")
                info_lines.append(f"     Confidence: {match['confidence']:.3f}")
                # Add size and position info for each pattern (compact format)
                width = match.get('pattern_width_px', 'N/A')
                height = match.get('pattern_height_px', 'N/A')
                min_x = match.get('pattern_min_x', 'N/A')
                max_x = match.get('pattern_max_x', 'N/A')
                min_y = match.get('pattern_min_y', 'N/A')
                max_y = match.get('pattern_max_y', 'N/A')
                info_lines.append(f"     Size: {width}x{height}px, X: {min_x}-{max_x}, Y: {min_y}-{max_y}")
            
            if len(self.matched_patterns) > 6:
                info_lines.append(f"  ... and {len(self.matched_patterns) - 6} more patterns")
        
        # Template matching info
        info_lines.append("")
        info_lines.append("Detection Parameters:")
        info_lines.append(f"Match threshold: {self.match_threshold}")
        info_lines.append(f"Max consecutive failures: {self.max_consecutive_failures}")
        info_lines.append(f"Calibration: {self.pixels_per_cm:.2f} px/cm")
        
        # Create side panel layout
        return self._add_side_panel_to_image(annotated_image, "E Pattern Detection", info_lines)
    
    def _add_side_panel_to_image(self, image, title, info_lines):
        """Add side panel to image with title and info text."""
        h, w = image.shape[:2]
        
        # Calculate panel width (30% of image width, minimum 350px, maximum 500px)
        panel_width = max(350, min(500, int(w * 0.3)))
        
        # Calculate required panel height
        line_height = 20
        title_lines = 2  # Title + spacing
        total_lines = title_lines + len(info_lines)
        panel_height = max(h, total_lines * line_height + 40)  # Ensure minimum height matches image
        
        # Add gap between image and panel
        gap_width = 10
        total_width = w + gap_width + panel_width
        
        # Create combined image with side panel and gap
        combined_image = np.zeros((panel_height, total_width, 3), dtype=np.uint8)
        
        # Copy original image to left side
        combined_image[:h, :w] = image
        
        # Create black panel on the right side (after gap)
        panel_start_x = w + gap_width
        cv2.rectangle(combined_image, (panel_start_x, 0), (panel_start_x + panel_width, panel_height), (0, 0, 0), -1)
        
        # Add title to panel
        title_y = 30
        cv2.putText(combined_image, title, (panel_start_x + 10, title_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add separator line
        cv2.line(combined_image, (panel_start_x + 10, title_y + 10), (panel_start_x + panel_width - 10, title_y + 10), 
                (100, 100, 100), 1)
        
        # Add info text lines
        if info_lines:
            start_y = title_y + 40
            max_chars_per_line = max(1, (panel_width - 20) // 7)  # Estimate chars that fit in panel
            
            current_y = start_y
            for line in info_lines:
                if len(line) <= max_chars_per_line:
                    cv2.putText(combined_image, line, (panel_start_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                               0.4, (200, 200, 200), 1, cv2.LINE_AA)
                    current_y += line_height
                else:
                    # Word wrap long lines
                    words = line.split(' ')
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        if len(test_line) <= max_chars_per_line:
                            current_line = test_line
                        else:
                            if current_line:
                                cv2.putText(combined_image, current_line, (panel_start_x + 10, current_y), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                                current_y += line_height
                                current_line = word
                            else:
                                # Single word too long, force break
                                cv2.putText(combined_image, word[:max_chars_per_line], 
                                           (panel_start_x + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                                           (200, 200, 200), 1, cv2.LINE_AA)
                                current_y += line_height
                                current_line = word[max_chars_per_line:]
                    
                    # Add remaining text
                    if current_line:
                        cv2.putText(combined_image, current_line, (panel_start_x + 10, current_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                        current_y += line_height
        
        return combined_image
    
    def get_detection_info(self):
        """Get information about the E-pattern detector."""
        target_height = self.pixels_per_cm * self.single_e_cm
        
        info = {
            'method': 'e_pattern_sequential_calibrated',
            'templates_loaded': len(self.templates),
            'template_names': list(self.templates.keys()),
            'calibration_pixel_per_cm': self.pixels_per_cm,
            'single_e_cm': self.single_e_cm,
            'target_template_height': target_height,
            'match_threshold': self.match_threshold,
            'max_consecutive_failures': self.max_consecutive_failures,
            'last_match_count': len(self.matched_patterns),
            'supports_flipped': True,
            'uses_dynamic_resizing': True,
            'validation_method': 'calibration_based_sizing'
        }
        
        # Add template scale information (scale-invariant approach)
        template_info = {}
        base_templates = {}
        for name, data in self.templates.items():
            # Group by base template name
            if 'original_size' in data:
                base_name = name.split('_scale_')[0].replace('_flipped', '')
                if base_name not in base_templates:
                    base_templates[base_name] = {
                        'base_size': data['original_size'],
                        'variants': []
                    }
                
                # Extract scale and orientation info
                scale_info = 'base'
                if '_scale_' in name:
                    scale_part = name.split('_scale_')[1]
                    scale_info = scale_part.split('_')[0] if '_' in scale_part else scale_part
                
                orientation = 'flipped' if '_flipped' in name else 'normal'
                base_templates[base_name]['variants'].append({
                    'name': name,
                    'scale': scale_info,
                    'orientation': orientation,
                    'size': data.get('image', np.array([])).shape if 'image' in data else 'unknown'
                })
        
        info['template_scale_info'] = base_templates
        
        return info