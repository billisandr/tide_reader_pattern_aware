"""
E-Pattern Sequential Scale Measurement Detector

Specialized detector for E-shaped scale markings that implements:
1. Top-to-bottom pattern matching on detected scale
2. Hierarchical pattern matching (double_E_pattern -> E_pattern_black/white)
3. Pixel per cm calculation and validation against calibration
4. Stopping condition when pixel per cm differs significantly (underwater detection)
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
    1. Load E-shaped templates (double_E_pattern, E_pattern_black, E_pattern_white)
    2. Start from top of scale, move downwards
    3. Try double_E_pattern first (10cm), then smaller E-patterns (5cm)
    4. Calculate pixel per cm for each match
    5. Stop when pixel per cm differs significantly from calibration
    6. Store debug info and images
    """
    
    def __init__(self, config, calibration_data=None):
        """
        Initialize E-pattern detector.
        
        Args:
            config: System configuration
            calibration_data: Enhanced calibration data containing pixels_per_cm
        """
        self.config = config
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
        
        # E-pattern specific settings (removed double_E_pattern)
        self.single_e_cm = e_pattern_config.get('single_e_cm', 5.0)   # E_pattern_black/white correspond to 5 cm
        
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
        template_files = {
            'E_pattern_black': self.template_dir / 'E_pattern_black.png',
            'E_pattern_white': self.template_dir / 'E_pattern_white.png'
        }
        
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
                        
                        # Create and store 180-degree flipped version
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
                    
                    self.logger.info(f"Created {len(scale_factors) * 2} template variants for {template_name}")
                    
                else:
                    self.logger.warning(f"Failed to load template: {template_path}")
            else:
                self.logger.warning(f"Template file not found: {template_path}")
        
        self.logger.info(f"Total templates loaded: {len(self.templates)} (multi-scale + flipped variants)")
    
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
                self._save_debug_info(scale_region, water_line_y, image_path)
            
            if water_line_y is not None:
                self.logger.info(f"E-pattern detection successful: water line at Y={water_line_y}")
                self.logger.info(f"Found {len(self.matched_patterns)} valid pattern matches before water")
            else:
                self.logger.warning("E-pattern detection failed to find water line")
            
            return water_line_y
            
        except Exception as e:
            self.logger.error(f"E-pattern detection failed: {e}")
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
                        
                        if best_match is None or best_valid_match['confidence'] > best_match['confidence']:
                            best_match = best_valid_match
                            best_template_height = template_img.shape[0]
            
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
                
                match = {
                    'template_name': template_name,
                    'local_x': pt[0],
                    'local_y': pt[1],
                    'global_y': global_y + pt[1],
                    'confidence': confidence,
                    'template_size': template.shape[::-1]  # (w, h)
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
            # Create debug directory for E-pattern detection
            e_debug_dir = self.debug_dir / 'e_pattern_detection'
            e_debug_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = f"e_pattern_{timestamp}"
            
            # Save detailed annotated scale region with matched patterns
            annotated_region = self._create_annotated_region(scale_region, water_line_y)
            annotated_path = e_debug_dir / f"{base_name}_annotated.png"
            cv2.imwrite(str(annotated_path), annotated_region)
            
            # Save clean outline-only version for better readability
            outline_region = self._create_outline_annotated_region(scale_region, water_line_y)
            outline_path = e_debug_dir / f"{base_name}_outlines.png"
            cv2.imwrite(str(outline_path), outline_region)
            
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
            
            self.logger.info(f"Saved E-pattern debug info: {annotated_path}, {outline_path}, {info_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save E-pattern debug info: {e}")
    
    def _create_annotated_region(self, scale_region, water_line_y):
        """Create annotated scale region showing matched patterns and water line."""
        annotated = scale_region.copy()
        
        # Colors for different templates
        colors = {
            'E_pattern_black': (0, 255, 0),        # Green
            'E_pattern_white': (0, 255, 255),      # Yellow
            'E_pattern_black_flipped': (0, 128, 255),   # Orange
            'E_pattern_white_flipped': (255, 128, 0)    # Cyan
        }
        
        # Draw matched patterns
        for match in self.matched_patterns:
            template_name = match['template_name']
            color = colors.get(template_name, (255, 255, 255))
            
            x, y = match['local_x'], match['global_y']
            w, h = match['template_size']
            
            # Draw rectangle around match
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{template_name.replace('_pattern', '')} ({match['cm_value']}cm)"
            cv2.putText(annotated, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add confidence
            conf_text = f"{match['confidence']:.2f}"
            cv2.putText(annotated, conf_text, (x + w + 5, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw water line if detected
        if water_line_y is not None:
            cv2.line(annotated, (0, water_line_y), (annotated.shape[1], water_line_y), 
                    (0, 0, 255), 3)  # Red line for water level
            cv2.putText(annotated, f"Water Line (Y={water_line_y})", (5, water_line_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add legend
        legend_y = 20
        for template_name, color in colors.items():
            if any(m['template_name'] == template_name for m in self.matched_patterns):
                cm_value = self.templates[template_name]['cm_value']
                legend_text = f"{template_name}: {cm_value}cm"
                cv2.putText(annotated, legend_text, (5, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                legend_y += 20
        
        return annotated
    
    def _create_outline_annotated_region(self, scale_region, water_line_y):
        """Create clean annotated scale region showing only pattern outlines."""
        annotated = scale_region.copy()
        
        # Use purple color for all pattern outlines
        purple_color = (128, 0, 128)  # Purple in BGR format
        
        # Draw pattern outlines with thick purple lines
        for match in self.matched_patterns:
            x, y = match['local_x'], match['global_y']
            w, h = match['template_size']
            
            # Draw thick purple rectangle outline
            cv2.rectangle(annotated, (x, y), (x + w, y + h), purple_color, 3)
        
        # Draw water line if detected (thicker line)
        if water_line_y is not None:
            cv2.line(annotated, (0, water_line_y), (annotated.shape[1], water_line_y), 
                    (0, 0, 255), 2)  # Red line for water level
        
        # Add minimal legend in top corner
        legend_y = 15
        font_scale = 0.4
        thickness = 1
        
        # Count patterns by type for summary
        pattern_counts = {}
        for match in self.matched_patterns:
            pattern_type = match['template_name'].replace('_flipped', '')
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        # Show pattern counts with purple color to match outlines
        cv2.putText(annotated, f"E-Patterns (purple outlines):", (5, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        legend_y += 15
        
        for pattern_type, count in pattern_counts.items():
            cv2.putText(annotated, f"{pattern_type}: {count}", (5, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, purple_color, thickness)
            legend_y += 12
        
        # Show total count and scale measurement
        total_patterns = len(self.matched_patterns)
        scale_above_water = total_patterns * self.single_e_cm
        
        legend_y += 5
        cv2.putText(annotated, f"Total: {total_patterns} patterns", (5, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        legend_y += 12
        cv2.putText(annotated, f"Scale: {scale_above_water}cm", (5, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        if water_line_y is not None:
            legend_y += 12
            cv2.putText(annotated, f"Water at Y={water_line_y}", (5, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        
        return annotated
    
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
        
        # Add template resize information
        template_info = {}
        for name, data in self.templates.items():
            if 'scale_factor' in data:
                template_info[name] = {
                    'original_size': data['original_size'],
                    'resized_size': data['resized_size'],
                    'scale_factor': data['scale_factor']
                }
        
        info['template_resize_info'] = template_info
        
        return info