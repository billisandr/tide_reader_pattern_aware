"""
Template Matching Water Level Detector

Uses template matching to identify and suppress scale markings, then
detects water interfaces in the remaining regions.
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from scipy import ndimage

class TemplateMatchingDetector:
    """
    Water level detector using template matching to suppress scale markings.
    
    Process:
    1. Load scale marking templates (numbers, lines, patterns)
    2. Match templates against the scale region
    3. Mask out detected markings
    4. Detect water interface in unmarked regions
    """
    
    def __init__(self, config, template_manager):
        """
        Initialize template matching detector.
        
        Args:
            config: System configuration
            template_manager: Template management instance
        """
        self.config = config
        self.template_manager = template_manager
        self.logger = logging.getLogger(__name__)
        
        # Template matching configuration
        template_config = config.get('detection', {}).get('pattern_aware', {}).get('template_matching', {})
        self.match_threshold = template_config.get('threshold', 0.7)
        self.max_templates = template_config.get('max_templates', 10)
        
        # Advanced preprocessing settings
        preprocessing_config = template_config.get('preprocessing', {})
        self.use_mean_shift = preprocessing_config.get('mean_shift_filtering', True)
        self.use_adaptive_threshold = preprocessing_config.get('adaptive_thresholding', True)  
        self.use_morphological_cleaning = preprocessing_config.get('morphological_cleaning', True)
        
        # NMS settings
        nms_config = template_config.get('nms', {})
        self.use_nms = nms_config.get('enabled', True)
        self.nms_threshold = nms_config.get('threshold', 0.4)
        self.nms_confidence_threshold = nms_config.get('confidence_threshold', 0.5)
        
        # Template-specific thresholds
        self.template_thresholds = template_config.get('template_thresholds', {})
        
        # Check environment variables for template source override
        env_template_source = os.environ.get('TEMPLATE_SOURCE', '').lower()
        if env_template_source in ['local', 'manager', 'both']:
            self.template_source = env_template_source
            self.logger.info(f"Template source overridden by environment variable: {env_template_source}")
        else:
            self.template_source = template_config.get('template_source', 'local')
        
        # Check environment variable for default templates
        env_use_defaults = os.environ.get('USE_DEFAULT_TEMPLATES', '').lower()
        if env_use_defaults in ['true', 'false']:
            self.use_default_templates = env_use_defaults == 'true'
            self.logger.info(f"Use default templates overridden by environment variable: {self.use_default_templates}")
        else:
            self.use_default_templates = template_config.get('use_default_templates', True)
        
        # Detection parameters
        self.min_water_interface_width = 0.6  # Minimum width for water interface (% of scale width)
        self.horizontal_emphasis = True       # Emphasize horizontal features
        
        # Initialize template storage
        self.templates = {}
        
        # Create default templates if enabled
        if self.use_default_templates and self.template_source in ['local', 'both']:
            self.create_templates()
        
        self.logger.info(f"Template matching detector initialized (threshold: {self.match_threshold}, source: {self.template_source})")
    
    def create_templates(self):
        """
        Create sophisticated template patterns for E-shaped markings and graduation lines
        Based on typical stadia rod marking patterns with improved accuracy
        """
        # Template for E-pattern (major graduations) - Enhanced version
        e_template = np.zeros((20, 15), dtype=np.uint8)
        # Create E-shape: horizontal lines at top, middle, bottom
        e_template[2:4, 2:13] = 255    # Top horizontal
        e_template[9:11, 2:8] = 255    # Middle horizontal  
        e_template[16:18, 2:13] = 255  # Bottom horizontal
        e_template[2:18, 2:4] = 255    # Vertical line
        self.templates['e_major'] = e_template
        
        # Template for simple line (minor graduations) 
        line_template = np.zeros((15, 3), dtype=np.uint8)
        line_template[:, 1] = 255  # Vertical line
        self.templates['line_minor'] = line_template
        
        # Template for thick line (intermediate graduations)
        thick_line = np.zeros((18, 5), dtype=np.uint8) 
        thick_line[:, 1:4] = 255
        self.templates['line_thick'] = thick_line
        
        # Add additional sophisticated templates for better detection
        
        # Template for number markings (common on stadia rods)
        number_template = np.zeros((24, 12), dtype=np.uint8)
        # Create a basic rectangular pattern for numbers
        number_template[4:20, 2:10] = 128  # Gray background for numbers
        number_template[6:8, 3:9] = 255    # Top line
        number_template[14:16, 3:9] = 255  # Bottom line
        self.templates['number_marking'] = number_template
        
        # Template for L-shaped markings (sometimes used for major graduations)
        l_template = np.zeros((16, 12), dtype=np.uint8)
        l_template[2:14, 2:4] = 255   # Vertical part of L
        l_template[12:14, 4:10] = 255 # Horizontal part of L
        self.templates['l_major'] = l_template
        
        self.logger.info(f"Created {len(self.templates)} sophisticated stadia rod templates")
        
        # Store template metadata for advanced matching (use config thresholds if available)
        default_thresholds = {
            'e_major': 0.6,
            'line_minor': 0.7,
            'line_thick': 0.65,
            'number_marking': 0.5,
            'l_major': 0.6
        }
        
        self.template_metadata = {}
        for template_name, default_threshold in default_thresholds.items():
            configured_threshold = self.template_thresholds.get(template_name, default_threshold)
            template_type = 'major' if 'major' in template_name else ('minor' if 'minor' in template_name else 'intermediate')
            priority = 1 if template_type == 'major' else (3 if template_type == 'minor' else 2)
            
            self.template_metadata[template_name] = {
                'type': template_type,
                'threshold': configured_threshold,
                'priority': priority
            }
    
    def _get_templates(self):
        """
        Get templates based on configured template source.
        
        Returns:
            list: List of template images
        """
        templates = []
        
        # Get local templates
        if self.template_source in ['local', 'both']:
            local_templates = list(self.templates.values()) if self.templates else []
            templates.extend(local_templates)
            if local_templates:
                self.logger.debug(f"Retrieved {len(local_templates)} local templates")
        
        # Get templates from template manager
        if self.template_source in ['manager', 'both'] and self.template_manager:
            try:
                manager_templates = self.template_manager.get_templates() if hasattr(self.template_manager, 'get_templates') else []
                if manager_templates:
                    templates.extend(manager_templates)
                    self.logger.debug(f"Retrieved {len(manager_templates)} templates from manager")
            except Exception as e:
                self.logger.warning(f"Failed to get templates from manager: {e}")
        
        # Apply max_templates limit
        if len(templates) > self.max_templates:
            templates = templates[:self.max_templates]
            self.logger.debug(f"Limited templates to {self.max_templates}")
        
        self.logger.debug(f"Total templates available: {len(templates)}")
        return templates
    
    def preprocess_scale_region(self, scale_region):
        """
        Preprocess the scale region for better template matching
        Based on advanced preprocessing from StadiaRodReader
        """
        # Convert to grayscale if needed
        if len(scale_region.shape) == 3:
            gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = scale_region.copy()
        
        # Apply mean shift filtering to reduce noise while preserving edges (configurable)
        if len(scale_region.shape) == 3 and self.use_mean_shift:
            try:
                filtered = cv2.pyrMeanShiftFiltering(scale_region, 20, 30)
                gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
                self.logger.debug("Applied mean shift filtering for noise reduction")
            except Exception as e:
                self.logger.warning(f"Mean shift filtering failed: {e}, using original")
                gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better contrast (configurable)
        if self.use_adaptive_threshold:
            try:
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                self.logger.debug("Applied adaptive thresholding")
            except Exception as e:
                self.logger.warning(f"Adaptive thresholding failed: {e}, using simple threshold")
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Simple thresholding as fallback
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.logger.debug("Used simple OTSU thresholding")
        
        # Apply morphological operations to clean up (configurable)
        if self.use_morphological_cleaning:
            kernel = np.ones((2,2), np.uint8)
            try:
                cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                self.logger.debug("Applied morphological cleaning")
            except Exception as e:
                self.logger.warning(f"Morphological operations failed: {e}")
                cleaned = binary
        else:
            cleaned = binary
        
        return gray, binary, cleaned
    
    def apply_non_maximum_suppression(self, matches, template_size, nms_threshold=None):
        """
        Remove overlapping detections using Non-Maximum Suppression
        Adapted from StadiaRodReader implementation
        """
        if not matches or not self.use_nms:
            return matches
        
        # Use configured NMS threshold if not provided
        if nms_threshold is None:
            nms_threshold = self.nms_threshold
        
        # Filter matches by confidence threshold first
        confidence_filtered = [m for m in matches if m['confidence'] >= self.nms_confidence_threshold]
        
        if not confidence_filtered:
            return []
        
        # Convert to format suitable for NMS
        boxes = []
        scores = []
        
        for match in confidence_filtered:
            x, y = match['position']
            w, h = template_size
            boxes.append([x, y, x+w, y+h])
            scores.append(match['confidence'])
        
        if not boxes:
            return []
        
        try:
            boxes = np.array(boxes, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            
            # Apply OpenCV's NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), self.nms_confidence_threshold, nms_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                filtered_matches = [matches[i] for i in indices]
                self.logger.debug(f"NMS reduced {len(matches)} matches to {len(filtered_matches)}")
                return filtered_matches
        except Exception as e:
            self.logger.warning(f"NMS failed: {e}, returning original matches")
        
        return matches
    
    def detect_waterline(self, scale_region):
        """
        Detect water line using template matching approach.
        
        Args:
            scale_region: Scale region image (BGR)
            
        Returns:
            int: Y-coordinate of detected water line (local to scale region)
        """
        if scale_region is None or scale_region.size == 0:
            return None
        
        try:
            # Get templates based on configuration
            templates = self._get_templates()
            
            if not templates:
                self.logger.debug("No templates available, using template-free detection")
                return self._detect_without_templates(scale_region)
            
            # Create mask of scale markings using templates
            marking_mask = self._create_marking_mask(scale_region, templates)
            
            # Apply mask to suppress markings
            masked_region = self._apply_marking_mask(scale_region, marking_mask)
            
            # Detect water interface in masked region
            waterline_y = self._detect_water_interface(masked_region, scale_region)
            
            if waterline_y is not None:
                self.logger.debug(f"Template matching detected waterline at Y={waterline_y}")
                return waterline_y
            else:
                self.logger.debug("Template matching failed to detect waterline")
                return None
                
        except Exception as e:
            self.logger.error(f"Template matching detection failed: {e}")
            return None
    
    def _create_marking_mask(self, scale_region, templates):
        """
        Create a mask of scale markings using advanced template matching.
        
        Args:
            scale_region: Scale region image
            templates: List of template images
            
        Returns:
            np.ndarray: Binary mask (0=marking, 255=clear)
        """
        # Apply advanced preprocessing
        gray, binary, cleaned = self.preprocess_scale_region(scale_region)
        height, width = gray.shape
        
        # Start with all areas available (white = available, black = marked)
        mask = np.ones_like(gray, dtype=np.uint8) * 255
        
        total_marking_count = 0
        all_detections = []
        
        # Use local templates with metadata for sophisticated matching
        local_templates = self.templates if self.template_source in ['local', 'both'] else {}
        
        for template_name, template in local_templates.items():
            try:
                # Get template metadata for adaptive thresholds
                metadata = getattr(self, 'template_metadata', {}).get(template_name, {})
                threshold = metadata.get('threshold', self.match_threshold)
                
                # Ensure template is grayscale
                if len(template.shape) == 3:
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                # Skip if template is too large for the region
                if template.shape[0] >= height * 0.8 or template.shape[1] >= width * 0.8:
                    continue
                
                # Perform template matching on both cleaned and original gray images
                matches_cleaned = []
                matches_gray = []
                
                # Match on cleaned image (better for high contrast markings)
                try:
                    result = cv2.matchTemplate(cleaned, template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= threshold)
                    
                    for pt in zip(*locations[::-1]):
                        confidence = result[pt[1], pt[0]]
                        matches_cleaned.append({
                            'position': pt,
                            'confidence': confidence,
                            'template': template_name,
                            'source': 'cleaned'
                        })
                except Exception as e:
                    self.logger.warning(f"Template matching on cleaned image failed for {template_name}: {e}")
                
                # Match on gray image (better for subtle markings)
                try:
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= threshold * 0.8)  # Slightly lower threshold for gray
                    
                    for pt in zip(*locations[::-1]):
                        confidence = result[pt[1], pt[0]]
                        matches_gray.append({
                            'position': pt,
                            'confidence': confidence,
                            'template': template_name,
                            'source': 'gray'
                        })
                except Exception as e:
                    self.logger.warning(f"Template matching on gray image failed for {template_name}: {e}")
                
                # Combine matches and apply NMS
                all_matches = matches_cleaned + matches_gray
                if all_matches:
                    template_size = template.shape[::-1]  # w,h
                    filtered_matches = self.apply_non_maximum_suppression(all_matches, template_size)
                    all_detections.extend(filtered_matches)
                    
                    # Mark matching areas in the mask
                    for match in filtered_matches:
                        x, y = match['position']
                        h, w = template.shape
                        
                        # Create a buffer around the match to ensure complete suppression
                        buffer = 3  # Slightly larger buffer for better suppression
                        x1 = max(0, x - buffer)
                        y1 = max(0, y - buffer)
                        x2 = min(width, x + w + buffer)
                        y2 = min(height, y + h + buffer)
                        
                        # Mark this area as a marking (black = marking)
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
                        total_marking_count += 1
                
            except Exception as e:
                self.logger.warning(f"Error matching template {template_name}: {e}")
                continue
        
        self.logger.debug(f"Advanced template matching found {total_marking_count} marking instances from {len(all_detections)} total detections")
        
        # Store detections for potential debugging/visualization
        if hasattr(self, 'debug_patterns') and self.debug_patterns:
            self.last_detections = all_detections
        
        return mask
    
    def _apply_marking_mask(self, scale_region, marking_mask):
        """
        Apply the marking mask to suppress detected markings.
        
        Args:
            scale_region: Original scale region
            marking_mask: Mask with markings marked as 0
            
        Returns:
            np.ndarray: Masked scale region
        """
        # Apply mask to original image
        masked_region = cv2.bitwise_and(scale_region, scale_region, mask=marking_mask)
        
        # Fill masked areas with background color to avoid edge artifacts
        background_color = self._estimate_background_color(scale_region, marking_mask)
        
        # Create inverse mask for filling
        inverse_mask = cv2.bitwise_not(marking_mask)
        
        # Fill marked areas with background color
        masked_region[inverse_mask > 0] = background_color
        
        return masked_region
    
    def _estimate_background_color(self, scale_region, marking_mask):
        """Estimate the background color of the scale."""
        # Use the most common color in unmasked areas
        available_pixels = scale_region[marking_mask > 0]
        
        if len(available_pixels) == 0:
            return [128, 128, 128]  # Default gray
        
        # Calculate mean color
        mean_color = np.mean(available_pixels, axis=0)
        return mean_color.astype(np.uint8)
    
    def _detect_water_interface(self, masked_region, original_region):
        """
        Detect water interface in the marking-suppressed region.
        
        Args:
            masked_region: Region with markings suppressed
            original_region: Original scale region for reference
            
        Returns:
            int: Y-coordinate of water interface
        """
        gray = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Use multiple approaches to find the water interface
        candidates = []
        
        # Approach 1: Horizontal gradient analysis
        gradient_candidate = self._detect_by_horizontal_gradient(gray)
        if gradient_candidate is not None:
            candidates.append(('gradient', gradient_candidate, 0.4))
        
        # Approach 2: Edge detection on masked region
        edge_candidate = self._detect_by_edge_analysis(gray)
        if edge_candidate is not None:
            candidates.append(('edge', edge_candidate, 0.3))
        
        # Approach 3: Intensity change detection
        intensity_candidate = self._detect_by_intensity_change(gray)
        if intensity_candidate is not None:
            candidates.append(('intensity', intensity_candidate, 0.3))
        
        # Select best candidate
        return self._select_best_candidate(candidates, height, width)
    
    def _detect_by_horizontal_gradient(self, gray):
        """Detect water interface using horizontal gradient analysis."""
        height, width = gray.shape
        
        best_y = None
        best_score = 0
        
        for y in range(5, height - 5):
            # Extract horizontal line
            line = gray[y, :]
            
            # Calculate gradient
            gradient = np.gradient(line)
            gradient_strength = np.sum(np.abs(gradient))
            
            # Look for strong horizontal features
            if gradient_strength > best_score:
                # Verify this is a continuous horizontal feature
                continuity = self._measure_horizontal_continuity(gray, y)
                
                if continuity > 0.6:  # At least 60% continuous
                    best_score = gradient_strength * continuity
                    best_y = y
        
        return best_y
    
    def _detect_by_edge_analysis(self, gray):
        """Detect water interface using edge detection."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find horizontal edges
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find the strongest horizontal edge
        height = horizontal_edges.shape[0]
        best_y = None
        best_score = 0
        
        for y in range(height):
            row_sum = np.sum(horizontal_edges[y, :])
            if row_sum > best_score and row_sum > 100:  # Minimum edge strength
                best_score = row_sum
                best_y = y
        
        return best_y
    
    def _detect_by_intensity_change(self, gray):
        """Detect water interface by looking for intensity changes."""
        height, width = gray.shape
        
        best_y = None
        best_score = 0
        
        # Look for significant brightness changes
        for y in range(10, height - 10):
            above_mean = np.mean(gray[y-5:y, :])
            below_mean = np.mean(gray[y:y+5, :])
            
            intensity_diff = abs(above_mean - below_mean)
            
            if intensity_diff > best_score and intensity_diff > 10:  # Minimum difference
                # Check if this spans most of the width
                width_coverage = self._measure_width_coverage(gray, y, intensity_diff / 2)
                
                if width_coverage > self.min_water_interface_width:
                    best_score = intensity_diff * width_coverage
                    best_y = y
        
        return best_y
    
    def _measure_horizontal_continuity(self, gray, y):
        """Measure how continuous a horizontal feature is at a given Y."""
        if y < 0 or y >= gray.shape[0]:
            return 0
        
        line = gray[y, :]
        width = len(line)
        
        # Look for continuous regions with similar intensity
        gradient = np.abs(np.gradient(line))
        low_gradient_pixels = np.sum(gradient < 10)  # Stable intensity regions
        
        return low_gradient_pixels / width
    
    def _measure_width_coverage(self, gray, y, threshold):
        """Measure what fraction of width shows consistent change at Y."""
        if y < 5 or y >= gray.shape[0] - 5:
            return 0
        
        width = gray.shape[1]
        consistent_pixels = 0
        
        for x in range(width):
            above_val = np.mean(gray[y-3:y, x])
            below_val = np.mean(gray[y:y+3, x])
            
            if abs(above_val - below_val) > threshold:
                consistent_pixels += 1
        
        return consistent_pixels / width
    
    def _select_best_candidate(self, candidates, height, width):
        """Select the best water interface candidate."""
        if not candidates:
            return None
        
        # Weight and combine candidates
        weighted_sum = 0
        total_weight = 0
        
        for method, y, weight in candidates:
            weighted_sum += y * weight
            total_weight += weight
        
        if total_weight > 0:
            final_y = int(weighted_sum / total_weight)
            
            # Bounds checking
            final_y = max(5, min(height - 5, final_y))
            
            self.logger.debug(f"Selected candidate Y={final_y} from {len(candidates)} candidates")
            return final_y
        
        return None
    
    def _detect_without_templates(self, scale_region):
        """
        Fallback detection when no templates are available.
        Uses basic horizontal feature detection.
        """
        self.logger.debug("Using template-free detection fallback")
        
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        
        # Simple horizontal gradient detection
        return self._detect_by_horizontal_gradient(gray)
    
    def get_detection_info(self):
        """Get information about the template matching detector."""
        local_template_count = len(self.templates)
        manager_template_count = 0
        
        if self.template_manager and hasattr(self.template_manager, 'get_template_count'):
            try:
                manager_template_count = self.template_manager.get_template_count()
            except:
                manager_template_count = 0
        
        total_templates = len(self._get_templates())
        
        return {
            'method': 'template_matching',
            'template_source': self.template_source,
            'match_threshold': self.match_threshold,
            'max_templates': self.max_templates,
            'local_templates': local_template_count,
            'manager_templates': manager_template_count,
            'total_available_templates': total_templates,
            'templates_loaded': total_templates > 0,
            'use_default_templates': self.use_default_templates
        }