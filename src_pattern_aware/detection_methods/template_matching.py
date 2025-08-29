"""
Template Matching Water Level Detector

Uses template matching to identify and suppress scale markings, then
detects water interfaces in the remaining regions.
"""

import cv2
import numpy as np
import logging
from pathlib import Path

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
        
        # Detection parameters
        self.min_water_interface_width = 0.6  # Minimum width for water interface (% of scale width)
        self.horizontal_emphasis = True       # Emphasize horizontal features
        
        self.logger.info(f"Template matching detector initialized (threshold: {self.match_threshold})")
    
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
            # Get templates from template manager
            templates = self.template_manager.get_templates()
            
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
        Create a mask of scale markings using template matching.
        
        Args:
            scale_region: Scale region image
            templates: List of template images
            
        Returns:
            np.ndarray: Binary mask (0=marking, 255=clear)
        """
        gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Start with all areas available (white = available, black = marked)
        mask = np.ones_like(gray, dtype=np.uint8) * 255
        
        marking_count = 0
        
        for i, template in enumerate(templates):
            if i >= self.max_templates:
                break
            
            try:
                # Ensure template is grayscale
                if len(template.shape) == 3:
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                # Skip if template is too large for the region
                if template.shape[0] >= height * 0.8 or template.shape[1] >= width * 0.8:
                    continue
                
                # Perform template matching
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                
                # Find matches above threshold
                locations = np.where(result >= self.match_threshold)
                
                # Mark matching areas in the mask
                for pt in zip(*locations[::-1]):  # Switch x,y coordinates
                    x, y = pt
                    h, w = template.shape
                    
                    # Create a buffer around the match to ensure complete suppression
                    buffer = 2
                    x1 = max(0, x - buffer)
                    y1 = max(0, y - buffer)
                    x2 = min(width, x + w + buffer)
                    y2 = min(height, y + h + buffer)
                    
                    # Mark this area as a marking (black = marking)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
                    marking_count += 1
                
            except Exception as e:
                self.logger.warning(f"Error matching template {i}: {e}")
                continue
        
        self.logger.debug(f"Template matching found {marking_count} marking instances")
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
        template_count = self.template_manager.get_template_count()
        
        return {
            'method': 'template_matching',
            'match_threshold': self.match_threshold,
            'max_templates': self.max_templates,
            'available_templates': template_count,
            'templates_loaded': template_count > 0
        }