"""
Morphological Water Level Detector

Uses morphological operations to detect horizontal water interfaces
while suppressing vertical scale markings.
"""

import cv2
import numpy as np
import logging

class MorphologicalDetector:
    """
    Water level detector using morphological operations.
    
    Process:
    1. Create horizontal and vertical morphological kernels
    2. Extract horizontal features (potential water interfaces)
    3. Suppress vertical features (scale markings)
    4. Find the strongest horizontal interface
    """
    
    def __init__(self, config):
        """
        Initialize morphological detector.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Morphological configuration
        morph_config = config.get('detection', {}).get('pattern_aware', {}).get('morphological', {})
        
        # Kernel sizes for morphological operations
        h_kernel_size = morph_config.get('horizontal_kernel_size', [40, 1])
        v_kernel_size = morph_config.get('vertical_kernel_size', [1, 40])
        
        # Create morphological kernels
        self.horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(h_kernel_size))
        self.vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(v_kernel_size))
        
        # Detection parameters
        self.min_interface_strength = 100    # Minimum strength for water interface
        self.continuity_threshold = 0.6      # Minimum continuity across width
        
        self.logger.info(f"Morphological detector initialized "
                        f"(h_kernel: {h_kernel_size}, v_kernel: {v_kernel_size})")
    
    def detect_waterline(self, scale_region):
        """
        Detect water line using morphological operations.
        
        Args:
            scale_region: Scale region image (BGR)
            
        Returns:
            int: Y-coordinate of detected water line (local to scale region)
        """
        if scale_region is None or scale_region.size == 0:
            return None
        
        try:
            gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
            
            # Extract horizontal and vertical features
            horizontal_features = self._extract_horizontal_features(gray)
            vertical_features = self._extract_vertical_features(gray)
            
            # Suppress vertical features to isolate water interfaces
            water_interface_image = self._isolate_water_interface(
                horizontal_features, vertical_features
            )
            
            # Find the best water interface position
            waterline_y = self._find_best_interface(water_interface_image, gray)
            
            if waterline_y is not None:
                self.logger.debug(f"Morphological detection found waterline at Y={waterline_y}")
                return waterline_y
            else:
                self.logger.debug("Morphological detection found no suitable waterline")
                return None
                
        except Exception as e:
            self.logger.error(f"Morphological detection failed: {e}")
            return None
    
    def _extract_horizontal_features(self, gray):
        """
        Extract horizontal features that could be water interfaces.
        
        Args:
            gray: Grayscale scale region
            
        Returns:
            np.ndarray: Image with horizontal features enhanced
        """
        # Morphological opening with horizontal kernel to extract horizontal features
        horizontal_features = cv2.morphologyEx(gray, cv2.MORPH_OPEN, self.horizontal_kernel)
        
        # Enhance the features using tophat operation
        horizontal_tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.horizontal_kernel)
        
        # Combine both approaches
        combined = cv2.add(horizontal_features, horizontal_tophat)
        
        return combined
    
    def _extract_vertical_features(self, gray):
        """
        Extract vertical features (scale markings) to suppress them.
        
        Args:
            gray: Grayscale scale region
            
        Returns:
            np.ndarray: Image with vertical features enhanced
        """
        # Morphological opening with vertical kernel to extract vertical features
        vertical_features = cv2.morphologyEx(gray, cv2.MORPH_OPEN, self.vertical_kernel)
        
        # Enhance vertical markings using tophat
        vertical_tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.vertical_kernel)
        
        # Combine both approaches
        combined = cv2.add(vertical_features, vertical_tophat)
        
        return combined
    
    def _isolate_water_interface(self, horizontal_features, vertical_features):
        """
        Isolate water interface by suppressing vertical features.
        
        Args:
            horizontal_features: Horizontal features image
            vertical_features: Vertical features image
            
        Returns:
            np.ndarray: Image with water interfaces isolated
        """
        # Subtract vertical features from horizontal features
        # This suppresses scale markings while preserving water interfaces
        water_interface = cv2.subtract(horizontal_features, vertical_features)
        
        # Apply morphological closing to connect broken interfaces
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        water_interface = cv2.morphologyEx(water_interface, cv2.MORPH_CLOSE, closing_kernel)
        
        # Remove noise with opening
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        water_interface = cv2.morphologyEx(water_interface, cv2.MORPH_OPEN, noise_kernel)
        
        return water_interface
    
    def _find_best_interface(self, water_interface_image, original_gray):
        """
        Find the best water interface position.
        
        Args:
            water_interface_image: Processed image with water interfaces
            original_gray: Original grayscale image for validation
            
        Returns:
            int: Y-coordinate of best water interface
        """
        height, width = water_interface_image.shape
        
        best_y = None
        best_score = 0
        
        # Analyze each horizontal line
        for y in range(5, height - 5):
            # Calculate horizontal strength at this line
            line_strength = np.sum(water_interface_image[y, :])
            
            if line_strength < self.min_interface_strength:
                continue
            
            # Measure continuity across the width
            continuity = self._measure_line_continuity(water_interface_image[y, :])
            
            if continuity < self.continuity_threshold:
                continue
            
            # Validate against original image
            validation_score = self._validate_interface_position(original_gray, y)
            
            # Calculate composite score
            composite_score = line_strength * continuity * validation_score
            
            if composite_score > best_score:
                best_score = composite_score
                best_y = y
        
        return best_y
    
    def _measure_line_continuity(self, line):
        """
        Measure how continuous a horizontal line is.
        
        Args:
            line: 1D array representing a horizontal line
            
        Returns:
            float: Continuity score (0-1)
        """
        if len(line) == 0:
            return 0
        
        # Count non-zero pixels (active pixels)
        active_pixels = np.count_nonzero(line)
        
        # Calculate continuity as ratio of active pixels
        continuity = active_pixels / len(line)
        
        return continuity
    
    def _validate_interface_position(self, original_gray, y):
        """
        Validate a potential water interface position against the original image.
        
        Args:
            original_gray: Original grayscale image
            y: Y-coordinate to validate
            
        Returns:
            float: Validation score (0-1)
        """
        height, width = original_gray.shape
        
        if y < 5 or y >= height - 5:
            return 0
        
        # Check for intensity difference above and below
        above_region = original_gray[max(0, y-5):y, :]
        below_region = original_gray[y:min(height, y+5), :]
        
        if above_region.size == 0 or below_region.size == 0:
            return 0
        
        above_mean = np.mean(above_region)
        below_mean = np.mean(below_region)
        
        # Water interfaces typically show intensity differences
        intensity_difference = abs(above_mean - below_mean)
        
        # Normalize intensity difference (0-1 scale)
        normalized_difference = min(intensity_difference / 50.0, 1.0)
        
        # Check for horizontal consistency
        horizontal_consistency = self._check_horizontal_consistency(original_gray, y)
        
        # Combine factors
        validation_score = normalized_difference * horizontal_consistency
        
        return validation_score
    
    def _check_horizontal_consistency(self, gray, y):
        """
        Check if a horizontal line shows consistent characteristics across its width.
        
        Args:
            gray: Grayscale image
            y: Y-coordinate of line to check
            
        Returns:
            float: Consistency score (0-1)
        """
        if y < 3 or y >= gray.shape[0] - 3:
            return 0
        
        height, width = gray.shape
        
        # Sample points along the line
        sample_points = min(10, width // 5)  # Sample every 5 pixels, max 10 points
        sample_indices = np.linspace(0, width-1, sample_points, dtype=int)
        
        intensity_differences = []
        
        for x in sample_indices:
            above_val = np.mean(gray[y-3:y, x])
            below_val = np.mean(gray[y:y+3, x])
            intensity_differences.append(abs(above_val - below_val))
        
        if not intensity_differences:
            return 0
        
        # Check consistency of intensity differences
        mean_diff = np.mean(intensity_differences)
        std_diff = np.std(intensity_differences)
        
        if mean_diff == 0:
            return 0
        
        # Consistency is high when standard deviation is low relative to mean
        consistency = max(0, 1.0 - (std_diff / max(mean_diff, 1)))
        
        return consistency
    
    def get_detection_info(self):
        """Get information about the morphological detector."""
        return {
            'method': 'morphological',
            'horizontal_kernel': self.horizontal_kernel.shape,
            'vertical_kernel': self.vertical_kernel.shape,
            'min_interface_strength': self.min_interface_strength,
            'continuity_threshold': self.continuity_threshold
        }