"""
Hybrid Water Level Detector

Combines both standard and pattern-aware detection methods with intelligent
fallback and comparison capabilities.
"""

import logging
from pathlib import Path

class HybridDetector:
    """
    Hybrid detector that can use both standard and pattern-aware methods.
    
    Features:
    - Runs both detection systems simultaneously
    - Compares results and selects the most reliable
    - Provides fallback protection
    - Detailed logging of both systems
    """
    
    def __init__(self, config, pixels_per_cm, enhanced_calibration_data=None, calibration_manager=None):
        """Initialize hybrid detector with both systems."""
        self.config = config
        self.pixels_per_cm = pixels_per_cm
        self.enhanced_calibration_data = enhanced_calibration_data
        self.calibration_manager = calibration_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize both detectors
        self._initialize_detectors()
        
        # Hybrid configuration
        self.hybrid_config = config.get('pattern_processing', {})
        self.comparison_enabled = True
        self.prefer_pattern = self.hybrid_config.get('prefer_pattern_aware', True)
        
        self.logger.info("Hybrid detector initialized with both standard and pattern-aware methods")
    
    def _initialize_detectors(self):
        """Initialize both standard and pattern-aware detectors."""
        try:
            # Initialize standard detector
            from src.water_level_detector import WaterLevelDetector
            self.standard_detector = WaterLevelDetector(
                self.config, self.pixels_per_cm, 
                self.enhanced_calibration_data, self.calibration_manager
            )
            self.logger.info("Standard detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize standard detector: {e}")
            self.standard_detector = None
        
        try:
            # Initialize pattern-aware detector
            from src_pattern_aware.pattern_water_detector import PatternWaterDetector
            self.pattern_detector = PatternWaterDetector(
                self.config, self.pixels_per_cm,
                self.enhanced_calibration_data, self.calibration_manager
            )
            self.logger.info("Pattern-aware detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pattern-aware detector: {e}")
            self.pattern_detector = None
    
    def process_image(self, image_path):
        """
        Process image using both detection systems and return best result.
        
        Args:
            image_path (str): Path to image to process
            
        Returns:
            dict: Processing result with comparison data
        """
        self.logger.info(f"Hybrid processing: {Path(image_path).name}")
        
        # Results storage
        standard_result = None
        pattern_result = None
        
        # Run standard detection
        if self.standard_detector:
            try:
                self.logger.info("Running standard detection...")
                standard_result = self.standard_detector.process_image(image_path)
                if standard_result:
                    self.logger.info(f"Standard detection: {standard_result['water_level_cm']:.1f}cm "
                                   f"(confidence: {standard_result.get('confidence', 0):.3f})")
                else:
                    self.logger.warning("Standard detection returned no result")
            except Exception as e:
                self.logger.error(f"Standard detection failed: {e}")
        
        # Run pattern-aware detection
        if self.pattern_detector:
            try:
                self.logger.info("Running pattern-aware detection...")
                pattern_result = self.pattern_detector.process_image(image_path)
                if pattern_result:
                    self.logger.info(f"Pattern-aware detection: {pattern_result['water_level_cm']:.1f}cm "
                                   f"(confidence: {pattern_result.get('confidence', 0):.3f})")
                else:
                    self.logger.warning("Pattern-aware detection returned no result")
            except Exception as e:
                self.logger.error(f"Pattern-aware detection failed: {e}")
        
        # Select best result
        selected_result = self._select_best_result(standard_result, pattern_result, image_path)
        
        # Add hybrid metadata
        if selected_result:
            selected_result['hybrid_data'] = {
                'detection_method': 'hybrid',
                'standard_available': standard_result is not None,
                'pattern_available': pattern_result is not None,
                'standard_result': standard_result['water_level_cm'] if standard_result else None,
                'pattern_result': pattern_result['water_level_cm'] if pattern_result else None,
                'selection_reason': self._get_selection_reason(standard_result, pattern_result)
            }
        
        return selected_result
    
    def _select_best_result(self, standard_result, pattern_result, image_path):
        """
        Select the best result from both detection systems.
        
        Selection criteria:
        1. If only one system succeeded, use that result
        2. If both succeeded, use confidence and preference settings
        3. If neither succeeded, return None
        """
        if not standard_result and not pattern_result:
            self.logger.error("Both detection systems failed")
            return None
        
        if not standard_result and pattern_result:
            self.logger.info("Selected pattern-aware result (standard failed)")
            return pattern_result
        
        if standard_result and not pattern_result:
            self.logger.info("Selected standard result (pattern-aware failed)")
            return standard_result
        
        # Both systems succeeded - compare results
        return self._compare_results(standard_result, pattern_result)
    
    def _compare_results(self, standard_result, pattern_result):
        """
        Compare results from both systems and select the best one.
        
        Comparison factors:
        1. Confidence scores
        2. Deviation from expected range
        3. User preference settings
        4. Result consistency
        """
        standard_confidence = standard_result.get('confidence', 0)
        pattern_confidence = pattern_result.get('confidence', 0)
        
        standard_level = standard_result['water_level_cm']
        pattern_level = pattern_result['water_level_cm']
        
        # Calculate difference between results
        level_difference = abs(standard_level - pattern_level)
        
        self.logger.info(f"Result comparison:")
        self.logger.info(f"  Standard: {standard_level:.1f}cm (confidence: {standard_confidence:.3f})")
        self.logger.info(f"  Pattern:  {pattern_level:.1f}cm (confidence: {pattern_confidence:.3f})")
        self.logger.info(f"  Difference: {level_difference:.1f}cm")
        
        # Selection logic
        if level_difference < 2.0:  # Results are very close (< 2cm)
            # Use preference setting
            if self.prefer_pattern:
                self.logger.info("Selected pattern-aware result (close results, preference)")
                return pattern_result
            else:
                self.logger.info("Selected standard result (close results, preference)")
                return standard_result
        
        elif level_difference < 5.0:  # Results are moderately close (< 5cm)
            # Use confidence scores
            if pattern_confidence > standard_confidence + 0.1:  # Pattern significantly more confident
                self.logger.info("Selected pattern-aware result (higher confidence)")
                return pattern_result
            else:
                self.logger.info("Selected standard result (higher/equal confidence)")
                return standard_result
        
        else:  # Results are very different (>= 5cm)
            # This indicates a problem - log warning and use higher confidence
            self.logger.warning(f"Large discrepancy between detection methods ({level_difference:.1f}cm)")
            
            if pattern_confidence > standard_confidence:
                self.logger.warning("Selected pattern-aware result (higher confidence despite discrepancy)")
                return pattern_result
            else:
                self.logger.warning("Selected standard result (higher confidence despite discrepancy)")
                return standard_result
    
    def _get_selection_reason(self, standard_result, pattern_result):
        """Get human-readable reason for result selection."""
        if not standard_result and not pattern_result:
            return "Both methods failed"
        elif not standard_result:
            return "Standard method failed"
        elif not pattern_result:
            return "Pattern-aware method failed"
        else:
            standard_level = standard_result['water_level_cm']
            pattern_level = pattern_result['water_level_cm']
            difference = abs(standard_level - pattern_level)
            
            if difference < 2.0:
                return "Close results - used preference setting"
            elif difference < 5.0:
                return "Moderate difference - used confidence scores"
            else:
                return "Large discrepancy - used highest confidence"
    
    def get_detection_info(self):
        """Get information about both detection systems."""
        info = {
            'detection_mode': 'hybrid',
            'comparison_enabled': self.comparison_enabled,
            'prefer_pattern': self.prefer_pattern,
            'standard_detector_available': self.standard_detector is not None,
            'pattern_detector_available': self.pattern_detector is not None
        }
        
        # Add standard detector info
        if self.standard_detector and hasattr(self.standard_detector, 'get_detection_info'):
            info['standard_detector_info'] = self.standard_detector.get_detection_info()
        
        # Add pattern detector info
        if self.pattern_detector and hasattr(self.pattern_detector, 'get_detection_info'):
            info['pattern_detector_info'] = self.pattern_detector.get_detection_info()
        
        return info
    
    def extract_and_save_templates(self, calibration_image_path):
        """Extract templates using pattern-aware detector if available."""
        if self.pattern_detector:
            return self.pattern_detector.extract_and_save_templates(calibration_image_path)
        else:
            self.logger.warning("Pattern-aware detector not available for template extraction")
            return 0