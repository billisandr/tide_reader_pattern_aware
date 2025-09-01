"""
Integrated Pattern Detector

Combines multiple pattern detection methods for robust water level detection.
Prioritizes E-pattern sequential detection when available.
"""

import cv2
import logging

# Import E-pattern detector
try:
    from .e_pattern_detector import EPatternDetector
except ImportError:
    from e_pattern_detector import EPatternDetector

class IntegratedPatternDetector:
    """
    Integrated pattern detector that combines multiple detection methods.
    
    This is a stub implementation that provides basic functionality
    while the full integrated system is being developed.
    """
    
    def __init__(self, config, detection_methods, pattern_classifier, debug_viz=None, calibration_data=None):
        """
        Initialize integrated pattern detector.
        
        Args:
            config: System configuration
            detection_methods: Dictionary of detection method instances
            pattern_classifier: Pattern classifier instance
            debug_viz: Debug visualizer instance
            calibration_data: Enhanced calibration data containing pixels_per_cm
        """
        self.config = config
        self.detection_methods = detection_methods
        self.pattern_classifier = pattern_classifier
        self.debug_viz = debug_viz
        self.calibration_data = calibration_data
        self.logger = logging.getLogger(__name__)
        
        # Initialize E-pattern detector if calibration data is available
        self.e_pattern_detector = None
        if calibration_data is not None:
            try:
                self.e_pattern_detector = EPatternDetector(config, calibration_data)
                pixels_per_cm = calibration_data.get('pixels_per_cm', 'unknown')
                self.logger.info(f"E-pattern sequential detector initialized successfully with pixels_per_cm: {pixels_per_cm}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize E-pattern detector: {e}")
        else:
            self.logger.warning("calibration_data not provided, E-pattern detector unavailable")
        
        self.logger.info(f"Integrated pattern detector initialized with {len(self.detection_methods)} methods"
                        f"{' + E-pattern detector' if self.e_pattern_detector else ''}")
    
    def detect_waterline(self, scale_region, image_path=None):
        """
        Detect water line using integrated pattern methods with E-pattern priority.
        
        Args:
            scale_region: Scale region image
            image_path: Original image path (for debugging)
            
        Returns:
            int: Y-coordinate of detected water line, or None if detection fails
        """
        self.logger.info("Starting integrated pattern detection")
        
        # Debug: Save preprocessed scale region for pattern analysis
        if self.debug_viz:
            self.debug_viz.save_debug_image(
                scale_region, 'pattern_preprocessing',
                info_text=f"Scale region for pattern analysis: {scale_region.shape[1]}x{scale_region.shape[0]}"
            )
        
        method_results = {}
        
        # PRIORITY 1: Try E-pattern sequential detector first
        if self.e_pattern_detector:
            try:
                self.logger.info("Trying E-pattern sequential detection (priority method)")
                result = self.e_pattern_detector.detect_waterline(scale_region, image_path)
                method_results['e_pattern_sequential'] = result
                
                if result is not None:
                    self.logger.info(f"E-pattern detection successful: Y={result}")
                    
                    # Debug: Save E-pattern result
                    if self.debug_viz:
                        debug_image = scale_region.copy()
                        if len(debug_image.shape) == 2:
                            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
                        
                        # Draw detected water line
                        cv2.line(debug_image, (0, result), (debug_image.shape[1], result), (0, 0, 255), 3)
                        cv2.putText(debug_image, f"E-Pattern Sequential: Y={result}", (10, result-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        self.debug_viz.save_debug_image(
                            debug_image, 'pattern_e_pattern_result',
                            info_text=f"E-pattern sequential detection: water line at Y={result}"
                        )
                    
                    # Return immediately on successful E-pattern detection
                    return result
                else:
                    self.logger.warning("E-pattern detection failed, falling back to other methods")
                    
            except Exception as e:
                self.logger.error(f"E-pattern detection failed with error: {e}")
                method_results['e_pattern_sequential'] = None
        
        # FALLBACK: Try other pattern methods
        self.logger.info("Trying fallback pattern detection methods")
        
        # Try methods in order of preference
        method_order = ['template_matching', 'morphological', 'frequency', 'contour', 'lsd']
        
        for method_name in method_order:
            if method_name in self.detection_methods:
                method = self.detection_methods[method_name]
                if hasattr(method, 'detect_waterline'):
                    try:
                        self.logger.debug(f"Trying {method_name} detection")
                        result = method.detect_waterline(scale_region)
                        method_results[method_name] = result
                        
                        if result is not None:
                            self.logger.info(f"Fallback method {method_name} successful: Y={result}")
                            
                            # Debug: Save method-specific result
                            if self.debug_viz:
                                debug_image = scale_region.copy()
                                if len(debug_image.shape) == 2:
                                    debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
                                
                                # Draw detected water line
                                cv2.line(debug_image, (0, result), (debug_image.shape[1], result), (0, 255, 255), 2)
                                cv2.putText(debug_image, f"{method_name}: Y={result}", (10, result-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                
                                self.debug_viz.save_debug_image(
                                    debug_image, f'pattern_{method_name}_result',
                                    info_text=f"{method_name} detected water line at Y={result}"
                                )
                            
                            return result
                    except Exception as e:
                        self.logger.warning(f"Method {method_name} failed: {e}")
                        method_results[method_name] = None
                        continue
        
        # Debug: Save summary of all method results
        if self.debug_viz and method_results:
            summary_info = []
            for method, result in method_results.items():
                status = f"Y={result}" if result is not None else "FAILED"
                summary_info.append(f"{method}: {status}")
            
            # Create summary visualization
            summary_image = scale_region.copy()
            if len(summary_image.shape) == 2:
                summary_image = cv2.cvtColor(summary_image, cv2.COLOR_GRAY2BGR)
            
            # Add text summary
            y_pos = 30
            for info in summary_info:
                cv2.putText(summary_image, info, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20
            
            self.debug_viz.save_debug_image(
                summary_image, 'pattern_methods_summary',
                info_text="All pattern detection methods failed"
            )
        
        self.logger.error("All pattern detection methods failed")
        return None
    
    def get_detection_info(self):
        """Get information about the integrated detector."""
        info = {
            'method': 'integrated_pattern_detector',
            'status': 'active_with_e_pattern_priority',
            'available_methods': list(self.detection_methods.keys()) if self.detection_methods else [],
            'e_pattern_detector': self.e_pattern_detector is not None,
            'calibration_data': self.calibration_data is not None
        }
        
        if self.e_pattern_detector:
            info['e_pattern_info'] = self.e_pattern_detector.get_detection_info()
        
        if self.calibration_data:
            info['pixels_per_cm'] = self.calibration_data.get('pixels_per_cm')
        
        return info