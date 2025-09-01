"""
Pattern-Aware Water Level Detector

Advanced water level detection designed to handle scales with complex markings,
numbers, and repetitive patterns using multiple pattern recognition techniques.
"""

import cv2
import numpy as np
import os
import logging
from datetime import datetime
from pathlib import Path

# Import debug visualizer - handle both relative and absolute imports
try:
    from .debug_visualizer import DebugVisualizer
except ImportError:
    from debug_visualizer import DebugVisualizer

# Import detection methods - handle both relative and absolute imports
try:
    from .detection_methods.template_matching import TemplateMatchingDetector
    from .detection_methods.morphological_detector import MorphologicalDetector
    from .detection_methods.frequency_analyzer import FrequencyAnalyzer
    from .detection_methods.lsd_detector import LSDDetector
    from .detection_methods.contour_analyzer import ContourAnalyzer
    from .detection_methods.integrated_detector import IntegratedPatternDetector
except ImportError:
    from detection_methods.template_matching import TemplateMatchingDetector
    from detection_methods.morphological_detector import MorphologicalDetector
    from detection_methods.frequency_analyzer import FrequencyAnalyzer
    from detection_methods.lsd_detector import LSDDetector
    from detection_methods.contour_analyzer import ContourAnalyzer
    from detection_methods.integrated_detector import IntegratedPatternDetector

# Import pattern analysis utilities - handle both relative and absolute imports
try:
    from .pattern_analysis.template_manager import TemplateManager
    from .pattern_analysis.pattern_classifier import PatternClassifier
except ImportError:
    from pattern_analysis.template_manager import TemplateManager
    from pattern_analysis.pattern_classifier import PatternClassifier

class PatternWaterDetector:
    """
    Pattern-aware water level detector for scales with complex markings.
    
    Uses multiple detection methods specifically designed to distinguish
    between scale markings (numbers, lines, patterns) and water interfaces.
    """
    
    def __init__(self, config, pixels_per_cm, enhanced_calibration_data=None, calibration_manager=None):
        """Initialize the pattern-aware water level detector."""
        self.config = config
        self.pixels_per_cm = pixels_per_cm
        self.enhanced_calibration_data = enhanced_calibration_data
        self.calibration_manager = calibration_manager
        self.logger = logging.getLogger(__name__)
        
        # Pattern-aware configuration
        self.pattern_config = config.get('detection', {}).get('pattern_aware', {})
        self.pattern_enabled = self.pattern_config.get('enabled', False)
        self.pattern_engine = self.pattern_config.get('engine', 'integrated_pattern')
        self.fallback_enabled = self.pattern_config.get('fallback_to_standard', True)
        
        # Processing mode
        self.processing_mode = config.get('pattern_processing', {}).get('mode', 'standard')
        self.debug_patterns = config.get('pattern_processing', {}).get('debug_patterns', False)
        self.save_templates = config.get('pattern_processing', {}).get('save_templates', True)
        
        # Detection parameters (inherited from original system)
        self.edge_low = config['detection']['edge_threshold_low']
        self.edge_high = config['detection']['edge_threshold_high']
        self.blur_kernel = config['detection']['blur_kernel_size']
        self.scale_height_cm = config['scale']['total_height']
        
        # Initialize debug visualizer with pattern-aware session prefix
        debug_enabled = config.get('debug', {}).get('enabled', False) and self.debug_patterns
        self.debug_viz = DebugVisualizer(config, enabled=debug_enabled, session_prefix='pattern_aware')
        
        # Initialize pattern analysis components
        self.template_manager = TemplateManager(config)
        self.pattern_classifier = PatternClassifier(config)
        
        # Initialize detection methods
        self._initialize_detection_methods()
        
        # Debug visualization is now handled by the DebugVisualizer class
        
        # Log configuration
        self._log_initialization()
    
    def _initialize_detection_methods(self):
        """Initialize all pattern detection methods."""
        self.detection_methods = {}
        
        # Template Matching Detector
        if self.pattern_config.get('template_matching', {}).get('enabled', True):
            self.detection_methods['template_matching'] = TemplateMatchingDetector(
                self.config, self.template_manager
            )
        
        # Morphological Detector
        if self.pattern_config.get('morphological', {}).get('enabled', True):
            self.detection_methods['morphological'] = MorphologicalDetector(self.config)
        
        # Frequency Analysis Detector
        if self.pattern_config.get('frequency_analysis', {}).get('enabled', True):
            self.detection_methods['frequency'] = FrequencyAnalyzer(self.config)
        
        # Line Segment Detector
        if self.pattern_config.get('line_detection', {}).get('enabled', True):
            self.detection_methods['lsd'] = LSDDetector(self.config)
        
        # Contour Analysis Detector
        if self.pattern_config.get('contour_analysis', {}).get('enabled', True):
            self.detection_methods['contour'] = ContourAnalyzer(self.config)
        
        # Integrated Pattern Detector (combines all methods)
        self.integrated_detector = IntegratedPatternDetector(
            self.config, self.detection_methods, self.pattern_classifier, self.debug_viz, self.enhanced_calibration_data
        )
        
        self.logger.info(f"Initialized {len(self.detection_methods)} pattern detection methods")
    
    
    def _log_initialization(self):
        """Log initialization information."""
        self.logger.info("Pattern-Aware Water Level Detector Initialized")
        self.logger.info(f"Pattern detection enabled: {self.pattern_enabled}")
        self.logger.info(f"Processing mode: {self.processing_mode}")
        self.logger.info(f"Pattern engine: {self.pattern_engine}")
        self.logger.info(f"Fallback to standard methods: {self.fallback_enabled}")
        self.logger.info(f"Debug patterns: {self.debug_patterns}")
        
        # Log calibration info
        if self.enhanced_calibration_data:
            method = self.enhanced_calibration_data.get('method', 'unknown')
            confidence = self.enhanced_calibration_data.get('confidence', 'unknown')
            self.logger.info(f"Using enhanced calibration: {method} (confidence: {confidence})")
        
        # Check for existing templates
        template_count = self.template_manager.get_template_count()
        if template_count > 0:
            self.logger.info(f"Loaded {template_count} existing scale marking templates")
        else:
            self.logger.info("No existing templates found - will extract from calibration data")
    
    def process_image(self, image_path):
        """
        Process an image to detect water level using pattern-aware methods.
        
        Args:
            image_path (str): Path to the image to process
            
        Returns:
            dict: Processing results including water level and confidence
        """
        import time
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            self.logger.info(f"Processing image with pattern-aware detection: {Path(image_path).name}")
            
            # Start debug session for this image
            self.debug_viz.start_image_debug(image_path)
            
            # Debug: Save original image
            self.debug_viz.save_debug_image(
                image, 'pattern_original',
                info_text=f"Original image: {image.shape[1]}x{image.shape[0]}"
            )
            
            # Extract scale region using existing calibration
            scale_region, scale_bounds = self._extract_scale_region(image)
            if scale_region is None:
                self.logger.error("Failed to extract scale region")
                return None
            
            # Debug: Save extracted scale region
            self.debug_viz.save_debug_image(
                scale_region, 'pattern_scale_region',
                info_text=f"Scale region: {scale_bounds}"
            )
            
            # Detect water line using pattern-aware methods
            water_line_y = self._detect_water_line_pattern_aware(scale_region, image_path)
            
            if water_line_y is None:
                if self.fallback_enabled:
                    self.logger.warning("Pattern detection failed, falling back to standard methods")
                    return self._fallback_to_standard_detection(image, image_path)
                else:
                    self.logger.error("Pattern detection failed and fallback disabled")
                    return None
            
            # Convert local coordinates to global coordinates
            global_y = water_line_y + scale_bounds['y_min']
            
            # Debug: Save water line detection result
            water_line_annotations = {
                'lines': [{
                    'x1': scale_bounds['x_min'],
                    'y1': global_y,
                    'x2': scale_bounds['x_max'],
                    'y2': global_y,
                    'color': (0, 255, 255),  # Yellow
                    'thickness': 3,
                    'label': f'Pattern Water Line (Y={global_y})'
                }],
                'rectangles': [{
                    'x': scale_bounds['x_min'],
                    'y': scale_bounds['y_min'],
                    'w': scale_bounds['x_max'] - scale_bounds['x_min'],
                    'h': scale_bounds['y_max'] - scale_bounds['y_min'],
                    'color': (0, 255, 0),
                    'thickness': 2,
                    'label': 'Scale Region'
                }]
            }
            
            self.debug_viz.save_debug_image(
                image, 'pattern_water_detection',
                annotations=water_line_annotations,
                info_text=f"Pattern-based water line at Y={global_y} (engine: {self.pattern_engine})"
            )
            
            # Calculate water level in cm
            water_level_cm = self._calculate_water_level(global_y, image.shape[0])
            
            # Create result
            result = {
                'timestamp': datetime.now(),
                'image_path': str(image_path),
                'water_level_cm': water_level_cm,
                'scale_above_water_cm': self.scale_height_cm - water_level_cm,
                'detection_method': 'pattern_aware',
                'pattern_engine': self.pattern_engine,
                'water_line_y': global_y,
                'confidence': 0.95,  # Pattern methods generally have high confidence
                'scale_bounds': scale_bounds
            }
            
            self.logger.info(f"Pattern detection successful: {water_level_cm:.1f}cm")
            
            # Save annotated image if configured
            processing_time = time.time() - start_time
            if self.config['processing'].get('save_processed_images', True):
                self.save_processed_image_enhanced(image, result, processing_time, image_path)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            
            # Save annotated image for failed detection if configured
            processing_time = time.time() - start_time
            if self.config['processing'].get('save_processed_images', True):
                try:
                    # Try to load image if not already loaded
                    if 'image' not in locals() or image is None:
                        image = cv2.imread(str(image_path))
                    if image is not None:
                        self.save_processed_image_enhanced(image, None, processing_time, image_path)
                except:
                    pass  # Don't let annotated image saving break the main flow
            
            if self.fallback_enabled:
                return self._fallback_to_standard_detection(image, image_path)
            return None
    
    def save_processed_image_enhanced(self, image, result, processing_time, image_path):
        """
        Save image with pattern-aware annotations showing detected water line and measurements.
        Enhanced version that handles both successful and failed detections.
        """
        import cv2
        from pathlib import Path
        from datetime import datetime
        
        annotated = image.copy()
        
        # Draw annotations based on result
        if result:
            # Successful detection
            water_line_y = result.get('water_line_y')
            scale_bounds = result.get('scale_bounds', {})
            
            # Draw water line (green for success)
            if water_line_y is not None:
                cv2.line(annotated, (0, water_line_y), (image.shape[1], water_line_y), (0, 255, 0), 3)
            
            # Draw scale bounds
            if scale_bounds:
                x_min = scale_bounds.get('x_min', 0)
                x_max = scale_bounds.get('x_max', image.shape[1])
                y_min = scale_bounds.get('y_min', 0)
                y_max = scale_bounds.get('y_max', image.shape[0])
                
                # Draw scale region rectangle
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # Add success text
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 30
            
            text = f"PATTERN-AWARE SUCCESS"
            cv2.putText(annotated, text, (10, y_offset), font, 0.8, (0, 255, 0), 2)
            y_offset += 35
            
            text = f"Water Level: {result['water_level_cm']:.1f}cm"
            cv2.putText(annotated, text, (10, y_offset), font, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            text = f"Scale Above Water: {result['scale_above_water_cm']:.1f}cm"
            cv2.putText(annotated, text, (10, y_offset), font, 0.7, (255, 255, 0), 2)
            y_offset += 30
            
            text = f"Engine: {result.get('pattern_engine', 'unknown')}"
            cv2.putText(annotated, text, (10, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
            
            text = f"Confidence: {result.get('confidence', 0.0):.2f}"
            cv2.putText(annotated, text, (10, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
            
        else:
            # Failed detection
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = 30
            
            text = "PATTERN DETECTION FAILED"
            cv2.putText(annotated, text, (10, y_offset), font, 0.8, (0, 0, 255), 2)
            y_offset += 35
            
            # Show attempted pattern engine
            text = f"Engine attempted: {self.pattern_engine}"
            cv2.putText(annotated, text, (10, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
            
            # Show fallback status
            if self.fallback_enabled:
                text = "Fallback to standard detection: enabled"
                cv2.putText(annotated, text, (10, y_offset), font, 0.6, (255, 255, 255), 1)
            else:
                text = "Fallback to standard detection: disabled"
                cv2.putText(annotated, text, (10, y_offset), font, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        # Add processing info (common for both success/failure)
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = image.shape[0] - 60
        
        text = f"Processing time: {processing_time:.2f}s"
        cv2.putText(annotated, text, (10, y_pos), font, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        text = f"Pixels/cm: {self.pixels_per_cm:.2f}"
        cv2.putText(annotated, text, (10, y_pos), font, 0.5, (200, 200, 200), 1)
        y_pos += 20
        
        text = f"Pattern-aware detection system"
        cv2.putText(annotated, text, (10, y_pos), font, 0.5, (200, 200, 200), 1)
        
        # Save annotated image
        image_format = self.config['processing'].get('image_format', 'jpg')
        if not image_format.startswith('.'):
            image_format = '.' + image_format
        
        # Use output directory for annotated images
        annotated_dir = Path('data/output/annotated')
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        # Include success/failure status in filename
        status = "success" if result else "failed"
        output_path = annotated_dir / f"annotated_pattern_{status}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{image_format}"
        success = cv2.imwrite(str(output_path), annotated)
        
        if success:
            self.logger.debug(f"Saved pattern-aware annotated image: {output_path}")
        else:
            self.logger.warning(f"Failed to save pattern-aware annotated image: {output_path}")
    
    def _extract_scale_region(self, image):
        """Extract scale region using existing calibration data."""
        if self.enhanced_calibration_data and 'scale_boundaries' in self.enhanced_calibration_data:
            bounds = self.enhanced_calibration_data['scale_boundaries']
            x_min, x_max = bounds['x_min'], bounds['x_max']
            y_min, y_max = bounds['y_min'], bounds['y_max']
        else:
            # Use config-based boundaries
            expected_pos = self.config['scale']['expected_position']
            x_min, x_max = expected_pos['x_min'], expected_pos['x_max']
            y_min, y_max = expected_pos['y_min'], expected_pos['y_max']
        
        # Extract region
        scale_region = image[y_min:y_max, x_min:x_max]
        
        scale_bounds = {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max
        }
        
        return scale_region, scale_bounds
    
    def _detect_water_line_pattern_aware(self, scale_region, image_path):
        """
        Detect water line using pattern-aware methods.
        
        Args:
            scale_region: Extracted scale region image
            image_path: Original image path for debugging
            
        Returns:
            int: Y-coordinate of detected water line (local to scale region)
        """
        if self.pattern_engine == 'integrated_pattern':
            return self.integrated_detector.detect_waterline(scale_region, image_path)
        
        # Single method detection
        elif self.pattern_engine in self.detection_methods:
            method = self.detection_methods[self.pattern_engine]
            return method.detect_waterline(scale_region)
        
        else:
            self.logger.error(f"Unknown pattern engine: {self.pattern_engine}")
            return None
    
    def _calculate_water_level(self, water_line_y, image_height):
        """Calculate water level in cm from pixel coordinates."""
        # Calculate distance from top of image to water line
        distance_pixels = water_line_y
        
        # Convert to cm using calibration
        water_level_cm = distance_pixels / self.pixels_per_cm
        
        # Adjust based on scale configuration
        if self.enhanced_calibration_data and 'scale_measurements' in self.enhanced_calibration_data:
            # Use enhanced calibration data
            scale_data = self.enhanced_calibration_data['scale_measurements']
            top_measurement = scale_data['top_measurement_cm']
            
            # Calculate actual water level based on scale position
            water_level_cm = top_measurement - water_level_cm
        else:
            # Use standard calculation
            water_level_cm = self.scale_height_cm - water_level_cm
        
        return max(0, water_level_cm)  # Ensure non-negative
    
    def _fallback_to_standard_detection(self, image, image_path):
        """Fallback to standard detection methods when pattern detection fails."""
        try:
            # Import standard detector
            from src.water_level_detector import WaterLevelDetector
            
            # Create standard detector
            standard_detector = WaterLevelDetector(
                self.config, self.pixels_per_cm, 
                self.enhanced_calibration_data, self.calibration_manager
            )
            
            self.logger.info("Using standard detection as fallback")
            return standard_detector.process_image(str(image_path))
            
        except Exception as e:
            self.logger.error(f"Fallback to standard detection failed: {e}")
            return None
    
    def extract_and_save_templates(self, calibration_image_path):
        """
        Extract scale marking templates from calibration image.
        
        Args:
            calibration_image_path (str): Path to calibration image
            
        Returns:
            int: Number of templates extracted
        """
        try:
            image = cv2.imread(str(calibration_image_path))
            if image is None:
                raise ValueError(f"Could not load calibration image: {calibration_image_path}")
            
            scale_region, _ = self._extract_scale_region(image)
            if scale_region is None:
                raise ValueError("Could not extract scale region from calibration image")
            
            # Extract templates using template manager
            template_count = self.template_manager.extract_templates_from_region(scale_region)
            
            if self.save_templates:
                self.template_manager.save_templates()
                self.logger.info(f"Extracted and saved {template_count} scale marking templates")
            
            return template_count
            
        except Exception as e:
            self.logger.error(f"Error extracting templates: {e}")
            return 0
    
    def get_detection_info(self):
        """Get information about available detection methods."""
        return {
            'pattern_enabled': self.pattern_enabled,
            'processing_mode': self.processing_mode,
            'pattern_engine': self.pattern_engine,
            'available_methods': list(self.detection_methods.keys()),
            'template_count': self.template_manager.get_template_count(),
            'fallback_enabled': self.fallback_enabled,
            'debug_enabled': self.debug_patterns
        }