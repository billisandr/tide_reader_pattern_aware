"""
Pattern-Aware Water Level Detector

Advanced water level detection designed to handle scales with complex markings,
numbers, and repetitive patterns using multiple pattern recognition techniques.
"""

import cv2
import numpy as np
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
    from .detection_methods.integrated_detector import IntegratedPatternDetector
except ImportError:
    from detection_methods.template_matching import TemplateMatchingDetector
    from detection_methods.morphological_detector import MorphologicalDetector
    from detection_methods.integrated_detector import IntegratedPatternDetector

# Pattern analysis utilities removed (were stub implementations)

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
        # Pattern classifier removed (was stub implementation)
        
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
                self.config
            )
        
        # Morphological Detector
        if self.pattern_config.get('morphological', {}).get('enabled', True):
            self.detection_methods['morphological'] = MorphologicalDetector(self.config)
        
        # Stub detectors removed (FrequencyAnalyzer, LSDDetector, ContourAnalyzer)
        
        # Integrated Pattern Detector (combines all methods)
        self.integrated_detector = IntegratedPatternDetector(
            self.config, self.detection_methods, None, self.debug_viz, self.enhanced_calibration_data
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
        
        self.logger.info("Template loading handled by individual detectors")
    
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
            water_line_y, detection_method_used = self._detect_water_line_pattern_aware(scale_region, image_path)
            
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
                'detection_method': detection_method_used or 'pattern_aware',
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
                    else:
                        self.logger.warning(f"Could not load image for failed detection annotation: {image_path}")
                except Exception as save_error:
                    self.logger.warning(f"Failed to save annotated image for failed detection: {save_error}")
                    # Don't let annotated image saving break the main flow
            
            if self.fallback_enabled:
                return self._fallback_to_standard_detection(image, image_path)
            return None
    
    def save_processed_image_enhanced(self, image, result, processing_time, image_path):
        """
        Save image with clean pattern-aware annotations and side panel information.
        Enhanced version that uses side panel format for decluttered display.
        """
        import cv2
        import numpy as np
        from pathlib import Path
        from datetime import datetime
        
        # Create clean annotated image (no text overlays on main image)
        annotated = image.copy()
        
        # Draw only essential visual annotations on the main image
        if result:
            # Successful detection - draw clean visual elements
            water_line_y = result.get('water_line_y')
            scale_bounds = result.get('scale_bounds', {})
            
            # Draw water line (green for success)
            if water_line_y is not None:
                cv2.line(annotated, (0, water_line_y), (image.shape[1], water_line_y), (0, 255, 0), 3)
            
            # Draw scale bounds (clean rectangle)
            if scale_bounds:
                x_min = scale_bounds.get('x_min', 0)
                x_max = scale_bounds.get('x_max', image.shape[1])
                y_min = scale_bounds.get('y_min', 0)
                y_max = scale_bounds.get('y_max', image.shape[0])
                
                # Draw scale region rectangle
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
        # Create side panel with comprehensive information
        status = "success" if result else "failed"
        info_lines = []
        
        if result:
            # Success information
            info_lines = [
                "PATTERN-AWARE DETECTION SUCCESS",
                "",
                f"Water Level: {result['water_level_cm']:.1f} cm",
                f"Scale Above Water: {result['scale_above_water_cm']:.1f} cm",
                f"Water Line Y: {result.get('water_line_y', 'N/A')} px",
                "",
                "Detection Details:",
                f"  Engine: {result.get('pattern_engine', 'unknown')}",
                f"  Method: {result.get('method', 'N/A')}",
                f"  Confidence: {result.get('confidence', 0.0):.3f}",
                "",
                "Scale Information:",
                f"  Pixels per cm: {self.pixels_per_cm:.2f}",
                f"  Scale height: {self.config['scale']['total_height']:.1f} cm"
            ]
            
            # Add scale bounds if available
            if result.get('scale_bounds'):
                bounds = result['scale_bounds']
                info_lines.extend([
                    "",
                    "Scale Region:",
                    f"  X: {bounds.get('x_min', 0)}-{bounds.get('x_max', 0)} px",
                    f"  Y: {bounds.get('y_min', 0)}-{bounds.get('y_max', 0)} px"
                ])
            
            # Add hybrid analysis info if available
            if result.get('hybrid_analysis'):
                hybrid = result['hybrid_analysis']
                info_lines.extend([
                    "",
                    "Hybrid Analysis:",
                    f"  Analysis performed: {hybrid.get('analysis_performed', False)}",
                    f"  Reason: {hybrid.get('reason', 'N/A')}",
                    f"  Confidence: {hybrid.get('confidence', 0.0):.3f}"
                ])
                
        else:
            # Failure information
            info_lines = [
                "PATTERN DETECTION FAILED",
                "",
                "Detection Attempt:",
                f"  Engine attempted: {self.pattern_engine}",
                f"  Fallback enabled: {'Yes' if self.fallback_enabled else 'No'}",
                "",
                "System Information:",
                f"  Pixels per cm: {self.pixels_per_cm:.2f}",
                f"  Scale height: {self.config['scale']['total_height']:.1f} cm"
            ]
        
        # Add common processing information
        info_lines.extend([
            "",
            "Processing Information:",
            f"  Processing time: {processing_time:.2f} seconds",
            f"  Image: {Path(image_path).name}",
            f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  System: Pattern-aware detection"
        ])
        
        # Create image with side panel using the same approach as debug visualizer
        final_image = self._add_side_panel_to_output(annotated, status.upper(), info_lines)
        
        # Save annotated image
        image_format = self.config['processing'].get('image_format', 'jpg')
        if not image_format.startswith('.'):
            image_format = '.' + image_format
        
        # Use output directory for annotated images
        annotated_dir = Path('data/output/annotated')
        annotated_dir.mkdir(parents=True, exist_ok=True)
        
        # Include original image name and success/failure status in filename
        original_name = Path(image_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = annotated_dir / f"{original_name}_annotated_pattern_{status}_{timestamp}{image_format}"
        success = cv2.imwrite(str(output_path), final_image)
        
        if success:
            self.logger.info(f"Saved pattern-aware annotated image: {output_path}")
        else:
            self.logger.error(f"Failed to save pattern-aware annotated image: {output_path}")
    
    def _add_side_panel_to_output(self, image, title, info_lines):
        """Add side panel to output image using the same format as debug visualizer."""
        h, w = image.shape[:2]
        
        # Calculate panel width (30% of image width, minimum 300px, maximum 500px)
        panel_width = max(300, min(500, int(w * 0.3)))
        
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
                if not line:  # Empty line for spacing
                    current_y += line_height // 2
                    continue
                
                # Word wrap long lines
                if len(line) > max_chars_per_line:
                    words = line.split(' ')
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        if len(test_line) <= max_chars_per_line:
                            current_line = test_line
                        else:
                            if current_line:
                                # Draw current line
                                cv2.putText(combined_image, current_line, (panel_start_x + 10, current_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                                current_y += line_height
                            current_line = word
                    
                    # Draw remaining text
                    if current_line:
                        cv2.putText(combined_image, current_line, (panel_start_x + 10, current_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                        current_y += line_height
                else:
                    # Short line, draw normally
                    cv2.putText(combined_image, line, (panel_start_x + 10, current_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    current_y += line_height
        
        return combined_image
    
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
            tuple: (Y-coordinate, detection_method) or (None, None) if failed
        """
        result = None
        method_used = None
        
        if self.pattern_engine == 'integrated_pattern':
            detection_result = self.integrated_detector.detect_waterline(scale_region, image_path)
            if isinstance(detection_result, dict):
                result = detection_result.get('y_position')
                method_used = detection_result.get('method')
            else:
                # Backwards compatibility
                result = detection_result
                method_used = 'integrated_pattern'
        
        # Single method detection
        elif self.pattern_engine in self.detection_methods:
            method = self.detection_methods[self.pattern_engine]
            result = method.detect_waterline(scale_region)
            method_used = self.pattern_engine if result is not None else None
        
        else:
            self.logger.error(f"Unknown pattern engine: {self.pattern_engine}")
        
        return result, method_used
    
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
        """Fallback functionality removed - pattern-aware system is now standalone."""
        self.logger.warning("Fallback to standard detection not available in standalone pattern-aware system")
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
            
            # Template extraction handled by individual detectors
            template_count = 0
            
            if self.save_templates:
                self.logger.info("Template saving handled by individual detectors")
            
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
            'template_count': 0,
            'fallback_enabled': self.fallback_enabled,
            'debug_enabled': self.debug_patterns
        }