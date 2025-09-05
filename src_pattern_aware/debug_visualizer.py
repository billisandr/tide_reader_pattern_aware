"""
Debug visualization module for water level detection system.
Provides comprehensive visual debugging with annotated images.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime


class DebugVisualizer:
    def __init__(self, config, enabled=False, session_prefix='pattern_aware'):
        self.config = config
        self.enabled = enabled
        self.session_prefix = session_prefix
        self.logger = logging.getLogger(__name__)
        self.debug_config = config.get('debug', {})
        
        # Debug settings
        self.save_images = self.debug_config.get('save_debug_images', True)
        # Use pattern-aware specific debug directory
        default_debug_dir = f'data/debug_{session_prefix}' if session_prefix != 'standard' else 'data/debug'
        self.output_dir = Path(self.debug_config.get('debug_output_dir', default_debug_dir))
        self.color = tuple(self.debug_config.get('annotation_color', [0, 255, 0]))  # BGR
        self.thickness = self.debug_config.get('annotation_thickness', 3)
        self.font_scale = self.debug_config.get('font_scale', 0.9)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.steps_to_save = set(self.debug_config.get('steps_to_save', []))
        
        # Current session info
        self.current_image_name = None
        self.session_dir = None
        self.created_step_dirs = set()  # Track which step directories have been created
        self.saved_images_count = 0     # Track number of images saved to prevent empty sessions
        
        if self.enabled and self.save_images:
            self.setup_debug_directory()
    
    def setup_debug_directory(self):
        """Create debug output directory structure."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f"{self.session_prefix}_debug_session_{timestamp}"
        self.session_dir = self.output_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Don't create step subdirectories yet - create them only when needed
        # This prevents empty folders from being left behind
        
        self.logger.info(f"Debug session directory: {self.session_dir}")
    
    def start_image_debug(self, image_path):
        """Initialize debugging for a new image."""
        if not self.enabled:
            return
        
        self.current_image_name = Path(image_path).stem
        self.logger.debug(f"Starting debug for image: {self.current_image_name}")
        
        # Generate legend.txt for the main session directory
        if hasattr(self, 'session_dir') and self.session_dir:
            self._save_main_legend()
    
    def _save_main_legend(self):
        """
        Save legend.txt file explaining colors and annotations used in main pattern-aware debug images.
        """
        if not hasattr(self, 'session_dir') or not self.session_dir:
            return
            
        legend_path = self.session_dir / "legend.txt"
        
        # Only create legend once per session
        if legend_path.exists():
            return
            
        legend_content = """PATTERN-AWARE DEBUG VISUALIZATION LEGEND
=============================================

This legend explains the colors, lines, and annotations used in the pattern-aware 
detection debug images in this session directory.

MAIN DEBUG IMAGES:
------------------

1. pattern_original
   - Original input image without any modifications
   - Shows the full image as received by the detector
   - No annotations - reference image

2. pattern_scale_region  
   - Extracted scale region for analysis
   - Cropped area where pattern detection occurs
   - Shows the calibrated region of interest
   - No annotations - cropped reference

3. pattern_preprocessing
   - Scale region prepared for pattern analysis
   - Same as scale region but indicates start of processing
   - Used by integrated detector for method comparison

4. pattern_water_detection (Final Result)
   Colors and Annotations:
   - WATERLINE - Horizontal line showing detected water level
     * Color depends on detection method used
     * Thickness: 2-3 pixels
     * Spans full width of scale region
     
   - SCALE REGION OUTLINE (if shown)
     * Color: Green (0, 255, 0) 
     * Thickness: 2 pixels
     * Shows boundaries of analyzed area

METHOD-SPECIFIC RESULTS:
------------------------

5. pattern_e_pattern_result
   Colors and Annotations:
   - DETECTED WATERLINE: Final E-pattern detection result
     * Color varies by confidence/method
     * Shows improved result if hybrid analysis was applied
     * Text label indicates Y position and method

6. pattern_[method_name]_result (e.g., pattern_template_matching_result)
   - Individual detection method results before final selection
   - YELLOW LINE: Method-specific waterline detection
     * Color: (0, 255, 255) - Yellow
     * Thickness: 2 pixels
   - Text overlay shows method name and Y position

7. pattern_methods_summary
   - Summary when all detection methods fail
   - Shows why detection failed
   - May contain diagnostic information

E-PATTERN DETECTION COLORS:
----------------------------

E-patterns (when visible in debug images):
- PURPLE RECTANGLES: Detected E-pattern templates
  * Color: (128, 0, 128) - Purple  
  * Thickness: 2 pixels
  * Shows bounding boxes around matched templates
  
- GREEN CIRCLES: Template centers (in some visualizations)
  * Color: (0, 255, 0) - Bright Green
  * Used in hybrid waterline analysis images

COORDINATE SYSTEM:
------------------
- Y=0 is at the top of the scale region
- Higher Y values = lower on the stadia rod scale
- All Y positions refer to scale region coordinates
- Final results are converted to global image coordinates

DETECTION PROCESS:
------------------
1. Original image → Scale region extraction
2. Scale region → Pattern preprocessing  
3. Pattern detection → Individual method results
4. Method results → Final waterline selection
5. Optional: Hybrid analysis for improved accuracy

FILES IN THIS SESSION:
----------------------
- Images: PNG/JPG files with visual annotations
- Text files: Detailed analysis data (if available)
- Subfolders: Specialized analysis (e.g., waterline_gradient_analysis)
- legend.txt: This file explaining visualizations

SPECIALIZED ANALYSIS DOCUMENTATION:
-----------------------------------
This session may include specialized analysis subfolders with additional debug images:

- waterline_gradient_analysis/: Advanced hybrid waterline detection analysis
  → See waterline_gradient_analysis/legend.txt for detailed gradient analysis documentation
  → Includes 4 specialized visualizations with purple/yellow/green color coding
  → Contains advanced gradient differential theory and candidate selection details
  → Covers: candidate regions, gradient analysis, verification summaries

For complete understanding of all debug visualizations, check both this legend
and any subfolder-specific legend.txt files.

OTHER ANALYSIS FOLDERS:
-----------------------
Additional specialized analysis folders may be created as needed, each with their
own legend.txt file explaining the specific visualizations and color coding used.

LEGEND CREATED: Generated automatically during pattern-aware detection
=============================================
"""
        
        try:
            with open(legend_path, 'w', encoding='utf-8') as f:
                f.write(legend_content)
            self.logger.debug(f"Saved main pattern-aware legend to {legend_path}")
        except Exception as e:
            self.logger.error(f"Failed to save main legend: {e}")
    
    def save_debug_image(self, image, step_name, annotations=None, info_text=None):
        """
        Save an annotated debug image.
        
        Args:
            image: OpenCV image (BGR format)
            step_name: Processing step name
            annotations: Dict with annotation instructions
            info_text: Additional text to display
        """
        if not self.enabled or not self.save_images or step_name not in self.steps_to_save:
            return
        
        if image is None:
            self.logger.warning(f"Cannot save debug image for step '{step_name}': image is None")
            return
        
        # Create step directory only when needed
        step_dir = self.session_dir / step_name
        if step_name not in self.created_step_dirs:
            step_dir.mkdir(exist_ok=True)
            self.created_step_dirs.add(step_name)
        
        # Create a copy for annotation
        debug_image = image.copy()
        
        # Convert grayscale to BGR if needed
        if len(debug_image.shape) == 2:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
        
        # Apply annotations
        if annotations:
            debug_image = self._apply_annotations(debug_image, annotations)
        
        # Add info text and title in side panel (skip for clean steps)
        if step_name != 'waterline_gradient_analysis_clean':
            debug_image = self._add_side_panel(debug_image, step_name, info_text)
        
        # Save image
        filename = f"{self.current_image_name}_{step_name}.jpg"
        output_path = step_dir / filename
        
        success = cv2.imwrite(str(output_path), debug_image)
        if success:
            self.saved_images_count += 1
            self.logger.debug(f"Saved debug image: {output_path}")
        else:
            self.logger.error(f"Failed to save debug image: {output_path}")
    
    def _apply_annotations(self, image, annotations):
        """Apply various annotations to the image."""
        annotated = image.copy()
        
        # Draw rectangles
        if 'rectangles' in annotations:
            for rect in annotations['rectangles']:
                pt1 = (int(rect['x']), int(rect['y']))
                pt2 = (int(rect['x'] + rect['w']), int(rect['y'] + rect['h']))
                color = rect.get('color', self.color)
                thickness = rect.get('thickness', self.thickness)
                label = rect.get('label', '')
                
                cv2.rectangle(annotated, pt1, pt2, color, thickness)
                
                if label:
                    # Add label above rectangle
                    label_pos = (pt1[0], pt1[1] - 10)
                    cv2.putText(annotated, label, label_pos, self.font, 
                               self.font_scale, color, 2, cv2.LINE_AA)
        
        # Draw contours
        if 'contours' in annotations:
            contours_data = annotations['contours']
            contours = contours_data.get('contours', [])
            color = contours_data.get('color', self.color)
            thickness = contours_data.get('thickness', self.thickness)
            
            cv2.drawContours(annotated, contours, -1, color, thickness)
        
        # Draw points
        if 'points' in annotations:
            for point in annotations['points']:
                center = (int(point['x']), int(point['y']))
                color = point.get('color', self.color)
                radius = point.get('radius', 5)
                label = point.get('label', '')
                
                cv2.circle(annotated, center, radius, color, -1)
                
                if label:
                    label_pos = (center[0] + 10, center[1])
                    cv2.putText(annotated, label, label_pos, self.font,
                               self.font_scale, color, 2, cv2.LINE_AA)
        
        # Draw lines
        if 'lines' in annotations:
            for line in annotations['lines']:
                pt1 = (int(line['x1']), int(line['y1']))
                pt2 = (int(line['x2']), int(line['y2']))
                color = line.get('color', self.color)
                thickness = line.get('thickness', self.thickness)
                label = line.get('label', '')
                
                cv2.line(annotated, pt1, pt2, color, thickness)
                
                if label:
                    # Add label at midpoint
                    mid_x = (pt1[0] + pt2[0]) // 2
                    mid_y = (pt1[1] + pt2[1]) // 2
                    cv2.putText(annotated, label, (mid_x, mid_y), self.font,
                               self.font_scale, color, 2, cv2.LINE_AA)
        
        # Draw text annotations
        if 'text' in annotations:
            for text_item in annotations['text']:
                position = (int(text_item['x']), int(text_item['y']))
                text = text_item['text']
                color = text_item.get('color', self.color)
                scale = text_item.get('scale', self.font_scale)
                
                cv2.putText(annotated, text, position, self.font,
                           scale, color, 2, cv2.LINE_AA)
        
        return annotated
    
    def _add_side_panel(self, image, step_name, info_text):
        """Add title and info text in a side panel next to the image."""
        h, w = image.shape[:2]
        
        # Prepare title
        title_text = step_name.replace('_', ' ').title()
        
        # Prepare info text lines
        info_lines = []
        if info_text:
            if isinstance(info_text, str):
                info_lines = info_text.split('\n')
            else:
                info_lines = info_text
        
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
        cv2.putText(combined_image, title_text, (panel_start_x + 10, title_y), self.font,
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
                    cv2.putText(combined_image, line, (panel_start_x + 10, current_y), self.font,
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
                                           self.font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                                current_y += line_height
                                current_line = word
                            else:
                                # Single word too long, force break
                                cv2.putText(combined_image, word[:max_chars_per_line], 
                                           (panel_start_x + 10, current_y), self.font, 0.4, 
                                           (200, 200, 200), 1, cv2.LINE_AA)
                                current_y += line_height
                                current_line = word[max_chars_per_line:]
                    
                    # Add remaining text
                    if current_line:
                        cv2.putText(combined_image, current_line, (panel_start_x + 10, current_y), 
                                   self.font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
                        current_y += line_height
        
        return combined_image
    
    def create_summary_image(self, results):
        """Create a summary image with key measurements and detection results."""
        if not self.enabled or not self.save_images:
            return
        
        # Create a summary canvas
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Add title
        title_text = f"{self.session_prefix.title()} Water Level Detection Summary"
        cv2.putText(canvas, title_text, (50, 50),
                   self.font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add measurement results
        y_pos = 100
        if results:
            info_lines = [
                f"Image: {self.current_image_name}",
                f"Water Level: {results.get('water_level_cm', 'N/A')} cm",
                f"Scale Above Water: {results.get('scale_above_water_cm', 'N/A')} cm",
                f"Confidence: {results.get('confidence', 'N/A')}",
                f"Processing Time: {results.get('processing_time', 'N/A')} seconds",
                f"Detection Method: {results.get('detection_method', 'N/A')}"
            ]
            
            for line in info_lines:
                cv2.putText(canvas, line, (50, y_pos), self.font,
                           0.6, (255, 255, 255), 1, cv2.LINE_AA)
                y_pos += 30
        
        # Save summary
        filename = f"{self.current_image_name}_summary.jpg"
        output_path = self.session_dir / filename
        success = cv2.imwrite(str(output_path), canvas)
        
        if success:
            self.saved_images_count += 1
            self.logger.debug(f"Saved summary image: {output_path}")
    
    def cleanup_session(self):
        """Clean up the debug session, removing empty directories if no images were saved."""
        if not self.enabled or not self.session_dir:
            return
        
        # If no images were saved, remove the entire session directory
        if self.saved_images_count == 0:
            try:
                import shutil
                shutil.rmtree(str(self.session_dir))
                self.logger.debug(f"Removed empty debug session directory: {self.session_dir}")
            except Exception as e:
                self.logger.warning(f"Could not remove empty debug session directory: {e}")
        else:
            self.logger.info(f"Debug session completed with {self.saved_images_count} images saved: {self.session_dir}")
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.cleanup_session()
    
    def is_enabled(self):
        """Check if debug mode is enabled."""
        return self.enabled
    
    def should_save_step(self, step_name):
        """Check if a specific step should be saved."""
        return self.enabled and step_name in self.steps_to_save