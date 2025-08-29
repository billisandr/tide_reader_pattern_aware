"""
Debug visualization module for water level detection system.
Provides comprehensive visual debugging with annotated images.
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime


class DebugVisualizer:
    def __init__(self, config, enabled=False):
        self.config = config
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)
        self.debug_config = config.get('debug', {})
        
        # Debug settings
        self.save_images = self.debug_config.get('save_debug_images', True)
        self.output_dir = Path(self.debug_config.get('debug_output_dir', 'data/debug'))
        self.color = tuple(self.debug_config.get('annotation_color', [0, 255, 0]))  # BGR
        self.thickness = self.debug_config.get('annotation_thickness', 2)
        self.font_scale = self.debug_config.get('font_scale', 0.7)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.steps_to_save = set(self.debug_config.get('steps_to_save', []))
        
        # Current session info
        self.current_image_name = None
        self.session_dir = None
        
        if self.enabled and self.save_images:
            self.setup_debug_directory()
    
    def setup_debug_directory(self):
        """Create debug output directory structure."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.output_dir / f"debug_session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create step subdirectories
        for step in self.steps_to_save:
            (self.session_dir / step).mkdir(exist_ok=True)
        
        self.logger.info(f"Debug session directory: {self.session_dir}")
    
    def start_image_debug(self, image_path):
        """Initialize debugging for a new image."""
        if not self.enabled:
            return
        
        self.current_image_name = Path(image_path).stem
        self.logger.debug(f"Starting debug for image: {self.current_image_name}")
    
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
        
        # Create a copy for annotation
        debug_image = image.copy()
        
        # Convert grayscale to BGR if needed
        if len(debug_image.shape) == 2:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
        
        # Apply annotations
        if annotations:
            debug_image = self._apply_annotations(debug_image, annotations)
        
        # Add info text
        if info_text:
            debug_image = self._add_info_text(debug_image, info_text)
        
        # Add step title
        debug_image = self._add_title(debug_image, step_name)
        
        # Save image
        filename = f"{self.current_image_name}_{step_name}.jpg"
        output_path = self.session_dir / step_name / filename
        
        success = cv2.imwrite(str(output_path), debug_image)
        if success:
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
                               self.font_scale, color, 1, cv2.LINE_AA)
        
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
                               self.font_scale, color, 1, cv2.LINE_AA)
        
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
                               self.font_scale, color, 1, cv2.LINE_AA)
        
        # Draw text annotations
        if 'text' in annotations:
            for text_item in annotations['text']:
                position = (int(text_item['x']), int(text_item['y']))
                text = text_item['text']
                color = text_item.get('color', self.color)
                scale = text_item.get('scale', self.font_scale)
                
                cv2.putText(annotated, text, position, self.font,
                           scale, color, 1, cv2.LINE_AA)
        
        return annotated
    
    def _add_info_text(self, image, info_text):
        """Add informational text to the bottom of the image."""
        h, w = image.shape[:2]
        
        # Split text into lines
        lines = info_text.split('\n') if isinstance(info_text, str) else info_text
        
        # Add background rectangle for better readability
        text_height = 25 * len(lines)
        cv2.rectangle(image, (0, h - text_height - 10), (w, h), (0, 0, 0), -1)
        
        # Add text lines
        for i, line in enumerate(lines):
            y_pos = h - text_height + (i * 25) + 15
            cv2.putText(image, line, (10, y_pos), self.font,
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return image
    
    def _add_title(self, image, title):
        """Add title at the top of the image."""
        title_text = f"Debug Step: {title.replace('_', ' ').title()}"
        
        # Add background rectangle
        cv2.rectangle(image, (0, 0), (len(title_text) * 12, 30), (0, 0, 0), -1)
        
        # Add title text
        cv2.putText(image, title_text, (10, 20), self.font,
                   0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        return image
    
    def create_summary_image(self, results):
        """Create a summary image with key measurements and detection results."""
        if not self.enabled or not self.save_images:
            return
        
        # Create a summary canvas
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(canvas, f"Water Level Detection Summary", (50, 50),
                   self.font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add measurement results
        y_pos = 100
        if results:
            info_lines = [
                f"Image: {self.current_image_name}",
                f"Water Level: {results.get('water_level_cm', 'N/A')} cm",
                f"Scale Above Water: {results.get('scale_above_water_cm', 'N/A')} cm",
                f"Confidence: {results.get('confidence', 'N/A')}",
                f"Processing Time: {results.get('processing_time', 'N/A')} seconds"
            ]
            
            for line in info_lines:
                cv2.putText(canvas, line, (50, y_pos), self.font,
                           0.6, (255, 255, 255), 1, cv2.LINE_AA)
                y_pos += 30
        
        # Save summary
        filename = f"{self.current_image_name}_summary.jpg"
        output_path = self.session_dir / filename
        cv2.imwrite(str(output_path), canvas)
        
        self.logger.debug(f"Saved summary image: {output_path}")
    
    def is_enabled(self):
        """Check if debug mode is enabled."""
        return self.enabled
    
    def should_save_step(self, step_name):
        """Check if a specific step should be saved."""
        return self.enabled and step_name in self.steps_to_save