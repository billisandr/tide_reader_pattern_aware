"""
Hybrid Waterline Analyzer

A completely separate analysis system that can optionally analyze E-pattern detection results
to improve waterline accuracy. This system does NOT modify any existing functionality and
operates as a post-processing step.

This analyzer:
1. Takes existing E-pattern detection results as input
2. Performs suspicious region and gradient analysis 
3. Saves results to separate debug directories
4. Returns improved waterline suggestion (optional to use)
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any


class HybridWaterlineAnalyzer:
    """
    Independent hybrid waterline analyzer that processes E-pattern detection results
    without modifying the original detection system.
    """
    
    def __init__(self, config: Dict[str, Any], debug_viz=None, logger: Optional[logging.Logger] = None, pixels_per_cm: float = None):
        """
        Initialize the hybrid analyzer.
        
        Args:
            config: System configuration
            debug_viz: Debug visualizer instance (to use same session directory)
            logger: Optional logger instance
            pixels_per_cm: Calibration data for pixel-to-cm conversion
        """
        self.config = config
        self.debug_viz = debug_viz
        self.logger = logger or logging.getLogger(__name__)
        self.pixels_per_cm = pixels_per_cm or 2.0  # Default fallback
        
        # Hybrid verification settings - with safe defaults
        verification_config = config.get('detection', {}).get('pattern_aware', {}).get('waterline_verification', {})
        
        # Core analysis parameters
        self.min_confidence_threshold = verification_config.get('min_pattern_confidence', 0.6)
        self.gradient_kernel_size = verification_config.get('gradient_kernel_size', 3)
        self.gradient_threshold = verification_config.get('gradient_threshold', 30)
        self.negative_gradient_threshold = verification_config.get('negative_gradient_threshold', 25)
        self.transition_search_height = verification_config.get('transition_search_height', 20)
        
        # Pattern analysis thresholds (consolidated configuration)
        pattern_config = verification_config.get('pattern_analysis', {})
        
        # Consecutive good pattern detection thresholds
        self.scale_consistency_threshold = pattern_config.get('scale_consistency_threshold', 0.15)
        self.size_consistency_threshold = pattern_config.get('size_consistency_threshold', 0.25)
        self.spacing_consistency_threshold = pattern_config.get('spacing_consistency_threshold', 0.50)
        self.min_consecutive_patterns = pattern_config.get('min_consecutive_patterns', 3)
        
        # Anomaly detection thresholds (applied after establishing baseline)
        self.scale_anomaly_threshold = pattern_config.get('scale_anomaly_threshold', 0.15)
        self.size_anomaly_threshold = pattern_config.get('size_anomaly_threshold', 0.20)
        self.aspect_ratio_anomaly_threshold = pattern_config.get('aspect_ratio_anomaly_threshold', 0.20)
        self.max_gap_ratio = pattern_config.get('max_gap_ratio', 2.0)
        
        # Underwater buffer configuration - percentage of average pattern height to search above last_good_y
        # This accounts for cases where last_good_y might be underwater
        self.underwater_buffer_percentage = pattern_config.get('underwater_buffer_percentage', 0.3)
        
        # Use debug visualizer's session directory if available
        self.debug_enabled = config.get('debug', {}).get('enabled', True) and debug_viz is not None
        
        self.logger.info(f"Hybrid Waterline Analyzer initialized as independent post-processor (pixels_per_cm: {self.pixels_per_cm:.2f})")
    
    def _pixel_to_cm(self, pixel_y: float) -> float:
        """
        Convert pixel Y coordinate to stadia rod reading in cm.
        
        Args:
            pixel_y: Y coordinate in pixels
            
        Returns:
            Stadia rod reading in cm
        """
        # For stadia rod: higher Y values = lower on scale = lower readings
        # This is a simplified conversion - may need adjustment based on specific scale setup
        scale_height_cm = self.config.get('scale', {}).get('total_height', 500.0)
        scale_height_pixels = scale_height_cm * self.pixels_per_cm
        
        # Convert Y pixel to reading (inverted: top of image = highest reading)
        reading_cm = scale_height_cm - (pixel_y / self.pixels_per_cm)
        return reading_cm
    
    def _cm_to_pixel(self, cm_reading: float) -> float:
        """
        Convert stadia rod reading in cm to pixel Y coordinate.
        
        Args:
            cm_reading: Stadia rod reading in cm
            
        Returns:
            Y coordinate in pixels
        """
        scale_height_cm = self.config.get('scale', {}).get('total_height', 500.0)
        pixel_y = (scale_height_cm - cm_reading) * self.pixels_per_cm
        return pixel_y
    
    def analyze_e_pattern_results(self, 
                                scale_region: np.ndarray,
                                e_pattern_matches: List[Dict[str, Any]], 
                                original_waterline_y: Optional[int],
                                image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze E-pattern detection results to suggest improved waterline position.
        
        Args:
            scale_region: The scale region image (same as used by E-pattern detector)
            e_pattern_matches: List of E-pattern matches from original detector
            original_waterline_y: Original waterline Y position from E-pattern detector
            image_path: Optional path to original image for debug naming
            
        Returns:
            Dictionary with analysis results and improved waterline suggestion
        """
        if len(e_pattern_matches) < 2:
            self.logger.warning("Need at least 2 E-patterns for hybrid analysis")
            return {
                'improved_waterline_y': original_waterline_y,
                'confidence': 0.0,
                'analysis_performed': False,
                'reason': 'insufficient_patterns'
            }
        
        self.logger.info(f"Starting hybrid analysis of {len(e_pattern_matches)} E-patterns")
        
        try:
            # Step 1: Analyze pattern continuity
            candidate_regions = self._analyze_pattern_continuity(e_pattern_matches)
            
            # Step 2: Analyze gradients in candidate regions  
            gradient_candidates = self._analyze_gradient_transitions(scale_region, candidate_regions, image_path)
            
            # Step 3: Select best waterline candidate
            analysis_result = self._select_best_waterline(
                candidate_regions, gradient_candidates, original_waterline_y
            )
            
            # Step 4: Save debug information if enabled
            if self.debug_enabled and self.debug_viz and image_path:
                self._save_hybrid_debug_info(
                    scale_region, e_pattern_matches, candidate_regions, 
                    gradient_candidates, analysis_result, image_path
                )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Hybrid analysis failed: {e}")
            return {
                'improved_waterline_y': original_waterline_y,
                'confidence': 0.0,
                'analysis_performed': False,
                'reason': f'analysis_error: {str(e)}'
            }
    
    def _analyze_pattern_continuity(self, e_pattern_matches: List[Dict[str, Any]]) -> List[Tuple[int, int, str, float]]:
        """
        Analyze E-pattern continuity to find waterline candidate regions around and below detected patterns.
        This ensures gradient analysis covers the most probable waterline locations.
        
        Returns:
            List of (y_min, y_max, reason, confidence) tuples for waterline candidate regions
        """
        # Handle edge case of no patterns
        if not e_pattern_matches:
            self.logger.warning("No E-patterns provided for analysis - cannot create scan regions")
            return []
        
        # Sort patterns by Y position
        sorted_patterns = sorted(e_pattern_matches, key=lambda p: p.get('center_y', p.get('global_y', 0)))
        
        # STEP 1: Analyze pattern sequence to find consecutive good patterns
        consecutive_good_patterns = self._find_consecutive_good_patterns(sorted_patterns)
        
        if consecutive_good_patterns < self.min_consecutive_patterns:
            self.logger.warning(f"Only {consecutive_good_patterns} consecutive good patterns found "
                              f"(minimum: {self.min_consecutive_patterns}) - using available patterns for analysis")
            # Even with fewer patterns, we should still create scan regions below the last pattern
            # Use all available patterns as baseline
        
        # Determine how many patterns to use for baseline
        patterns_to_use = max(consecutive_good_patterns, 1)  # Use at least 1 pattern
        patterns_to_use = min(patterns_to_use, len(sorted_patterns))  # Don't exceed available patterns
        
        self.logger.info(f"Using {patterns_to_use} patterns for baseline establishment")
        
        # STEP 2: Calculate baseline from available patterns  
        baseline_patterns = sorted_patterns[:patterns_to_use]
        baseline_sizes = []
        baseline_aspects = []
        baseline_scales = []
        baseline_heights = []
        
        for pattern in baseline_patterns:
            template_size = pattern.get('template_size', (20, 15))
            size = template_size[0] * template_size[1]
            aspect_ratio = template_size[1] / template_size[0] if template_size[0] > 0 else 1.0
            scale_factor = pattern.get('scale_factor', 1.0)
            # Get pattern height for buffer calculations
            pattern_height = pattern.get('pattern_height_px', template_size[1] if len(template_size) > 1 else 15)
            
            baseline_sizes.append(size)
            baseline_aspects.append(aspect_ratio)
            baseline_scales.append(scale_factor)
            baseline_heights.append(pattern_height)
        
        baseline_size = np.median(baseline_sizes)
        baseline_aspect = np.median(baseline_aspects)
        baseline_scale = np.median(baseline_scales)
        average_pattern_height = np.median(baseline_heights) if baseline_heights else 20.0
        
        # Calculate expected spacing from good patterns
        spacings = []
        for i in range(len(baseline_patterns) - 1):
            current_y = baseline_patterns[i].get('center_y', baseline_patterns[i].get('global_y', 0))
            next_y = baseline_patterns[i + 1].get('center_y', baseline_patterns[i + 1].get('global_y', 0))
            spacings.append(abs(next_y - current_y))
        
        baseline_spacing = np.median(spacings) if spacings else 50.0
        
        # Position after which we expect anomalies to indicate water interface
        # Use the BOTTOM of the last good pattern as the reference point
        last_good_pattern = baseline_patterns[-1]
        last_good_center_y = last_good_pattern.get('center_y', last_good_pattern.get('global_y', 0))
        last_good_height = last_good_pattern.get('pattern_height_px', 
                                               last_good_pattern.get('template_size', (20, 15))[1] if 
                                               last_good_pattern.get('template_size', (20, 15)) else 15)
        last_good_pattern_y = last_good_center_y + (last_good_height // 2)  # Bottom of last good pattern
        
        self.logger.info(f"Baseline established from {len(baseline_patterns)} patterns: "
                        f"Size={baseline_size:.0f}, Scale={baseline_scale:.2f}, Spacing={baseline_spacing:.1f}")
        self.logger.info(f"Last good pattern bottom Y={last_good_pattern_y} (center: {last_good_center_y})")
        self.logger.info(f"Will create systematic scan regions below LOWEST detected pattern")
        
        # STEP 3: Create systematic waterline candidate regions AROUND the lowest E-pattern baseline
        # The most probable waterline location is around the baseline, not just strictly below it
        
        candidate_regions = []
        
        # Find the LOWEST (highest Y value) pattern - this is our reference point
        lowest_pattern = sorted_patterns[-1]  # Last in sorted list has highest Y
        lowest_center_y = lowest_pattern.get('center_y', lowest_pattern.get('global_y', 0))
        lowest_height = lowest_pattern.get('pattern_height_px', 
                                         lowest_pattern.get('template_size', (20, 15))[1] if 
                                         lowest_pattern.get('template_size', (20, 15)) else 15)
        lowest_pattern_bottom_y = lowest_center_y + (lowest_height // 2)  # Bottom of lowest pattern
        
        # Calculate buffer for region around baseline
        baseline_buffer = average_pattern_height * self.underwater_buffer_percentage
        region_height = self.transition_search_height * 2  # Make regions larger for better gradient detection
        
        self.logger.info(f"Using LOWEST pattern at center Y={lowest_center_y}, bottom Y={lowest_pattern_bottom_y}")
        self.logger.info(f"Creating candidate regions AROUND baseline with {baseline_buffer:.1f}px buffer")
        
        # MOST IMPORTANT: Create primary candidate region AROUND the baseline
        # This covers the most probable waterline location: around the last detected pattern
        primary_region_start = lowest_pattern_bottom_y - baseline_buffer  # Start ABOVE bottom of pattern
        primary_region_end = lowest_pattern_bottom_y + baseline_buffer + region_height  # Extend well below
        candidate_regions.append((primary_region_start, primary_region_end, "around_baseline", 1.0))
        
        # Create additional candidate regions extending further downward for completeness
        current_scan_y = primary_region_end
        region_count = 1
        max_additional_regions = 2  # Fewer regions since primary region covers most probable area
        
        while region_count <= max_additional_regions:
            next_region_y_min = current_scan_y
            next_region_y_max = next_region_y_min + region_height
            candidate_regions.append((next_region_y_min, next_region_y_max, f"extended_scan_{region_count}", 0.6))
            
            current_scan_y = next_region_y_max
            region_count += 1
        
        # Store analysis results for waterline enforcement and debug info
        self.first_anomaly_y = primary_region_start  # Primary candidate region start
        self.last_good_pattern_y = lowest_pattern_bottom_y  # Use LOWEST pattern bottom as reference
        self.consecutive_good_patterns = patterns_to_use  # Use actual patterns used, not consecutive count
        self.average_pattern_height = average_pattern_height
        
        self.logger.info(f"Created {len(candidate_regions)} waterline candidate regions")
        self.logger.info(f"Primary region (around baseline): Y={primary_region_start:.1f} to Y={primary_region_end:.1f}")
        self.logger.info(f"All regions cover Y={primary_region_start:.1f} to Y={current_scan_y:.1f}")
        
        return candidate_regions
    
    def _find_consecutive_good_patterns(self, sorted_patterns: List[Dict[str, Any]]) -> int:
        """
        Find the number of consecutive good patterns from the start of the sequence.
        Good patterns have consistent scale factors, sizes, and spacing.
        
        Returns:
            int: Number of consecutive good patterns from start
        """
        if len(sorted_patterns) < 2:
            return len(sorted_patterns)
        
        # Calculate initial metrics from first pattern
        first_pattern = sorted_patterns[0]
        first_template_size = first_pattern.get('template_size', (20, 15))
        first_size = first_template_size[0] * first_template_size[1]
        first_aspect = first_template_size[1] / first_template_size[0] if first_template_size[0] > 0 else 1.0
        first_scale = first_pattern.get('scale_factor', 1.0)
        
        consecutive_count = 1  # First pattern is always "good" by definition
        
        # Check each subsequent pattern
        for i in range(1, len(sorted_patterns)):
            current = sorted_patterns[i]
            previous = sorted_patterns[i-1]
            
            # Get pattern metrics
            current_template_size = current.get('template_size', (20, 15))
            current_size = current_template_size[0] * current_template_size[1]
            current_aspect = current_template_size[1] / current_template_size[0] if current_template_size[0] > 0 else 1.0
            current_scale = current.get('scale_factor', 1.0)
            
            # Check scale factor consistency (most important)
            scale_change = abs(current_scale - first_scale) / first_scale if first_scale > 0 else 0
            if scale_change > self.scale_consistency_threshold:
                self.logger.debug(f"Pattern {i} breaks sequence: scale change {scale_change:.2f} > {self.scale_consistency_threshold} "
                                f"({current_scale:.2f} vs {first_scale:.2f})")
                break
            
            # Check size consistency
            size_change = abs(current_size - first_size) / first_size if first_size > 0 else 0
            if size_change > self.size_consistency_threshold:
                self.logger.debug(f"Pattern {i} breaks sequence: size change {size_change:.2f} > {self.size_consistency_threshold}")
                break
            
            # Check spacing consistency with previous pattern
            current_y = current.get('center_y', current.get('global_y', 0))
            previous_y = previous.get('center_y', previous.get('global_y', 0))
            current_spacing = abs(current_y - previous_y)
            
            # If we have multiple patterns, check against established spacing
            if i >= 2:
                # Calculate average spacing from previous good patterns
                spacings = []
                for j in range(i-1):
                    y1 = sorted_patterns[j].get('center_y', sorted_patterns[j].get('global_y', 0))
                    y2 = sorted_patterns[j+1].get('center_y', sorted_patterns[j+1].get('global_y', 0))
                    spacings.append(abs(y2 - y1))
                avg_spacing = sum(spacings) / len(spacings)
                
                spacing_change = abs(current_spacing - avg_spacing) / avg_spacing if avg_spacing > 0 else 0
                if spacing_change > self.spacing_consistency_threshold:
                    self.logger.debug(f"Pattern {i} breaks sequence: spacing change {spacing_change:.2f} > {self.spacing_consistency_threshold} "
                                    f"({current_spacing:.1f} vs avg {avg_spacing:.1f})")
                    break
            
            # Pattern passed all tests - increment consecutive count
            consecutive_count += 1
            self.logger.debug(f"Pattern {i} is good: scale={current_scale:.2f}, size={current_size:.0f}")
        
        return consecutive_count
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, str, float]]) -> List[Tuple[int, int, str, float]]:
        """Merge overlapping suspicious regions."""
        if len(regions) <= 1:
            return regions
        
        sorted_regions = sorted(regions, key=lambda r: r[0])
        merged = []
        current = sorted_regions[0]
        
        for next_region in sorted_regions[1:]:
            if current[1] >= next_region[0]:  # Overlapping
                # Merge regions
                current = (
                    current[0],  # Keep y_min
                    max(current[1], next_region[1]),  # Extend y_max
                    f"{current[2]}+{next_region[2]}",  # Combine reasons
                    max(current[3], next_region[3])  # Take higher confidence
                )
            else:
                merged.append(current)
                current = next_region
        
        merged.append(current)
        return merged
    
    def _analyze_gradient_transitions(self, 
                                    scale_region: np.ndarray, 
                                    candidate_regions: List[Tuple[int, int, str, float]],
                                    image_path: Optional[str] = None) -> List[Tuple[int, float]]:
        """
        Analyze gradient transitions in waterline candidate regions with enforced waterline positioning.
        
        Returns:
            List of (y_position, confidence) tuples for waterline candidates
        """
        if not candidate_regions:
            return []
        
        # Convert to grayscale if needed
        if len(scale_region.shape) == 3:
            gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = scale_region.copy()
        
        waterline_candidates = []
        
        # Prepare gradient analysis data for logging
        gradient_analysis_data = {
            'image_path': image_path,
            'regions': [],
            'overall_gradient_profile': None,
            'overall_gradient_differences': None
        }
        
        # Calculate overall gradient profile for the entire image for logging
        if self.debug_enabled and image_path:
            grad_y_full = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.gradient_kernel_size)
            grad_magnitude_full = np.abs(grad_y_full)
            gradient_profile_full = np.mean(grad_magnitude_full, axis=1)
            gradient_diff_full = np.diff(gradient_profile_full) if len(gradient_profile_full) > 1 else np.array([])
            
            gradient_analysis_data['overall_gradient_profile'] = gradient_profile_full
            gradient_analysis_data['overall_gradient_differences'] = gradient_diff_full
        
        # Since we've fixed suspicious regions to only be BELOW last_good_y, 
        # the constraint is simpler: waterline must be at or below last_good_y
        last_good_y = getattr(self, 'last_good_pattern_y', None)
        first_anomaly_y = getattr(self, 'first_anomaly_y', None)
        average_pattern_height = getattr(self, 'average_pattern_height', 20.0)
        
        # Calculate buffer: configurable percentage of average pattern height for edge cases
        underwater_buffer = average_pattern_height * self.underwater_buffer_percentage
        
        # Since regions are now properly defined below last_good_y, use a more conservative constraint
        if last_good_y is not None:
            # Allow small buffer for measurement precision, but regions should already be below last_good_y
            min_allowed_y = last_good_y - (underwater_buffer * 0.5)  # Reduced buffer since regions are corrected
            constraint_desc = f"last good pattern Y={last_good_y} with reduced buffer (-{underwater_buffer * 0.5:.1f}px)"
        else:
            min_allowed_y = first_anomaly_y
            constraint_desc = f"first anomaly Y={first_anomaly_y}"
        
        if min_allowed_y is not None:
            self.logger.info(f"Waterline constraint: Y >= {min_allowed_y:.1f} ({constraint_desc})")
            self.logger.info(f"Regions now properly defined below last good pattern at Y={last_good_y}")
        
        for y_min, y_max, reason, region_confidence in candidate_regions:
            self.logger.debug(f"Analyzing gradients in candidate region Y={y_min}-{y_max} (reason: {reason})")
            
            # Extract region of interest
            y_start = max(0, int(y_min))
            y_end = min(gray.shape[0], int(y_max))
            
            # ENFORCE: Only analyze regions at or below the buffered constraint
            if min_allowed_y is not None:
                if y_end <= min_allowed_y:
                    self.logger.debug(f"Skipping region Y={y_min}-{y_max} - entirely above constraint at Y={min_allowed_y:.1f}")
                    continue
                # Constrain region to be at or below the buffered position  
                y_start = max(y_start, int(min_allowed_y))
            
            if y_end - y_start < 3:
                continue
                
            roi = gray[y_start:y_end, :]
            
            # Calculate VERTICAL gradients (Y-axis changes) to detect horizontal waterline transitions
            # This is correct for waterline detection: look for vertical color changes, not horizontal ones
            grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=self.gradient_kernel_size)
            grad_magnitude = np.abs(grad_y)
            
            # Get gradient profile (average per row) - this shows Y-axis transitions
            gradient_profile = np.mean(grad_magnitude, axis=1)
            
            # Store region gradient data for logging
            region_data = {
                'y_start': y_start,
                'y_end': y_end,
                'reason': reason,
                'region_confidence': region_confidence,
                'gradient_profile': gradient_profile.copy() if self.debug_enabled and image_path else None,
                'gradient_differences': np.diff(gradient_profile) if len(gradient_profile) > 1 and self.debug_enabled and image_path else None,
                'candidates_found': []
            }
            if self.debug_enabled and image_path:
                gradient_analysis_data['regions'].append(region_data)
            
            # Find transitions with improved negative gradient differential detection
            if len(gradient_profile) > 1:
                gradient_diff = np.diff(gradient_profile)
                
                # Improved waterline detection: prioritize topmost negative gradient differential in first candidate region
                is_first_region = "around_baseline" in reason
                topmost_negative_found = False
                
                for i in range(1, len(gradient_diff) - 1):
                    gradient_diff_value = gradient_diff[i]
                    
                    # Check for significant gradient changes
                    is_significant_change = abs(gradient_diff_value) > self.gradient_threshold
                    is_negative_differential = gradient_diff_value < 0 and abs(gradient_diff_value) > self.negative_gradient_threshold
                    
                    if is_significant_change:
                        # Calculate confidence
                        local_gradient = gradient_profile[i] if i < len(gradient_profile) else 0
                        surrounding_variance = np.var(gradient_profile[max(0, i-3):min(len(gradient_profile), i+4)])
                        
                        # Base confidence on gradient strength and local variance
                        gradient_confidence = min(local_gradient / 100.0, 1.0) * min(surrounding_variance / 50.0, 1.0)
                        
                        # IMPROVED: Give highest priority to negative gradient differentials in first region
                        if is_first_region and is_negative_differential:
                            gradient_confidence *= 2.0  # Significantly boost confidence for negative differentials
                            
                            # If this is the topmost negative differential in first region, give it max priority
                            if not topmost_negative_found:
                                gradient_confidence *= 1.5  # Additional boost for topmost negative
                                topmost_negative_found = True
                                self.logger.info(f"Found topmost negative gradient differential at Y={y_start + i} "
                                               f"(diff: {gradient_diff_value:.1f}, threshold: {self.negative_gradient_threshold})")
                        
                        # Apply region-based confidence scaling for non-negative differentials
                        elif "around_baseline" in reason:
                            gradient_confidence *= 1.2  # Higher confidence for first region below patterns
                        elif "extended_scan" in reason:
                            gradient_confidence *= 0.9  # Slightly lower confidence for extended regions
                        
                        waterline_y = y_start + i
                        
                        # Final check: waterline must be at or below the buffered constraint
                        if min_allowed_y is not None and waterline_y < min_allowed_y:
                            self.logger.debug(f"Rejected gradient candidate at Y={waterline_y} - above constraint Y={min_allowed_y:.1f}")
                            continue
                        
                        final_confidence = min(gradient_confidence, 3.0)  # Allow higher confidence for negative differentials
                        waterline_candidates.append((waterline_y, final_confidence))
                        
                        # Store candidate info in region data for logging  
                        is_topmost_negative = is_first_region and is_negative_differential and not topmost_negative_found
                        if self.debug_enabled and image_path and gradient_analysis_data['regions']:
                            gradient_analysis_data['regions'][-1]['candidates_found'].append({
                                'y_position': waterline_y,
                                'confidence': final_confidence,
                                'local_gradient': local_gradient,
                                'surrounding_variance': surrounding_variance,
                                'gradient_diff_value': gradient_diff_value,
                                'is_negative_differential': is_negative_differential,
                                'is_topmost_negative': is_topmost_negative
                            })
                        
                        detection_type = "NEGATIVE DIFF" if is_negative_differential else "GENERAL"
                        self.logger.debug(f"Valid gradient transition ({detection_type}) at Y={waterline_y}, "
                                        f"confidence: {final_confidence:.3f}, diff: {gradient_diff_value:.1f}")
        
        # Sort by confidence and remove duplicates
        unique_candidates = list(set(waterline_candidates))
        sorted_candidates = sorted(unique_candidates, key=lambda x: x[1], reverse=True)
        
        if min_allowed_y is not None and sorted_candidates:
            self.logger.info(f"Found {len(sorted_candidates)} valid waterline candidates below first anomaly")
        
        # Save detailed gradient analysis to text file
        if self.debug_enabled and image_path and self.debug_viz:
            self._save_gradient_analysis_text_file(gradient_analysis_data, sorted_candidates)
        
        return sorted_candidates
    
    def _select_best_waterline(self, 
                             suspicious_regions: List[Tuple[int, int, str, float]],
                             gradient_candidates: List[Tuple[int, float]],
                             original_waterline_y: Optional[int]) -> Dict[str, Any]:
        """
        Select the best waterline candidate from analysis results.
        """
        if not gradient_candidates:
            self.logger.info("No gradient candidates found")
            return {
                'improved_waterline_y': original_waterline_y,
                'confidence': 0.0,
                'analysis_performed': True,
                'reason': 'no_gradient_candidates',
                'suspicious_regions': suspicious_regions,
                'gradient_candidates': []
            }
        
        # Get best candidate
        best_y, best_confidence = gradient_candidates[0]
        
        # Final validation: ensure waterline is after last good pattern 
        # Since regions are now properly defined below last_good_y, use consistent reduced buffer
        last_good_y = getattr(self, 'last_good_pattern_y', None)
        first_anomaly_y = getattr(self, 'first_anomaly_y', None)
        average_pattern_height = getattr(self, 'average_pattern_height', 20.0)
        underwater_buffer = average_pattern_height * self.underwater_buffer_percentage
        
        if last_good_y is not None:
            min_allowed_y = last_good_y - (underwater_buffer * 0.5)  # Same reduced buffer as gradient analysis
            constraint_desc = f"buffered last good pattern Y={min_allowed_y:.1f} (original: {last_good_y}, reduced buffer)"
        else:
            min_allowed_y = first_anomaly_y
            constraint_desc = f"first anomaly Y={min_allowed_y}"
        
        if min_allowed_y is not None and best_y < min_allowed_y:
            self.logger.warning(f"Best waterline candidate Y={best_y} is above {constraint_desc} - rejecting")
            return {
                'improved_waterline_y': original_waterline_y,
                'confidence': 0.0,
                'analysis_performed': True,
                'reason': 'waterline_above_constraint',
                'suspicious_regions': suspicious_regions,
                'gradient_candidates': gradient_candidates
            }
        
        # Check if confidence meets threshold
        if best_confidence >= self.min_confidence_threshold:
            # Use improved waterline if significantly different or original failed
            if original_waterline_y is None or abs(best_y - original_waterline_y) > 10:
                self.logger.info(f"Hybrid analysis suggests improved waterline: Y={best_y} "
                               f"(original: {original_waterline_y}, confidence: {best_confidence:.3f})")
                return {
                    'improved_waterline_y': best_y,
                    'confidence': best_confidence,
                    'analysis_performed': True,
                    'reason': 'improved_detection',
                    'original_waterline_y': original_waterline_y,
                    'improvement_delta': abs(best_y - original_waterline_y) if original_waterline_y else 0,
                    'suspicious_regions': suspicious_regions,
                    'gradient_candidates': gradient_candidates
                }
            else:
                self.logger.info(f"Hybrid analysis confirms original waterline: Y={original_waterline_y}")
                return {
                    'improved_waterline_y': original_waterline_y,
                    'confidence': best_confidence,
                    'analysis_performed': True,
                    'reason': 'confirmed_original',
                    'suspicious_regions': suspicious_regions,
                    'gradient_candidates': gradient_candidates
                }
        else:
            self.logger.warning(f"Best candidate confidence too low: {best_confidence:.3f} < {self.min_confidence_threshold}")
            return {
                'improved_waterline_y': original_waterline_y,
                'confidence': best_confidence,
                'analysis_performed': True,
                'reason': 'low_confidence',
                'suspicious_regions': suspicious_regions,
                'gradient_candidates': gradient_candidates
            }
    
    def _save_hybrid_debug_info(self,
                               scale_region: np.ndarray,
                               e_pattern_matches: List[Dict[str, Any]],
                               suspicious_regions: List[Tuple[int, int, str, float]],
                               gradient_candidates: List[Tuple[int, float]],
                               analysis_result: Dict[str, Any],
                               image_path: str):
        """Save hybrid analysis debug images using the existing debug visualizer."""
        try:
            # 1. Save suspicious regions visualization
            suspicious_regions_image = self._create_suspicious_regions_visualization(
                scale_region, e_pattern_matches, suspicious_regions, analysis_result
            )
            
            # Create detailed info text for side panel
            candidate_info_lines = [
                f"Analysis Result: {analysis_result['reason']}",
                f"Confidence: {analysis_result['confidence']:.3f}",
                f"E-patterns analyzed: {len(e_pattern_matches)}",
                f"Suspicious regions found: {len(suspicious_regions)}",
                "",
                "Configuration:",
                f"  Size anomaly threshold: {self.size_anomaly_threshold}",
                f"  Aspect ratio anomaly threshold: {self.aspect_ratio_anomaly_threshold}",
                f"  Max gap ratio: {self.max_gap_ratio}",
                f"  Underwater buffer: {self.underwater_buffer_percentage:.1%}",
                ""
            ]
            
            # Add pattern sequence analysis info
            consecutive_patterns = getattr(self, 'consecutive_good_patterns', 0)
            last_good_y = getattr(self, 'last_good_pattern_y', None)
            average_pattern_height = getattr(self, 'average_pattern_height', 20.0)
            
            if consecutive_patterns > 0:
                underwater_buffer = average_pattern_height * self.underwater_buffer_percentage
                buffered_constraint = last_good_y - underwater_buffer if last_good_y else None
                
                candidate_info_lines.extend([
                    "",
                    "Pattern Sequence Analysis:",
                    f"  Consecutive good patterns: {consecutive_patterns}",
                    f"  Last good pattern Y: {last_good_y}",
                    f"  Average pattern height: {average_pattern_height:.1f}px",
                    f"  Underwater buffer: {underwater_buffer:.1f}px ({self.underwater_buffer_percentage:.1%})",
                    f"  Buffered constraint Y: {buffered_constraint:.1f}" if buffered_constraint else "  No buffered constraint"
                ])
            
            # Add details for each suspicious region
            for i, (y_min, y_max, reason, confidence) in enumerate(suspicious_regions):
                stadia_min = self._pixel_to_cm(y_min)
                stadia_max = self._pixel_to_cm(y_max)
                candidate_info_lines.extend([
                    f"Region {i+1}:",
                    f"  Y-range: {y_min:.1f} - {y_max:.1f}",
                    f"  Stadia range: {stadia_max:.1f} - {stadia_min:.1f} cm",
                    f"  Reason: {reason.replace('_', ' ')}",
                    f"  Confidence: {confidence:.3f}"
                ])
            
            self.debug_viz.save_debug_image(
                suspicious_regions_image, 'waterline_candidate_regions',
                info_text='\n'.join(candidate_info_lines)
            )
            
            # 2. Save gradient analysis visualization
            gradient_analysis_image = self._create_gradient_analysis_visualization(
                scale_region, suspicious_regions, gradient_candidates
            )
            
            # Create detailed gradient info
            last_good_y = getattr(self, 'last_good_pattern_y', None)
            average_pattern_height = getattr(self, 'average_pattern_height', 20.0)
            underwater_buffer = average_pattern_height * self.underwater_buffer_percentage
            buffered_constraint = last_good_y - underwater_buffer if last_good_y else None
            
            gradient_info_lines = [
                f"Gradient candidates found: {len(gradient_candidates)}",
                f"Gradient threshold: {self.gradient_threshold}",
                f"Negative gradient threshold: {self.negative_gradient_threshold}",
                f"Pixels per cm: {self.pixels_per_cm:.3f}",
                f"Kernel size: {self.gradient_kernel_size}",
                f"Search height: {self.transition_search_height}px",
                "",
                "Waterline Constraints:",
                f"  Last good pattern Y: {last_good_y}" if last_good_y else "  No last good pattern",
                f"  Underwater buffer: {underwater_buffer:.1f}px ({self.underwater_buffer_percentage:.1%})" if last_good_y else "",
                f"  Min allowed Y: {buffered_constraint:.1f}" if buffered_constraint else "  No Y constraint",
                ""
            ]
            
            # Add top gradient candidates with stadia readings
            if gradient_candidates:
                gradient_info_lines.append("Top Waterline Candidates:")
                for i, (y_pos, confidence) in enumerate(gradient_candidates[:10]):
                    stadia_reading = self._pixel_to_cm(y_pos)
                    gradient_info_lines.append(f"  {i+1}: Y={y_pos:.1f} ({stadia_reading:.1f} cm), confidence={confidence:.3f}")
            else:
                gradient_info_lines.append("No waterline candidates found")
            
            self.debug_viz.save_debug_image(
                gradient_analysis_image, 'waterline_gradient_analysis',
                info_text='\n'.join(gradient_info_lines)
            )
            
            # 2b. Save clean gradient analysis visualization (without annotations)
            clean_gradient_analysis_image = self._create_clean_gradient_analysis_visualization(scale_region)
            self.debug_viz.save_debug_image(
                clean_gradient_analysis_image, 'waterline_gradient_analysis_clean',
                info_text=None
            )
            
            # 3. Create verification analysis summary
            verification_summary = self._create_verification_summary_image(
                scale_region, e_pattern_matches, suspicious_regions, 
                gradient_candidates, analysis_result
            )
            
            # Create comprehensive summary info
            original_y = analysis_result.get('original_waterline_y', 'N/A')
            improved_y = analysis_result['improved_waterline_y']
            
            summary_info_lines = [
                "WATERLINE VERIFICATION SUMMARY",
                "",
                f"Pixels per cm: {self.pixels_per_cm:.3f}",
                ""
            ]
            
            # Add original waterline info with stadia reading
            if original_y != 'N/A' and original_y is not None:
                original_stadia = self._pixel_to_cm(original_y)
                summary_info_lines.append(f"Original waterline: Y={original_y} ({original_stadia:.1f} cm)")
            else:
                summary_info_lines.append("Original waterline: N/A")
            
            # Add improved waterline info with stadia reading  
            if improved_y is not None:
                improved_stadia = self._pixel_to_cm(improved_y)
                summary_info_lines.append(f"Improved waterline: Y={improved_y:.1f} ({improved_stadia:.1f} cm)")
            else:
                summary_info_lines.append("Improved waterline: N/A")
                
            summary_info_lines.extend([
                "",
                f"Analysis result: {analysis_result['reason']}",
                f"Confidence: {analysis_result['confidence']:.3f}"
            ])
            
            if 'improvement_delta' in analysis_result:
                delta_pixels = analysis_result['improvement_delta']
                delta_cm = delta_pixels / self.pixels_per_cm
                summary_info_lines.append(f"Improvement delta: {delta_pixels:.1f} pixels ({delta_cm:.2f} cm)")
            
            # Add buffer information to summary
            last_good_y = getattr(self, 'last_good_pattern_y', None)
            average_pattern_height = getattr(self, 'average_pattern_height', 20.0)
            underwater_buffer = average_pattern_height * self.underwater_buffer_percentage if last_good_y else 0
            
            summary_info_lines.extend([
                "",
                "Pattern Analysis:",
                f"  E-patterns: {len(e_pattern_matches)}",
                f"  Suspicious regions: {len(suspicious_regions)}",
                f"  Gradient candidates: {len(gradient_candidates)}",
                "",
                "Buffer Configuration:",
                f"  Underwater buffer: {self.underwater_buffer_percentage:.1%} of pattern height",
                f"  Buffer size: {underwater_buffer:.1f}px" if underwater_buffer > 0 else "  No buffer applied"
            ])
            
            self.debug_viz.save_debug_image(
                verification_summary, 'waterline_verification_analysis',
                info_text='\n'.join(summary_info_lines)
            )
            
            self.logger.info(f"Saved hybrid debug images via debug visualizer")
            
        except Exception as e:
            self.logger.error(f"Failed to save hybrid debug info: {e}")
    
    def _create_suspicious_regions_visualization(self,
                                               scale_region: np.ndarray,
                                               e_pattern_matches: List[Dict[str, Any]], 
                                               suspicious_regions: List[Tuple[int, int, str, float]],
                                               analysis_result: Dict[str, Any]) -> np.ndarray:
        """Create clean visualization showing suspicious regions and patterns for side panel display."""
        # Create color version
        if len(scale_region.shape) == 2:
            debug_image = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2BGR)
        else:
            debug_image = scale_region.copy()
        
        height, width = debug_image.shape[:2]
        
        # Colors - using distinct colors for clear visual identification
        colors = {
            'valid_pattern': (0, 255, 0),      # Green - valid E-patterns
            'suspicious_pattern': (0, 165, 255),  # Orange - E-patterns in suspicious regions
            'suspicious_region': (0, 255, 255),   # Yellow - suspicious regions
            'waterline': (0, 0, 255),            # Red - detected waterline
        }
        
        # Draw E-patterns as clean rectangles
        for i, match in enumerate(e_pattern_matches):
            x = match.get('local_x', match.get('x', 0))
            y = match.get('global_y', match.get('y', 0))
            template_size = match.get('template_size', (15, 20))
            w_match, h_match = template_size[1], template_size[0]  # width, height
            
            # Check if pattern is in suspicious region
            pattern_y = match.get('center_y', y + h_match // 2)
            is_suspicious = any(
                region_y_min <= pattern_y <= region_y_max
                for region_y_min, region_y_max, _, _ in suspicious_regions
            )
            
            color = colors['suspicious_pattern'] if is_suspicious else colors['valid_pattern']
            
            # Draw pattern box with simple numbering
            cv2.rectangle(debug_image, (x, y), (x + w_match, y + h_match), color, 2)
            
            # Add simple pattern number (minimal text overlay)
            cv2.putText(debug_image, f"E{i+1}", (x + 2, max(15, y - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw suspicious regions as semi-transparent overlays
        for i, (y_min, y_max, reason, confidence) in enumerate(suspicious_regions):
            y_start = max(0, int(y_min))
            y_end = min(height, int(y_max))
            
            # Semi-transparent overlay
            overlay = debug_image.copy()
            cv2.rectangle(overlay, (0, y_start), (width, y_end), colors['suspicious_region'], -1)
            cv2.addWeighted(debug_image, 0.8, overlay, 0.2, 0, debug_image)
            
            # Clean region border with simple numbering
            cv2.rectangle(debug_image, (0, y_start), (width, y_end), colors['suspicious_region'], 2)
            cv2.putText(debug_image, f"R{i+1}", (5, y_start + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['suspicious_region'], 1)
        
        # Draw waterline as clean line
        waterline_y = analysis_result['improved_waterline_y']
        if waterline_y is not None:
            cv2.line(debug_image, (0, waterline_y), (width, waterline_y), colors['waterline'], 3)
        
        return debug_image
    
    def _create_gradient_analysis_visualization(self,
                                              scale_region: np.ndarray,
                                              suspicious_regions: List[Tuple[int, int, str, float]],
                                              gradient_candidates: List[Tuple[int, float]]) -> np.ndarray:
        """Create clean gradient analysis visualization for side panel display."""
        if len(scale_region.shape) == 3:
            gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = scale_region.copy()
        
        # Calculate Y-axis gradients for waterline detection visualization  
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.gradient_kernel_size)
        grad_magnitude = np.abs(grad_y)
        
        # Normalize for visualization
        grad_vis = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad_color = cv2.applyColorMap(grad_vis, cv2.COLORMAP_JET)
        
        # Create side-by-side comparison without title text overlays
        combined = np.hstack([
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            grad_color
        ])
        
        height, width = combined.shape[:2]
        
        # Add clean suspicious region overlays
        for i, (y_min, y_max, reason, confidence) in enumerate(suspicious_regions):
            y_start = max(0, int(y_min))
            y_end = min(height, int(y_max))
            
            # Draw clean borders on both sides without text
            cv2.rectangle(combined, (0, y_start), (width // 2, y_end), (0, 255, 255), 2)
            cv2.rectangle(combined, (width // 2, y_start), (width, y_end), (0, 255, 255), 2)
            
            # Simple region number only
            cv2.putText(combined, f"R{i+1}", (5, y_start + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Add clean gradient candidate lines
        for i, (y_pos, confidence) in enumerate(gradient_candidates[:5]):
            if 0 <= y_pos < height:
                cv2.line(combined, (0, y_pos), (width, y_pos), (255, 0, 255), 2)
                cv2.putText(combined, f"G{i+1}", (width - 30, y_pos - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        return combined
    
    def _create_clean_gradient_analysis_visualization(self,
                                                    scale_region: np.ndarray) -> np.ndarray:
        """Create clean gradient analysis visualization without annotations."""
        if len(scale_region.shape) == 3:
            gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = scale_region.copy()
        
        # Calculate Y-axis gradients for waterline detection visualization  
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.gradient_kernel_size)
        grad_magnitude = np.abs(grad_y)
        
        # Normalize for visualization
        grad_vis = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad_color = cv2.applyColorMap(grad_vis, cv2.COLORMAP_JET)
        
        # Create side-by-side comparison without any annotations
        combined = np.hstack([
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            grad_color
        ])
        
        return combined
    
    def _create_verification_summary_image(self,
                                         scale_region: np.ndarray,
                                         e_pattern_matches: List[Dict[str, Any]],
                                         suspicious_regions: List[Tuple[int, int, str, float]],
                                         gradient_candidates: List[Tuple[int, float]],
                                         analysis_result: Dict[str, Any]) -> np.ndarray:
        """Create clean verification summary image for side panel display."""
        if len(scale_region.shape) == 2:
            summary_image = cv2.cvtColor(scale_region, cv2.COLOR_GRAY2BGR)
        else:
            summary_image = scale_region.copy()
        
        height, width = summary_image.shape[:2]
        
        # Draw E-pattern locations as simple circles
        for i, pattern in enumerate(e_pattern_matches):
            center_y = pattern.get('center_y', pattern.get('global_y', 0))
            if 0 <= center_y < height:
                cv2.circle(summary_image, (width // 4, center_y), 4, (0, 255, 0), 2)
                cv2.putText(summary_image, f"E{i+1}", (width // 4 + 8, center_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Draw clean suspicious regions
        for i, (y_min, y_max, reason, confidence) in enumerate(suspicious_regions):
            y_start = max(0, int(y_min))
            y_end = min(height, int(y_max))
            
            # Semi-transparent overlay
            overlay = summary_image.copy()
            cv2.rectangle(overlay, (0, y_start), (width, y_end), (0, 255, 255), -1)
            summary_image = cv2.addWeighted(summary_image, 0.8, overlay, 0.2, 0)
            
            # Clean border with simple numbering
            cv2.rectangle(summary_image, (0, y_start), (width, y_end), (0, 255, 255), 2)
            cv2.putText(summary_image, f"R{i+1}", (5, y_start + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Draw original waterline as clean line
        original_y = analysis_result.get('original_waterline_y')
        if original_y is not None and 0 <= original_y < height:
            cv2.line(summary_image, (0, original_y), (width, original_y), (0, 0, 255), 3)
        
        # Draw improved waterline (if different) as clean line
        improved_y = analysis_result['improved_waterline_y']
        if improved_y is not None and improved_y != original_y and 0 <= improved_y < height:
            cv2.line(summary_image, (0, improved_y), (width, improved_y), (255, 0, 255), 3)
        
        # Add top gradient candidates as simple lines
        for i, (y_pos, confidence) in enumerate(gradient_candidates[:3]):
            if 0 <= y_pos < height:
                cv2.line(summary_image, (width//2, y_pos), (width, y_pos), (128, 128, 255), 1)
                cv2.putText(summary_image, f"G{i+1}", (width - 25, y_pos - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 255), 1)
        
        return summary_image
    
    def _save_gradient_analysis_text_file(self, gradient_analysis_data: Dict[str, Any], sorted_candidates: List[Tuple[int, float]]):
        """
        Save detailed gradient analysis data to a text file in the waterline_gradient_analysis directory.
        """
        try:
            from datetime import datetime
            from pathlib import Path
            import os
            
            # Extract image name from path
            image_path = gradient_analysis_data.get('image_path', 'unknown_image')
            if image_path:
                image_name = Path(image_path).stem
            else:
                image_name = 'unknown_image'
            
            # Use debug visualizer's session directory to maintain consistency
            if hasattr(self.debug_viz, 'session_dir') and self.debug_viz.session_dir:
                base_debug_dir = self.debug_viz.session_dir
            else:
                # Fallback to creating our own directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_debug_dir = Path("data/debug") / f"pattern_aware_debug_session_{timestamp}"
            
            # Create waterline_gradient_analysis subdirectory
            gradient_debug_dir = Path(base_debug_dir) / "waterline_gradient_analysis"
            gradient_debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Create text file name
            text_file_name = f"{image_name}_waterline_gradient_analysis.txt"
            text_file_path = gradient_debug_dir / text_file_name
            
            # Prepare analysis content
            content_lines = []
            content_lines.append("=" * 80)
            content_lines.append(f"WATERLINE GRADIENT ANALYSIS REPORT")
            content_lines.append(f"Image: {image_name}")
            content_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content_lines.append("=" * 80)
            content_lines.append("")
            
            # Overall gradient analysis
            content_lines.append("OVERALL IMAGE GRADIENT ANALYSIS:")
            content_lines.append("-" * 50)
            
            overall_profile = gradient_analysis_data.get('overall_gradient_profile')
            overall_differences = gradient_analysis_data.get('overall_gradient_differences')
            
            if overall_profile is not None and overall_differences is not None:
                content_lines.append(f"Image height: {len(overall_profile)} pixels")
                content_lines.append(f"Gradient kernel size: {self.gradient_kernel_size}")
                content_lines.append(f"Gradient threshold: {self.gradient_threshold}")
                content_lines.append(f"Negative gradient threshold: {self.negative_gradient_threshold}")
                content_lines.append(f"Pixels per cm calibration: {self.pixels_per_cm:.3f}")
                content_lines.append(f"Scale total height: {self.config.get('scale', {}).get('total_height', 500.0)} cm")
                content_lines.append("")
                
                content_lines.append("GRADIENT DIFFERENCES BY Y POSITION:")
                content_lines.append("Y_Position\tStadia_Reading_cm\tGradient_Value\tGradient_Difference\tAbove_Threshold\tNegative_Diff")
                content_lines.append("-" * 110)
                
                for y_pos in range(len(overall_profile)):
                    gradient_value = overall_profile[y_pos]
                    stadia_reading_cm = self._pixel_to_cm(y_pos)
                    
                    # Get gradient difference (if available)
                    if y_pos < len(overall_differences):
                        grad_diff = overall_differences[y_pos]
                        above_threshold = "YES" if abs(grad_diff) > self.gradient_threshold else "NO"
                        is_negative_diff = "YES" if grad_diff < 0 and abs(grad_diff) > self.negative_gradient_threshold else "NO"
                    else:
                        grad_diff = 0.0
                        above_threshold = "N/A"
                        is_negative_diff = "N/A"
                    
                    content_lines.append(f"{y_pos:8d}\t{stadia_reading_cm:14.1f}\t{gradient_value:14.3f}\t{grad_diff:17.3f}\t{above_threshold:>13}\t{is_negative_diff:>12}")
                
                content_lines.append("")
                content_lines.append("SUMMARY STATISTICS:")
                content_lines.append(f"  Max gradient value: {np.max(overall_profile):.3f}")
                content_lines.append(f"  Min gradient value: {np.min(overall_profile):.3f}")
                content_lines.append(f"  Mean gradient value: {np.mean(overall_profile):.3f}")
                content_lines.append(f"  Std gradient value: {np.std(overall_profile):.3f}")
                
                if len(overall_differences) > 0:
                    content_lines.append(f"  Max gradient difference: {np.max(np.abs(overall_differences)):.3f}")
                    content_lines.append(f"  Positions above threshold: {np.sum(np.abs(overall_differences) > self.gradient_threshold)}")
                
                content_lines.append("")
            else:
                content_lines.append("Overall gradient analysis data not available.")
                content_lines.append("")
            
            # Region-specific analysis
            content_lines.append("REGION-SPECIFIC GRADIENT ANALYSIS:")
            content_lines.append("-" * 50)
            
            regions = gradient_analysis_data.get('regions', [])
            for i, region in enumerate(regions):
                content_lines.append(f"\nREGION {i+1}:")
                content_lines.append(f"  Y Range: {region['y_start']} - {region['y_end']}")
                content_lines.append(f"  Reason: {region['reason']}")
                content_lines.append(f"  Region Confidence: {region['region_confidence']:.3f}")
                
                region_profile = region.get('gradient_profile')
                region_differences = region.get('gradient_differences')
                
                if region_profile is not None and region_differences is not None:
                    content_lines.append(f"  Region height: {len(region_profile)} pixels")
                    content_lines.append("")
                    content_lines.append(f"  Region Gradient Details (Y positions {region['y_start']}-{region['y_end']}):")
                    content_lines.append("  Y_Pos\tStadia_Reading_cm\tGradient_Value\tGradient_Difference\tAbove_Threshold\tNegative_Diff")
                    content_lines.append("  " + "-" * 105)
                    
                    for j in range(len(region_profile)):
                        y_global = region['y_start'] + j
                        gradient_value = region_profile[j]
                        stadia_reading_cm = self._pixel_to_cm(y_global)
                        
                        if j < len(region_differences):
                            grad_diff = region_differences[j]
                            above_threshold = "YES" if abs(grad_diff) > self.gradient_threshold else "NO"
                            is_negative_diff = "YES" if grad_diff < 0 and abs(grad_diff) > self.negative_gradient_threshold else "NO"
                        else:
                            grad_diff = 0.0
                            above_threshold = "N/A"
                            is_negative_diff = "N/A"
                        
                        content_lines.append(f"  {y_global:5d}\t{stadia_reading_cm:14.1f}\t{gradient_value:14.3f}\t{grad_diff:17.3f}\t{above_threshold:>13}\t{is_negative_diff:>12}")
                
                # Add candidate information
                candidates = region.get('candidates_found', [])
                if candidates:
                    content_lines.append(f"\n  WATERLINE CANDIDATES FOUND IN THIS REGION:")
                    for j, candidate in enumerate(candidates):
                        y_position = candidate['y_position']
                        stadia_reading = self._pixel_to_cm(y_position)
                        content_lines.append(f"    Candidate {j+1}:")
                        content_lines.append(f"      Y Position: {y_position}")
                        content_lines.append(f"      Stadia Reading: {stadia_reading:.1f} cm")
                        content_lines.append(f"      Confidence: {candidate['confidence']:.3f}")
                        content_lines.append(f"      Local Gradient: {candidate['local_gradient']:.3f}")
                        content_lines.append(f"      Surrounding Variance: {candidate['surrounding_variance']:.3f}")
                        content_lines.append(f"      Gradient Diff Value: {candidate['gradient_diff_value']:.3f}")
                        content_lines.append(f"      Is Negative Differential: {candidate.get('is_negative_differential', False)}")
                        content_lines.append(f"      Is Topmost Negative: {candidate.get('is_topmost_negative', False)}")
                else:
                    content_lines.append(f"\n  No waterline candidates found in this region.")
            
            # Final candidates summary
            content_lines.append("")
            content_lines.append("FINAL WATERLINE CANDIDATES:")
            content_lines.append("-" * 50)
            if sorted_candidates:
                content_lines.append("Rank\tY_Position\tStadia_Reading_cm\tConfidence")
                content_lines.append("-" * 50)
                for i, (y_pos, confidence) in enumerate(sorted_candidates):
                    stadia_reading = self._pixel_to_cm(y_pos)
                    content_lines.append(f"{i+1:4d}\t{y_pos:10d}\t{stadia_reading:14.1f}\t{confidence:10.3f}")
            else:
                content_lines.append("No valid waterline candidates found.")
            
            content_lines.append("")
            content_lines.append("=" * 80)
            content_lines.append("END OF ANALYSIS")
            content_lines.append("=" * 80)
            
            # Write to file
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))
            
            self.logger.info(f"Saved detailed gradient analysis to: {text_file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save gradient analysis text file: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")