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
    
    def __init__(self, config: Dict[str, Any], debug_viz=None, logger: Optional[logging.Logger] = None):
        """
        Initialize the hybrid analyzer.
        
        Args:
            config: System configuration
            debug_viz: Debug visualizer instance (to use same session directory)
            logger: Optional logger instance
        """
        self.config = config
        self.debug_viz = debug_viz
        self.logger = logger or logging.getLogger(__name__)
        
        # Hybrid verification settings - with safe defaults
        verification_config = config.get('detection', {}).get('pattern_aware', {}).get('waterline_verification', {})
        
        # Core analysis parameters
        self.min_confidence_threshold = verification_config.get('min_pattern_confidence', 0.6)
        self.gradient_kernel_size = verification_config.get('gradient_kernel_size', 3)
        self.gradient_threshold = verification_config.get('gradient_threshold', 30)
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
        
        # Use debug visualizer's session directory if available
        self.debug_enabled = config.get('debug', {}).get('enabled', True) and debug_viz is not None
        
        self.logger.info("Hybrid Waterline Analyzer initialized as independent post-processor")
    
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
            suspicious_regions = self._analyze_pattern_continuity(e_pattern_matches)
            
            # Step 2: Analyze gradients in suspicious regions  
            gradient_candidates = self._analyze_gradient_transitions(scale_region, suspicious_regions)
            
            # Step 3: Select best waterline candidate
            analysis_result = self._select_best_waterline(
                suspicious_regions, gradient_candidates, original_waterline_y
            )
            
            # Step 4: Save debug information if enabled
            if self.debug_enabled and self.debug_viz and image_path:
                self._save_hybrid_debug_info(
                    scale_region, e_pattern_matches, suspicious_regions, 
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
        Analyze E-pattern continuity to find suspicious regions ONLY after consecutive good patterns.
        This ensures gradient analysis only runs where we expect water interface transitions.
        
        Returns:
            List of (y_min, y_max, reason, confidence) tuples for suspicious regions
        """
        # Sort patterns by Y position
        sorted_patterns = sorted(e_pattern_matches, key=lambda p: p.get('center_y', p.get('global_y', 0)))
        
        # STEP 1: Analyze pattern sequence to find consecutive good patterns
        consecutive_good_patterns = self._find_consecutive_good_patterns(sorted_patterns)
        
        if consecutive_good_patterns < self.min_consecutive_patterns:
            self.logger.warning(f"Only {consecutive_good_patterns} consecutive good patterns found - "
                              f"insufficient for reliable waterline analysis (minimum: {self.min_consecutive_patterns})")
            return []
        
        self.logger.info(f"Found {consecutive_good_patterns} consecutive good patterns - establishing baseline")
        
        # STEP 2: Calculate baseline from consecutive good patterns ONLY
        baseline_patterns = sorted_patterns[:consecutive_good_patterns]
        baseline_sizes = []
        baseline_aspects = []
        baseline_scales = []
        
        for pattern in baseline_patterns:
            template_size = pattern.get('template_size', (20, 15))
            size = template_size[0] * template_size[1]
            aspect_ratio = template_size[1] / template_size[0] if template_size[0] > 0 else 1.0
            scale_factor = pattern.get('scale_factor', 1.0)
            
            baseline_sizes.append(size)
            baseline_aspects.append(aspect_ratio)
            baseline_scales.append(scale_factor)
        
        baseline_size = np.median(baseline_sizes)
        baseline_aspect = np.median(baseline_aspects)
        baseline_scale = np.median(baseline_scales)
        
        # Calculate expected spacing from good patterns
        spacings = []
        for i in range(len(baseline_patterns) - 1):
            current_y = baseline_patterns[i].get('center_y', baseline_patterns[i].get('global_y', 0))
            next_y = baseline_patterns[i + 1].get('center_y', baseline_patterns[i + 1].get('global_y', 0))
            spacings.append(abs(next_y - current_y))
        
        baseline_spacing = np.median(spacings) if spacings else 50.0
        
        # Position after which we expect anomalies to indicate water interface
        last_good_pattern_y = baseline_patterns[-1].get('center_y', baseline_patterns[-1].get('global_y', 0))
        
        self.logger.info(f"Baseline established from {len(baseline_patterns)} patterns: "
                        f"Size={baseline_size:.0f}, Scale={baseline_scale:.2f}, Spacing={baseline_spacing:.1f}")
        self.logger.info(f"Looking for water interface anomalies after Y={last_good_pattern_y}")
        
        # STEP 3: Find suspicious regions ONLY in patterns after the consecutive good series
        suspicious_regions = []
        first_anomaly_y = None
        
        # Only analyze patterns after the established good sequence
        for i in range(consecutive_good_patterns, len(sorted_patterns)):
            current = sorted_patterns[i]
            current_y = current.get('center_y', current.get('global_y', 0))
            
            # Skip patterns that come before our established baseline
            if current_y <= last_good_pattern_y:
                continue
                
            # Analyze this pattern for anomalies
            anomaly_found, anomaly_type, confidence = self._check_pattern_anomaly(
                current, baseline_size, baseline_aspect, baseline_scale, baseline_spacing
            )
            
            if anomaly_found:
                # Determine region bounds for gradient analysis
                if i < len(sorted_patterns) - 1:
                    next_pattern = sorted_patterns[i + 1]
                    next_y = next_pattern.get('center_y', next_pattern.get('global_y', 0))
                    y_max = next_y + self.transition_search_height
                else:
                    # Last pattern - extend search area downward
                    y_max = current_y + self.transition_search_height * 2
                
                y_min = max(last_good_pattern_y, current_y - self.transition_search_height)
                
                suspicious_regions.append((y_min, y_max, anomaly_type, confidence))
                
                # Track first anomaly after good patterns
                if first_anomaly_y is None:
                    first_anomaly_y = current_y
                
                self.logger.info(f"Water interface anomaly detected at Y={current_y}: {anomaly_type} "
                               f"(confidence: {confidence:.3f}) - enabling gradient analysis")
        
        # Store analysis results for waterline enforcement and debug info
        self.first_anomaly_y = first_anomaly_y
        self.last_good_pattern_y = last_good_pattern_y
        self.consecutive_good_patterns = consecutive_good_patterns
        
        if first_anomaly_y is not None:
            self.logger.info(f"First water interface anomaly at Y={first_anomaly_y}")
        else:
            self.logger.info("No water interface anomalies detected after good pattern sequence")
        
        # Merge overlapping regions
        merged_regions = self._merge_overlapping_regions(suspicious_regions)
        self.logger.info(f"Found {len(merged_regions)} suspicious regions")
        
        return merged_regions
    
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
    
    def _check_pattern_anomaly(self, pattern: Dict[str, Any], baseline_size: float, 
                             baseline_aspect: float, baseline_scale: float, 
                             baseline_spacing: float) -> Tuple[bool, str, float]:
        """
        Check if a pattern shows anomalies compared to baseline.
        
        Returns:
            Tuple of (anomaly_found, anomaly_type, confidence)
        """
        template_size = pattern.get('template_size', (20, 15))
        size = template_size[0] * template_size[1]
        aspect_ratio = template_size[1] / template_size[0] if template_size[0] > 0 else 1.0
        scale_factor = pattern.get('scale_factor', 1.0)
        
        # Check scale factor anomalies (highest priority)
        scale_change = abs(scale_factor - baseline_scale) / baseline_scale if baseline_scale > 0 else 0
        if scale_change > self.scale_anomaly_threshold or scale_factor < baseline_scale * 0.7:
            confidence = min(scale_change * 2, 1.0)
            return True, "scale_factor_anomaly", confidence
        
        # Check size anomalies
        size_change = abs(size - baseline_size) / baseline_size if baseline_size > 0 else 0
        if size_change > self.size_anomaly_threshold:
            confidence = min(size_change, 1.0)
            return True, "size_anomaly", confidence
        
        # Check aspect ratio anomalies
        aspect_change = abs(aspect_ratio - baseline_aspect) / baseline_aspect if baseline_aspect > 0 else 0
        if aspect_change > self.aspect_ratio_anomaly_threshold:
            confidence = min(aspect_change, 1.0)
            return True, "aspect_ratio_change", confidence
        
        return False, "none", 0.0
    
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
                                    suspicious_regions: List[Tuple[int, int, str, float]]) -> List[Tuple[int, float]]:
        """
        Analyze gradient transitions in suspicious regions with enforced waterline positioning.
        
        Returns:
            List of (y_position, confidence) tuples for waterline candidates
        """
        if not suspicious_regions:
            return []
        
        # Convert to grayscale if needed
        if len(scale_region.shape) == 3:
            gray = cv2.cvtColor(scale_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = scale_region.copy()
        
        waterline_candidates = []
        
        # CRITICAL RULE: Waterline must be after last good pattern in sequence
        last_good_y = getattr(self, 'last_good_pattern_y', None)
        first_anomaly_y = getattr(self, 'first_anomaly_y', None)
        
        # Use the stricter constraint (last good pattern position)
        min_allowed_y = last_good_y if last_good_y is not None else first_anomaly_y
        
        if min_allowed_y is not None:
            self.logger.info(f"Enforcing waterline constraint: Y >= {min_allowed_y} "
                           f"({'last good pattern' if last_good_y else 'first anomaly'} position)")
        
        for y_min, y_max, reason, region_confidence in suspicious_regions:
            self.logger.debug(f"Analyzing gradients in region Y={y_min}-{y_max} (reason: {reason})")
            
            # Extract region of interest
            y_start = max(0, int(y_min))
            y_end = min(gray.shape[0], int(y_max))
            
            # ENFORCE: Only analyze regions at or below first anomaly
            if min_allowed_y is not None:
                if y_end <= min_allowed_y:
                    self.logger.debug(f"Skipping region Y={y_min}-{y_max} - entirely above first anomaly at Y={min_allowed_y}")
                    continue
                # Constrain region to be below first anomaly
                y_start = max(y_start, min_allowed_y)
            
            if y_end - y_start < 3:
                continue
                
            roi = gray[y_start:y_end, :]
            
            # Calculate horizontal gradients
            grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=self.gradient_kernel_size)
            grad_magnitude = np.abs(grad_x)
            
            # Get gradient profile (average per row)
            gradient_profile = np.mean(grad_magnitude, axis=1)
            
            # Find transitions
            if len(gradient_profile) > 1:
                gradient_diff = np.diff(gradient_profile)
                
                for i in range(1, len(gradient_diff) - 1):
                    if abs(gradient_diff[i]) > self.gradient_threshold:
                        # Calculate confidence
                        local_gradient = gradient_profile[i] if i < len(gradient_profile) else 0
                        surrounding_variance = np.var(gradient_profile[max(0, i-3):min(len(gradient_profile), i+4)])
                        
                        gradient_confidence = min(local_gradient / 100.0, 1.0) * min(surrounding_variance / 50.0, 1.0)
                        
                        # Boost confidence based on region reason (scale factor anomalies are most reliable)
                        if "scale_factor_anomaly" in reason:
                            gradient_confidence *= 1.5  # Highest boost for scale anomalies
                        elif "gap_detected" in reason:
                            gradient_confidence *= 1.2
                        elif "size_anomaly" in reason:
                            gradient_confidence *= 1.1
                        
                        waterline_y = y_start + i
                        
                        # Final check: waterline must be below first anomaly
                        if min_allowed_y is not None and waterline_y < min_allowed_y:
                            self.logger.debug(f"Rejected gradient candidate at Y={waterline_y} - above first anomaly")
                            continue
                        
                        final_confidence = min(gradient_confidence, 1.0)
                        waterline_candidates.append((waterline_y, final_confidence))
                        
                        self.logger.debug(f"Valid gradient transition at Y={waterline_y}, confidence: {final_confidence:.3f}")
        
        # Sort by confidence and remove duplicates
        unique_candidates = list(set(waterline_candidates))
        sorted_candidates = sorted(unique_candidates, key=lambda x: x[1], reverse=True)
        
        if min_allowed_y is not None and sorted_candidates:
            self.logger.info(f"Found {len(sorted_candidates)} valid waterline candidates below first anomaly")
        
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
        last_good_y = getattr(self, 'last_good_pattern_y', None)
        first_anomaly_y = getattr(self, 'first_anomaly_y', None)
        min_allowed_y = last_good_y if last_good_y is not None else first_anomaly_y
        
        if min_allowed_y is not None and best_y < min_allowed_y:
            constraint_type = 'last good pattern' if last_good_y else 'first anomaly'
            self.logger.warning(f"Best waterline candidate Y={best_y} is above {constraint_type} Y={min_allowed_y} - rejecting")
            return {
                'improved_waterline_y': original_waterline_y,
                'confidence': 0.0,
                'analysis_performed': True,
                'reason': 'waterline_above_first_anomaly',
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
            suspicious_info_lines = [
                f"Analysis Result: {analysis_result['reason']}",
                f"Confidence: {analysis_result['confidence']:.3f}",
                f"E-patterns analyzed: {len(e_pattern_matches)}",
                f"Suspicious regions found: {len(suspicious_regions)}",
                "",
                "Configuration:",
                f"  Size anomaly threshold: {self.size_anomaly_threshold}",
                f"  Aspect ratio anomaly threshold: {self.aspect_ratio_anomaly_threshold}",
                f"  Max gap ratio: {self.max_gap_ratio}",
                ""
            ]
            
            # Add pattern sequence analysis info
            consecutive_patterns = getattr(self, 'consecutive_good_patterns', 0)
            last_good_y = getattr(self, 'last_good_pattern_y', None)
            
            if consecutive_patterns > 0:
                suspicious_info_lines.extend([
                    "",
                    "Pattern Sequence Analysis:",
                    f"  Consecutive good patterns: {consecutive_patterns}",
                    f"  Last good pattern Y: {last_good_y}",
                    f"  Analysis after Y: {last_good_y}"
                ])
            
            # Add details for each suspicious region
            for i, (y_min, y_max, reason, confidence) in enumerate(suspicious_regions):
                suspicious_info_lines.extend([
                    f"Region {i+1}:",
                    f"  Y-range: {y_min:.1f} - {y_max:.1f}",
                    f"  Reason: {reason.replace('_', ' ')}",
                    f"  Confidence: {confidence:.3f}"
                ])
            
            self.debug_viz.save_debug_image(
                suspicious_regions_image, 'waterline_suspicious_regions',
                info_text='\n'.join(suspicious_info_lines)
            )
            
            # 2. Save gradient analysis visualization
            gradient_analysis_image = self._create_gradient_analysis_visualization(
                scale_region, suspicious_regions, gradient_candidates
            )
            
            # Create detailed gradient info
            gradient_info_lines = [
                f"Gradient candidates found: {len(gradient_candidates)}",
                f"Gradient threshold: {self.gradient_threshold}",
                f"Kernel size: {self.gradient_kernel_size}",
                f"Search height: {self.transition_search_height}px",
                ""
            ]
            
            # Add top gradient candidates
            for i, (y_pos, confidence) in enumerate(gradient_candidates[:10]):
                gradient_info_lines.append(f"Candidate {i+1}: Y={y_pos:.1f}, confidence={confidence:.3f}")
            
            self.debug_viz.save_debug_image(
                gradient_analysis_image, 'waterline_gradient_analysis',
                info_text='\n'.join(gradient_info_lines)
            )
            
            # 3. Create verification analysis summary
            verification_summary = self._create_verification_summary_image(
                scale_region, e_pattern_matches, suspicious_regions, 
                gradient_candidates, analysis_result
            )
            
            # Create comprehensive summary info
            summary_info_lines = [
                "WATERLINE VERIFICATION SUMMARY",
                "",
                f"Original waterline Y: {analysis_result.get('original_waterline_y', 'N/A')}",
                f"Improved waterline Y: {analysis_result['improved_waterline_y']}",
                f"Analysis result: {analysis_result['reason']}",
                f"Confidence: {analysis_result['confidence']:.3f}"
            ]
            
            if 'improvement_delta' in analysis_result:
                summary_info_lines.append(f"Improvement delta: {analysis_result['improvement_delta']:.1f} pixels")
            
            summary_info_lines.extend([
                "",
                "Pattern Analysis:",
                f"  E-patterns: {len(e_pattern_matches)}",
                f"  Suspicious regions: {len(suspicious_regions)}",
                f"  Gradient candidates: {len(gradient_candidates)}"
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
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.gradient_kernel_size)
        grad_magnitude = np.abs(grad_x)
        
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