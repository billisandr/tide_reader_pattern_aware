"""
Hybrid Waterline Integration

Optional post-processing integration that can be called after E-pattern detection
to provide improved waterline suggestions without modifying existing code.

This integration:
- Is completely optional and non-intrusive
- Can be called from existing detection flows
- Preserves all original functionality  
- Creates separate debug outputs
- Returns suggestions that can be used or ignored
"""

import logging
from typing import Dict, List, Optional, Any
from .hybrid_waterline_analyzer import HybridWaterlineAnalyzer


def analyze_e_pattern_waterline(config: Dict[str, Any],
                               scale_region,
                               e_pattern_detector,
                               original_waterline_y: Optional[int],
                               image_path: Optional[str] = None,
                               debug_viz=None,
                               logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Optional post-processing function that can improve E-pattern waterline detection.
    
    This function can be called after E-pattern detection is complete to get
    an improved waterline suggestion. It does not modify any existing functionality.
    
    Args:
        config: System configuration
        scale_region: The same scale region used by E-pattern detector
        e_pattern_detector: The E-pattern detector instance (to access results)
        original_waterline_y: Original waterline Y position from E-pattern detector
        image_path: Optional path to original image for debug naming
        debug_viz: Optional debug visualizer instance for saving debug images
        logger: Optional logger instance
        
    Returns:
        Dictionary with analysis results:
        - improved_waterline_y: Suggested improved waterline Y position
        - confidence: Confidence in the improvement (0.0-1.0)
        - analysis_performed: Whether analysis was actually performed
        - reason: Reason for the result (improved_detection, confirmed_original, etc.)
        - original_waterline_y: Original waterline for reference
        - improvement_delta: Difference in pixels (if improved)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Check if hybrid analysis is enabled
    verification_config = config.get('detection', {}).get('pattern_aware', {}).get('waterline_verification', {})
    if not verification_config.get('enabled', False):
        logger.debug("Hybrid waterline verification disabled in config")
        return {
            'improved_waterline_y': original_waterline_y,
            'confidence': 0.0,
            'analysis_performed': False,
            'reason': 'disabled_in_config'
        }
    
    try:
        # Extract E-pattern matches from detector
        e_pattern_matches = getattr(e_pattern_detector, 'matched_patterns', [])
        
        if len(e_pattern_matches) < 2:
            logger.debug("Not enough E-patterns for hybrid analysis")
            return {
                'improved_waterline_y': original_waterline_y,
                'confidence': 0.0,
                'analysis_performed': False,
                'reason': 'insufficient_patterns'
            }
        
        # Initialize hybrid analyzer with debug visualizer for session directory
        analyzer = HybridWaterlineAnalyzer(config, debug_viz, logger)
        
        # Perform analysis
        result = analyzer.analyze_e_pattern_results(
            scale_region=scale_region,
            e_pattern_matches=e_pattern_matches,
            original_waterline_y=original_waterline_y,
            image_path=image_path
        )
        
        logger.info(f"Hybrid analysis completed: {result['reason']} "
                   f"(confidence: {result['confidence']:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Hybrid waterline analysis failed: {e}")
        return {
            'improved_waterline_y': original_waterline_y,
            'confidence': 0.0,
            'analysis_performed': False,
            'reason': f'analysis_error: {str(e)}'
        }


def apply_hybrid_waterline_improvement(original_waterline_y: Optional[int],
                                     hybrid_analysis_result: Dict[str, Any],
                                     acceptance_threshold: float = 0.6,
                                     logger: Optional[logging.Logger] = None) -> int:
    """
    Helper function to decide whether to use the hybrid waterline improvement.
    
    Args:
        original_waterline_y: Original waterline from E-pattern detection
        hybrid_analysis_result: Result from analyze_e_pattern_waterline()
        acceptance_threshold: Minimum confidence to accept improvement
        logger: Optional logger
        
    Returns:
        Final waterline Y position (improved or original)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not hybrid_analysis_result.get('analysis_performed', False):
        logger.debug(f"Hybrid analysis not performed: {hybrid_analysis_result.get('reason', 'unknown')}")
        return original_waterline_y
    
    improved_y = hybrid_analysis_result['improved_waterline_y'] 
    confidence = hybrid_analysis_result['confidence']
    reason = hybrid_analysis_result['reason']
    
    if reason == 'improved_detection' and confidence >= acceptance_threshold:
        improvement_delta = hybrid_analysis_result.get('improvement_delta', 0)
        logger.info(f"Applying hybrid waterline improvement: from Y={original_waterline_y} to Y={improved_y} "
                   f"(delta={improvement_delta:.1f}px, confidence={confidence:.3f})")
        return improved_y
    elif reason == 'confirmed_original':
        logger.info(f"Hybrid analysis confirms original waterline: Y={original_waterline_y}")
        return original_waterline_y
    else:
        logger.debug(f"Hybrid improvement not applied: {reason} (confidence={confidence:.3f})")
        return original_waterline_y


# Example usage function (for demonstration)
def example_integration_with_e_pattern_detector(config, scale_region, image_path, logger):
    """
    Example showing how to integrate hybrid analysis with existing E-pattern detection.
    
    This shows the pattern for non-intrusive integration:
    1. Run original E-pattern detection unchanged
    2. Optionally run hybrid analysis as post-processing
    3. Use improved result or stick with original
    """
    from .detection_methods.e_pattern_detector import EPatternDetector
    
    # Step 1: Run original E-pattern detection (unchanged)
    e_detector = EPatternDetector(config)
    original_waterline_y = e_detector.detect_waterline(scale_region, image_path)
    
    # Step 2: Optionally run hybrid analysis
    hybrid_result = analyze_e_pattern_waterline(
        config=config,
        scale_region=scale_region, 
        e_pattern_detector=e_detector,
        original_waterline_y=original_waterline_y,
        image_path=image_path,
        debug_viz=None,  # Pass debug visualizer if available
        logger=logger
    )
    
    # Step 3: Decide whether to use improvement
    final_waterline_y = apply_hybrid_waterline_improvement(
        original_waterline_y=original_waterline_y,
        hybrid_analysis_result=hybrid_result,
        logger=logger
    )
    
    return final_waterline_y, hybrid_result