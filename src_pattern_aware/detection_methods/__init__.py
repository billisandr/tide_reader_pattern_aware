"""
Detection Methods Module

Individual detection algorithms for pattern-aware water level detection.
Each method is designed to handle scales with repetitive markings and patterns.
"""

from .template_matching import TemplateMatchingDetector
from .morphological_detector import MorphologicalDetector
from .integrated_detector import IntegratedPatternDetector

__all__ = [
    'TemplateMatchingDetector',
    'MorphologicalDetector',
    'IntegratedPatternDetector'
]