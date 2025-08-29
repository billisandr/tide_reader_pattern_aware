"""
Detection Methods Module

Individual detection algorithms for pattern-aware water level detection.
Each method is designed to handle scales with repetitive markings and patterns.
"""

from .template_matching import TemplateMatchingDetector
from .morphological_detector import MorphologicalDetector  
from .frequency_analyzer import FrequencyAnalyzer
from .lsd_detector import LSDDetector
from .contour_analyzer import ContourAnalyzer
from .integrated_detector import IntegratedPatternDetector

__all__ = [
    'TemplateMatchingDetector',
    'MorphologicalDetector', 
    'FrequencyAnalyzer',
    'LSDDetector',
    'ContourAnalyzer',
    'IntegratedPatternDetector'
]