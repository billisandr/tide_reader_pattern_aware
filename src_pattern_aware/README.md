# Pattern-Aware Water Level Detection System

*Advanced water level detection for stadia rods using OpenCV pattern_matching*

> Disclaimer: This work is part of a non-funded prototype research idea conducted at the [SenseLAB](http://senselab.tuc.gr/) of the [TUC](https://www.tuc.gr/el/archi).

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Status-Beta-orange.svg)](https://github.com/)

## Overview

The Pattern-Aware Water Level Detection System is an advanced module designed specifically for measuring water levels on scales that have complex markings, numbers, and repetitive patterns. Traditional detection methods often struggle with these scales because they cannot adequately distinguish between scale markings (text, numbers, lines) and actual water interfaces.

This system uses multiple pattern recognition techniques to solve this fundamental problem, providing accurate water level measurements even on scales with heavy visual noise.

**Key Problem Solved:** Traditional computer vision approaches fail on scales with regular markings because they detect the markings as "edges" or "interfaces" rather than the actual water surface. This system learns to recognize and suppress scale-specific patterns while accurately detecting the true water interface.

## Table of Contents

- [Overview](#overview)
- [Key Applications](#key-applications)
- [Features](#features)
- [Implementation Status](#implementation-status)
- [Quick Start Guide](#quick-start-guide)
- [Architecture](#architecture)
- [Detection Methods](#detection-methods)
- [Configuration](#configuration)
- [Usage](#usage)
- [Integration with Standard System](#integration-with-standard-system)
- [Development Status](#development-status)
- [Future Work](#future-work)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)
- [Contributing](#contributing)

**Key Applications:**

- **Tide gauges with numerical markings** - Scales with numbers every centimeter
- **Industrial scales with text overlays** - Equipment with printed specifications
- **Laboratory equipment** - Precision scales with fine markings
- **Ruler-based measurements** - Physical rulers with cm/inch markings
- **Vintage or weathered scales** - Older scales with faded or irregular markings
- **Multi-language scales** - Scales with text in various languages
- **Research instruments** - Scientific equipment with complex measurement annotations
- **Maritime applications** - Port and harbor water level monitoring
- **Environmental monitoring** - River and lake level measurement with marked structures

## Features

### Core Pattern Recognition Capabilities (WORKING)

- **Template-Based Marking Suppression** - Learns and suppresses scale-specific marking patterns
- **Morphological Interface Detection** - Separates horizontal water interfaces from vertical markings
- **Multi-Method Integration Framework** - Combines multiple detection approaches for reliability
- **Intelligent Fallback Protection** - Automatically falls back to standard detection when needed
- **Hybrid Detection Mode** - Runs both pattern-aware and standard detection simultaneously
- **Configuration-Driven Operation** - Extensively configurable through YAML and environment variables
- **Zero-Risk Integration** - Completely separate from standard system, no interference

### Advanced Features (WORKING)

- **Scale-Specific Adaptation** - Adapts detection parameters to individual scale characteristics
- **Confidence Scoring** - Provides detailed reliability metrics for all detections
- **Comprehensive Logging** - Detailed logging of pattern analysis and decision processes
- **Debug Visualization Framework** - Foundation for pattern analysis debugging
- **Performance Monitoring** - Tracks detection method performance and accuracy

### Planned Advanced Features (FUTURE WORK)

- **Automated Template Learning** - Extract marking templates directly from calibration images
- **Advanced Pattern Classification** - Machine learning-based pattern recognition
- **Template Persistence System** - Save, version, and share learned templates
- **Real-Time Processing Optimization** - Optimized for continuous monitoring applications
- **Multi-Scale Support** - Handle multiple scales in single image
- **Adaptive Threshold Learning** - Self-adjusting detection parameters

## Implementation Status

### WORKING FEATURES

#### Phase 1: Infrastructure (COMPLETE)

- **Directory Structure** - Complete module organization
- **Base PatternWaterDetector Class** - Main detection framework
- **Configuration Integration** - Pattern-aware settings in config.yaml
- **System Selection Logic** - Environment and config-based mode selection
- **Hybrid Detection** - Simultaneous standard and pattern-aware detection

#### Phase 2: Detection Methods (PARTIAL)

- **Template Matching Detector** - WORKING - Suppresses learned marking patterns
- **Morphological Detector** - WORKING - Horizontal vs vertical feature separation

### FUTURE WORK FEATURES

#### Phase 2: Remaining Detection Methods (PLANNED)

- **Frequency Analysis Detector** - FFT-based periodicity analysis for rejecting repetitive patterns
- **Line Segment Detector (LSD)** - Precise line detection with geometric filtering
- **Contour Analysis Detector** - Shape-based water interface detection
- **Integrated Pattern Detector** - Multi-method consensus system

#### Phase 3: Pattern Analysis (PLANNED)

- **Template Extraction** - Automated marking template extraction from calibration images
- **Pattern Classification** - Intelligent classification of markings vs water interfaces
- **Template Management** - Template storage, versioning, and reuse system
- **Enhanced Calibration** - Pattern-aware calibration extensions

#### Phase 4: Integration & Testing (PLANNED)

- **Advanced Debug Visualization** - Pattern analysis debug images
- **Performance Optimization** - Speed and accuracy improvements
- **Comprehensive Testing** - Validation across different scale types

## Installation

### Prerequisites

The pattern-aware system shares the same requirements as the standard system:

- **Python 3.9+** with pip
- **OpenCV 4.9+** for computer vision operations
- **NumPy** for numerical computations
- **PyYAML** for configuration management

### System Dependencies

```bash
# Core dependencies (same as standard system)
pip install opencv-python numpy scipy
pip install PyYAML Pillow matplotlib pandas
pip install sqlalchemy python-dotenv

# Or simply:
pip install -r requirements.txt
```

### Pattern-Aware System Installation

The pattern-aware system is already included in the project structure. No additional installation is required beyond the standard system dependencies.

```bash
# Verify installation
python -c "from src_pattern_aware.pattern_water_detector import PatternWaterDetector; print('Pattern-aware system ready')"
```

### Directory Structure Setup

```bash
# Ensure pattern template directories exist
mkdir -p data/pattern_templates/scale_markings
mkdir -p data/pattern_templates/calibration_patterns
mkdir -p data/debug/pattern_analysis
```

## Quick Start Guide

### Step 1: Enable Pattern-Aware Detection

**Method 1: Environment Variable (Recommended for Testing)**

```bash
# Windows Command Prompt
set PATTERN_AWARE_MODE=true
python src_pattern_aware/main_pattern_aware.py
```

**Method 2: Configuration File**

```yaml
# In config.yaml
pattern_processing:
  mode: 'pattern_aware'  # Options: 'standard', 'pattern_aware', 'hybrid'

detection:
  pattern_aware:
    enabled: true
```

### Step 2: Run Detection

```bash
# Using pattern-aware entry point
python src_pattern_aware/main_pattern_aware.py

# Or using hybrid mode (runs both systems)
set PATTERN_AWARE_MODE=hybrid
python src_pattern_aware/main_pattern_aware.py
```

### Step 3: Monitor Results

**Expected Log Output:**

```
INFO - Starting Pattern-Aware Water Level Detection System
INFO - Selected detection system: pattern_aware
INFO - Pattern-Aware Water Level Detector Initialized
INFO - Initialized 2 pattern detection methods
INFO - Template matching detector initialized (threshold: 0.7)
INFO - Morphological detector initialized (h_kernel: [40, 1], v_kernel: [1, 40])
```

### Step 4: Compare with Standard Detection

```bash
# Run hybrid mode for comparison
set PATTERN_AWARE_MODE=hybrid
python src_pattern_aware/main_pattern_aware.py
```

**Hybrid Output Example:**

```
INFO - Hybrid processing: IMG_0154.JPG
INFO - Running standard detection...
INFO - Standard detection: 301.3cm (confidence: 0.886)
INFO - Running pattern-aware detection...
INFO - Pattern-aware detection: 445.5cm (confidence: 0.950)
INFO - Selected pattern-aware result (higher confidence)
```

## Architecture

### System Structure

```
src_pattern_aware/
├── main_pattern_aware.py              # Entry point with system selection
├── pattern_water_detector.py          # Main pattern-aware detector
├── hybrid_detector.py                 # Hybrid detection (both systems)
├── detection_methods/                 # Individual detection algorithms
│   ├── template_matching.py           # WORKING: Template-based suppression
│   ├── morphological_detector.py      # WORKING: Horizontal interface detection
│   ├── frequency_analyzer.py          # PLANNED: FFT periodicity analysis
│   ├── lsd_detector.py               # PLANNED: Line Segment Detector
│   ├── contour_analyzer.py           # PLANNED: Geometric contour analysis
│   └── integrated_detector.py        # PLANNED: Multi-method integration
├── pattern_analysis/                  # PLANNED: Pattern recognition utilities
│   ├── marking_extractor.py          # Template extraction from calibration
│   ├── pattern_classifier.py         # Marking vs water classification
│   └── template_manager.py           # Template persistence & management
└── utils/                            # PLANNED: Pattern-specific utilities
    ├── image_processing.py           # Pattern image processing
    ├── geometric_utils.py            # Geometric analysis
    └── frequency_utils.py            # FFT utilities
```

### Data Flow

```
Image Input
     │
     ▼
┌─────────────────┐
│ System Selector │ ── Environment vars or config.yaml
│ (main_pattern_ │    determine detection mode
│ aware.py)      │
└─────────────────┘
     │
     ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Standard        │    │ Pattern-Aware   │    │ Hybrid          │
│ Detection       │    │ Detection       │    │ Detection       │
│ (fallback)      │    │ (primary)       │    │ (comparison)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Template        │ ── Learn scale markings
                    │ Matching        │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Morphological   │ ── Horizontal vs vertical
                    │ Detection       │    feature separation
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Result          │
                    │ Integration     │
                    └─────────────────┘
```

## Detection Methods

### Template Matching (WORKING)

**Purpose:** Learn and suppress scale-specific markings

**Process:**

1. Load scale marking templates from calibration
2. Match templates against scale region
3. Mask out detected markings
4. Detect water interface in unmarked areas

**Configuration:**

```yaml
detection:
  pattern_aware:
    template_matching:
      enabled: true
      threshold: 0.7                    # Template match confidence
      max_templates: 10                 # Maximum templates to store
      template_source: 'local'          # Template source: 'local', 'manager', 'both'
      use_default_templates: true       # Create default stadia rod templates
```

**Dual Template Sourcing:**

The template matching system supports flexible template sourcing:

- **'local'**: Uses built-in default templates (E-patterns, lines, thick lines)
- **'manager'**: Uses external template manager (when available)
- **'both'**: Combines local and external templates for maximum coverage

**Environment Variable Override:**

```bash
set TEMPLATE_SOURCE=both              # Override config template source
set USE_DEFAULT_TEMPLATES=true        # Override default template creation
```

**Status:** FULLY IMPLEMENTED AND WORKING

### Morphological Detection (WORKING)

**Purpose:** Separate horizontal water interfaces from vertical scale markings

**Process:**

1. Create horizontal and vertical morphological kernels
2. Extract horizontal features (water interfaces)
3. Suppress vertical features (scale markings)
4. Find strongest horizontal interface

**Configuration:**

```yaml
detection:
  pattern_aware:
    morphological:
      enabled: true
      horizontal_kernel_size: [40, 1]  # Horizontal feature detection
      vertical_kernel_size: [1, 40]    # Vertical feature suppression
```

**Status:** FULLY IMPLEMENTED AND WORKING

### Frequency Analysis (PLANNED)

**Purpose:** Reject periodic marking patterns using FFT analysis

**Process:**

1. Analyze frequency content of horizontal lines
2. Identify periodic patterns (markings)
3. Select lines with low periodicity (water interfaces)

**Status:** NOT YET IMPLEMENTED

### Line Segment Detector (PLANNED)

**Purpose:** Precise line detection with geometric filtering

**Process:**

1. Use OpenCV's Line Segment Detector
2. Filter for horizontal, continuous lines
3. Validate water interface characteristics

**Status:** NOT YET IMPLEMENTED

### Contour Analysis (PLANNED)

**Purpose:** Geometric shape-based interface detection

**Process:**

1. Analyze contour properties
2. Filter for water interface geometry
3. Reject text/number shapes

**Status:** NOT YET IMPLEMENTED

## Configuration

### Pattern Processing Mode

```yaml
pattern_processing:
  mode: 'standard'                    # Options: 'standard', 'pattern_aware', 'hybrid'
  debug_patterns: false               # Save pattern analysis debug images
  save_templates: true                # Save extracted templates for reuse
  template_directory: 'data/pattern_templates/scale_markings'
```

### Detection Method Configuration

```yaml
detection:
  pattern_aware:
    enabled: false                    # Master enable/disable
    engine: 'integrated_pattern'      # Pattern detection engine (PLANNED)
    fallback_to_standard: true        # Fall back to standard methods if pattern fails
    
    template_matching:                # WORKING
      enabled: true
      threshold: 0.7
      max_templates: 10
    
    morphological:                    # WORKING
      enabled: true
      horizontal_kernel_size: [40, 1]
      vertical_kernel_size: [1, 40]
    
    frequency_analysis:               # PLANNED
      enabled: true
      periodicity_threshold: 0.3
    
    line_detection:                   # PLANNED
      enabled: true
      min_line_length_ratio: 0.6
      max_angle_deviation: 10
    
    contour_analysis:                 # PLANNED
      enabled: true
      min_aspect_ratio: 10
      min_width_ratio: 0.5
```

### Environment Variables

```bash
# Primary mode selection
PATTERN_AWARE_MODE=true              # Enable pattern-aware detection
PATTERN_AWARE_MODE=hybrid            # Enable hybrid mode (both systems)
PATTERN_AWARE_MODE=false             # Force standard detection

# Template configuration
TEMPLATE_SOURCE=local                # Template source: 'local', 'manager', 'both'
USE_DEFAULT_TEMPLATES=true           # Enable/disable default stadia rod templates

# Debug options
PATTERN_DEBUG=true                   # Enable pattern analysis debugging
DEBUG_MODE=true                      # Enable general debug mode
```

## Usage

### Basic Usage

```bash
# Standard detection (existing system)
python src/main.py

# Pattern-aware detection
set PATTERN_AWARE_MODE=true
python src_pattern_aware/main_pattern_aware.py

# Hybrid detection (compare both systems)
set PATTERN_AWARE_MODE=hybrid
python src_pattern_aware/main_pattern_aware.py
```

### Integration with Existing Workflow

The pattern-aware system is designed to be a drop-in replacement:

```bash
# Replace your existing command:
# python src/main.py

# With pattern-aware version:
set PATTERN_AWARE_MODE=true
python src_pattern_aware/main_pattern_aware.py
```

All output formats (CSV, JSON, database) remain the same.

### Template Extraction

#### Interactive E-Template Extractor (Recommended)
Extract E-shaped templates interactively with visual feedback:

```bash
cd src_pattern_aware
python interactive_template_extractor.py
```

This tool extracts black and white E-shaped templates (5cm markings) with immediate visual feedback.

#### Programmatic Template Extraction
```bash
# Extract templates from calibration image
python -c "
from src_pattern_aware.pattern_water_detector import PatternWaterDetector
detector = PatternWaterDetector(config, pixels_per_cm, calib_data)
detector.extract_and_save_templates('data/calibration/calibration_image.jpg')
"
```

## Integration with Standard System

### Zero-Risk Architecture

The pattern-aware system is completely separate from the standard system:

- **Separate directory:** `src_pattern_aware/` vs `src/`
- **No modifications** to existing code
- **Independent operation** - can be disabled without affecting standard system
- **Fallback protection** - automatically uses standard detection if pattern detection fails

### Migration Strategy

1. **Test phase:** Use hybrid mode to compare results
2. **Validation phase:** Run pattern-aware mode on known images
3. **Production phase:** Switch to pattern-aware mode for problematic scales
4. **Rollback capability:** Can instantly return to standard system if needed

### Configuration Coexistence

Both systems share the same config.yaml file but use different sections:

```yaml
# Standard system configuration (unchanged)
detection:
  method: 'integrated'
  forced_method: 'enhanced_gradient'

# Pattern-aware system configuration (new)
pattern_processing:
  mode: 'standard'  # Disabled by default

detection:
  pattern_aware:
    enabled: false  # Disabled by default
```

## Development Status

### Current State: FUNCTIONAL PROTOTYPE

The system currently provides:

- **Working template matching detection** - Can suppress learned scale markings
- **Working morphological detection** - Separates horizontal from vertical features
- **Complete configuration integration** - Full YAML and environment variable support
- **Hybrid detection capability** - Can run both systems and compare results
- **Fallback protection** - Automatic fallback to standard methods

### Limitations

- **Limited detection methods** - Only 2 of 5 planned methods implemented
- **No template learning** - Templates must be manually provided
- **Basic pattern classification** - Advanced pattern analysis not yet implemented
- **No performance optimization** - Focus has been on functionality over speed

### Testing Status

- **Architecture testing** - System selection and configuration loading tested
- **Individual method testing** - Template matching and morphological detection tested
- **Integration testing** - NOT YET COMPLETED
- **Performance testing** - NOT YET COMPLETED
- **Comparative analysis** - NOT YET COMPLETED

## Future Work

### High Priority

1. **Complete Phase 2** - Implement remaining detection methods (frequency, LSD, contour)
2. **Template Learning** - Automated template extraction from calibration images
3. **Integration Testing** - Comprehensive testing across different scale types
4. **Performance Optimization** - Speed and memory usage improvements

### Medium Priority

1. **Advanced Pattern Classification** - Machine learning-based pattern recognition
2. **Template Management System** - Version control and sharing of templates
3. **Enhanced Debug Visualization** - Comprehensive pattern analysis debug images
4. **Configuration Validation** - Input validation and error handling

### Low Priority

1. **Multi-Scale Support** - Handle multiple scales in single image
2. **Real-Time Optimization** - Video stream processing capabilities
3. **Cloud Integration** - Template sharing and collaborative learning
4. **Mobile Support** - Smartphone-based detection

## Project Structure

```
src_pattern_aware/                     # Pattern-aware detection module
├── README.md                          # This documentation
├── main_pattern_aware.py              # Main entry point with system selection
├── pattern_water_detector.py          # Core pattern-aware detector
├── hybrid_detector.py                 # Hybrid detection (both systems)
├── detection_methods/                 # Individual detection algorithms
│   ├── template_matching.py           # WORKING: Template-based suppression
│   ├── morphological_detector.py      # WORKING: Horizontal interface detection
│   ├── frequency_analyzer.py          # PLANNED: FFT periodicity analysis
│   ├── lsd_detector.py               # PLANNED: Line Segment Detector
│   ├── contour_analyzer.py           # PLANNED: Geometric contour analysis
│   └── integrated_detector.py        # PLANNED: Multi-method integration
├── pattern_analysis/                  # PLANNED: Pattern recognition utilities
│   ├── marking_extractor.py          # Template extraction from calibration
│   ├── pattern_classifier.py         # Marking vs water classification
│   └── template_manager.py           # Template persistence & management
└── utils/                            # PLANNED: Pattern-specific utilities
    ├── image_processing.py           # Pattern image processing
    ├── geometric_utils.py            # Geometric analysis
    └── frequency_utils.py            # FFT utilities

data/pattern_templates/               # Pattern template storage
├── scale_markings/                   # Learned scale marking templates
└── calibration_patterns/             # Calibration-derived patterns
```

## Troubleshooting

### Common Issues

**"Pattern detection not enabled" error:**

- **Solution**: Check `PATTERN_AWARE_MODE` environment variable
- **Solution**: Verify `pattern_processing.mode` in config.yaml
- **Solution**: Ensure `detection.pattern_aware.enabled: true`
- **Verification**: Look for log message "Pattern-Aware Water Level Detector Initialized"

**"No templates available" warning:**

```
WARNING - No templates available, using template-free detection
```

- **Status**: Currently normal - template learning not yet implemented
- **Workaround**: System will use morphological detection methods
- **Impact**: Reduced accuracy on heavily marked scales, but still functional

**"Both detection systems failed" error:**

- **Check**: Calibration data validity using standard system first
- **Check**: Scale region extraction in debug images
- **Debug**: Enable `DEBUG_MODE=true` for detailed logging
- **Solution**: Verify scale boundaries in config.yaml match image content

**Performance issues:**

- **Cause**: Current implementation prioritizes functionality over speed
- **Workaround**: Use standard system for time-critical applications
- **Timeline**: Performance optimizations planned for Phase 4

**ImportError: Cannot import pattern detection modules:**

```
ImportError: cannot import name 'PatternWaterDetector' from 'src_pattern_aware.pattern_water_detector'
```

- **Solution**: Ensure you're running from project root directory
- **Solution**: Verify all `__init__.py` files are present in pattern detection directories
- **Solution**: Check Python path includes both `src/` and `src_pattern_aware/`

### Debug Information

**Enable comprehensive debugging:**

```bash
# Windows Command Prompt
set DEBUG_MODE=true
set PATTERN_DEBUG=true
python src_pattern_aware/main_pattern_aware.py

# Alternative: Create .env file
echo "DEBUG_MODE=true" > .env
echo "PATTERN_DEBUG=true" >> .env
```

**Expected debug output for successful detection:**

```
INFO - Starting Pattern-Aware Water Level Detection System
INFO - Selected detection system: pattern_aware
INFO - Pattern-Aware Water Level Detector Initialized
INFO - Initialized 2 pattern detection methods
DEBUG - Template matching found 0 marking instances
DEBUG - Morphological detection found waterline at Y=245
DEBUG - Selected candidate Y=245 from 2 candidates
INFO - Pattern detection successful: 445.5cm
```

**Expected debug output for hybrid mode:**

```
INFO - Selected detection system: hybrid
INFO - Hybrid detector initialized with both standard and pattern-aware methods
INFO - Hybrid processing: IMG_0154.JPG
INFO - Running standard detection...
INFO - Standard detection: 301.3cm (confidence: 0.886)
INFO - Running pattern-aware detection...
INFO - Pattern-aware detection: 445.5cm (confidence: 0.950)
INFO - Result comparison:
INFO -   Standard: 301.3cm (confidence: 0.886)
INFO -   Pattern:  445.5cm (confidence: 0.950)
INFO -   Difference: 144.2cm
WARNING - Large discrepancy between detection methods (144.2cm)
INFO - Selected pattern-aware result (higher confidence despite discrepancy)
```

### Log Analysis

**Successful pattern detection:**

```
INFO - Processing image with pattern-aware detection: IMG_0154.JPG
INFO - Template matching detector initialized (threshold: 0.7)
INFO - Morphological detector initialized (h_kernel: [40, 1], v_kernel: [1, 40])
DEBUG - Template matching found 0 marking instances
DEBUG - Using template-free detection fallback
DEBUG - Morphological detection found waterline at Y=245
INFO - Pattern detection successful: 445.5cm
```

**Fallback to standard detection:**

```
WARNING - Pattern detection failed, falling back to standard methods
INFO - Using standard detection as fallback
INFO - Standard detection: 301.3cm (confidence: 0.886)
```

**Template detection (future - when implemented):**

```
INFO - Loaded 5 existing scale marking templates
DEBUG - Template matching found 12 marking instances
DEBUG - Morphological detection found waterline at Y=245
DEBUG - Validation against template-suppressed region: PASSED
INFO - Pattern-aware detection: 445.5cm (confidence: 0.970)
```

### Visual Debug Analysis

When `PATTERN_DEBUG=true` is enabled, debug images are saved to:

```
data/debug/pattern_analysis/pattern_session_YYYYMMDD_HHMMSS/
├── original/                    # Original input images
├── template_matching/           # Template matching debug images
├── morphological/              # Morphological operation results
├── pattern_suppression/        # Before/after pattern suppression
└── final_detection/            # Final detection results with annotations
```

**What to check in pattern debug images:**

- **Template matching**: Verify detected markings are correctly identified
- **Morphological**: Check horizontal vs vertical feature separation
- **Pattern suppression**: Ensure scale markings are properly masked
- **Final detection**: Confirm water interface is correctly identified

## Performance

### Current Performance (Prototype)

- **Processing Speed:** ~3-5 seconds per image (not optimized)
- **Memory Usage:** ~300-400MB (higher than standard system)
- **Accuracy:** Variable - depends on scale type and markings
- **Reliability:** High - fallback protection ensures results

### Planned Improvements

- **Speed Optimization:** Target 2x faster processing
- **Memory Optimization:** Reduce memory usage by 50%
- **Accuracy Validation:** Comprehensive accuracy testing
- **Batch Processing:** Optimize for multiple image processing

### Comparison with Standard System

| Metric | Standard System | Pattern-Aware (Current) | Pattern-Aware (Target) |
|--------|----------------|-------------------------|------------------------|
| Speed | ~2-3 seconds | ~3-5 seconds | ~2-4 seconds |
| Memory | ~200MB | ~300-400MB | ~250MB |
| Accuracy (simple scales) | High | Medium | High |
| Accuracy (complex scales) | Low | Medium-High | High |
| Reliability | High | High (with fallback) | High |

## Development

### Setting Up Development Environment

1. **Clone and setup:**

```bash
git clone <repo-url>
cd tide-level-img-proc-backup

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Development workspace:**

```bash
# Create development directories
mkdir -p data/{pattern_templates,debug/pattern_analysis}
mkdir -p logs

# Verify pattern-aware system
python -c "from src_pattern_aware import PatternWaterDetector; print('Development environment ready')"
```

### Development Workflow

**Phase 2: Complete Detection Methods**

```bash
# Template for new detection method
cp src_pattern_aware/detection_methods/template_matching.py \
   src_pattern_aware/detection_methods/frequency_analyzer.py

# Update __init__.py imports
# Implement detect_waterline() method
# Add configuration options
# Test with debug mode
```

**Phase 3: Pattern Analysis Components**

```bash
# Implement template extraction
# Add pattern classification
# Create template management system
# Integrate with calibration workflow
```

### Testing New Detection Methods

```bash
# Test individual method
python -c "
from src_pattern_aware.detection_methods.template_matching import TemplateMatchingDetector
import cv2, yaml
with open('config.yaml') as f: config = yaml.safe_load(f)
detector = TemplateMatchingDetector(config, None)
print(detector.get_detection_info())
"

# Test integration
set PATTERN_DEBUG=true
python src_pattern_aware/main_pattern_aware.py

# Compare with standard system
set PATTERN_AWARE_MODE=hybrid
python src_pattern_aware/main_pattern_aware.py
```

### Code Architecture Guidelines

**Detection Method Structure:**

```python
class NewDetectionMethod:
    def __init__(self, config):
        # Load configuration
        # Initialize parameters
        # Setup logging
        
    def detect_waterline(self, scale_region):
        # Main detection logic
        # Return Y-coordinate or None
        
    def get_detection_info(self):
        # Return method information
```

**Configuration Integration:**

```yaml
detection:
  pattern_aware:
    new_method:
      enabled: true
      parameter1: value1
      parameter2: value2
```

## Contributing

### Development Guidelines

1. **No emojis or icons** - All code, documentation, and logging must be text-only
2. **Maintain separation** - Do not modify existing `src/` directory
3. **Comprehensive testing** - Test all new detection methods thoroughly
4. **Configuration-driven** - Make features configurable via YAML
5. **Backward compatibility** - Ensure fallback to standard system always works
6. **Modular design** - Each detection method should be independent
7. **Error handling** - Graceful degradation and detailed logging

### Code Style

- **Follow PEP 8** - Standard Python style guidelines
- **Comprehensive logging** - Use structured logging throughout
- **Type hints** - Add type annotations where applicable
- **Detailed docstrings** - Document all classes and methods
- **Error handling** - Include try/catch blocks and graceful degradation
- **Configuration validation** - Validate all configuration parameters

### Testing Requirements

**Unit Tests:**

```python
# Test individual detection methods
def test_template_matching_detector():
    config = load_test_config()
    detector = TemplateMatchingDetector(config, None)
    result = detector.detect_waterline(test_scale_region)
    assert result is not None
    assert 0 <= result < test_scale_region.shape[0]
```

**Integration Tests:**

```python
# Test complete pattern-aware system
def test_pattern_aware_integration():
    detector = PatternWaterDetector(config, pixels_per_cm)
    result = detector.process_image('test_image.jpg')
    assert result['detection_method'] == 'pattern_aware'
```

**Comparison Tests:**

```python
# Compare with standard system
def test_hybrid_comparison():
    hybrid = HybridDetector(config, pixels_per_cm)
    result = hybrid.process_image('test_image.jpg')
    assert 'standard_result' in result['hybrid_data']
    assert 'pattern_result' in result['hybrid_data']
```

### Documentation Requirements

- **Update README** - Document all new features and configuration options
- **Code documentation** - Comprehensive docstrings with examples
- **Configuration documentation** - Explain all YAML parameters
- **Usage examples** - Provide real-world usage scenarios
- **Implementation notes** - Explain algorithmic decisions and trade-offs
- **Limitation documentation** - Clearly state what doesn't work yet

### Pull Request Guidelines

1. **Feature branches** - Create feature-specific branches
2. **Clear descriptions** - Explain what the PR accomplishes
3. **Test coverage** - Include comprehensive tests
4. **Documentation updates** - Update relevant documentation
5. **Performance notes** - Document any performance implications
6. **Breaking changes** - Clearly mark any API changes

### Implementation Priorities

**High Priority (Phase 2 completion):**

1. Frequency Analysis Detector
2. Line Segment Detector (LSD)
3. Contour Analysis Detector
4. Integrated Pattern Detector

**Medium Priority (Phase 3):**

1. Template extraction from calibration
2. Pattern classification system
3. Template management and persistence
4. Enhanced calibration integration

**Low Priority (Phase 4):**

1. Performance optimization
2. Advanced debug visualization
3. Comprehensive testing suite
4. Documentation improvements

## License

This project is licensed under the BSD 3-Clause License - same as the main water level detection system. See [LICENSE](../LICENSE.md) file for details.

## Acknowledgments

This pattern-aware detection system builds upon:

- **Main water level detection system** - Provides the foundation and fallback capabilities
- **OpenCV computer vision library** - Core image processing and computer vision operations
- **NumPy scientific computing** - Numerical operations and array processing
- **SciPy scientific library** - Advanced mathematical functions and algorithms

**Pattern Recognition Techniques:**

- **Template Matching** - Classical computer vision approach for pattern recognition
- **Morphological Operations** - Mathematical morphology for image structure analysis
- **Frequency Domain Analysis** - Signal processing techniques for pattern detection
- **Line Segment Detection** - Geometric feature extraction methods
- **Contour Analysis** - Shape-based object recognition approaches

**Research Context:**
This work is part of prototype research conducted at the [SenseLAB](http://senselab.tuc.gr/) of the [Technical University of Crete](https://www.tuc.gr/en/).

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Pattern-specific issues**: Use the "pattern-aware" label for issues related to this module
- **Documentation**: Additional technical documentation in the `/documentation` directory
- **Research collaboration**: Contact SenseLAB for research partnerships

### Getting Help

**For pattern-aware detection issues:**

1. Check this README first
2. Enable debug mode and check logs
3. Try hybrid mode to compare with standard detection
4. Create GitHub issue with "pattern-aware" label

**For general system issues:**

1. Refer to the main system [README](../README.md)
2. Check standard system functionality first
3. Use fallback to standard detection if needed

---

*The Pattern-Aware Water Level Detection System is specifically designed for scales with complex markings and patterns. For simple scales without markings, the standard detection system may be more appropriate and efficient.*
