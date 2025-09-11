# Water Level Measurement System: Edge & Color Detection

*Automated tide/water level detection using computer vision*

> Disclaimer: This work is part of a non-funded prototype research idea conducted at the [SenseLAB](http://senselab.tuc.gr/) of the [Technical University of Crete](https://www.tuc.gr/el/archi).

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green.svg)](https://opencv.org/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Use Guide](#quick-use-guide)
- [Installation](#installation)
- [Configuration](#configuration)
- [Operation Guide and Instructions](#operation-guide-and-instructions)
  - [Enhanced Scale Configuration Analysis](#enhanced-scale-configuration-analysis)
  - [Calibration System](#calibration-system)
  - [Processing Images](#processing-images)
  - [Visual Debugging](#visual-debugging)
  - [Water Detection Methods](#water-detection-methods)
  - [Scale Detection and Color Options](#scale-detection-and-color-options)
- [Data Export Options](#data-export-options)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Deployment Options](#deployment-options)
- [Architecture](#architecture)
- [License](#license)

## Overview

This system provides automated water/tide level detection and measurement using computer vision techniques. Designed for fixed camera setups with calibrated measurement scales, it processes images continuously and stores precise measurements in a database.

## Features

Automated processing with visual calibration, multiple detection methods, GUI interface, database exports, and comprehensive debugging capabilities.

## Quick Use Guide

This guide provides a step-by-step workflow for getting the system running quickly:

### Step 1: Setup and Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd tide-level-img-proc-backup

# Install Python dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/input data/calibration data/output data/debug
```

### Step 2: Run Enhanced Scale Analysis

Before calibration, analyze your scale setup:

```bash
# Place your calibration image as data/calibration/calibration_image.jpg
# Run the interactive analysis tool
python src/calibration/analyze_scale_photo.py
```

**Follow the interactive prompts:**

1. Enter actual scale readings (top and waterline positions)
2. Click 4 scale corners (even if partially underwater)  
3. Mark waterline position (left and right edges)
4. Optionally sample scale colors and water color
5. Review generated config.yaml suggestions

**Apply the results:** Copy the suggested values to your `config.yaml`

### Step 3: Configure Detection Method

Edit `config.yaml` to set detection method and forced method:

```yaml
detection:
  method: 'integrated'              # Always run all 4 methods
  forced_method: 'enhanced_gradient' # But use this method's result
  # Options: 'enhanced_gradient', 'edge', 'color', 'gradient', null
  
  # Environment variables for testing
  USE_GUI_SELECTOR: true                  # Enable folder selection GUI
  DEBUG_MODE: true                        # Enable debug images
  CALIBRATION_MODE: false                 # Set to true for calibration
```

### Step 4: Set Environment Variables

```bash
# Windows Command Prompt
set USE_GUI_SELECTOR=true        # Enable folder selection GUI
set DEBUG_MODE=true              # Enable debug images and detailed logging
set CALIBRATION_MODE=true        # Run calibration mode
set PROCESS_INTERVAL=60          # Processing interval in seconds
```

Or create a `.env` file:

```bash
USE_GUI_SELECTOR=true
DEBUG_MODE=true  
CALIBRATION_MODE=false
PROCESS_INTERVAL=60
```

### Step 5: Run System Calibration

```bash
# Run calibration mode to generate calibration.yaml
set CALIBRATION_MODE=true & python src/main.py
```

**Calibration generates:**

- `data/calibration/calibration.yaml` with precise pixel/cm ratio
- Enhanced waterline reference data
- Scale marking analysis (if available)

### Step 6: Add Test Images

```bash
# Place test images in any folder (e.g., data/input_tests/)
# System will prompt for folder selection if USE_GUI_SELECTOR=true
```

### Step 7: Run Water Level Detection

```bash
# Run main detection system
python src/main.py
```

**When prompted:** Select your image folder via GUI dialog

### Step 8: Monitor Logs and Results

**Key log messages to watch for:**

```bash
# Successful startup
INFO - Using enhanced waterline-aware calibration (confidence: 0.98)
INFO - FORCED METHOD CONFIGURED: Will always use 'enhanced_gradient' method

# Detection results for each image
INFO - Detection summary: 4 method(s) found results
INFO -   enhanced_gradient: Y=17, confidence=0.874
INFO -   edge: Y=274, confidence=0.563  
INFO -   color: Y=0, confidence=0.267
INFO -   gradient: Y=278, confidence=1.000
INFO - FORCED METHOD: Using 'enhanced_gradient' method (Y=17, confidence=0.874)

# Final measurements
INFO - Processed IMG_0154.JPG: Water level = 44.55cm
```

### Step 9: Check Debug Images

Debug images are saved in `data/debug/debug_session_TIMESTAMP/`:

```bash
# Enhanced debug images structure:
data/debug/debug_session_20250829_102135/
├── original/                      # Original input images
├── preprocessed/                  # Resized images with metadata
├── scale_detection/              # Scale boundary detection (blue rectangles)
├── scale_bounds_enhanced/        # Enhanced scale detection with method info
├── scale_region_extracted/       # Extracted scale regions for analysis
├── edges/                        # Canny edge detection results
├── contours/                     # Hough line detection results
├── gradient_analysis/            # Vertical gradient analysis (Sobel)
├── water_detection/              # Water detection results with annotations
├── water_color_mask/            # Water color mask visualization
├── waterline_within_scale/      # Waterline detection within scale bounds
├── integrated_detection_methods/ # Multi-method comparison visualization
├── final_result/                # Final annotated results with measurements
└── [conditional folders]        # Additional folders based on detection methods used
```

**What to check in debug images:**

- Scale boundaries are correctly detected (blue rectangles)
- Waterline detection is accurate (green lines)
- Final measurements are overlaid correctly

### Step 10: Review Output Data

Check generated measurement data:

```bash
# View output files
data/output/
├── measurements.db                         # Main SQLite database
├── measurements_20250829_102137.csv       # CSV export
├── measurements_20250829_102137.json      # JSON export  
├── measurements_20250829_102137.db        # Database backup
└── annotated/                              # Visual output images
    ├── annotated_success_20250829_102140.jpg    # Successful detection
    ├── annotated_failed_20250829_102142.jpg     # Failed detection (diagnostic)
    └── annotated_success_20250829_102145.jpg    # Another successful detection
```

#### Understanding Annotated Images

**Important**: Annotated images are created for ALL processed images, not just successful detections:

- **Success Images** (`annotated_success_*.jpg`): Show successful water level detection with:
  - Green water line overlay
  - Blue scale boundary lines  
  - Measurement text (water level, scale above water)
  - Processing time and calibration info

- **Failed Images** (`annotated_failed_*.jpg`): Show diagnostic information for failed detections:
  - Red water line (if detected) or "No water line detected"
  - Blue scale boundaries (if detected) or "Scale bounds not detected"
  - "DETECTION FAILED" status text
  - Processing diagnostics to help troubleshoot issues

**Key Point**: Even if some images fail detection, you'll still get annotated output showing what was attempted. This helps debug why certain images couldn't be processed successfully.

**Configuration**: Enable/disable annotated image creation in `config.yaml`:

```yaml
processing:
  save_processed_images: true  # Set to false to disable annotated images
```

### Step 11: Run Cleanup (Optional)

```bash
# Run the automated cleanup script
.\cleanup.bat

# Or manually clean specific directories:
# - Clean debug images (optional - useful for disk space)
# - Reset processed image tracking (to reprocess same images) 
# - Clean processed images directory
```

## Common Troubleshooting

- **Scale not detected:** Disable image resizing in config: `resize_width: null`
- **Waterline inaccurate:** Use `forced_method: 'enhanced_gradient'` for clear water
- **No images processed:** Check folder permissions and database path
- **Calibration fails:** Ensure `data/calibration/calibration_image.jpg` exists

> **Note:** This guide covers the essential workflow from setup to measurement results. For detailed configuration options, see the full documentation sections below.

## Installation

### Prerequisites

- **Python 3.9+** with pip
- **Docker & Docker Compose** (for containerized deployment)

### System Dependencies

The system requires several Python packages for image processing:

```bash
# Core dependencies
pip install opencv-python numpy scipy
pip install PyYAML Pillow matplotlib pandas
pip install sqlalchemy python-dotenv
```

Or simply:

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file with the following settings:

```bash
# Processing Configuration
USE_GUI_SELECTOR=false          # Enable GUI directory selection
CALIBRATION_MODE=false          # Run in calibration mode
PROCESS_INTERVAL=60            # Processing interval (seconds)
DB_PATH=./data/output/measurements.db # Database file path

# Web Server (Docker deployment)
SECRET_KEY=your-secret-key-here
ADMIN_USER=admin
ADMIN_PASS=secure-password
```

### System Configuration (`config.yaml`)

```yaml
# Camera and Scale Configuration
scale:
  # Known height of scale in cm
  total_height: 45.0   # Corrected from dm scale reading
  # Width of scale in cm (for reference)
  width: 8.0
  # Scale position in image (if fixed)
  expected_position:
    x_min: 81
    x_max: 181
    y_min: 1
    y_max: 603
  
  # RGB/HSV color-based scale detection
  color_detection:
    enabled: true
    adaptive_thresholding: true
    debug_color_masks: true
    
    # Scale color definitions (HSV ranges)
    scale_colors:
      yellow:
        enabled: true
        hsv_lower: [20, 100, 100]
        hsv_upper: [30, 255, 255]
        description: "Yellow scale background"
      
      white:
        enabled: true
        hsv_lower: [1, 0, 0]
        hsv_upper: [164, 108, 255]
        description: "White scale background"
      
      blue:
        enabled: true
        hsv_lower: [94, 12, 155]
        hsv_upper: [114, 112, 255]
        description: "Blue scale background"

      black:
        enabled: true
        hsv_lower: [97, 0, 0]
        hsv_upper: [129, 92, 87]
        description: "Black scale markings"
      
      red:
        enabled: true
        hsv_lower: [0, 100, 100]
        hsv_upper: [10, 255, 255]
        description: "Red scale background"
      
      red_alt:
        enabled: true
        hsv_lower: [170, 100, 100]
        hsv_upper: [180, 255, 255]
        description: "Red scale background (alt range)"
      
      white_markings:
        enabled: true
        hsv_lower: [0, 0, 220]
        hsv_upper: [180, 20, 255]
        description: "White scale markings on colored backgrounds"
    
    # Morphological operations for mask cleanup
    morphology:
      kernel_size: 3
      close_iterations: 1
      open_iterations: 1

calibration:
  # Pixels per centimeter (will be calculated during calibration)
  pixels_per_cm: null
  # Reference water level for calibration (cm)
  reference_water_level: 50.0
  # Use enhanced calibration data with waterline gradient analysis
  use_enhanced_data: true
  enhanced_data_path: 'data/calibration/calibration.yaml'

detection:
  # Water line detection parameters
  edge_threshold_low: 50
  edge_threshold_high: 150
  blur_kernel_size: 7
  min_contour_area: 500
  
  # Water detection method: 'edge', 'color', 'gradient'
  method: 'edge'              # Choose detection method (see Water Detection Methods section)
  
  # Color-based detection (HSV ranges for water) - used when method: 'color'
  water_hsv_lower: [100, 50, 50]   # Lower HSV bounds for water color detection
  water_hsv_upper: [130, 255, 255] # Upper HSV bounds for water color detection
  
  # Pattern-aware detection configuration (used with pattern_processing.mode: 'pattern_aware')
  pattern_aware:
    # Hybrid waterline verification system
    waterline_verification:
      enabled: true                   # Enable hybrid waterline verification
      min_pattern_confidence: 0.7     # Minimum confidence for waterline detection
      gradient_kernel_size: 3         # Sobel kernel size for gradient analysis
      gradient_threshold: 10          # Threshold for significant gradient changes
      transition_search_height: 20    # Search height around suspicious regions (pixels)
      
      # Consolidated pattern analysis thresholds
      pattern_analysis:
        # Consecutive good pattern detection thresholds
        scale_consistency_threshold: 0.15    # Scale factor consistency for consecutive good patterns (±15%)
        size_consistency_threshold: 0.25     # Size consistency for consecutive good patterns (±25%) 
        spacing_consistency_threshold: 0.50  # Spacing consistency for consecutive good patterns (±50%)
        min_consecutive_patterns: 3          # Minimum consecutive good patterns required
        
        # Anomaly detection thresholds (applied after establishing baseline)
        scale_anomaly_threshold: 0.15        # Scale factor change to flag suspicious regions (±15%)
        size_anomaly_threshold: 0.20         # Size change to flag suspicious regions (±20%)
        aspect_ratio_anomaly_threshold: 0.20 # Aspect ratio change to flag suspicious regions (±20%)
        max_gap_ratio: 2.0                   # Max gap between patterns (2x expected spacing)

processing:
  # Image processing settings
  resize_width: null  # WARNING: Set to null to disable resizing (recommended)
                      # If enabled (e.g., 800), scale coordinates may become incorrect
                      # Use enhanced calibration workflow to avoid coordinate mismatch
  save_processed_images: true
  image_format: 'jpg'
  
output:
  # Output settings - Enable/disable export formats
  csv_export: true      # Export measurements to timestamped CSV files
  json_export: true     # Export measurements to timestamped JSON files  
  database: true        # Create timestamped database backup copies

debug:
  # Visual debugging options - can be overridden by DEBUG_MODE environment variable
  enabled: false            # Set to true to enable debug mode via config
  save_debug_images: true
  debug_output_dir: 'data/debug'
  annotation_color: [0, 255, 0]  # Green BGR
  annotation_thickness: 2
  font_scale: 0.7
  steps_to_save:
    # Core debugging steps (always available)
    - 'original'                      # Original input images
    - 'preprocessed'                  # Resized/prepared images
    - 'scale_detection'              # Scale boundary detection
    - 'scale_bounds_enhanced'        # Enhanced scale detection with method info
    - 'scale_region_extracted'       # Extracted scale regions
    - 'edges'                        # Canny edge detection
    - 'contours'                     # Hough line detection
    - 'gradient_analysis'            # Vertical gradient analysis
    - 'water_detection'              # Water detection results
    - 'integrated_detection_methods' # Multi-method comparison
    - 'final_result'                 # Final annotated results
    
    # Enhanced color-based detection steps (conditional)
    - 'color_mask_blue'              # Blue color detection masks
    - 'color_mask_black'             # Black color detection masks  
    - 'color_mask_combined'          # Combined color masks
    - 'hsv_conversion'               # HSV color space conversion
    - 'water_color_mask'            # Water color mask visualization
    - 'masked_grayscale'             # Color-filtered grayscale images
    
    # Enhanced edge detection steps (conditional)
    - 'edges_color_enhanced'         # Color-enhanced edge detection
    - 'edges_masked_gray'            # Edges on color-masked grayscale
    - 'edges_multi_channel'          # Combined edges from RGB channels
    - 'edges_hue_transitions'        # Hue channel transition edges
    - 'edges_individual_colors'      # Edges from individual color masks
    - 'edges_final_combined'         # Final combined color-enhanced edges
    
    # Analysis and validation steps (conditional)
    - 'scale_contours_analysis'      # Contour scoring for scale detection
    - 'waterline_within_scale'       # Waterline detection within scale bounds
```

## Operation Guide and Instructions

### Enhanced Scale Configuration Analysis

**Before calibration**, use the enhanced scale analysis tool to determine optimal configuration values with waterline detection:

```bash
# 1. Place your scale image as data/calibration/calibration_image.jpg
# 2. Run the enhanced scale analysis tool
python src/calibration/analyze_scale_photo.py
```

The interactive analysis tool allows you to mark the waterline position, enter actual scale readings, select scale boundaries, and optionally configure colors for detection.

**Interactive Analysis Workflow:**

1. **Image display**: Shows calibration image window before requesting measurements
2. **Scale measurement input**: Enter actual scale readings while viewing the image:
   - Enter scale reading at TOP of visible scale (e.g., 485 cm)
   - Enter scale reading at current WATERLINE position (e.g., 420 cm)
3. **Scale boundary selection**: Click 4 corner points of FULL visible scale (top-left, top-right, bottom-left, bottom-right)
   - Select entire visible scale even if parts are underwater
4. **Waterline marking**: Click 2 points on left and right edges of scale at waterline position
5. **Interactive color sampling** (optional): Click on scale background color, then on marking/text color
6. **Interactive water sampling** (optional): Click on water color (if visible) for detection calibration
7. **Automatic analysis**: Edge detection and color analysis
8. **Waterline gradient analysis**: Advanced analysis of color differences above/below waterline
9. **Enhanced config generation**: Provides ready-to-use config.yaml values plus accurate calibration data
10. **Summary display**: Clear summary of suggested config.yaml changes at the end

### Calibration System

The system provides two integrated calibration workflows that work together to provide accurate water level measurements.

#### Workflow 1: Enhanced Interactive Analysis + Calibration (Recommended)

**Best for**: New setups, precise measurements, waterline-aware calibration, partially submerged scales

**Enhanced Complete Process**:

```bash
# Step 1: Enhanced interactive scale analysis with waterline detection
python src/calibration/analyze_scale_photo.py
# → Select 4 corners of FULL visible scale (even if partially underwater)
# → Mark waterline position on scale (2 points: left and right edges)
# → Enter actual scale readings at top and waterline positions
# → System calculates precise pixels/cm from real measurements
# → Choose scale background/marking colors (optional)
# → Sample water color (optional)
# → Generates config.yaml suggestions AND accurate calibration data

# Step 2: Apply the enhanced analysis results
# - Update your config.yaml with suggested values
# - Enhanced calibration.yaml is automatically generated with waterline data

# Step 3: Run calibration (uses enhanced waterline-aware data)
set CALIBRATION_MODE=true & python src/main.py
# → Loads enhanced calibration.yaml with waterline reference
# → Uses actual measured cm/pixel ratio
# → Ready for highly accurate water level detection
```

This method uses actual scale measurements for highest accuracy and incorporates the current waterline position into calibration.

#### Workflow 2: Direct Configuration Calibration (Legacy)

**Best for**: Quick setup, when you already have precise config.yaml settings

**Process**:

```bash
# Step 1: Ensure config.yaml has accurate scale settings
# scale:
#   total_height: 45.0   # Corrected from dm scale reading        # Actual scale height in cm
#   expected_position:         # Approximate scale location
#     x_min: 75
#     x_max: 190
#     y_min: 1
#     y_max: 605

# Step 2: Place calibration image
# - Save image as: data/calibration/calibration_image.jpg

# Step 3: Run direct calibration
set CALIBRATION_MODE=true & python src/main.py
# → Uses config.yaml values to generate calibration.yaml
# → Calculates pixels_per_cm from scale dimensions
```

#### Calibration Data Integration

**How the methods work together**:

1. **Data Generation**: Both workflows create `data/calibration/calibration.yaml`
2. **Source Tracking**: File includes metadata about which method generated it
3. **Consistent Format**: Same data structure regardless of generation method
4. **Automatic Updates**: Each calibration run updates the file with current settings

**Recent Improvements**: Fixed YAML serialization, improved image display timing, enhanced waterline gradient analysis, and better error handling.

**Enhanced generated calibration.yaml contains**:

```yaml
pixels_per_cm: 12.546                  # Precise conversion factor from real measurements
image_path: data/calibration/...       # Source calibration image
calibration_date: '2024-08-26T...'     # Generation timestamp
image_dimensions:                      # Image metadata
  width: 3024
  height: 4032
  resized: false
scale_measurements:                    # Actual scale readings
  top_measurement_cm: 48.5            # Scale reading at top (corrected from dm)
  waterline_measurement_cm: 42.0      # Scale reading at waterline (corrected from dm)
  measurement_difference_cm: 6.5       # Actual measured difference (corrected from dm)
  current_water_level_cm: 42.0        # Current water level (corrected from dm)
reference_points:                      # Enhanced reference points
  top_of_scale: {x: 75, y: 45}        # Top of scale boundary
  bottom_of_scale: {x: 190, y: 580}   # Bottom of scale boundary
  waterline:                          # Waterline position data
    x_left: 78
    y_left: 245
    x_right: 185
    y_right: 247
    y_average: 246
scale_boundaries:                      # Complete boundary data
  x_min: 75
  x_max: 190
  y_min: 45
  y_max: 580
  x_min_pct: 0.025                    # Percentage coordinates
  x_max_pct: 0.063
  y_min_pct: 0.011
  y_max_pct: 0.144
calibration_method: 'enhanced_interactive_waterline'  # Enhanced method
confidence: 0.98                       # Higher confidence with real measurements
notes: 'Generated by enhanced analyze_scale_photo.py with waterline detection...'
```

#### Method Selection Guide

**Use Enhanced Interactive Analysis When**:

- Setting up system for first time
- Scale is partially underwater/submerged
- Need waterline-aware calibration
- Want maximum measurement accuracy with real scale readings
- Scale boundaries are unclear or complex
- Need water color detection setup
- Scale position varies between images
- **Recommended for all new installations**

**Use Direct Calibration When**:

- Config.yaml is already precisely configured from previous interactive analysis
- Scale position and size are completely consistent
- Quick calibration update needed (not recommended for accuracy)
- Batch processing multiple identical setups

#### Calibration Validation

After either method, verify calibration accuracy:

```bash
# Test with debug mode to see detection results
set DEBUG_MODE=true & python src/main.py
# Check generated debug images in data/debug/
# Verify scale detection boundaries and water line detection
```

### Processing Images

**Batch processing:**

```bash
# Place images in data/input/ directory
python src/main.py
```

**GUI directory selection:**

```bash
set USE_GUI_SELECTOR=true & python src/main.py
# Select directory via file dialog
```

**Visual debugging mode:**

```bash
# Option 1: Enable via environment variable (overrides config)
set DEBUG_MODE=true & python src/main.py

# Option 2: Enable via config.yaml (set debug.enabled: true)
python src/main.py
# Generates step-by-step annotated debug images
```

**Docker deployment:**

```bash
docker compose up -d
# Access web interface at https://localhost:5000
```

### Visual Debugging

The system includes a comprehensive visual debugging mode that creates annotated images showing each step of the processing pipeline.

**Enable debug mode:**

```bash
# Via environment variable
set DEBUG_MODE=true & python src/main.py

# Or in .env file
DEBUG_MODE=true
```

**Debug output structure:**

```
data/debug/
└── debug_session_20250822_160530/
    # Core detection pipeline folders (always present)
    ├── original/                      # Input images
    ├── preprocessed/                  # Resized images with metadata
    ├── scale_detection/              # Scale boundary detection
    ├── scale_bounds_enhanced/        # Enhanced scale detection with method info
    ├── scale_region_extracted/       # Extracted scale regions
    ├── edges/                        # Canny edge detection results
    ├── contours/                     # Hough line detection results
    ├── gradient_analysis/            # Vertical gradient analysis
    ├── water_detection/              # Water detection results
    ├── integrated_detection_methods/ # Multi-method comparison
    ├── final_result/                 # Final annotated results
    
    # Enhanced detection folders (conditional - when color detection enabled)
    ├── hsv_conversion/               # HSV color space conversion
    ├── color_mask_blue/             # Blue color detection masks
    ├── color_mask_black/            # Black color detection masks
    ├── color_mask_combined/         # Combined color masks
    ├── water_color_mask/            # Water color mask visualization
    ├── masked_grayscale/            # Color-filtered grayscale images
    ├── edges_color_enhanced/        # Color-enhanced edge detection
    ├── edges_masked_gray/           # Edges on color-masked grayscale
    ├── edges_multi_channel/         # Combined edges from RGB channels
    ├── edges_hue_transitions/       # Hue channel transition edges
    ├── edges_individual_colors/     # Edges from individual color masks
    ├── edges_final_combined/        # Final combined color-enhanced edges
    └── waterline_within_scale/      # Waterline detection within scale bounds
```

**Advanced Color-Based Analysis:**

- **Color masks** for scale types (blue, black markings; various backgrounds)
- **HSV conversion** showing color space transformation for analysis
- **Water color masks** highlighting detected water regions
- **Enhanced edge detection** combining RGB channels and color transitions
- **Color-filtered analysis** showing masked grayscale and individual color processing

**Detection Pipeline Information:**

- **Processing parameters** including thresholds, kernel sizes, and calibration data
- **Method selection** showing which detection approach was used (integrated/forced)
- **Timing information** and performance metrics for each processing step
- **Confidence scores** for individual detection methods and final consensus
- **Scale measurement calculations** with pixel-to-cm conversions and reference points

**Configure debug options** in `config.yaml`:

```yaml
debug:
  enabled: false              # Set to true, or use DEBUG_MODE=true environment variable
  save_debug_images: true
  debug_output_dir: 'data/debug'
  steps_to_save:
    # Core steps (always available)
    - 'original'
    - 'preprocessed'
    - 'scale_detection'
    - 'edges'
    - 'contours'
    - 'gradient_analysis'
    - 'water_detection'
    - 'integrated_detection_methods'
    - 'final_result'
    
    # Enhanced steps (conditional on color detection)
    - 'hsv_conversion'
    - 'color_mask_combined'
    - 'edges_color_enhanced'
    - 'scale_contours_analysis'
```

### Water Detection Methods

The system provides three distinct water detection methods with **enhanced multi-color-space analysis** for improved accuracy, especially for clear water scenarios:

```yaml
detection:
  method: 'gradient'  # Options: 'edge', 'color', 'gradient' (recommended for clear water)
```

**Enhanced Detection Features**: Multi-color-space analysis, clear water optimization, edge-gradient fusion, and enhanced calibration integration.

**Method Selection Guide:**

- **`gradient`** **RECOMMENDED**: Best for clear water and enhanced accuracy
  - **Multi-color-space gradient analysis** across 5 color spaces (RGB, HSV, LAB, YUV, Grayscale)
  - **Clear water darkening detection** - identifies 10-40% brightness reduction patterns
  - **Edge-based validation** using horizontal gradient analysis
  - **Texture analysis** - detects water's smoothing effects
  - **Enhanced calibration integration** - uses waterline reference data
  - **High confidence scoring** based on multiple validation factors
  - **Best for**: Clear water, subtle transitions, enhanced calibration setups

- **`edge`**: Best for clear water-air interfaces with sharp contrast
  - Uses Canny edge detection with optional color enhancement
  - Reliable for typical water level detection with strong boundaries
  - Works well with various lighting conditions
  - **Best for**: Sharp water interfaces, good lighting

- **`color`**: Best for colored water (algae, sediment, specific water conditions)
  - Uses HSV color ranges to identify water regions: `water_hsv_lower/upper`
  - Ideal when water has distinct color different from background
  - Configure HSV ranges using the analyze_scale_photo.py tool
  - **Best for**: Distinctly colored water, non-clear water conditions

**Enhanced Gradient Method Details:**

The enhanced gradient method uses a comprehensive **5-layer analysis system**:

1. **Multi-Color-Space Consistency (35% weight)**:
   - Grayscale brightness analysis
   - HSV Value channel analysis  
   - LAB Lightness analysis (perceptually accurate)
   - YUV Luminance analysis
   - Validates darkening patterns across all color spaces

2. **BGR Channel Analysis (20% weight)**:
   - Individual Blue, Green, Red channel analysis
   - Compares with calibrated expected values
   - Detects subtle color shifts per channel

3. **Edge Gradient Validation (20% weight)**:
   - Prioritizes horizontal gradients (waterline characteristic)
   - Uses Sobel, Scharr, and Laplacian edge detection
   - Validates transitions across multiple scales

4. **Spatial Width Consistency (15% weight)**:
   - Ensures uniform pattern across waterline width
   - Samples multiple points along waterline
   - Validates expected darkening consistency

5. **Texture Variance Analysis (10% weight)**:
   - Detects water's surface smoothing effects
   - Analyzes texture changes above/below waterline
   - Validates expected variance reduction patterns

**Enhanced Calibration Integration:**

When using enhanced calibration data (`use_enhanced_data: true`), the gradient method gains access to:

- **Waterline reference position** for validation
- **Expected darkening ratios** specific to your setup
- **Above/below water color characteristics**
- **Calibrated brightness differences** from real measurements
- **95% base confidence** vs 80% for standard methods

**Color-Based Water Detection Setup:**
When using `method: 'color'`, configure the HSV ranges for your water:

```yaml
detection:
  method: 'color'
  water_hsv_lower: [100, 50, 50]   # Adjust based on your water color
  water_hsv_upper: [130, 255, 255] # Use analyze_scale_photo.py to calibrate
```

Use the interactive analysis tool to determine optimal HSV ranges:

```bash
python src/calibration/analyze_scale_photo.py
# Step 4: Click on water color for HSV range calibration
```

### Scale Detection and Color Options

The system uses advanced RGB/HSV color filtering to detect various types of measurement scales:

**Supported Scale Types:**

- **Yellow scales with blue markings**
- **White scales with black markings**
- **Red scales with white markings**
- **Custom color combinations** via configuration

**Color Detection Features:**

- **Multi-channel edge detection** combining RGB and HSV analysis
- **Adaptive thresholding** for varying lighting conditions  
- **Morphological operations** for noise reduction and gap filling
- **Contour scoring** based on size, aspect ratio, location, and color overlap
- **Fallback methods** when color detection fails

**Color Configuration:**
Each color is defined by HSV ranges for robust detection across lighting conditions:

```yaml
scale_colors:
  yellow:             # HSV: [20-30, 100-255, 100-255]
  white:              # HSV: [0-180, 0-30, 200-255]  
  red:                # HSV: [0-10, 100-255, 100-255]
  red_alt:            # HSV: [170-180, 100-255, 100-255] (red wraparound)
  blue:               # HSV: [100-120, 100-255, 50-255]
  black:              # HSV: [0-180, 0-255, 0-50]
  white_markings:     # HSV: [0-180, 0-20, 220-255] (high brightness, low saturation)
```

The color detection can be **enabled/disabled per color** and includes **extensive debug visualization** showing mask generation, edge enhancement, and contour analysis.

## Data Export Options

The system automatically exports measurement data in multiple formats based on your `config.yaml` settings. All exports are created in the `data/output/` directory with timestamped filenames to prevent overwrites.

### Export Configuration

```yaml
output:
  csv_export: true      # Export to CSV format
  json_export: true     # Export to JSON format  
  database: true        # Create database backups
```

### Export Formats

**CSV Export (`csv_export: true`)**

- Creates `measurements_YYYYMMDD_HHMMSS.csv` files
- Includes all measurement data in tabular format
- **New**: `detection_method` column shows which method was actually used (e.g., 'e_pattern_sequential', 'enhanced_gradient', 'standard', etc.)
- Compatible with Excel, data analysis tools
- Best for: Spreadsheet analysis, reporting, graphs, method performance tracking

**JSON Export (`json_export: true`)**

- Creates `measurements_YYYYMMDD_HHMMSS.json` files
- Structured data with proper type handling
- Easy parsing for web applications, APIs
- Best for: Integration with other systems, custom analysis

**Database Backup (`database: true`)**

- Creates `measurements_YYYYMMDD_HHMMSS.db` backup files
- Complete SQLite database copies
- Preserves all data including processing metadata
- Best for: Data archiving, system recovery

### Export Behavior

Exports occur automatically after processing with timestamped filenames, conditional format creation, and error handling.

**Example Output:**

```
data/output/
├── measurements.db                    # Main database (always present)
├── measurements_20250826_143022.csv  # CSV export (if enabled)
├── measurements_20250826_143022.json # JSON export (if enabled)  
└── measurements_20250826_143022.db   # Database backup (if enabled)
```

**Disable Exports:**
Set any format to `false` to disable:

```yaml
output:
  csv_export: false     # No CSV files created
  json_export: false    # No JSON files created
  database: false       # No database backups
```

## Project Structure

```
tide-level-img-proc-backup/
├── src/                             # Source code
│   ├── calibration/                 # Calibration tools
│   │   ├── analyze_scale_photo.py   # Interactive scale analysis tool        
│   │   └── quick_scale_analysis.py  # Quick scale analysis utility
│   ├── main.py                      # Main application entry point
│   ├── water_level_detector.py      # Core processing logic
│   ├── calibration.py               # Calibration management
│   ├── database.py                  # Database operations
│   ├── debug_visualizer.py          # Debug image generation
│   └── utils.py                     # Utility functions
├── data/                            # Data directories
│   ├── input/                       # Input images
│   ├── processed/                   # Processed images  
│   ├── calibration/                 # Calibration data
│   ├── output/                      # Export files
│   └── debug/                       # Debug images (when enabled)
├── logs/                            # Application logs
├── config.yaml                      # System configuration
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container configuration
├── docker-compose.yml               # Deployment configuration
├── .env.example                     # Environment template
├── LICENSE.md                       # License file
├── cleanup.bat                      # Windows cleanup script
├── cleanup.ps1                      # PowerShell cleanup script
└── README.md                        # Documentation
```

## Development

### Setting Up Development Environment

1. **Clone repository:**

```bash
git clone <repo-url>
cd tide-level-img-proc
```

2. **Create virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Setup development environment:**

```bash
# Create local configuration
cp .env.example .env

# Create data directories
mkdir -p data/{input,processed,calibration,output} logs
```

### Running Tests

```bash
# 1. Scale configuration analysis
python src/calibration/analyze_scale_photo.py

# 2. Run with debug logging
python src/main.py

# 3. Test GUI interface
set USE_GUI_SELECTOR=true & python src/main.py

# 4. Test calibration
set CALIBRATION_MODE=true & python src/main.py

# 5. Test color detection with debug
set DEBUG_MODE=true & python src/main.py
```

## Troubleshooting

### Common Issues

**"System not calibrated" error:**

- Ensure calibration file exists in `data/calibration/calibration.yaml`
- Check calibration file path in logs
- Run enhanced calibration: `python src/calibration/analyze_scale_photo.py` then `set CALIBRATION_MODE=true & python src/main.py`

**Scale detection completely missing/wrong:**

- **Most likely cause**: Image resizing mismatch with hardcoded coordinates
- **Immediate solution**: Disable resizing in config.yaml: `resize_width: null`
- **Root cause**: When resizing is enabled, hardcoded scale coordinates become incorrect
- **Important**: Current system uses absolute pixel coordinates that don't scale properly
- **Best practice**: Always disable resizing (`resize_width: null`) until relative coordinate system is implemented
- **Workaround**: Run `python src/calibration/analyze_scale_photo.py` to get correct coordinates for your setup

**Waterline detection inaccurate:**

- **For clear water**: Use `method: 'gradient'` in config.yaml (recommended)
- Use enhanced calibration workflow: `python src/calibration/analyze_scale_photo.py`
- Mark waterline position during calibration for reference
- Enter actual scale measurements for precise cm/pixel calculation
- Ensure scale is properly outlined even if partially underwater
- Enable enhanced calibration: `use_enhanced_data: true` in config.yaml
- **Clear water optimization**: The enhanced gradient method specifically handles water that darkens scale background

**Missing annotated images (fewer than expected):**

- **Symptom**: All images show in debug session, but only some have annotated images in `data/output/annotated/`
- **Root cause**: Previous versions only created annotated images for 100% successful detections
- **What this means**:
  - If scale detection OR water line detection failed → no annotated image created
  - Even partial success (e.g., found scale but no water line) → no annotated image
  - Debug images still created for all processed images regardless of success/failure
- **Solution**: Updated system now creates annotated images for ALL processed images:
  - `annotated_success_*.jpg` for successful detections (green lines, measurements)
  - `annotated_failed_*.jpg` for failed detections (red lines, diagnostic info)
- **Configuration**: Ensure `processing.save_processed_images: true` in config.yaml
- **Benefit**: You can now see what went wrong with failed detections instead of getting no visual feedback

**No images being processed:**

- Verify image directory permissions
- Check database path configuration
- Clear database if paths have changed: `rm data/output/measurements.db`

**Enhanced calibration tool not starting:**

- Ensure display is available (not headless environment)
- Check OpenCV GUI support: `python -c "import cv2; cv2.imshow('test', cv2.imread('image.jpg')); cv2.waitKey(1000)"`
- For WSL: Enable X11 forwarding or use Windows Python
- Run from project root directory

**GUI directory selector not appearing:**

- Ensure `USE_GUI_SELECTOR=true` in environment
- Check that `.env` file is being loaded
- Verify display environment (especially in Docker)

**Docker container won't start:**

- Check SSL certificate files exist
- Verify environment variables are set
- Check port 5000 is available

### Debug Mode

Enable detailed logging:

```bash
# Modify src/main.py temporarily
setup_logging(logging.DEBUG)
```

Enable visual debugging:

```bash
set DEBUG_MODE=true & python src/main.py
```

View logs:

```bash
tail -f logs/water_level.log
```

### Visual Debug Analysis

When troubleshooting detection issues, use visual debugging to verify scale detection, examine edge detection, validate line detection, and check final measurements.

## Deployment Options

> **Important Notice**: The Docker functionality has not been thoroughly tested and may require adjustments for production use. Docker configuration is maintained to facilitate potential future needs for dockerized deployment scenarios. For reliable operation, local installation is recommended.

### Production Docker Deployment

1. **🔴 Setup SSL certificates:**

```bash
# Generate self-signed certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Or use existing certificates
cp your-cert.pem cert.pem
cp your-key.pem key.pem
```

2. **Configure environment:**

```bash
# Production .env file
SECRET_KEY=production-secret-key
ADMIN_USER=your-username  
ADMIN_PASS=secure-password
DB_PATH=/app/data/output/measurements.db
```

3. **Deploy with Docker Compose:**

```bash
docker compose -f docker-compose.yml up -d
```

**For GUI applications in Docker (X11 forwarding):**

```bash
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix your-image
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Image Input    │───▶│  Water Level     │───▶│  Database       │
│  (Files/Upload) │    │  Detection       │    │  Storage        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Calibration     │
                       │  Management      │ 
                       └──────────────────┘
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) file for details.
